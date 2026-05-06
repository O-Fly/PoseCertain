"""
A+B 实验专项诊断分析脚本
针对：同时开启 BAYESIAN_LOC + MC_DROPOUT 的 evaluate() 输出 CSV
诊断目标：找出 Z 轴误差爆炸的根本原因
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 字体设置：防止中文乱码，全部图表输出改为英文
# ============================================================
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 工具函数
# ============================================================

def safe_sqrt(series):
    """对方差列安全开根号（防止负值）"""
    return np.sqrt(np.maximum(series, 0))


def sparsification_curve(df, unc_col, err_col, fractions):
    """
    按不确定性从高到低排序，逐步剔除最不确定的样本，
    计算剩余样本的平均误差。
    返回：误差曲线列表
    """
    df_sorted = df.sort_values(by=unc_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    errors = []
    for f in fractions:
        cutoff = int(n * (1.0 - f))
        subset = df_sorted.iloc[cutoff:]
        errors.append(subset[err_col].mean())
    return errors


def oracle_curve(df, err_col, fractions):
    """按真实误差从高到低排序（Oracle上界）"""
    df_sorted = df.sort_values(by=err_col, ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    errors = []
    for f in fractions:
        cutoff = int(n * (1.0 - f))
        subset = df_sorted.iloc[cutoff:]
        errors.append(subset[err_col].mean())
    return errors


def random_curve(df, err_col, fractions, seed=42):
    """随机剔除基线"""
    rng = np.random.RandomState(seed)
    errors = []
    for f in fractions:
        idx = rng.choice(len(df), size=int(len(df) * f), replace=False)
        errors.append(df[err_col].iloc[idx].mean())
    return errors


def ause(unc_curve, oracle_curve_vals, fractions):
    """
    计算 AUSE (Area Under Sparsification Error)
    值越小说明不确定性估计越接近 Oracle
    """
    unc_arr = np.array(unc_curve)
    ora_arr = np.array(oracle_curve_vals)
    diff = unc_arr - ora_arr
    return np.trapz(diff, fractions)


# ============================================================
# 主分析函数
# ============================================================

def analyze_ab_experiment(csv_path, experiment_name="A+B_Experiment"):
    print(f"\n{'='*60}")
    print(f"  A+B 实验专项诊断: {experiment_name}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}\n")

    # ---------- 读取数据 ----------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ 数据加载成功，共 {len(df)} 条样本")
    print(f"   列名: {list(df.columns)}\n")

    # ---------- 输出目录 ----------
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    output_dir = os.path.join(base_dir, f"AB_diagnosis_{experiment_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 结果保存至: {output_dir}\n")

    sns.set_theme(style="whitegrid", font_scale=1.1)
    AXES = ['X', 'Y', 'Z']
    COLORS = {'X': '#4C72B0', 'Y': '#DD8452', 'Z': '#C44E52'}

    # ---------- 检查列是否存在 ----------
    has_aleatoric = 'Loc_Aleatoric_Unc_X' in df.columns
    has_epistemic = 'Loc_Epistemic_Unc_X' in df.columns
    has_total     = 'Loc_Total_Unc_X' in df.columns
    has_ori_unc   = 'Ori_Mutual_Info' in df.columns
    has_gt_xyz    = all(c in df.columns for c in ['GT_X', 'GT_Y', 'GT_Z'])
    has_err_xyz   = all(c in df.columns for c in ['Loc_Error_X', 'Loc_Error_Y', 'Loc_Error_Z'])

    print(f"  Aleatoric 不确定性列: {'✅' if has_aleatoric else '❌'}")
    print(f"  Epistemic 不确定性列: {'✅' if has_epistemic else '❌'}")
    print(f"  Total 不确定性列:     {'✅' if has_total else '❌'}")
    print(f"  GT XYZ 列:            {'✅' if has_gt_xyz else '❌'}")
    print(f"  Error XYZ 列:         {'✅' if has_err_xyz else '❌'}\n")

    # ---------- 衍生列 ----------
    for ax in AXES:
        if has_aleatoric:
            df[f'Aleatoric_Std_{ax}'] = safe_sqrt(df[f'Loc_Aleatoric_Unc_{ax}'])
        if has_epistemic:
            df[f'Epistemic_Std_{ax}'] = safe_sqrt(df[f'Loc_Epistemic_Unc_{ax}'])
        if has_total:
            df[f'Total_Std_{ax}'] = safe_sqrt(df[f'Loc_Total_Unc_{ax}'])

    # 3D 合成不确定性（分轴方差相加再开根，物理意义：3D 预测球的半径估计）
    if has_aleatoric:
        df['Aleatoric_Std_3D'] = safe_sqrt(
            df['Loc_Aleatoric_Unc_X'] + df['Loc_Aleatoric_Unc_Y'] + df['Loc_Aleatoric_Unc_Z']
        )
    if has_epistemic:
        df['Epistemic_Std_3D'] = safe_sqrt(
            df['Loc_Epistemic_Unc_X'] + df['Loc_Epistemic_Unc_Y'] + df['Loc_Epistemic_Unc_Z']
        )
    if has_total:
        df['Total_Std_3D'] = safe_sqrt(
            df['Loc_Total_Unc_X'] + df['Loc_Total_Unc_Y'] + df['Loc_Total_Unc_Z']
        )

    # ============================================================
    # 【诊断一】Z 轴专项：预测值 vs 真值散点图
    # 目标：判断 Z 轴误差是系统性偏移、尺度压缩还是随机噪声
    # ============================================================
    print("📊 [诊断一] Z轴预测 vs 真值散点图...")

    if has_gt_xyz and has_err_xyz:
        # evaluate() 里存的是 abs(pred - gt)，用 GT 和 |err| 做近似分析（只看误差大小分布）

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'[Diag-1] Absolute Location Error vs GT Coordinate ({experiment_name})',
                     fontsize=14, fontweight='bold')

        for i, ax_name in enumerate(AXES):
            err_col = f'Loc_Error_{ax_name}'
            gt_col  = f'GT_{ax_name}'

            ax = axes[i]
            ax.scatter(df[gt_col], df[err_col], alpha=0.3, s=8, color=COLORS[ax_name])
            ax.axhline(df[err_col].mean(), color='red', linestyle='--',
                       linewidth=1.5, label=f'Mean={df[err_col].mean():.4f}m')
            ax.axhline(df[err_col].median(), color='orange', linestyle='-.',
                       linewidth=1.5, label=f'Median={df[err_col].median():.4f}m')

            corr_p, _ = pearsonr(df[gt_col], df[err_col])
            ax.set_title(f'{ax_name}-Axis: |Error| vs GT Coord\nPearson={corr_p:.3f}', fontsize=12)
            ax.set_xlabel(f'GT_{ax_name} (m)')
            ax.set_ylabel(f'|Loc_Error_{ax_name}| (m)')
            ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_xyz_error_vs_gt.png'), dpi=300)
        plt.close()
        print("   ✅ 已保存: 1_xyz_error_vs_gt.png")

        # 各轴误差分布直方图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'[Diag-1b] Location Error Distribution per Axis ({experiment_name})',
                     fontsize=13)
        for i, ax_name in enumerate(AXES):
            err_col = f'Loc_Error_{ax_name}'
            axes[i].hist(df[err_col], bins=60, color=COLORS[ax_name],
                         edgecolor='white', alpha=0.8)
            axes[i].axvline(df[err_col].mean(), color='red', linestyle='--',
                            linewidth=2, label=f'Mean={df[err_col].mean():.4f}m')
            axes[i].axvline(df[err_col].median(), color='orange', linestyle='-.',
                            linewidth=2, label=f'Median={df[err_col].median():.4f}m')
            axes[i].set_title(f'{ax_name}-Axis Error Distribution (std={df[err_col].std():.4f}m)')
            axes[i].set_xlabel('Absolute Error (m)')
            axes[i].set_ylabel('Count')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1b_xyz_error_histogram.png'), dpi=300)
        plt.close()
        print("   ✅ 已保存: 1b_xyz_error_histogram.png\n")

    # ============================================================
    # 【诊断二】Aleatoric vs Epistemic 不确定性对比（分轴）
    # 目标：判断两种不确定性在 Z 轴上是否异常虚高
    # ============================================================
    print("📊 [诊断二] Aleatoric vs Epistemic 不确定性分布对比...")

    if has_aleatoric and has_epistemic:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'[Diag-2] Aleatoric vs Epistemic Uncertainty Distribution ({experiment_name})',
                     fontsize=14, fontweight='bold')

        for i, ax_name in enumerate(AXES):
            ale_col = f'Aleatoric_Std_{ax_name}'
            epi_col = f'Epistemic_Std_{ax_name}'

            # 上行：分布直方图（对数坐标更容易看出尾部）
            axes[0, i].hist(df[ale_col], bins=50, alpha=0.6, color='steelblue',
                            label='Aleatoric Std', density=True)
            axes[0, i].hist(df[epi_col], bins=50, alpha=0.6, color='tomato',
                            label='Epistemic Std', density=True)
            axes[0, i].set_title(f'{ax_name}-Axis Uncertainty Distribution')
            axes[0, i].set_xlabel('Std (m)')
            axes[0, i].set_ylabel('Density (log scale)')
            axes[0, i].legend()
            axes[0, i].set_yscale('log')

            ale_mean = df[ale_col].mean()
            epi_mean = df[epi_col].mean()
            axes[0, i].axvline(ale_mean, color='steelblue', linestyle='--', linewidth=1.5)
            axes[0, i].axvline(epi_mean, color='tomato',    linestyle='--', linewidth=1.5)
            axes[0, i].text(0.98, 0.95,
                            f'Ale mean={ale_mean:.4f}\nEpi mean={epi_mean:.4f}',
                            transform=axes[0, i].transAxes,
                            ha='right', va='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 下行：Aleatoric vs Epistemic 散点图（看两者是否相关）
            axes[1, i].scatter(df[ale_col], df[epi_col], alpha=0.2, s=6,
                               color=COLORS[ax_name])
            corr_p, _ = pearsonr(df[ale_col], df[epi_col])
            axes[1, i].set_title(f'{ax_name}-Axis: Aleatoric vs Epistemic\nPearson={corr_p:.3f}')
            axes[1, i].set_xlabel('Aleatoric Std (m)')
            axes[1, i].set_ylabel('Epistemic Std (m)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_aleatoric_vs_epistemic.png'), dpi=300)
        plt.close()
        print("   ✅ 已保存: 2_aleatoric_vs_epistemic.png\n")

    # ============================================================
    # 【诊断三】不确定性 vs 真实误差相关性（核心有效性验证）
    # 目标：验证不确定性是否真的能预测误差大小
    # ============================================================
    print("📊 [诊断三] 不确定性 vs 真实误差相关性...")

    if has_err_xyz:
        unc_types = {}
        if has_aleatoric:
            unc_types['Aleatoric'] = {ax: f'Aleatoric_Std_{ax}' for ax in AXES}
        if has_epistemic:
            unc_types['Epistemic'] = {ax: f'Epistemic_Std_{ax}' for ax in AXES}
        if has_total:
            unc_types['Total']     = {ax: f'Total_Std_{ax}'     for ax in AXES}

        n_types = len(unc_types)
        if n_types > 0:
            fig, axes = plt.subplots(n_types, 3, figsize=(18, 5 * n_types))
            if n_types == 1:
                axes = axes.reshape(1, 3)
            fig.suptitle(f'[Diag-3] Uncertainty vs True Error ({experiment_name})',
                         fontsize=14, fontweight='bold')

            corr_summary = {}
            for row_idx, (unc_name, unc_cols) in enumerate(unc_types.items()):
                corr_summary[unc_name] = {}
                for col_idx, ax_name in enumerate(AXES):
                    unc_col = unc_cols[ax_name]
                    err_col = f'Loc_Error_{ax_name}'

                    ax = axes[row_idx, col_idx]
                    ax.scatter(df[unc_col], df[err_col], alpha=0.25, s=6,
                               color=COLORS[ax_name])

                    sp_corr, _ = spearmanr(df[unc_col], df[err_col])
                    pe_corr, _ = pearsonr(df[unc_col],  df[err_col])
                    corr_summary[unc_name][ax_name] = {
                        'Spearman': sp_corr, 'Pearson': pe_corr
                    }

                    ax.set_title(f'{unc_name} {ax_name}-Axis\n'
                                 f'Spearman={sp_corr:.3f}  Pearson={pe_corr:.3f}',
                                 fontsize=11)
                    ax.set_xlabel(f'{unc_name} Std {ax_name} (m)')
                    ax.set_ylabel(f'True Error {ax_name} (m)')

                    # 颜色提示：相关性好坏（绿=好，橙=边缘，红=差）
                    bg_color = '#e8f5e9' if sp_corr > 0.3 else ('#fff3e0' if sp_corr > 0.1 else '#ffebee')
                    ax.set_facecolor(bg_color)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '3_uncertainty_vs_error.png'), dpi=300)
            plt.close()
            print("   ✅ 已保存: 3_uncertainty_vs_error.png")

            # 打印相关性汇总表
            print("\n   📋 相关性汇总（Spearman）:")
            print(f"   {'Type':<12} {'X-Axis':>10} {'Y-Axis':>10} {'Z-Axis':>10}")
            print(f"   {'-'*44}")
            for unc_name, axes_corr in corr_summary.items():
                row = f"   {unc_name:<12}"
                for ax_name in AXES:
                    val = axes_corr[ax_name]['Spearman']
                    flag = '✅' if val > 0.3 else ('⚠️' if val > 0.1 else '❌')
                    row += f" {val:>8.3f}{flag}"
                print(row)
            print()

    # ============================================================
    # 【诊断四】稀疏化曲线对比（三种不确定性 + 分轴）
    # 目标：评估哪种不确定性对误差的排序能力最强
    # ============================================================
    print("📊 [诊断四] 稀疏化曲线对比（Aleatoric / Epistemic / Total）...")

    fractions = np.linspace(1.0, 0.1, 50)

    # 4-1: 3D 总误差的稀疏化曲线
    fig, ax = plt.subplots(figsize=(9, 6))
    oracle_3d = oracle_curve(df, 'Loc_Error(m)', fractions)
    random_3d = random_curve(df, 'Loc_Error(m)', fractions)
    ax.plot(fractions, oracle_3d, 'g--', linewidth=2.5, label='Oracle (True Error)')
    ax.plot(fractions, random_3d, 'k:',  linewidth=2.0, label='Random Baseline')

    ause_results = {}
    if has_aleatoric:
        ale_3d = sparsification_curve(df, 'Aleatoric_Std_3D', 'Loc_Error(m)', fractions)
        ax.plot(fractions, ale_3d, color='steelblue', linewidth=2.5, label='Aleatoric (logvar head)')
        ause_results['Aleatoric'] = ause(ale_3d, oracle_3d, fractions)
    if has_epistemic:
        epi_3d = sparsification_curve(df, 'Epistemic_Std_3D', 'Loc_Error(m)', fractions)
        ax.plot(fractions, epi_3d, color='tomato', linewidth=2.5, label='Epistemic (MC Dropout)')
        ause_results['Epistemic'] = ause(epi_3d, oracle_3d, fractions)
    if has_total:
        tot_3d = sparsification_curve(df, 'Total_Std_3D', 'Loc_Error(m)', fractions)
        ax.plot(fractions, tot_3d, color='purple', linewidth=2.5, label='Total (Ale + Epi)')
        ause_results['Total'] = ause(tot_3d, oracle_3d, fractions)

    ax.invert_xaxis()
    ax.set_title(f'[Diag-4a] 3D Location Error Sparsification Curve ({experiment_name})',
                 fontsize=13)
    ax.set_xlabel('Fraction of Data Retained')
    ax.set_ylabel('Mean Location Error (m)')
    ax.legend()

    if ause_results:
        ause_text = '\n'.join([f'AUSE({k})={v:.4f}' for k, v in ause_results.items()])
        ax.text(0.02, 0.98, ause_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4a_sparsification_3d.png'), dpi=300)
    plt.close()
    print("   ✅ 已保存: 4a_sparsification_3d.png")

    # 4-2: Z 轴专项稀疏化曲线（最关键）
    if has_err_xyz:
        fig, ax = plt.subplots(figsize=(9, 6))
        oracle_z = oracle_curve(df, 'Loc_Error_Z', fractions)
        random_z = random_curve(df, 'Loc_Error_Z', fractions)
        ax.plot(fractions, oracle_z, 'g--', linewidth=2.5, label='Oracle')
        ax.plot(fractions, random_z, 'k:',  linewidth=2.0, label='Random')

        ause_z = {}
        if has_aleatoric:
            ale_z = sparsification_curve(df, 'Aleatoric_Std_Z', 'Loc_Error_Z', fractions)
            ax.plot(fractions, ale_z, color='steelblue', linewidth=2.5, label='Aleatoric Z')
            ause_z['Aleatoric_Z'] = ause(ale_z, oracle_z, fractions)
        if has_epistemic:
            epi_z = sparsification_curve(df, 'Epistemic_Std_Z', 'Loc_Error_Z', fractions)
            ax.plot(fractions, epi_z, color='tomato', linewidth=2.5, label='Epistemic Z')
            ause_z['Epistemic_Z'] = ause(epi_z, oracle_z, fractions)
        if has_total:
            tot_z = sparsification_curve(df, 'Total_Std_Z', 'Loc_Error_Z', fractions)
            ax.plot(fractions, tot_z, color='purple', linewidth=2.5, label='Total Z')
            ause_z['Total_Z'] = ause(tot_z, oracle_z, fractions)

        ax.invert_xaxis()
        ax.set_title(f'[Diag-4b] Z-Axis Error Sparsification Curve ({experiment_name})',
                     fontsize=13)
        ax.set_xlabel('Fraction of Data Retained')
        ax.set_ylabel('Mean Z Error (m)')
        ax.legend()

        if ause_z:
            ause_text = '\n'.join([f'AUSE({k})={v:.4f}' for k, v in ause_z.items()])
            ax.text(0.02, 0.98, ause_text, transform=ax.transAxes,
                    va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4b_sparsification_z_axis.png'), dpi=300)
        plt.close()
        print("   ✅ 已保存: 4b_sparsification_z_axis.png\n")

    # ============================================================
    # 【诊断五】Z 轴不确定性 vs GT_Z 距离关系
    # 目标：判断 Z 轴不确定性是否随距离单调增大（合理）
    #        还是与距离无关（方差头失效）
    # ============================================================
    print("📊 [诊断五] Z轴不确定性 vs 距离关系...")

    if has_gt_xyz:
        unc_z_cols = {}
        if has_aleatoric: unc_z_cols['Aleatoric'] = 'Aleatoric_Std_Z'
        if has_epistemic: unc_z_cols['Epistemic'] = 'Epistemic_Std_Z'
        if has_total:     unc_z_cols['Total']     = 'Total_Std_Z'

        n_cols = len(unc_z_cols)
        if n_cols > 0:
            fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
            if n_cols == 1:
                axes = [axes]
            fig.suptitle(f'[Diag-5] Z-Axis Uncertainty vs Target Distance ({experiment_name})',
                         fontsize=13)

            for idx, (unc_name, unc_col) in enumerate(unc_z_cols.items()):
                axes[idx].scatter(df['GT_Z'], df[unc_col], alpha=0.3, s=8, color=COLORS['Z'])
                sns.regplot(data=df, x='GT_Z', y=unc_col, ax=axes[idx],
                            scatter=False, color='black',
                            line_kws={"linestyle": "--", "linewidth": 1.5})
                corr_p, _ = pearsonr(df['GT_Z'],  df[unc_col])
                corr_s, _ = spearmanr(df['GT_Z'], df[unc_col])
                axes[idx].set_title(f'{unc_name} Z-Axis Uncertainty vs GT_Z\n'
                                    f'Pearson={corr_p:.3f}  Spearman={corr_s:.3f}')
                axes[idx].set_xlabel('GT_Z (m) - Target Distance')
                axes[idx].set_ylabel(f'{unc_name} Std Z (m)')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '5_z_uncertainty_vs_distance.png'), dpi=300)
            plt.close()
            print("   ✅ 已保存: 5_z_uncertainty_vs_distance.png\n")

    # ============================================================
    # 【汇总报告】打印关键统计数据
    # ============================================================
    print("\n" + "="*60)
    print(f"  📋 关键统计汇总 ({experiment_name})")
    print("="*60)

    print(f"\n  ▶ 整体误差:")
    print(f"    Loc Error (3D) — Mean: {df['Loc_Error(m)'].mean():.4f}m  "
          f"Median: {df['Loc_Error(m)'].median():.4f}m  "
          f"Std: {df['Loc_Error(m)'].std():.4f}m")
    print(f"    Ori Error      — Mean: {df['Ori_Error(deg)'].mean():.4f}deg  "
          f"Median: {df['Ori_Error(deg)'].median():.4f}deg")

    if has_err_xyz:
        print(f"\n  ▶ 分轴误差:")
        for ax_name in AXES:
            col = f'Loc_Error_{ax_name}'
            print(f"    {ax_name}轴: Mean={df[col].mean():.4f}m  "
                  f"Median={df[col].median():.4f}m  "
                  f"Std={df[col].std():.4f}m  "
                  f"Max={df[col].max():.4f}m")

    if has_aleatoric:
        print(f"\n  ▶ Aleatoric 不确定性 (logvar头):")
        for ax_name in AXES:
            col = f'Aleatoric_Std_{ax_name}'
            print(f"    {ax_name}轴: Mean={df[col].mean():.4f}  "
                  f"Median={df[col].median():.4f}  "
                  f"Max={df[col].max():.4f}")

    if has_epistemic:
        print(f"\n  ▶ Epistemic 不确定性 (MC Dropout):")
        for ax_name in AXES:
            col = f'Epistemic_Std_{ax_name}'
            print(f"    {ax_name}轴: Mean={df[col].mean():.4f}  "
                  f"Median={df[col].median():.4f}  "
                  f"Max={df[col].max():.4f}")

    if ause_results:
        print(f"\n  ▶ AUSE (越小越好，越接近Oracle):")
        for k, v in ause_results.items():
            print(f"    {k}: {v:.6f}")

    print("\n" + "="*60)
    print(f"  ✅ 全部分析完成！共生成图表保存至:")
    print(f"     {output_dir}")
    print("="*60 + "\n")

    return df


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # A+B 实验的 CSV 路径
    CSV_AB = "models/logs/speed20260421T1251/Speed_evaluation_results.csv"
    df_result = analyze_ab_experiment(CSV_AB, experiment_name="AB_BayesianLoc_MCDropout")
