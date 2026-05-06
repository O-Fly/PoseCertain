import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import os


def analyze_epistemic_csv(csv_path):
    print(f"🚀 开始分析 MC-Dropout 认知不确定性数据: {csv_path}")

    # 1. 读取数据并设置输出路径 (保存在 CSV 同级目录下)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")
    df = pd.read_csv(csv_path)

    base_dir = os.path.dirname(os.path.abspath(csv_path))
    output_dir = os.path.join(base_dir, "epistemic_analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 所有分析结果将保存在: {output_dir}")

    sns.set_theme(style="whitegrid")

    # 2. 数据准备：计算位置的总 Epistemic 标准差
    if 'Loc_Epistemic_Unc_X' in df.columns:
        df['Loc_Epistemic_Std'] = np.sqrt(
            df['Loc_Epistemic_Unc_X'] +
            df['Loc_Epistemic_Unc_Y'] +
            df['Loc_Epistemic_Unc_Z']
        )
    else:
        print("⚠️ 未找到位置不确定性列，请检查 CSV 文件。")
        return

    # 姿态的 Epistemic 不确定性对应互信息 (Mutual Information)
    has_ori_unc = 'Ori_Mutual_Info' in df.columns

    # ====================================================================
    # 维度一：稀疏化曲线 (Sparsification Plot) - 位置与姿态双轨分析
    # ====================================================================
    print("📊 [1/3] 正在生成维度一：稀疏化曲线 (Location & Orientation)...")
    fractions_retained = np.linspace(1.0, 0.1, 50)

    fig, axes = plt.subplots(1, 2 if has_ori_unc else 1, figsize=(14 if has_ori_unc else 8, 6))
    if not has_ori_unc: axes = [axes]

    # --- 位置稀疏化曲线 ---
    df_loc_unc = df.sort_values(by='Loc_Epistemic_Std', ascending=False)
    df_loc_err = df.sort_values(by='Loc_Error(m)', ascending=False)

    loc_unc_errs = [df_loc_unc['Loc_Error(m)'].iloc[int(len(df) * (1 - f)):].mean() for f in fractions_retained]
    loc_opt_errs = [df_loc_err['Loc_Error(m)'].iloc[int(len(df) * (1 - f)):].mean() for f in fractions_retained]
    loc_rnd_errs = [df['Loc_Error(m)'].sample(frac=f, random_state=42).mean() for f in fractions_retained]

    axes[0].plot(fractions_retained, loc_unc_errs, label='Drop by Epistemic (Ours)', linewidth=2.5, color='darkorange')
    axes[0].plot(fractions_retained, loc_opt_errs, label='Drop by True Error (Oracle)', linewidth=2.5, linestyle='--',
                 color='green')
    axes[0].plot(fractions_retained, loc_rnd_errs, label='Random Drop', linewidth=2.5, linestyle=':', color='gray')
    axes[0].invert_xaxis()
    axes[0].set_title('Location Sparsification Plot', fontsize=14)
    axes[0].set_xlabel('Fraction of Data Retained', fontsize=12)
    axes[0].set_ylabel('Mean Location Error (m)', fontsize=12)
    axes[0].legend()

    # --- 姿态稀疏化曲线 ---
    if has_ori_unc:
        uncertainty_metric = 'Ori_Entropy' # 你也可以试试 'Ori_Entropy'
        df_ori_unc = df.sort_values(by=uncertainty_metric, ascending=False)
        df_ori_err = df.sort_values(by='Ori_Error(deg)', ascending=False)

        ori_unc_errs = [df_ori_unc['Ori_Error(deg)'].iloc[int(len(df) * (1 - f)):].mean() for f in fractions_retained]
        ori_opt_errs = [df_ori_err['Ori_Error(deg)'].iloc[int(len(df) * (1 - f)):].mean() for f in fractions_retained]
        ori_rnd_errs = [df['Ori_Error(deg)'].sample(frac=f, random_state=42).mean() for f in fractions_retained]

        axes[1].plot(fractions_retained, ori_unc_errs, label='Drop by Epistemic (Ours)', linewidth=2.5,
                     color='darkorange')
        axes[1].plot(fractions_retained, ori_opt_errs, label='Drop by True Error (Oracle)', linewidth=2.5,
                     linestyle='--', color='green')
        axes[1].plot(fractions_retained, ori_rnd_errs, label='Random Drop', linewidth=2.5, linestyle=':', color='gray')
        axes[1].invert_xaxis()
        axes[1].set_title('Orientation Sparsification Plot', fontsize=14)
        axes[1].set_xlabel('Fraction of Data Retained', fontsize=12)
        axes[1].set_ylabel('Mean Orientation Error (deg)', fontsize=12)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_sparsification_plot.png'), dpi=300)
    plt.close()

    # ====================================================================
    # 维度二：误差解耦 (Epistemic vs. GT_Z 距离)
    # ====================================================================
    print("📊 [2/3] 正在生成维度二：误差解耦分析 (Epistemic vs GT_Z)...")
    fig, axes = plt.subplots(1, 2 if has_ori_unc else 1, figsize=(14 if has_ori_unc else 7, 6))
    if not has_ori_unc: axes = [axes]

    # 位置不确定性 vs 距离
    corr_p_loc, _ = pearsonr(df['GT_Z'], df['Loc_Epistemic_Std'])
    sns.scatterplot(data=df, x='GT_Z', y='Loc_Epistemic_Std', ax=axes[0], alpha=0.5, color='purple')
    sns.regplot(data=df, x='GT_Z', y='Loc_Epistemic_Std', ax=axes[0], scatter=False, color='black',
                line_kws={"linestyle": "--"})
    axes[0].set_title(f'Loc Epistemic vs Distance (Pearson: {corr_p_loc:.3f})', fontsize=13)
    axes[0].set_xlabel('Ground Truth Z (m)')
    axes[0].set_ylabel('Location Epistemic Std (m)')

    # 姿态不确定性 vs 距离
    if has_ori_unc:
        corr_p_ori, _ = pearsonr(df['GT_Z'], df['Ori_Mutual_Info'])
        sns.scatterplot(data=df, x='GT_Z', y='Ori_Mutual_Info', ax=axes[1], alpha=0.5, color='teal')
        sns.regplot(data=df, x='GT_Z', y='Ori_Mutual_Info', ax=axes[1], scatter=False, color='black',
                    line_kws={"linestyle": "--"})
        axes[1].set_title(f'Ori Epistemic vs Distance (Pearson: {corr_p_ori:.3f})', fontsize=13)
        axes[1].set_xlabel('Ground Truth Z (m)')
        axes[1].set_ylabel('Orientation Mutual Info')

    plt.suptitle('Epistemic Uncertainty Decoupling (Low correlation with distance is expected)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_epistemic_vs_distance.png'), dpi=300)
    plt.close()

    ## ===================================================================
    # 维度三：难例挖掘 (Top 15 Hard Examples)
    # ====================================================================
    print("📊 [3/3] 正在生成维度三：难例挖掘...")

    # 提取位置最不确定的前 15 个
    top_15_loc = df.sort_values(by='Loc_Epistemic_Std', ascending=False).head(15)
    loc_cols = ['Image_Name', 'Loc_Epistemic_Std', 'Loc_Error(m)', 'GT_Z']
    top_15_loc[loc_cols].to_csv(os.path.join(output_dir, '3_top15_hard_loc.csv'), index=False)

    # 提取姿态最不确定的前 15 个
    if has_ori_unc:
        top_15_ori = df.sort_values(by='Ori_Mutual_Info', ascending=False).head(15)
        ori_cols = ['Image_Name', 'Ori_Mutual_Info', 'Ori_Error(deg)', 'GT_Z']
        top_15_ori[ori_cols].to_csv(os.path.join(output_dir, '3_top15_hard_ori.csv'), index=False)

    print("\n✅ 所有分析完成！图表已保存至文件夹:", output_dir)


if __name__ == "__main__":
    # 替换为你的 CSV 绝对路径或相对路径
    CSV_FILE = "models/logs/speed20260413T1059/Speed_evaluation_results.csv"
    analyze_epistemic_csv(CSV_FILE)
