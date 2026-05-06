#只针对打开bayesian_loc但是关闭MC_Dropout的evaluate出来的csv文件，分析位置不确定性数据的有效性
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os


def analyze_uncertainty(csv_path):
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. 确保结果保存在输入的 CSV 同级目录下
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    output_dir = os.path.join(base_dir, "uncertainty_analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved to: {output_dir}\n")

    sns.set_theme(style="whitegrid")

    # 计算总的 3D 空间不确定性 (标准差 Std) 和 总误差
    # 方差(Variance)相加后开根号得到总标准差(Std)
    df['Total_Aleatoric_Std'] = np.sqrt(
        df['Loc_Aleatoric_Unc_X'] +
        df['Loc_Aleatoric_Unc_Y'] +
        df['Loc_Aleatoric_Unc_Z']
    )

    # ==========================================
    # 图 1: X, Y, Z 轴的 Error vs Uncertainty (散点图)
    # 意义：验证网络在各个轴上输出的不确定性是否与真实误差正相关
    # ==========================================
    print("[1/4] Generating XYZ Error vs Uncertainty plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes_info = [('X', 'Loc_Aleatoric_Unc_X', 'Loc_Error_X'),
                 ('Y', 'Loc_Aleatoric_Unc_Y', 'Loc_Error_Y'),
                 ('Z', 'Loc_Aleatoric_Unc_Z', 'Loc_Error_Z')]

    for i, (axis_name, unc_col, err_col) in enumerate(axes_info):
        # 将方差转为标准差(Std)，使其与误差(m)量纲一致
        std_val = np.sqrt(df[unc_col])
        sns.scatterplot(x=std_val, y=df[err_col], ax=axes[i], alpha=0.5, s=20, color='blue')

        # 计算斯皮尔曼相关系数
        corr, _ = spearmanr(std_val, df[err_col])

        axes[i].set_title(f'{axis_name}-Axis: Error vs Uncertainty\nSpearman Corr: {corr:.3f}')
        axes[i].set_xlabel(f'Predicted Std {axis_name} (m)')
        axes[i].set_ylabel(f'True Error {axis_name} (m)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_uncertainty_xyz.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 2: X, Y, Z 轴的 真实坐标(GT) vs 不确定性 (散点图)
    # 意义：分析目标在视野边缘(X,Y偏大)或距离极远(Z偏大)时，不确定性如何变化
    # ==========================================
    print("[2/4] Generating XYZ Ground Truth vs Uncertainty plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    gt_info = [('X', 'GT_X', 'Loc_Aleatoric_Unc_X'),
               ('Y', 'GT_Y', 'Loc_Aleatoric_Unc_Y'),
               ('Z', 'GT_Z', 'Loc_Aleatoric_Unc_Z')]

    for i, (axis_name, gt_col, unc_col) in enumerate(gt_info):
        std_val = np.sqrt(df[unc_col])
        sns.scatterplot(x=df[gt_col], y=std_val, ax=axes[i], alpha=0.5, s=20, color='green')

        axes[i].set_title(f'{axis_name}-Axis: GT Coordinate vs Uncertainty')
        axes[i].set_xlabel(f'Ground Truth {axis_name} (m)')
        axes[i].set_ylabel(f'Predicted Std {axis_name} (m)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gt_vs_uncertainty_xyz.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 3: 稀疏化曲线 (Sparsification Plot)
    # 意义：评估不确定性排序的质量。蓝线越贴近绿线，说明不确定性估计越完美。
    # ==========================================
    print("[3/4] Generating Sparsification Plot...")
    fractions_retained = np.linspace(1.0, 0.1, 50)

    # 按照预测的不确定性剔除 (Ours)
    df_sorted_by_unc = df.sort_values(by='Total_Aleatoric_Std', ascending=False)
    unc_errors = [df_sorted_by_unc['Loc_Error(m)'].iloc[int(len(df) * (1 - frac)):].mean() for frac in
                  fractions_retained]

    # 按照真实的误差剔除 (Optimal/Oracle)
    df_sorted_by_err = df.sort_values(by='Loc_Error(m)', ascending=False)
    opt_errors = [df_sorted_by_err['Loc_Error(m)'].iloc[int(len(df) * (1 - frac)):].mean() for frac in
                  fractions_retained]

    # 随机剔除 (Random Baseline)
    rand_errors = [df['Loc_Error(m)'].sample(frac=frac, random_state=42).mean() for frac in fractions_retained]

    plt.figure(figsize=(8, 6))
    plt.plot(fractions_retained, unc_errors, label='Drop by Predicted Uncertainty (Ours)', linewidth=2.5, color='blue')
    plt.plot(fractions_retained, opt_errors, label='Drop by True Error (Optimal)', linewidth=2.5, linestyle='--',
             color='green')
    plt.plot(fractions_retained, rand_errors, label='Random Drop', linewidth=2.5, linestyle=':', color='gray')

    plt.gca().invert_xaxis()  # X轴反转 (1.0 -> 0.1)
    plt.title('Sparsification Plot (Error vs Data Retained)')
    plt.xlabel('Fraction of Data Retained')
    plt.ylabel('Mean Location Error (m)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'sparsification_plot.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 4: X, Y, Z 轴不确定性分布 (箱线图 Boxplot)
    # 意义：直观展示 Z 轴(深度)的不确定性在整体分布上远大于 X 和 Y 轴
    # ==========================================
    print("[4/4] Generating XYZ Uncertainty Boxplot...")
    plt.figure(figsize=(8, 6))

    # 将方差转换为标准差(Std)用于展示，量纲为米(m)，更直观
    df['Std_X'] = np.sqrt(df['Loc_Aleatoric_Unc_X'])
    df['Std_Y'] = np.sqrt(df['Loc_Aleatoric_Unc_Y'])
    df['Std_Z'] = np.sqrt(df['Loc_Aleatoric_Unc_Z'])

    # 宽表转长表
    df_melted = df.melt(value_vars=['Std_X', 'Std_Y', 'Std_Z'],
                        var_name='Axis', value_name='Predicted_Std')

    sns.boxplot(data=df_melted, x='Axis', y='Predicted_Std', showfliers=False, palette="Set2")
    plt.title('Uncertainty (Std) Distribution across X, Y, Z axes')
    plt.ylabel('Predicted Standard Deviation (m)')
    plt.xlabel('Spatial Axis')
    plt.savefig(os.path.join(output_dir, 'uncertainty_boxplot.png'), dpi=300)
    plt.close()

    # ==========================================
    # 难例挖掘 (Hard Example Mining)
    # ==========================================
    print("\n" + "=" * 60)
    print("🎯 HARD EXAMPLE MINING (Top 15 Most Uncertain Images)")
    print("=" * 60)
    top_hard_examples = df_sorted_by_unc.head(15)
    print(top_hard_examples[['Image_Name', 'Loc_Error(m)', 'Total_Aleatoric_Std', 'GT_Z']].to_string(index=False))

    top_hard_examples.to_csv(os.path.join(output_dir, 'hard_examples.csv'), index=False)
    print(f"\n[SUCCESS] Analysis complete! All plots saved to: {output_dir}")


if __name__ == "__main__":
    # 替换为你实际生成的 CSV 文件名或绝对路径
    CSV_FILE_PATH = "models/logs/speed20260403T1849/Speed_evaluation_results_xyzall.csv"
    if os.path.exists(CSV_FILE_PATH):
        analyze_uncertainty(CSV_FILE_PATH)
    else:
        print(f"Error: Could not find {CSV_FILE_PATH}. Please check the path.")
