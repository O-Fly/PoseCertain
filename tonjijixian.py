import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────
CSV_PATH = "models/logs/speed20260402T1259/Speed_evaluation_results.csv"  # ← 改为你的实际路径

COL_LOC = "Loc_Error(m)"
COL_ORI = "Ori_Error(deg)"
COL_ESA = "ESA_Score"

# ─────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

loc = df[COL_LOC].dropna()
ori = df[COL_ORI].dropna()
esa = df[COL_ESA].dropna()

# ─────────────────────────────────────────
# 2. 计算统计指标
# ─────────────────────────────────────────
stats = {
    "位置误差 / m":  [loc.mean(), loc.median(), loc.std()],
    "姿态误差 / deg": [ori.mean(), ori.median(), ori.std()],
    "ESA Score":     [esa.mean(), esa.median(), esa.std()],
}

stats_df = pd.DataFrame(stats, index=["均值", "中位数", "标准差"]).T
stats_df = stats_df.round(4)

print("=" * 55)
print("        表 3.6  基线模型在测试集上的误差统计结果")
print("=" * 55)
print(f"{'指标':<16}{'均值':>10}{'中位数':>10}{'标准差':>10}")
print("-" * 55)
for name, row in stats_df.iterrows():
    print(f"{name:<16}{row['均值']:>10.4f}{row['中位数']:>10.4f}{row['标准差']:>10.4f}")
print("=" * 55)

# ─────────────────────────────────────────
# 3. 绘制误差分布直方图
# ─────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "SimHei",   # 支持中文
    "axes.unicode_minus": False,
    "font.size":         11,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("基线模型位置误差与姿态误差分布图", fontsize=14, fontweight="bold", y=1.01)

def plot_hist(ax, data, xlabel, color, title):
    n_bins = min(20, max(10, len(data) // 5))
    counts, edges, patches = ax.hist(
        data, bins=n_bins, color=color, edgecolor="white",
        linewidth=0.6, alpha=0.85, zorder=2
    )
    mean_val   = data.mean()
    median_val = data.median()
    std_val    = data.std()

    # 均值线
    ax.axvline(mean_val,   color="#d62728", linestyle="--", linewidth=1.8,
               label=f"均值 = {mean_val:.4f}", zorder=3)
    # 中位数线
    ax.axvline(median_val, color="#2ca02c", linestyle=":",  linewidth=1.8,
               label=f"中位数 = {median_val:.4f}", zorder=3)
    # ±1 std 阴影
    ax.axvspan(mean_val - std_val, mean_val + std_val,
               alpha=0.12, color="#d62728", label=f"±1σ = {std_val:.4f}", zorder=1)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

plot_hist(axes[0], loc, "位置误差 / m",   "#4C72B0", "(a) 位置误差分布")
plot_hist(axes[1], ori, "姿态误差 / deg", "#DD8452", "(b) 姿态误差分布")

plt.tight_layout()

# ─────────────────────────────────────────
# 4. 保存图片
# ─────────────────────────────────────────
OUTPUT_FIG = r"C:\Users\lenovo\Desktop\Pose Estimation\Ursonet\UrsoNet-master\error_distribution.png"
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
print(f"\n图表已保存至：{OUTPUT_FIG}")
plt.show()