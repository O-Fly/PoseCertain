import pandas as pd
import glob
import sys

# ── 配置：把这里改成你的 CSV 所在目录 ──────────────────────
CSV_PATTERN = "C:/Users/lenovo/Desktop/compare/*.csv"
# ─────────────────────────────────────────────────────────

files = glob.glob(CSV_PATTERN)

if not files:
    print("❌ 未找到任何 CSV 文件，请检查 CSV_PATTERN 路径")
    sys.exit(1)

print("========== 各文件统计结果 ==========")
for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\n📄 {f}")

        if "Loc_Error(m)" in df.columns:
            loc_median = df["Loc_Error(m)"].dropna().median()
            print(f"  Loc_Error(m)   中位数：{loc_median:.6f} m")
        else:
            print(f"  ⚠️  未找到 'Loc_Error(m)' 列")

        if "Ori_Error(deg)" in df.columns:
            ori_median = df["Ori_Error(deg)"].dropna().median()
            print(f"  Ori_Error(deg) 中位数：{ori_median:.6f} deg")
        else:
            print(f"  ⚠️  未找到 'Ori_Error(deg)' 列")

    except Exception as e:
        print(f"  ❌ 读取失败：{e}")
