import os
import re
import pandas as pd
from glob import glob

BASE_DIR = "/home/wanglin/Projects/onerec/onerec_v2/DRAGON/ablation/results"
OUTPUT_CSV = os.path.join(BASE_DIR, "ablation_summary.csv")
target_k = 10


def parse_result_file(path):
    result = {}
    name = os.path.basename(path).replace(".txt", "")

    # === 文件名解析 ===
    m_dataset = re.search(r"DRAGON_([a-zA-Z0-9]+)", name)
    result["dataset"] = m_dataset.group(1) if m_dataset else "unknown"

    # 提取 ablation 参数配置
    for key in ["homo", "div", "align", "res"]:
        m = re.search(fr"{key}-([A-Za-z]+)", name, re.IGNORECASE)
        result[key] = m.group(1) if m else None

    # === 读取内容 ===
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # === Test 部分匹配 ===
    test_block = re.search(r"Test:\s*([\s\S]+?)config_dict", content)
    if not test_block:
        print(f"⚠️ 无法在文件中找到 Test 部分: {path}")
        return None

    test_block = test_block.group(1)

    # === 提取指标 ===
    metrics = ["map", "ndcg", "precision", "recall"]
    ks = [1, 3, 5, 10, 20, 50]
    for metric in metrics:
        line = re.search(rf"{metric}\s+([0-9.\s]+)", test_block)
        if line:
            numbers = line.group(1).split()
            for i, v in enumerate(numbers[:len(ks)], start=1):
                result[f"{metric}@{ks[i-1]}"] = float(v)

    return result or None


# === 扫描所有结果文件 ===
records = []
for path in sorted(glob(os.path.join(BASE_DIR, "*.txt"))):
    parsed = parse_result_file(path)
    if parsed:
        records.append(parsed)

print(f"🧾 共解析成功 {len(records)} 个文件")
if not records:
    raise ValueError("❌ 没有成功解析任何结果文件，请检查路径或文件格式。")

# === 转数据表 ===
df = pd.DataFrame(records)
print("表头：", df.columns.tolist())

# ✅ 配置列（去掉 filename，只保留 dataset）
config_cols = ["dataset", "homo", "div", "align", "res"]
metric_cols = [c for c in df.columns if c not in config_cols]

# ✅ 自定义排序函数（保证 @1,@3,@5,@10,@20,@50 顺序）
def sort_key(col):
    m = re.match(r"([a-zA-Z_]+)@(\d+)", col)
    if not m:
        return (col, 0)
    metric, num = m.groups()
    order_map = {"map": 0, "ndcg": 1, "precision": 2, "recall": 3}
    return (order_map.get(metric, 99), int(num))

metric_cols_sorted = sorted(metric_cols, key=sort_key)
df = df[config_cols + metric_cols_sorted]



# === 只显示每个 metric 在 @k 的结果 ===


# 基础配置列
config_cols = ["dataset", "homo", "div", "align", "res"]

# 选出包含指定 @k 的所有列
metric_cols_k = [c for c in df.columns if f"@{target_k}" in c]

# 重新组成 DataFrame
df_k = df[config_cols + metric_cols_k]


# === 保存输出 ===
df_k.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 已保存到: {OUTPUT_CSV}")
print(df_k.head())







# === 🎨 可视化整个表格为带颜色的图片 ===
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns
import numpy as np

# === [5] 🎨 生成带热力颜色的表格 ===

print("🎨 正在生成每个数据集的热力表...")

df_vis_master = df_k.copy()
for col in ["homo", "div", "align", "res"]:
    df_vis_master[col] = df_vis_master[col].map({True: "✔️", "True": "✔️", False: "×", "False": "×"})

info_cols = ["dataset", "homo", "div", "align", "res"]
metric_cmaps = {
    "map": "Blues",
    "ndcg": "Greens",
    "precision": "Oranges",
    "recall": "Purples",
}
metric_cols = [c for c in df_k.columns if "@" in c]

# 对每个数据集单独绘制
for dataset_name, df_sub in df_k.groupby("dataset"):
    print(f"  ➜ 数据集: {dataset_name}, 样本数: {len(df_sub)}")

    df_vis = df_vis_master[df_vis_master["dataset"] == dataset_name].copy()

    # 🎨 针对该数据集计算颜色梯度（独立归一化）
    color_blocks = []
    for m in metric_cols:
        metric_type = m.split("@")[0]
        cmap = sns.color_palette(metric_cmaps.get(metric_type, "Greys"), as_cmap=True)
        values = df_sub[m].values
        vmin, vmax = values.min(), values.max()
        norm = (values - vmin) / (vmax - vmin + 1e-12)
        hex_colors = [to_hex(cmap(v)) for v in norm]
        color_blocks.append(hex_colors)
    color_df = pd.DataFrame({m: color_blocks[i] for i, m in enumerate(metric_cols)}).T.T
    color_df.index = df_sub.index

    # === 建立表格 ===
    plt.figure(figsize=(len(df_sub.columns) * 1.2, 0.35 * len(df_sub)))
    cell_text = df_vis[df_vis["dataset"] == dataset_name].values
    col_labels = df_vis.columns.tolist()

    table = plt.table(cellText=cell_text,
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    # 无边框
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)

    # 调整列宽
    col_widths = {
        "dataset": 1.2, "homo": 0.5, "div": 0.5,
        "align": 0.5, "res": 0.5,
    }
    default_width = 1.0
    for (r, c), cell in table.get_celld().items():
        if c < len(col_labels):
            col_name = col_labels[c]
            factor = col_widths.get(col_name, default_width)
            cell.set_width(cell.get_width() * factor)

    # 按行上色
    col_indices = {c: i for i, c in enumerate(col_labels)}
    for r in range(len(df_vis[df_vis["dataset"] == dataset_name])):
        for m in metric_cols:
            val = df_sub.loc[df_sub.index[r], m]
            color = color_df.loc[df_sub.index[r], m]
            table[(r + 1, col_indices[m])].set_facecolor(color)
            text_color = "white" if np.mean(sns.color_palette([color])[0]) < 0.5 else "black"
            table[(r + 1, col_indices[m])].get_text().set_color(text_color)
            table[(r + 1, col_indices[m])].get_text().set_text(f"{val:.4f}")

    plt.axis("off")
    plt.title(f"{dataset_name.upper()} -- Ablation Performance @ {target_k}",
              fontsize=14, weight="bold", pad=20)

    output_pdf = os.path.join(BASE_DIR, f"ablation_heatmap_{dataset_name}_at{target_k}.pdf")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()
    print(f"    ✅ 已生成: {output_pdf}")

print("🎯 所有数据集热力图生成完毕。")