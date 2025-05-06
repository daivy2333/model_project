import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    # 去除所有 (...) 和 [...] 中的内容，包括括号本身
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    # 替换多个空格为一个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 文件夹路径
csv_dir = Path("transformer!/gpt/csv")
output_path = Path("transformer!/gpt/input.txt")

# 收集所有文本
all_texts = []

# 遍历文件夹下所有 CSV 文件
for csv_file in csv_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)
    # 自动选择合适的文本列
    column = 'transcript' if 'transcript' in df.columns else 'content' if 'content' in df.columns else None
    if column is None:
        print(f"⚠️ 跳过文件（无合适文本列）: {csv_file.name}")
        continue

    # 清洗并添加标记
    cleaned = df[column].dropna().apply(clean_text)
    marked = cleaned.apply(lambda x: f"<s>{x}<eos>")
    all_texts.extend(marked.tolist())

# 合并所有文本并保存
full_text = "\n".join(all_texts)
output_path.write_text(full_text, encoding='utf-8')
print(f"✅ 所有 TED 演讲已清洗合并到：{output_path}")
