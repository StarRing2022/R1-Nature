import pyarrow.parquet as pq
import json
import pandas as pd


def parquet_to_json(input_file, output_file):
    # 读取 Parquet 文件
    df = pd.read_parquet(input_file)
    # 将 DataFrame 转换为 JSON 格式并保存到文件
    if "input" in df.columns:
        df.drop(columns=["input"], inplace=True)
    if "output" in df.columns:
        df["output"] = df["output"].str.replace("<thought>", "<think>")
        df["output"] = df["output"].str.replace("<\/thought>", "</think>")
       
    df.to_json(output_file, orient='records', force_ascii=False, indent=4)

    print(f"数据已成功从 {input_file} 转换并保存到 {output_file}")

# 示例用法
input_file = './data/magpie-reason-train-00000-of-00001.parquet'  # 输入的 Parquet 文件路径
output_file = './data/magpie-r1.json'     # 输出的 JSON 文件路径
parquet_to_json(input_file, output_file)