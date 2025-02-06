import json

# 打开原始的JSON文件
input_file_path = "./data/alpaca_r1_data_zh-local.json"  # 替换为你的原始文件路径
output_file_path = "./data/alpaca_r1_data_zh-localpost.json"  # 替换为你想要保存的新文件路径

# 读取JSON文件
with open(input_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 遍历数据并修改"output"字段
for item in data:
    if "output" in item:
        # 找到</think>\n\n并插入<answer>
        modified_output = item["output"].replace("</think>\n\n", "</think>\n\n<answer>")
        # 在内容最后加上</answer>
        modified_output += "</answer>"
        item["output"] = modified_output

# 将修改后的数据保存到新的JSON文件
with open(output_file_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("处理完成，新文件已保存到", output_file_path)