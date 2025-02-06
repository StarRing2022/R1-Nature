import json
from openai import OpenAI
import time

def zeng_chat(usrprompt):
    API_SECRET_KEY = "XXX"
    BASE_URL = "XXX"

    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    chat_completion = client.chat.completions.create(
        model='XXX',
        messages = [
            {
                "role": "user",
                "content": usrprompt
            }
        ],
    )
    #print(chat_completion)
    answer = chat_completion.choices[0].message.content
    return answer

def modify_json_file(input_file, output_file):
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # 打开输出文件，准备写入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入 JSON 数组的开头
        outfile.write("[\n")
        
        # 遍历数据，逐条处理并写入文件
        for i, item in enumerate(data):
            # 删除 "input" 字段（如果存在）
            if "input" in item:
                del item["input"]
            # 替换 "output" 字段的内容
            if "output" in item:
                item["output"] = zeng_chat(usrprompt="你需要面对这题的内部推理内容放入到<think>...</think>，把答案放进<answer>...</answer>。问题："+item['instruction'])
            
            # 将当前项写入文件
            with open(output_file, 'a+', encoding='utf-8') as outfile:
                json.dump(item, outfile, ensure_ascii=False, indent=4)

                print("题项："+str(i+1))
            
                # 如果不是最后一项，添加逗号
                if i < len(data) - 1:
                    outfile.write(",\n")
                else:
                    outfile.write("\n")  # 最后一项后不加逗号
        
        # 写入 JSON 数组的结尾
        outfile.write("]\n")

    print(f"数据已成功从 {input_file} 修改并保存到 {output_file}")

# 示例用法
input_file = './data/alpaca_gpt4_data_zh.json'  # 输入的原始 JSON 文件路径
output_file = './data/alpaca_r1_data_zh-remote.json'  # 输出的修改后的 JSON 文件路径
modify_json_file(input_file, output_file)