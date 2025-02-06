import json

def jsonl_to_json(input_file, output_file):
    # 打开输入的.jsonl文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        # 逐行读取并解析每行的JSON对象
        data = [json.loads(line) for line in infile]
    
    
    for item in data:
        item['instruction'] = item.pop('prompt')

        item['response'] = item['response'].replace("<Thought>","<think>")
        item['response'] = item['response'].replace("</Thought>","</think>")
        item['response'] = item['response'].replace("<Output>","<answer>")
        item['response'] = item['response'].replace("</Output>","</answer>")
        #print(item['response'])
        item['output'] = item.pop('response')


    # 将解析后的数据保存为一个JSON文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"数据已成功从 {input_file} 转换并保存到 {output_file}")

# 示例用法
input_file = './data/openo1-SFT.jsonl'  # 输入的.jsonl文件路径
output_file = './data/openr1-SFT.json'   # 输出的JSON文件路径
jsonl_to_json(input_file, output_file)