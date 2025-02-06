from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen2.5-0.5B-Instruct"
model_name = "Qwen2.5-1.5B-Instruct"
model_name = "Qwen2.5-3B-Instruct"
model_name = "Qwen2.5-7B-Instruct-1M"

# model_name = "Llama-3.2-1B-Instruct"
# model_name = "Llama-3.2-3B-Instruct"
# model_name = "DeepSeek-R1-Distill-Qwen-1.5B"

# model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "DeepSeek-R1-Distill-Qwen-7B"
# model_name = "DeepSeek-R1-Distill-Llama-8B"

# model_name = "./model-out/llama3.2-1b-r1"
# model_name = "./model-out/llama3.2-3b-r1"
# model_name = "./model-out/qwen2.5-0.5b-r1"
# model_name = "./model-out/qwen2.5-1.5b-r1"
model_name = "./model-out/deepqwen2.5-1.5b-r1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "介绍一下你自己" #对可解释性研究也有巨大帮助，像这种非常简单的，思考内容和输出内容是一致的
# prompt = "写一篇关于DeepNexa R1的说明书"
# prompt = "保持健康的三个提示。"
# prompt = "天上有多少颗星星?"
prompt = "如何证明爱因斯坦质能方程，要求出现数学表达式"
messages = [
    {"role": "system", "content": "你是一个由StarRing开发有用的AI助手，名为DeepNexa R1。在回答问题时，要发挥你的思维链，尽量回答。"},
    {"role": "user", "content": "你要把这题的内部推理内容放入到<think>...</think>，而将推理的答案放入到<answer>...</answer>。问题是："+prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)