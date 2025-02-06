import torch
from peft import PeftModel
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer



device = "cuda"
#model_dir = 'Llama-3.2-1B-Instruct'
#model_dir = 'Llama-3.2-3B-Instruct'
#model_dir = 'Qwen2.5-0.5B-Instruct'
#model_dir = 'Qwen2.5-1.5B-Instruct'
model_dir = 'DeepSeek-R1-Distill-Qwen-1.5B'

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

model= PeftModel.from_pretrained(model, "./lora-out")
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def evaluate(
    instruction,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    max_new_tokens=800
):
    #由于目前模型的强大，其实无论prompt选择常见格式的哪一种，都可以识别
    prompt = generate_prompt(instruction, input=None)
    
    messages = [
        {"role": "system", "content": "你是一个有用的人工智能助手。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
   
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )


    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


gr.Interface(
    fn=evaluate,#接口函数
    inputs=[
        gr.components.Textbox(
            lines=2, label="问题", placeholder="给我讲解一道推理题~"
        ),
        gr.components.Slider(minimum=0.1, maximum=4.0, value=0.6, label="创造力"),
        gr.components.Slider(minimum=0.05, maximum=1.0, value=0.9, label="P参数"),
        gr.components.Slider(minimum=1, maximum=1000, step=1, value=50, label="K参数"),
        gr.components.Slider(minimum=1.0, maximum=2.0, step=0.05, value=1.2, label="惩罚参数"),
        gr.components.Slider(
            minimum=1, maximum=2048, step=1, value=1024, label="上下文长度"
        ),
    ],
    outputs=[
        gr.components.Textbox(
            lines=15,
            label="Output",
        )
    ],
    title="ChatUni",
    description="Chat,Your Own World",
).launch()