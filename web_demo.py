from transformers import AutoTokenizer,AutoModel
import torch
from peft import get_peft_model, LoraConfig, TaskType
import gradio as gr

model = AutoModel.from_pretrained(
    "./chatglm6b-dddd", trust_remote_code=True).half().cuda()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value',],
)
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(
    "./chatglm6b-dddd", 
    trust_remote_code=True
)
# 在这里加载lora模型，注意修改chekpoint
peft_path = "test004/checkpoint-2000/chatglm-lora.pt"
model.load_state_dict(torch.load(peft_path), strict=False)
model.eval()

dialogue_history = []

def generate_response(user_input):
    dialogue_history.append('用户：' + user_input)

    model_input = '\n'.join(dialogue_history)
    response,history = model.chat(tokenizer=tokenizer, query=model_input,max_length=300)

    model_reply = response
    dialogue_history.append('模型：' + model_reply)
    return model_reply

iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="ChatGLM模型",
    description="与ChatGLM模型进行对话。",
    theme="default"
)

iface.launch()
