from transformers import AutoTokenizer,AutoModel
# from thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType


model = AutoModel.from_pretrained(
    "./chatglm6b-dddd", trust_remote_code=True).half().cuda()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value',],
)
model = get_peft_model(model, peft_config)

# 在这里加载lora模型，注意修改chekpoint
peft_path = "test004/checkpoint-2000/chatglm-lora.pt"
model.load_state_dict(torch.load(peft_path), strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./chatglm6b-dddd", trust_remote_code=True)

text ="谷氨酰转肽酶水平会因吸毒或饮酒而升高吗?"

with torch.autocast("cuda"):
    res, history = model.chat(tokenizer=tokenizer, query=text,max_length=300)
    print(res)