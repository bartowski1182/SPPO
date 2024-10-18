from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained(
    "internlm/internlm2-7b-reward",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "internlm/internlm2-7b-reward", trust_remote_code=True
)
