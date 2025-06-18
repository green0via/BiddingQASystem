import torch
# from modelscope import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import BGEM3FlagModel
from pymilvus.model.reranker import BGERerankFunction
from langchain_community.chat_models import ChatZhipuAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
glm_model = ChatZhipuAI(model="glm-4-air-250414", temperature=0.5)
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
gemini_temp = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)

# qwen_tokenizer = AutoTokenizer.from_pretrained(
#     'Qwen/Qwen3-8B',
#     trust_remote_code=True
# )
# qwen_model = AutoModelForCausalLM.from_pretrained(
#     'Qwen/Qwen3-8B',
#     trust_remote_code=True,
#     torch_dtype=torch.float16
# ).cuda()

bgem3_model = BGEM3FlagModel(
    'BAAI/bge-m3',
    use_fp16=False,
    pooling_method='cls',
    devices=['cuda:0']
)

bge_rf = BGERerankFunction(
    model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
    device="cuda:0" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

def qwen_llm(prompt_str: str) -> str:
    print("Qwen...")
    if hasattr(prompt_str, 'to_string'):
        prompt_str = prompt_str.to_string()
    # ✅ 确保输入是字符串
    assert isinstance(prompt_str, str), f"Expected string, got {type(prompt_str)}"
    messages = [
        {"role": "user", "content": prompt_str}
    ]
    text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    inputs = qwen_tokenizer([text], return_tensors='pt').to(qwen_model.device)
    outputs = qwen_model.generate(**inputs, max_new_tokens=32768)  # ✅ 注意 max_new_tokens 拼写
    # return tokenizer.decode(outputs[0], skip_special_tokens=True).split('<think>\n\n</think>\n\n')[-1]
    return qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)