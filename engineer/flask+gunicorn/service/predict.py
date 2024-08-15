# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
from utils import get_root_path
from modelscope.hub.snapshot_download import snapshot_download

class Qwen7BInfer:
    def __init__(self):
        self.model = os.path.join(get_root_path(), 'autodl-tmp/model/qwen/Qwen2-7B-Instruct')
        self.llm = None
        
    def async_download_model(self):
        target_path = os.path.join(get_root_path(), 'autodl-tmp/model')
        print(f"下载模型目标路径{target_path}")
        try:
            snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir=target_path)
        except Exception as e:
            print("下载模型失败")
            raise e
        
        print("下载模型成功！")
        self.load_model()
        
    def load_model(self, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
        self.llm = LLM(model=self.model, tokenizer=tokenizer, max_model_len=max_model_len, trust_remote_code=True, gpu_memory_utilization=0.5, enforce_eager=True)
        print("加载模型成功！")
        

    def get_completion(self, prompts, tokenizer=None, max_tokens=512, temperature=0.1, top_p=0.95, max_model_len=2048):
        # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
        print(prompts)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        # 初始化 vLLM 推理引擎
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

G_model = Qwen7BInfer()

if __name__ == "__main__":
    # 初始化 vLLM 推理引擎
    model = '/root/autodl-tmp/qwen/Qwen2-7B-Instruct'  # 指定模型路径
    # model="qwen/Qwen2-7B-Instruct" # 指定模型名称，自动下载模型
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    text = ["你好，帮我介绍一下什么时大语言模型。",
            "可以给我将一个有趣的童话故事吗？"]
    # messages = [
    #     {"role": "system", "content": "你是一个有用的助手。"},
    #     {"role": "user", "content": prompt}
    # ]
    # 作为聊天模板的消息，不是必要的。
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1,
                             max_model_len=2048)

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
