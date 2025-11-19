import asyncio
import os
import json
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from openai import OpenAI
from openai import AsyncOpenAI
# from dashscope import TextEmbedding  # 阿里云 DashScope embedding
import logging
import time
# 日志设置
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 阿里云 API Key（从环境变量或硬编码；建议用 os.getenv）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-f653945d68514f5aa5e7f6ddc7bc04fb")
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY  # DashScope 自动使用
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 本地 llama.cpp 客户端
llm_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-needed"
)

vision_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8082/v1",
    api_key="sk-no-key-needed"
)

async def main():
    # Neo4j 配置
    os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "20250923"

    # RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="/data_nvme/home/rag_data/rag_storage",
        parser="mineru",  # 你的解析器
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):  
        messages = []  
        if system_prompt:  
            messages.append({"role": "system", "content": system_prompt})  
        
        # 修改: 确保 history_messages 不为 None  
        if history_messages:  
            messages.extend(history_messages)  
        
        messages.append({"role": "user", "content": prompt})  
    
        response = await llm_client.chat.completions.create(  
            model="Qwen3-VL-8B-Instruct-GGUF",  
            messages=messages,  
            temperature=kwargs.get("temperature", 0.7),  
            top_p=kwargs.get("top_p", 0.8),  
            max_tokens=kwargs.get("max_tokens", 4096),  
        )  
        return response.choices[0].message.content

    async def vision_model_func(prompt, system_prompt=None, history_messages=None,   
                     image_data=None, messages=None, **kwargs):  
        # 修改 1: 如果提供了 messages 格式(框架内部用)  
        if messages:  
            response = await vision_client.chat.completions.create(  
                model="Qwen3-VL-8B-Instruct-GGUF",  
                messages=messages,  
                temperature=kwargs.get("temperature", 0.7),  # 添加默认参数  
                top_p=kwargs.get("top_p", 0.8),  
                max_tokens=kwargs.get("max_tokens", 4096),  
            )  
            return response.choices[0].message.content  
    
        # 修改 2: base64 图像数据处理  
        elif image_data:  
            content = [  
                {"type": "text", "text": prompt},  
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}  
            ]  
            
            # 构建消息列表  
            msg_list = []  
            if system_prompt:  
                msg_list.append({"role": "system", "content": system_prompt})  
            msg_list.append({"role": "user", "content": content})  
    
            response = await vision_client.chat.completions.create(  
                model="Qwen3-VL-8B-Instruct-GGUF",  
                messages=msg_list,  
                temperature=kwargs.get("temperature", 0.7),  
                top_p=kwargs.get("top_p", 0.8),  
                max_tokens=kwargs.get("max_tokens", 4096),  
            )  
            return response.choices[0].message.content  
        
        # 修改 3: 纯文本回退 - 确保传递正确的参数  
        else:  
            # 如果 history_messages 为 None,传递空列表  
            return await llm_model_func(  
                prompt,   
                system_prompt,   
                history_messages if history_messages is not None else [],   
                **kwargs  
            )

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-v3",
            api_key=DASHSCOPE_API_KEY,
            base_url=base_url,
        ),
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "graph_storage": "Neo4JStorage",
            "vector_db_storage_cls_kwargs": {
                "dim": 1024,  # 匹配 embedding_dim
                "storage_path": "/data_nvme/home/rag_data/vector_storage",
            }
        }
    )

    # 读取 JSON content_list
    def read_json_to_list(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data

    # 读取 content_list
    content_list = read_json_to_list(
        file_path="/data_nvme/home/rag_data/output/mercedes-e_user_manual/auto/mercedes-e_user_manual_content_list_fix.json"
    )

    # 可选：小测试（前 100 项）
    # content_list = content_list[:100]
    # logger.info(f"Testing with first {len(content_list)} items")

    # 插入 content_list
    await rag.insert_content_list(
        content_list=content_list,
        file_path="mercedes-e_user_manual.pdf",
        split_by_character=None,
        split_by_character_only=False,
        doc_id=None,
        display_stats=True
    )

    # 查询示例
    modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
    for m in modes:
        start = time.time()
        text_result = await rag.aquery("驾驶员显示屏有哪些东西？", mode=m)
        end = time.time()
        time_shift = end - start
        print(f"文本查询结果:{text_result}, 查询模式:{m}, 消耗时间:{time_shift}")
    # text_result = await rag.aquery("驾驶员显示屏有哪些东西？", mode="local")
    # print("文本查询结果:", text_result)

if __name__ == "__main__":
    asyncio.run(main())