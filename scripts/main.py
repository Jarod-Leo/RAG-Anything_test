import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # 设置 API 配置
    api_key = "sk-f653945d68514f5aa5e7f6ddc7bc04fb"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 可选

    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # 选择解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 定义 LLM 模型函数
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "qwen3-vl-plus",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # 定义视觉模型函数用于图像处理
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # 如果提供了messages格式（用于多模态VLM增强查询），直接使用
        if messages:
            return openai_complete_if_cache(
                "qwen3-vl-plus",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 传统单图片格式
        elif image_data:
            return openai_complete_if_cache(
                "qwen3-vl-flash",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 纯文本格式
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定义嵌入函数
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-v4",
            api_key=api_key,
            base_url=base_url
        ),
    )
    embeddings = await embedding_func(["测试文本"])
    print("嵌入维度:", len(embeddings[0]))


    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "vector_db_storage_cls_kwargs": {
                "dim": 1024,  # 向量维度
                "storage_path": "./vector_storage",  # 存储路径
            }
        }
    )

    # 处理文档
    await rag.process_document_complete(
        file_path="./document/mercedes-e_user_manual.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # 查询处理后的内容
    # 纯文本查询 - 基本知识库搜索
    text_result = await rag.aquery(
        "文档的主要内容是什么？",
        mode="hybrid"
    )
    print("文本查询结果:", text_result)
    text_result2 = await rag.aquery(query="触摸感应式控制元件的主要功能是什么？", mode="hybrid")
    print("文本查询结果2:", text_result2)
    # # 多模态查询 - 包含具体多模态内容的查询
    # multimodal_result = await rag.aquery_with_multimodal(
    #     "分析这个性能数据并解释与现有文档内容的关系",
    #     multimodal_content=[{
    #         "type": "table",
    #         "table_data": """系统,准确率,F1分数
    #                         RAGAnything,95.2%,0.94
    #                         基准方法,87.3%,0.85""",
    #         "table_caption": "性能对比结果"
    #     }],
    #     mode="hybrid"
    # )
    # print("多模态查询结果:", multimodal_result)

if __name__ == "__main__":
    asyncio.run(main())