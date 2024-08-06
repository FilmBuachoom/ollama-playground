"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pipelines.utilities_function.indexer import run_pyserini_faiss_index
from pyserini.search import FaissSearcher


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        work_dir    = "/app/pipelines/pipelines" 
        input_path  = f"{work_dir}/data/document/sample.jsonl"
        output_path = f"{work_dir}/data/index/"
        print("INDEXING ...")
        run_pyserini_faiss_index(input_path, output_path)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        work_dir = "/app/pipelines/pipelines" 
        searcher = FaissSearcher(
            f"{work_dir}/data/index/",
            # 'facebook/dpr-question_encoder-multiset-base'
            'castorini/tct_colbert-v2-hnp-msmarco'
        )
        hits = searcher.search(messages[0]["content"])

        for i in range(0, 5):
            print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')

        return "Hello, world!"