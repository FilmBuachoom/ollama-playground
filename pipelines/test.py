"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, ServiceContext
from llama_index.core import load_index_from_storage

# import utility functions
from pipelines.utils.on_startup import check_index_files_exist
from pipelines.utils.rag        import main_rag


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://ollama:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            temperature=0, 
            max_tokens=512
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            temperature=0.75,
            context_window=2048,
            request_timeout=600
        )

        # This function is called when the server is started.
        global documents, index, work_dir, index_dir, storage_context, service_context, llm

        self.work_dir   = "/app/pipelines"
        self.index_dir  = f"{self.work_dir}/data/index"

        # Service context
        Settings.service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model, llm=Settings.llm)
        
        # Check the index files exist in the given directory
        self.storage_context = check_index_files_exist(
            embed_model=Settings.embed_model, 
            docs_dir=f"{self.work_dir}/data/document", 
            index_dir=self.index_dir
        )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # print(messages)
        print(user_message)

        # load index
        load_index = load_index_from_storage(self.storage_context, service_context=Settings.service_context)

        # RAG function
        response = main_rag(
            service_context=Settings.service_context,
            index=load_index, 
            top_k=2, 
            user_message=user_message
        )

        # return response.response_gen
        return response