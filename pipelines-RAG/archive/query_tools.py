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
import os, torch

from pydantic import BaseModel

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.response_synthesizers import TreeSummarize

# import utility functions
from pipelines.utils.agent_utils import VectorStoreManager, RewritingInput


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        TEXT_EMBEDDING_INFERENCE_BASE_URL: str

    def __init__(self):
        self.query_engine_tools1, self.query_engine_tools2 = None, None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "gemma2:2b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "TEXT_EMBEDDING_INFERENCE_BASE_URL": os.getenv("TEXT_EMBEDDING_INFERENCE_BASE_URL", "http://localhost:8081"),
            }
        )

    async def on_startup(self):
        """This function is called when the server is started."""
        # setup llm
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            temperature=0.4,
            context_window=4096,
            request_timeout=600
        )
        print(">>> Settings up LLM model successfull.")

        # Setting embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-small", embed_batch_size=32, device="cuda")
        print(">>> Settings up embedding model successfull.")

        # load retriever tool
        global query_engine_tools1, query_engine_tools2
        self.query_engine_tools1, self.query_engine_tools2 = VectorStoreManager(path_to_folder="/app/pipelines/data/embedding").load_vector_store_idx()
        print(">>> Load index successfull.")

        # load rewriting process
        # self.rewriting = RewritingInput()

        # build agent
        # self.agent = ReActAgent.from_tools(query_engine_tools, llm=Settings.llm, verbose=True, max_iterations=20)

        # # initialize router query engine (single selection, pydantic)
        # query_engine = RouterQueryEngine(
        #     selector=LLMSingleSelector.from_defaults(llm=Settings.llm), 
        #     query_engine_tools=query_engine_tools,
        #     summarizer=TreeSummarize,
        #     verbose=True
        # )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response."""
        print(f">>> User message:\n\t{user_message}\n\n")
        # print(f">>> Tool: \n\t{self.query_engine_tools1}\n\n")
        result = self.query_engine_tools1.query(user_message)
        # print(result)
        return result.response_gen