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
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent

# import utility functions
from pipelines.utils.agent_utils import VectorStoreManager, RewritingInput


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        TEXT_EMBEDDING_INFERENCE_BASE_URL: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://ollama:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "TEXT_EMBEDDING_INFERENCE_BASE_URL": os.getenv("TEXT_EMBEDDING_INFERENCE_BASE_URL", "http://embedding-inference:11435"),
            }
        )

    async def on_startup(self):
        """This function is called when the server is started."""
        # setup llm
        Settings.embed_model = TextEmbeddingsInference(
            base_url=self.valves.TEXT_EMBEDDING_INFERENCE_BASE_URL,
            model_name="thenlper/gte-small",
            auth_token=None,
            timeout=120,
            embed_batch_size=32,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            temperature=0.4,
            context_window=4096,
            request_timeout=600
        )

        # defind global variables
        global query_engine_tools, rewriting, agent

        # load retriever tool
        query_engine_tools = VectorStoreManager(path_to_folder="/app/pipelines/data/embedding").load_query_engine_tool()
        print(f">>> query_engine_tools: {query_engine_tools}")

        # load rewriting process
        rewriting = RewritingInput()

        # build agent
        agent = ReActAgent.from_tools(query_engine_tools, llm=Settings.llm, verbose=True, max_iterations=20)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response."""
        # print(messages)
        print(user_message)
        # query = rewriting.rewrite(query=user_message)
        result = agent.chat(user_message)

        return str(result.response)
