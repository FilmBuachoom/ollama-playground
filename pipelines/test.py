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
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# import utility functions
from pipelines.utils.on_startup import check_index_files_exist


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
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global documents, index, work_dir, index_dir, storage_context, service_context

        self.work_dir   = "/app/pipelines"
        self.index_dir  = f"{self.work_dir}/data/index"

        # Service context
        Settings.service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model, llm=Settings.llm)
        
        # Check the index files exist in the given directory
        self.storage_context = check_index_files_exist(
            service_context=Settings.service_context, 
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

        # # Service context
        # service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model, llm=Settings.llm)

        # load index
        load_index = load_index_from_storage(self.storage_context, service_context=Settings.service_context)

        retriever   = load_index.as_retriever(similarity_top_k=5)
        nodes       = retriever.retrieve(user_message)
        for i in range(3):
            print(f'Score:{nodes[i].score}')
            print(f'Context:{nodes[i].text}')
            print('='*100)

        # # # configure retriever
        # # retriever = VectorIndexRetriever(index=load_index, similarity_top_k=3)

        # # # configure response synthesizer
        # # response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", verbose=True)

        # # # assemble query engine
        # # query_engine = RetrieverQueryEngine(
        # #     retriever=retriever,
        # #     response_synthesizer=response_synthesizer,
        # #     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        # # )
        # USER_CHAT_TEMPLATE = '<start_of_turn>system\nYou are a question answering assistant. Answer the question as truthfully and helpfully as possible.คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<end_of_turn>'
        # MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'
        # promp = f'''
        #         <s>[INST] <<SYS>>
        #         You are a question answering assistant. Answer the question as truthful and helpful as possible
        #         คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด
        #         <</SYS>>

        #         Answer the question based only on the following context:
        #         {prompt}

        #         [/INST]
        #     '''
        # tourism_prompt_template = """
        #     {{- range $i, $_ := .Messages }}
        #     {{- $last := eq (len (slice $.Messages $i)) 1 }}
        #     {{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
        #     {{ .Content }}<end_of_turn>
        #     {{ if $last }}<start_of_turn>model
        #     {{ end }}
        #     {{- else if eq .Role "assistant" }}<start_of_turn>model
        #     {{ .Content }}{{ if not $last }}<end_of_turn>
        #     {{ end }}
        #     {{- end }}
        #     {{- end }}
        # """

        # system_role = """You are a question answering assistant. Answer the question as truthfully and helpfully as possible.
        # คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด"""

        # messages = [
        #     {"Role": "system", "Content": system_role},
        #     {"Role": "user", "Content": user_message},
        #     # {"Role": "assistant", "Content": "The top tourist attractions in Paris include the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées."},
        # ]

        # for message in messages:
        #     role = message["Role"]
        #     content = message["Content"]
        #     print(f"<start_of_turn>{role}\n{content}<end_of_turn>")

        query_engine = load_index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        # return response.response_gen
        return response.response_gen