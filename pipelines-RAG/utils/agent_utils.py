# import
import os, re
from llama_index.core import Settings, load_index_from_storage, StorageContext, PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter


# Main class
class VectorStoreManager:
    def __init__(self, path_to_folder: str, top_k: int = 5):
        """Initialize VectorStoreManager with a path and top_k value."""
        self.top_k          = top_k
        self.llm            = Settings.llm
        self.embed_model    = Settings.embed_model
        self.raw_doc_path   = "/app/pipelines/data/raw"
        self.vector_idx_1_path = os.path.join(path_to_folder, "multiple_chunk_size", "1024")
        self.vector_idx_2_path = os.path.join(path_to_folder, "vector_store_index_amazing_th")

    def vector_indexing(self, path_to_raw, documents):
        """Indexing documents"""
        print(">>> Indexing document ...")
        if path_to_raw == "Lonely Planet Thailand-Lonely Planet.pdf":
            # indexing lonely planet
            use_page = {
                "need_to_know": [23],
                "first_time_th": [24, 25, 26],
                "month_by_month": [31, 32, 33],
                "understand_th": [i for i in range(704+1, 757+1)],
                "survival_guild": [i for i in range(758+1, 787+1)]
            }

            # project documents
            documents = [documents[i] for i in [page for key in use_page for page in use_page[key]]]

            # chunk size
            chunk_size, chunk_overlap = 1024, 32
            persis_path = self.vector_idx_1_path

        else:
            # indexing amazing Thailand
            chunk_size, chunk_overlap = 512, 32
            persis_path = self.vector_idx_2_path

        # # Token test spliter
        # text_splitter = TokenTextSplitter(
        #     separator=" ",
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     backup_separators=["\n"]
        # )

        # Sentence spliter
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Transformer process Vector Index for answering specific context questions
        vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, transformations=[node_parser, self.embed_model])

        # persis vector index
        vector_index.storage_context.persist(persist_dir=persis_path)
        StorageContext.from_defaults(persist_dir=persis_path)

    def load_document(self, path_to_raw):
        """Load document"""
        compose_path_to_raw = os.path.join(self.raw_doc_path, path_to_raw)
        if os.path.isdir(compose_path_to_raw):
            reader = SimpleDirectoryReader(input_dir=compose_path_to_raw)
            
        elif os.path.isfile(compose_path_to_raw):
            reader = SimpleDirectoryReader(input_files=[compose_path_to_raw])
            
        self.vector_indexing(path_to_raw=path_to_raw, documents=reader.load_data()) 

    def load_vector_store_idx(self):
        """Load vector indices from different sources."""
        # Load from source 1 (Lonely Planet)
        if not os.path.exists(self.vector_idx_1_path):
            self.load_document(path_to_raw="Lonely Planet Thailand-Lonely Planet.pdf")

        storage_context_1 = StorageContext.from_defaults(persist_dir=self.vector_idx_1_path)
        vector_store_index_1 = load_index_from_storage(storage_context_1)
        query_eng_1 = vector_store_index_1.as_query_engine(similarity_top_k=self.top_k, llm=self.llm, streaming=True)

        # Load from source 2 (Amazing TH)
        if not os.path.exists(self.vector_idx_2_path):
            self.load_document(path_to_raw="amazing_thailand")

        storage_context = StorageContext.from_defaults(persist_dir=self.vector_idx_2_path)
        vector_store_index_2 = load_index_from_storage(storage_context)
        query_eng_2 = vector_store_index_2.as_query_engine(similarity_top_k=self.top_k, llm=self.llm, streaming=True)

        # Return both retrievers
        return query_eng_1, query_eng_2

    def build_query_routing(self, tools):
        """Build query engine tool based on the loaded vector indices."""
        # Define and return the QueryEngine
        query_engine_lst = list()
        for tool in tools.values():
            query_engine_tool = QueryEngineTool(
                query_engine=tool["tool"], metadata=ToolMetadata(name=tool["name"], description=tool["description"])
            )
            query_engine_lst.append(query_engine_tool)

        return query_engine_lst

    def load_query_engine_tool(self):
        """Main function to load query engine for agent"""
        # Load the vector store indices
        query_eng_1, query_eng_2 = self.load_vector_store_idx()

        # Adding description
        tools = {
            "tool_1": {
                "name": "lonely_planet_vector_c1024",
                "tool": query_eng_1,
                "description": (
                    "Useful for retrieving specific context about things that tourists need to know in Thailand."
                    "Thai culture or etiquette. Include aspects like local customs, significant festivals, etiquette,"
                    "traditional foods, basic language phrases, religious practices, and guidelines for respect at "
                    "historical landmarks"
                )
            },
            
            "tool_2": {
                "name": "amazing_th",
                "tool": query_eng_2,
                "description": "Useful for retrieving specific context about attractions in Thailand, include details such as"
                "opening hours, admission fees, facilities, exact location, transportation options, type of attraction, "
                "special events, dress code, and best visit times."
            }
        }
        query_engine_tools = self.build_query_routing(tools=tools)

        # return result
        return query_engine_tools


class RewritingInput:
    def __init__(self):
        """Initialize VectorStoreManager with a path and top_k value."""
        self.llm = Settings.llm
        self.prompt_template = {
            'check_question_prompt': (
                "Classify the following text: '{query}' as a question or not. Respond only with 'Yes' "
                "if it is a question, or 'No' if it is not a question."    
            ),
            
            'classify_prompt': (
                "Given the user query, classify the main topic as either 'Other', 'Culture', or 'Attraction'. "
                "Note that queries about suggestions, recommendations, or any topics that do not specifically relate "
                "to social norms, traditions, etiquette, specific places, landmarks, or sightseeing points should be classified as 'Other'. "
                "Queries about social norms, traditions, and etiquette fall under 'Culture' "
                "Queries about specific places, landmarks, or sightseeing points fall under 'Attraction'. "
                "Return only 'Other', 'Culture', or 'Attraction' based on the user's input.\n\n"
                "Here is the user query: {query}\n\n"
                "Category:"

            ),
            
            'rewriting_prompt': {
                'Attraction': (
                    "Generate a detailed question about attractions in Thailand in 64 words. Expand the query to include details such as "
                    "opening hours, admission fees, facilities, exact location, transportation options, type of attraction, "
                    "special events, dress code, and best visit times. Format your output in plain text.\n\n"
                    "Here is the user query: {query}\n\n"
                    "Question: "
                ),
                'Culture': (
                    "Generate a detailed question about Thai culture or etiquette in 64 words. Include aspects like local customs, "
                    "significant festivals, etiquette, traditional foods, basic language phrases, religious practices, and "
                    "guidelines for respect at historical landmarks. Format your output in plain text.\n\n"
                    "Here is the user query: {query}\n\n"
                    "Question: "
                )
            }
        }

    
    def validate_question(self, query: str):
        "Entry point for validate input query, does the query is question."
        # build prompt
        check_question_prompt_tmpl = PromptTemplate(self.prompt_template["check_question_prompt"].format(query=query))

        # classification
        is_question = self.llm.predict(check_question_prompt_tmpl).strip()
        
        # return result
        return is_question

    
    def rewrite(self, query: str):
        "Entry point for rewriting, triggered by a StartEvent with `query`."
        # check does the query is question
        is_question = self.validate_question(query=query)
        print(f">>> Input query:\n\t{query}\n\n")
        print(f">>> Does input is question:\n\t{is_question}\n\n")
        if is_question == "No":
            return query
            
        # classification the query with llm
        classification_prompt_tmpl = PromptTemplate(self.prompt_template["classify_prompt"].format(query=query))
        query_class = self.llm.predict(classification_prompt_tmpl).strip()
        print(f">>> User query about:\n\t{query_class}\n\n")

        # recheck classification
        if len(re.findall(r'\b(recommend(?:s|ed|ation)?|suggest(?:s|ion)?)\b', query, re.IGNORECASE)) != 0 and query_class != "Other":
            query_class = 'Other'
            print(f">>> User query about (re-check):\n\t{query_class}\n\n")
            
       # generate question based on topic
        if query_class in ['Attraction', 'Culture']:
            rewriting_prompt = PromptTemplate(self.prompt_template['rewriting_prompt'][query_class].format(query=query))
            rewriting_query = Settings.llm.predict(rewriting_prompt).split("Question: ")[-1].strip()
            # rewriting_query += ". Try to use tools first."
            print(f">>> Rewriting and expanding query:\n\t{rewriting_query}\n\n")
        else:
            return query

        # return
        return rewriting_query