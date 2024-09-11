# import
import os
from llama_index.core import Settings, QueryBundle, PromptTemplate, load_index_from_storage, StorageContext
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step, Event
from llama_index.core.retrievers import QueryFusionRetriever, RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore


# output schema class
class RewritingEvent(Event):
    """Result of running rewriting"""
    rewrite_query: str

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""
    nodes: list[NodeWithScore]
    gen_queries: list[QueryBundle]


# main class RAG
class RAGWorkflow(Workflow):
    prompt_template = {
        'classify_prompt': (
            "Given the user query, classify the main topic as either 'Culture', 'Attraction', or 'Other'. "
            "Note that queries about social norms, traditions, and etiquette fall under 'Culture.' "
            "Queries about specific places, landmarks, or sightseeing points fall under 'Attraction'. "
            "All other queries that do not fit into these categories should be classified as 'Other'. "
            "Return only 'Attraction', 'Culture', or 'Other' based on the user's input.\n\n"
            "Here is the user query: {query}\n\n"
            "Category:"
        ),
        'rewriting_prompt': {
            'Attraction': (
                "Generate a detailed question about attractions in Thailand. Expand the query to include details such as "
                "opening hours, admission fees, facilities, exact location, transportation options, type of attraction, "
                "special events, dress code, and best visit times. Format your output in plain text.\n\n"
                "Here is the user query: {query}\n\n"
                "Question: "
            ),
            'Culture': (
                "Generate a detailed question about Thai culture or etiquette. Include aspects like local customs, "
                "significant festivals, etiquette, traditional foods, basic language phrases, religious practices, and "
                "guidelines for respect at historical landmarks. Format your output in plain text.\n\n"
                "Here is the user query: {query}\n\n"
                "Question: "
            )
        },
        'summary_memo_prompt': (
            "Summarize the following conversation into 512 words or less while retaining key information:\n\n"
            "{previous_memory}"
        )
    }

    @step
    async def rewrite(self, ctx: Context, ev: StartEvent) -> RewritingEvent | None:
        if ev.get("retriever") is None:
            return None
        await ctx.set("retriever", ev.get("retriever"))
        
        query = ev.get("query")
        if not query:
            return None

        # Classify the query
        classify_prompt = PromptTemplate(self.prompt_template['classify_prompt'].format(query=query))
        query_class = Settings.llm.predict(classify_prompt).strip()
        print(f">>> Input query:\n\t{query}\n\n")
        print(f">>> Query classified as:\n\t{query_class}\n\n")
        
        # Generate question based on topic
        if query_class in ['Attraction', 'Culture']:
            rewriting_prompt = PromptTemplate(self.prompt_template['rewriting_prompt'][query_class].format(query=query))
            rewriting_query = Settings.llm.predict(rewriting_prompt).split("Question: ")[-1].strip()
        else:
            rewriting_query = query

        return RewritingEvent(rewrite_query=rewriting_query)

    @step
    async def retrieve(self, ctx: Context, ev: RewritingEvent) -> RetrieverEvent | None:
        retriever = await ctx.get("retriever")
        query = ev.rewrite_query
        print(f">>> Rewriting query:\n\t{query}\n\n")

        if not query or retriever is None:
            print("Missing retriever or query, please check the setup!")
            return None

        nodes = await retriever.aretrieve(query)
        print(f">>> Retrieved:\n\t{len(nodes)} nodes.\n\n")
        return RetrieverEvent(nodes=nodes, gen_queries=query)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        summarizer = CompactAndRefine(llm=Settings.llm, streaming=False, verbose=True)
        query = ev.gen_queries

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)
    

# main class
class VectorStoreManager:
    def __init__(self, path_to_folder: str, top_k: int = 5):
        """Initialize VectorStoreManager with a path and top_k value."""
        self.path_to_folder = path_to_folder
        self.top_k = top_k

    def load_vector_store_idx(self):
        """Load vector indices from different sources."""
        # Load from source 1 (Lonely Planet)
        vector_store_index_1 = dict()
        for chunk in [128, 256, 512, 1024]:
            sub_index_dir = os.path.join(self.path_to_folder, "multiple_chunk_size", str(chunk))
            storage_context = StorageContext.from_defaults(persist_dir=sub_index_dir)
            vector_store_index = load_index_from_storage(storage_context)
            vector_store_index_1[f"vector_store_index_1_c{chunk}"] = vector_store_index.as_query_engine()

        # Create QueryFusionRetriever for source 1
        retriever_1 = QueryFusionRetriever(
            retrievers=[vector_store_index_1[i] for i in vector_store_index_1],
            llm=Settings.llm,
            similarity_top_k=self.top_k,
            num_queries=1,  # set to 1 to disable query generation
            use_async=True,
            verbose=False
        )

        # Load from source 2 (Amazing TH)
        index_2_dir = os.path.join(self.path_to_folder, "vector_store_index")
        storage_context = StorageContext.from_defaults(persist_dir=index_2_dir)
        vector_store_index_2 = load_index_from_storage(storage_context)
        retriever_2 = vector_store_index_2.as_query_engine(similarity_top_k=self.top_k)

        # Return both retrievers
        return retriever_1, retriever_2

    def build_query_routing(self):
        """Build retriever routing based on the loaded vector indices."""
        # Load the vector store indices
        retriever_1, retriever_2 = self.load_vector_store_idx()

        # Initialize tools for each retriever
        vector_tool_1 = RetrieverTool.from_defaults(
            retriever=retriever_1,
            description="Useful for retrieving specific context about things that tourists need to know in Thailand."
        )
        
        vector_tool_2 = RetrieverTool.from_defaults(
            retriever=retriever_2,
            description="Useful for retrieving specific context about Thailand attractions."
        )

        # Define and return the RouterRetriever
        router_retriever = RouterRetriever(
            selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
            retriever_tools=[vector_tool_1, vector_tool_2],
            verbose=True
        )

        return router_retriever