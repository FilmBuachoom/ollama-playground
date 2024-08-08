# Import
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import ServiceContext, get_response_synthesizer, ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

# Retrieve function
def retrieve_func(index, top_k, text_qa_template):
    # Configure retriever
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    print(f"VectorIndexRetriever: {vector_retriever}")

    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="refine",
        # service_context=service_context,
        text_qa_template=text_qa_template,
        # refine_template=refine_template,
        use_async=False,
        streaming=False,
    )

    # Assemble query engine
    vector_query_engine = RetrieverQueryEngine(retriever=vector_retriever, response_synthesizer=response_synthesizer)
    print(f"RetrieverQueryEngine: {vector_retriever}")

    # # Retriever result
    # docs = vector_retriever.retrieve(user_message)
    # for i in range(top_k):
    #     print(f'Score:{docs[i].score}')
    #     print(f'Context:{docs[i].text}')
    #     print('='*100)
    
    return vector_query_engine

# Main function RAG
def main_rag(llm, embed_model, index, top_k, user_message):

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    #the context and question as a "user" message.
    chat_text_qa_msgs = [
        (
            """\
            <start_of_turn>system
            You are a tourism guide that provides accurate and helpful information based only on the following query documents. Always include references to the documents used in your response.<end_of_turn>
            <start_of_turn>user
            Context:
            {context_str}
            
            Question:
            {query_str}
            <end_of_turn>
            <start_of_turn>model
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    # Retriever
    vector_query_engine = retrieve_func(index, top_k, text_qa_template)

    # Generation
    response = vector_query_engine.query(user_message)

    return "TEST"
