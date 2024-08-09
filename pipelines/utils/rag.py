# Import
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import ServiceContext, get_response_synthesizer, ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

# Prompt function
def create_prompt(user_message, vector_retriever):
    # Config prompt
    context_str = "Here is the context from the document(s)..."

    qa_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "Relevant Document: {relevant_document}\n"
        "---------------------\n"
        "Given the context information and the relevant document, and not prior knowledge, "
        "answer the question: {query_str}\n"
    )

    # Render the initial QA prompt
    qa_prompt_str = qa_prompt_str.format(context_str=context_str, relevant_document=vector_retriever, query_str=user_message)
    print(f"qa_prompt_str: {qa_prompt_str}")

    refine_prompt_str = (
        "We have the opportunity to refine the original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "Relevant Document: {relevant_document}\n"
        "------------\n"
        "Given the new context and the relevant document, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
    )

    # Render the refine prompt
    refine_prompt_str = refine_prompt_str.format(context_msg="Additional context provided...", relevant_document=vector_retriever, query_str=user_message, existing_answer="Original answer to refine")
    print(f"refine_prompt_str: {refine_prompt_str}")

    # Text QA Prompt
    chat_text_qa_msgs = [
        (
            "system",
            "You are a tourism guide that provides accurate and helpful information based only on the following query documents. Always include references to the documents used in your response. คุณคือไกด์นำเที่ยวที่ให้ข้อมูลที่ถูกต้องและเป็นประโยชน์โดยอิงจากเอกสารคำถามเท่านั้น และคืนค่าการอ้างอิงถึงเอกสารที่ใช้ในคำตอบของคุณ",
        ),
        ("user", qa_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    # Refine Prompt
    chat_refine_msgs = [
        (
            "system",
            "You are a tourism guide that provides accurate and helpful information based only on the following query documents. Always include references to the documents used in your response. คุณคือไกด์นำเที่ยวที่ให้ข้อมูลที่ถูกต้องและเป็นประโยชน์โดยอิงจากเอกสารคำถามเท่านั้น และคืนค่าการอ้างอิงถึงเอกสารที่ใช้ในคำตอบของคุณ",
        ),
        ("user", refine_prompt_str),
    ]
    refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

    return text_qa_template, refine_template

# Retrieve function
def retrieve_func(index, top_k, user_message, service_context, text_qa_template):
    # Configure retriever
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    print(f"VectorIndexRetriever: {vector_retriever}")

    # Retriever result
    relevance_docs = vector_retriever.retrieve(user_message)
    print(relevance_docs[:top_k])
    # for i in range(top_k):
    #     print(f'Score:{relevance_docs[i].score}')
    #     print(f'Context:{relevance_docs[i].text}')
    #     print('='*100)

    # Get prompt
    text_qa_template, refine_template = create_prompt(user_message=user_message, vector_retriever=vector_retriever)

    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="refine",
        service_context=service_context,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        use_async=False,
        streaming=False,
    )

    # Assemble query engine
    vector_query_engine = RetrieverQueryEngine(retriever=vector_retriever, response_synthesizer=response_synthesizer)
    print(f"RetrieverQueryEngine: {vector_retriever}")
    
    return vector_query_engine

# Main function RAG
def main_rag(service_context, index, top_k, user_message):
    # Retriever
    vector_query_engine = retrieve_func(index, top_k, user_message, service_context, text_qa_template=None)

    # Generation
    response = vector_query_engine.query(user_message)
    print(f"Response: {response}")
    # response = "test"

    return str(response)
