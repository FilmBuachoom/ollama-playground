# import
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

# Tramsformation function
def transformation_func(embed_model, documents):
    # defind values
    chunk_size, chunk_overlap = 1024, 20

    # Token test spliter
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        backup_separators=["\n"]
    )

    # Sentence spliter
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Transformer process
    index = VectorStoreIndex.from_documents(documents, show_progress=True, transformations=[text_splitter, node_parser, embed_model])
    return index


# Check the index files exist in the given directory
def check_index_files_exist(embed_model, docs_dir, index_dir):
    """Check if the specified files exist in the given directory."""
    files_to_check = [
        "default__vector_store.json",
        "docstore.json",
        "graph_store.json",
        "image__vector_store.json",
        "index_store.json"
    ]
    missing_files = [filename for filename in files_to_check if not os.path.exists(os.path.join(index_dir, filename))]

    
    if not missing_files:
        print("All index files exist.")
    else:
        print(f"The following index files are missing: {', '.join(missing_files)}")

        # Load document(s)
        documents = SimpleDirectoryReader(input_files=[f"{docs_dir}/allattractions_50.csv"]).load_data()
        print(f"Number of document(s): {len(documents)}")

        # Transformer process
        index = transformation_func(embed_model, documents)

        # Persisting to disk
        index.storage_context.persist(persist_dir=index_dir)

    # Storage index directory
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    return storage_context