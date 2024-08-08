# import
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter


# Check the index files exist in the given directory
def check_index_files_exist(service_context, embed_model, docs_dir, index_dir):
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

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                embed_model,
            ],
        )

        nodes = pipeline.run(documents=documents)

        # Create vector store index
        vector_store_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context, node_parser=nodes)

        # vector_store = FaissVectorStore(faiss_index=faiss_index)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # index = VectorStoreIndex(nodes=nodes,storage_context=storage_context, service_context = service_context)

        # # Indexing
        # index = VectorStoreIndex.from_documents(documents)

        # Persisting to disk
        vector_store_index.storage_context.persist(persist_dir=index_dir)

    # Storage index directory
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    return storage_context