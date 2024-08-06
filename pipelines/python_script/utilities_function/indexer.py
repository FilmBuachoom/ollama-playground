# import
import os, sys, subprocess

# faiss indexer
def run_pyserini_faiss_index(input_path, output_path):
    index_dir = "/app/pipelines/pipelines/data/index"

    # Check if the output directory exists
    if not os.path.exists(index_dir):
        print(f"Creating output directory at {index_dir}")
        os.makedirs(index_dir)

    # # Check index file exist
    # index_file_name = f"/data/index/{os.path.basename(input_path)}.xx"
    # if os.path.exists(index_file_name):
    #     print(f"{index_file_name} is exist.")
    
    # else:
    # Run the Pyserini FAISS indexing command
    command = [
        'python', '-m', 'pyserini.encode',
        'input', 
            '--corpus', input_path,
            '--fields', 'text',
            '--delimiter', '\n',
            '--shard-id', '0',
            '--shard-num', '1',
        'output',
            '--embeddings', output_path,
            '--to-faiss',
        'encoder',
            '--encoder', 'castorini/tct_colbert-v2-hnp-msmarco',
            '--fields', 'text',
            '--batch', '32',
            '--fp16'
    ]

    # Run the command
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Encoding completed successfully:")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to complete encoding:")
        print("Error:", e.stderr)