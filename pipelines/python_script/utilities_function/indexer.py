# import
import os, subprocess

# faiss indexer
def run_pyserini_faiss_index(input_path, output_path):
    print("Indexing ...")
    index_dir = "/app/pipelines/pipelines/data/index"

    # Check if the output directory exists
    if not os.path.exists(index_dir):
        print(f"Creating output directory at {index_dir}")
        os.makedirs(index_dir)
    
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