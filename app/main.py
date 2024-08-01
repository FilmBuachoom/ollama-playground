import os
from llama_index.llms.ollama import Ollama

def test():
    model_name = os.getenv("MODEL_NAME")
    print(f"Model name: {model_name}")
    llm = Ollama(
        model=model_name,       # model name
        context_window=2048,    # increase the context window
        request_timeout=600     # timeout for making http request to Ollama API server
    )
    question = input("Enter your quest: ")
    response = llm.complete(question)
    print(f"Model response: {response}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test()
        # elif sys.argv[1] == 'check_cuda':
        #     check_cuda()
        else:
            print(f"No function named {sys.argv[1]} available.")