from llama_cpp import Llama
import sys

try:
    llm = Llama(
        model_path="C:/Users/User/AppData/Local/llama.cpp/TheBloke_Mistral-7B-Instruct-v0.2-GGUF_mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_threads=12,   
        n_ctx=1024,     
        verbose=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Simple interactive loop
while True:
    try:
        sys.stdout.flush()
        print("################################################################")
        print("###### Rayen said ##############################################")
        print("################################################################")
        prompt = input(": ")
        if prompt.lower() in ["exit", "quit"]:
            break

        print(":", end=" ", flush=True)
        print("################################################################")
        print("###### Local Hosted Ai Added ###################################")
        print("################################################################")
        for token in llm(prompt, max_tokens=300, echo=False, stream=True):
            # Print each token as it's generated
            print(token["choices"][0]["text"], end="", flush=True)
        print("\n")  # new line after response

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
