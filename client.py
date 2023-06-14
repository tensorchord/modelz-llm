import argparse

import openai

openai.api_base = "http://localhost:8000"
openai.api_key = "test"
openai.debug = True


def chat():
    chat_completion = openai.ChatCompletion.create(
        model="fastchat-t5-3b-v1.0",
        messages=[
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am a student"},
            {"role": "user", "content": "What do you learn?"},
            {"role": "assistant", "content": "I learn math"},
            {"role": "user", "content": "Do you like english?"},
        ],
        max_tokens=100,
    )
    print(chat_completion)


def embedding():
    emb = openai.Embedding.create(
        input=["Once upon a time", "There was a frog", "Who lived in a well"],
        model="text-embedding-ada-002",
    )
    print(emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chat", action="store_true")
    parser.add_argument("-e", "--embedding", action="store_true")
    args = parser.parse_args()

    if args.chat:
        chat()
    if args.embedding:
        embedding()
