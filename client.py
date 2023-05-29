import openai

openai.api_base = "http://localhost:8000/"
openai.api_key = "test"
openai.debug = True

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
