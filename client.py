import openai

openai.api_base = "http://localhost:8000/v1"
openai.debug = True

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}]
)
