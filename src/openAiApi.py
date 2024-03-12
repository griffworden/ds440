import os
from openai import OpenAI

client = OpenAI(
    api_key= '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' # os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4",
)

print(chat_completion.choices[0].message.content)