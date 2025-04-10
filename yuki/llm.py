from google import genai
from google.genai import types
from pydantic import BaseModel

class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

class GoogleChatAI:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"


    def user_message(self, content):
        return Message("user", content)
        
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=content)]
        )

    def ai_message(self, content):
        return Message("model", content)
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=content)]
        )
    
    def convert_messages(self, messages: Message):
        contents = []
        for message in messages:
            contents.append(
                types.Content(
                    role=message.role,
                    parts=[types.Part.from_text(text=message.content)]
                )
            )
        return contents

    def invoke(self, contents, prompt=None, system_instruction="", temperature=1.0, top_p=0.9, top_k=40, max_output_tokens=8192):
        if prompt:
            contents = prompt
        else:
            contents = self.convert_messages(contents)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                response_mime_type="text/plain"
            )
        )

        return response.text.strip()

    def invoke_json(self, prompt, schema):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        return response.parsed
