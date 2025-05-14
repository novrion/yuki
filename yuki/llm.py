from datetime import datetime
from pydantic import BaseModel
from google.genai import types
from google import genai
import os

class Message:
    def __init__(self, content, role, timestamp=None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().timestamp()

    @classmethod
    def from_dict(cls, data):
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"]
        )

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

class GoogleChatAI:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-preview-04-17"

    def convert_messages(self, messages: list[Message]):
        contents = []
        for message in messages:
            contents.append(
                types.Content(
                    role=message.role,
                    parts=[types.Part.from_text(text=message.content)]
                )
            )
        return contents


    def invoke(self, prompt=None, contents=None, system_instruction="", temperature=0.9, top_p=0.9, top_k=40, max_output_tokens=8192):
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


    def invoke_json(self, prompt, schema, system_instruction=""):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        return response.parsed
