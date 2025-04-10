from google import genai
from google.genai import types
from pydantic import BaseModel

class llm:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"


    def generate_text(self, msgs, prompt=None, system_instruction="", temperature=1.0, top_p=0.9, top_k=40, max_output_tokens=8192):
        if prompt:
            contents = prompt
        else:
            contents = []
            for msg in msgs:
                contents.append(
                    types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

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

    def generate_json(self, prompt, schema):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        return response.parsed
