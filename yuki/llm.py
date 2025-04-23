import os
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel

LOGS_DIR = "./logs/"
for dir in [LOGS_DIR]:
    os.makedirs(dir, exist_ok=True)

LOG_PATH = LOGS_DIR + "llm.log"
for file in [LOG_PATH]:
    if not os.path.exists(file):
        open(file, 'w').close()



class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.time = datetime.now()



class GoogleChatAI:
    def __init__(self, api_key):
        self.log("\n---------- Starting new GoogleChatAI instance... ----------\n", timestamp=False)
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-preview-04-17"


    def log(self, text, timestamp=True, sub_log=False):
        with open(LOG_PATH, 'a', encoding='utf-8') as file:
            if sub_log:
                file.write(f"                    \t{text}\n")
            elif timestamp:
                file.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')}\t{text}\n")
            else:
                file.write(f"{text}\n")


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


    def invoke(self, prompt=None, contents=None, system_instruction="", temperature=0.9, top_p=0.9, top_k=40, max_output_tokens=8192):
        self.log("invoke() - API call")
        
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
        self.log("invoke_json() - API call")
        
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
