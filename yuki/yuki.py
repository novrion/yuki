import os
from dotenv import load_dotenv
import json

from llm import llm
from prompts import (
    EPISODIC_PROMPT_TEMPLATE
)

load_dotenv()
API_KEY = os.getenv("API_KEY")

class Yuki:
    def __init__(self, user_name):
        self.llm = llm(API_KEY)
        self.vdb = vdb()
        self.user_name = user_name
        self.system_instruction = ""
        self.msgs = []


    def print_msgs(self):
        for msg in msgs:
            print(f"{msg["role"]}:\t{msg["content"]}\n")


    def msg(self, user_msg):
        self.msgs.append({
            "role": "user",
            "content": user_msg
        })

        response = self.llm.generate_text(
            msgs=self.msgs,
            system_instruction=self.system_instruction
        )

        self.msgs.append({
            "role": "model",
            "content": response
        })

        return response


    def format_conversation(self):
        conversation = []
        for msg in msgs:
            role = self.user_name.upper() if msg["role"] == "user" else "YUKI"
            conversation.append(f"{role}: {msg["content"]}")
        return "\n".join(conversation)


    # Generate json using llm that acts as the episodic memory
    # Create a vector database and add the data from the json to the vector database
    # Create function to query vector database with episodic memory
    # Create a function to regenerate the system instruction with new episodic memory for instance
    # https://github.com/ALucek/agentic-memory/blob/main/agentic_memory.ipynb
    def update_episodic_memory(self):
        prompt = EPISODIC_PROMPT_TEMPLATE.format(conversation=self.format_conversation())
        self.llm.generate_text(prompt=prompt)
