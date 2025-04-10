import os
from dotenv import load_dotenv
import json
import chromadb
from datetime import datetime

from llm import llm, BaseModel
from prompts import (
    BASE_INSTRUCTION,
    EPISODIC_PROMPT_TEMPLATE,
)

load_dotenv()
API_KEY = os.getenv("API_KEY")

class Yuki:
    def __init__(self, ai_name, user_name):
        self.llm = llm(API_KEY)

        self.ai_name = ai_name
        self.user_name = user_name

        self.msgs = []
        self.conversations = []
        self.what_worked = set()
        self.what_to_avoid = set()

        # Vector Database
        self.vdb_client = chromadb.PersistentClient(path="./vdb")
        self.vdb_episodic = self.vdb_client.get_or_create_collection(
            name="episodic_memory",
            metadata={
                "description": "Collection of historical conversations and takeaways."
            }
        )
        self.vdb_semantic = self.vdb_client.get_or_create_collection(
            name="semantic_memory",
            metadata={
                "description": "Collection of documents (RAG)."
            }
        )



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
            system_instruction=self.construct_system_instruction(user_msg)
        )

        self.msgs.append({
            "role": "model",
            "content": response
        })

        return response


    def format_conversation(self):
        conversation = []
        for msg in self.msgs:
            role = self.user_name.upper() if msg["role"] == "user" else "YUKI"
            conversation.append(f"{role}: {msg["content"]}")
        return "\n".join(conversation)











    # https://github.com/ALucek/agentic-memory/blob/main/agentic_memory.ipynb
    # https://www.youtube.com/watch?v=VKPngyO0iKg


    #
    # Episodic Memory
    #

    def update_episodic(self):

        class episodic_schema(BaseModel):
            context_tags: list[str]
            conversation_summary: str
            what_worked: str
            what_to_avoid: str

        conversation = self.format_conversation()
        prompt = EPISODIC_PROMPT_TEMPLATE.format(conversation=conversation)
        mem: episodic_schema = self.llm.generate_json(
            prompt=prompt,
            schema=episodic_schema
        )

        self.vdb_episodic.add(
            documents=[conversation],
            metadatas={
                "what_worked": mem.what_worked,
                "what_to_avoid": mem.what_to_avoid
            },
            ids=[f"{datetime.now().timestamp()}"]
        )

 
    def query_episodic(self, query):
        return self.vdb_episodic.query(
            query_texts=[query],
            n_results=1
        )


    def construct_episodic_instruction(self, query):
        mem = self.query_episodic(query)
        if not mem["documents"][0]:
            return ""

        current_conversation_match = mem["documents"][0][0].replace('\n', '\\n')
        self.what_worked.update(mem["metadatas"][0][0]["what_worked"].split('. '))
        self.what_to_avoid.update(mem["metadatas"][0][0]["what_to_avoid"].split('. '))

        if len(self.conversations) >= 4:
            previous_conversations = self.conversations[-4:]
        else:
            previous_conversations = self.conversations

        return f"""You recall similar conversations with {self.user_name}, here are the details:

Current Conversation Match: {current_conversation_match}
Previous Conversations: {" | ".join(previous_conversations)}
What has worked well: {" ".join(self.what_worked)}
what to avoid: {" ".join(self.what_to_avoid)}

Use these memories as context for your response to {self.user_name}."""



    #
    # System Instruction Construction
    #

    def construct_system_instruction(self, user_msg):
        base_instruction = BASE_INSTRUCTION.format(ai_name=self.ai_name, user_name=self.user_name)
        episodic_instruction = self.construct_episodic_instruction(user_msg)

        final = base_instruction + "\n\n" + episodic_instruction
        print(final)
        return final
