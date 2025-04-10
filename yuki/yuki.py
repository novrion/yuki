import os
from dotenv import load_dotenv
from datetime import datetime
import chromadb

from llm import GoogleChatAI, Message, BaseModel
from prompts import (
    BASE_INSTRUCTION,
    EPISODIC_INSTRUCTION,
    UPDATE_EPISODIC,
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class Yuki:
    def __init__(self, ai_name, user_name):
        self.ai_name = ai_name
        self.user_name = user_name
 
        self.llm = GoogleChatAI(api_key=GOOGLE_API_KEY)

        self.messages = []
        self.system_instruction = ""

        self.what_worked = set()
        self.what_to_avoid = set()

        self.vdb_client = chromadb.PersistentClient(path="./vdb")
        self.vdb_episodic = self.vdb_client.get_or_create_collection(name="episodic_memory")
        self.vdb_semantic = self.vdb_client.get_or_create_collection(name="semantic_memory")

    def chat(self):
        while True:
            user_input = input(f"{self.user_name}: ")
            if user_input.lower() == "exit":
                self.update_episodic_memory()
                break

            user_message = self.llm.user_message(user_input)

            system_instruction = self.create_system_instruction(user_message)

            self.messages.append(user_message)

            response = self.llm.invoke(
                contents=self.messages,
                system_instruction=system_instruction
            )

            print(f"{self.ai_name}: ", response)

            ai_message = self.llm.ai_message(response)
            self.messages.append(ai_message)


    #
    # Episodic Memory
    #

    def format_conversation(self):
        conversation = []
        for message in self.messages:
            role = self.user_name.upper() if message.role == "user" else self.ai_name.upper()
            conversation.append(f"{role}: {message.content}")
        return "\n".join(conversation)


    def update_episodic_memory(self):

        class episodic_schema(BaseModel):
            context_tags: list[str]
            conversation_summary: str
            what_worked: str
            what_to_avoid: str

        conversation = self.format_conversation()
        prompt = UPDATE_EPISODIC.format(conversation=conversation)
        mem: episodic_schema = self.llm.invoke_json(
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


    def create_episodic_instruction(self, user_message: Message):
        mem = self.query_episodic(user_message.content)
        if not mem["documents"][0]:
            return ""

        self.what_worked.update(mem["metadatas"][0][0]["what_worked"].split('. '))
        self.what_to_avoid.update(mem["metadatas"][0][0]["what_to_avoid"].split('. '))

        current_conversation_match = mem["documents"][0][0].replace('\n', '\\n')
        what_worked = " ".join(self.what_worked)
        what_to_avoid = " ".join(self.what_to_avoid)

        return EPISODIC_INSTRUCTION.format(
            user_name=self.user_name,
            current_conversation_match=current_conversation_match,
            what_worked=what_worked,
            what_to_avoid=what_to_avoid
        )



    #
    # Semantic Memory
    #

    #
    # System Instruction
    #

    def create_system_instruction(self, user_message: Message):
        base_instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )

        episodic_instruction = self.create_episodic_instruction(user_message)

        return base_instruction + "\n\n" + episodic_instruction











def main():
    yuki = Yuki("Yuki", "Elias")
    yuki.chat()

if __name__ == "__main__":
    main()
