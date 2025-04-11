#https://github.com/ALucek/agentic-memory/blob/main/agentic_memory.ipynb

import os
from dotenv import load_dotenv
from datetime import datetime
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from chunking_evaluation.chunking import RecursiveTokenChunker

from llm import GoogleChatAI, Message, BaseModel
from prompts import (
    BASE_INSTRUCTION,
    EPISODIC_INSTRUCTION,
    SEMANTIC_INSTRUCTION,
    UPDATE_EPISODIC,
)

DOCS_DIR = "./docs/"
os.makedirs(DOCS_DIR, exist_ok=True)

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

        self.load_docs()

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
    
    def load_pdf(self, path):
        loader = PyPDFLoader(path)
        pages = []
        for page in loader.load():
            pages.append(page)

        document = " ".join(page.page_content for page in pages)
        
        recursive_character_chunker = RecursiveTokenChunker(
            chunk_size=800,
            chunk_overlap=0,
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        recursive_character_chunks = recursive_character_chunker.split_text(document)

        # Delete existant chunks with same path
        self.vdb_semantic.delete(where={"path": path})

        for i, chunk in enumerate(recursive_character_chunks):
            self.vdb_semantic.add(
                documents=[chunk],
                metadatas={
                    "path": path
                },
                ids=[f"{i}"]
            )


    def load_docs(self):
        docs = []
        for doc_name in os.listdir(DOCS_DIR):
            doc_path = os.path.join(DOCS_DIR, doc_name)
            docs.append(doc_path)

        self.vdb_semantic.delete(where={"path": "./docs/MOSES C++ Technical Specification (2025).pdf"})

        for doc in docs:
            if doc.endswith(".pdf"):
                self.load_pdf(doc)


    def query_semantic(self, query):
        return self.vdb_semantic.query(
            query_texts=[query],
            n_results=15
        )


    def create_semantic_instruction(self, user_message: Message):
        mem = self.query_semantic(user_message.content)
        if not mem["documents"][0][0]:
            return ""

        paths = [i["path"] for i in mem["metadatas"][0]]
        documents = mem["documents"][0]

        chunks = ""
        for i, chunk in enumerate(documents):
            chunks += f"\nCHUNK {i+1} (from {paths[i]}):\n"
            chunks += chunk.strip()

        return SEMANTIC_INSTRUCTION.format(
            user_name=self.user_name,
            chunks=chunks
        )



    #
    # System Instruction
    #

    def create_system_instruction(self, user_message: Message):
        base_instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )
        episodic_instruction = self.create_episodic_instruction(user_message)
        semantic_instruction = self.create_semantic_instruction(user_message)

        instruction = base_instruction + "\n\n" + episodic_instruction + "\n\n" + semantic_instruction
        return instruction











def main():
    yuki = Yuki("Yuki", "Elias")
    yuki.chat()

if __name__ == "__main__":
    main()
