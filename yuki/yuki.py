#https://github.com/ALucek/agentic-memory/blob/main/agentic_memory.ipynb

# TODO: add more conversations (more query outputs) for episodic memory.
# TODO: decay based long-term memory system (should check it out): https://arxiv.org/pdf/2305.10250
# TODO: add autonomous decision-making
# TODO: Fine-tuning...
# TODO: Does the AI have a good long-term memory of the user as of now?
# TODO: Better time awareness. Perhaps add timestamps to all messages. But don't show in vdb and previous conversations (except for the start and end times + dates perhaps)

import os
from dotenv import load_dotenv
from datetime import datetime
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from chunking_evaluation.chunking import RecursiveTokenChunker

from llm import GoogleChatAI, Message, BaseModel
from prompts import (
    BASE_INSTRUCTION,
    AWARENESS_INSTRUCTION,
    EPISODIC_INSTRUCTION,
    SEMANTIC_INSTRUCTION,
    PROCEDURAL_INSTRUCTION,
    UPDATE_EPISODIC,
    UPDATE_PROCEDURAL,
)

DOCS_DIR = "./docs/"
for dir in [DOCS_DIR]:
    os.makedirs(dir, exist_ok=True)

PROCEDURAL_PATH = "./procedural_memory"
PREVIOUS_CONVERSATIONS_PATH = "./previous_conversations"
for file in [PROCEDURAL_PATH, PREVIOUS_CONVERSATIONS_PATH]:
    if not os.path.exists(file):
        open(file, 'w').close()

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class Yuki:
    def __init__(self, ai_name, user_name):
        self.ai_name = ai_name
        self.user_name = user_name
 
        self.llm = GoogleChatAI(api_key=GOOGLE_API_KEY)

        self.messages = []
        self.system_instruction = ""
        self.previous_conversations = []

        self.what_worked = set()
        self.what_to_avoid = set()

        self.vdb_client = chromadb.PersistentClient(path="./vdb")
        self.vdb_episodic = self.vdb_client.get_or_create_collection(name="episodic_memory")
        self.vdb_semantic = self.vdb_client.get_or_create_collection(name="semantic_memory")

        self.load_docs()
        self.load_previous_conversations()



    def chat(self):
        while True:
            user_input = input(f"{self.user_name}: ")
            if user_input.lower() == "exit":
                self.update_episodic_memory()
                self.update_procedural_memory()
                break

            user_message = self.llm.user_message(user_input)
            system_instruction = self.create_system_instruction(user_message)
            semantic_context = self.llm.user_message(self.create_semantic_context(user_message))

            self.messages.append(user_message)

            response = self.llm.invoke(
                contents=[semantic_context, *self.messages],
                system_instruction=system_instruction,
            )

            print(f"{self.ai_name}: ", response)
            ai_message = self.llm.ai_message(response)
            
            self.messages.append(ai_message)



    #
    # Awareness
    #

    def create_awareness_instruction(self):
        time = datetime.now().strftime('%H:%M on %A, %d %B')
        return AWARENESS_INSTRUCTION.format(
            time=time
        )



    #
    # Episodic Memory
    #

    def format_conversation(self):
        conversation = []

        conversation.append(f"[CONVERSATION START TIME: {self.messages[0].time.strftime('%H:%M on %A, %d %B')}]")
        for message in self.messages:
            role = self.user_name.upper() if message.role == "user" else self.ai_name.upper()
            conversation.append(f"{role}: {message.content}")
        conversation.append(f"[CONVERSATION END TIME: {self.messages[-1].time.strftime('%H:%M on %A, %d %B')}]")

        return "\n".join(conversation)


    def store_conversations(self):
        convos = self.previous_conversations
        if len(convos) > 2:
            convos = convos[-2:]

        convos_escaped = [convo.replace('\n', '\\n') for convo in convos]
        with open(PREVIOUS_CONVERSATIONS_PATH, 'w', encoding='utf-8') as file:
            for convo in convos_escaped:
                file.write(f"{convo}\n")


    def load_previous_conversations(self):
        with open(PREVIOUS_CONVERSATIONS_PATH, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for convo in lines:
            self.previous_conversations.append(convo.replace('\\n', '\n').strip())


    def update_episodic_memory(self):

        class episodic_schema(BaseModel):
            context_tags: list[str]
            conversation_summary: str
            what_worked: str
            what_to_avoid: str

        conversation = self.format_conversation()
        self.previous_conversations.append(conversation)
        self.store_conversations()

        prompt = UPDATE_EPISODIC.format(
            conversation=conversation,
            ai_name=self.ai_name,
            user_name=self.user_name
        )
        mem: episodic_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=episodic_schema
        )

        self.vdb_episodic.add(
            documents=[conversation],
            metadatas={
                "timestamp": f"datetime.now().timestamp()",
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
        previous_conversations = [i.replace('\n', '\\n') for i in self.previous_conversations]

        if current_conversation_match in previous_conversations:
            previous_conversations.remove(current_conversation_match)

        previous_conversations = " | ".join(previous_conversations)
        what_worked = " ".join(self.what_worked)
        what_to_avoid = " ".join(self.what_to_avoid)

        return EPISODIC_INSTRUCTION.format(
            user_name=self.user_name,
            current_conversation_match=current_conversation_match,
            previous_conversations=previous_conversations,
            what_worked=what_worked,
            what_to_avoid=what_to_avoid
        )



    #
    # Semantic Memory
    #

    def load_chunks(self, chunks, path):
        self.vdb_semantic.delete(where={"path": path})
        for chunk in chunks:
            self.vdb_semantic.add(
                documents=[chunk],
                metadatas={
                    "path": path
                },
                ids=[f"{datetime.now().timestamp()}"]
            )

    def get_chunks(self, document):
        recursive_character_chunker = RecursiveTokenChunker(
            chunk_size=800,
            chunk_overlap=0,
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        return recursive_character_chunker.split_text(document)

    
    def load_pdf(self, path):
        loader = PyPDFLoader(path)
        pages = []
        for page in loader.load():
            pages.append(page)

        document = " ".join(page.page_content for page in pages)
        self.load_chunks(self.get_chunks(document), path)


    def load_text(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            document = file.read()
        self.load_chunks(self.get_chunks(document), path)


    def load_docs(self):
        docs = []
        for doc_name in os.listdir(DOCS_DIR):
            doc_path = os.path.join(DOCS_DIR, doc_name)
            docs.append(doc_path)

        for doc in docs:
            if doc.endswith(".pdf"):
                self.load_pdf(doc)
            else:
                self.load_text(doc)


    def query_semantic(self, query):
        return self.vdb_semantic.query(
            query_texts=[query],
            n_results=5
        )


    def create_semantic_context(self, user_message: Message):
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
    # Procedural Memory
    #
    
    def update_procedural_memory(self):
        with open(PROCEDURAL_PATH, 'r', encoding='utf-8') as file:
            current_takeaways = file.read().strip()

        prompt = UPDATE_PROCEDURAL.format(
            ai_name=self.ai_name,
            current_takeaways=current_takeaways,
            what_worked=". ".join(self.what_worked),
            what_to_avoid=". ".join(self.what_to_avoid)
        )

        procedural_memory = self.llm.invoke(
            prompt=prompt,
            temperature=0
        )

        with open(PROCEDURAL_PATH, 'w', encoding='utf-8') as file:
            file.write(procedural_memory)

    def create_procedural_instruction(self):
        with open(PROCEDURAL_PATH, 'r', encoding='utf-8') as file:
            procedural_memory = file.read().strip()

        if not procedural_memory:
            return ""

        return PROCEDURAL_INSTRUCTION.format(
            user_name=self.user_name,
            procedural_memory=procedural_memory
        )



    #
    # System Instruction
    #

    def create_system_instruction(self, user_message: Message):
        base_instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )
        awareness_instruction = self.create_awareness_instruction()
        episodic_instruction = self.create_episodic_instruction(user_message)
        procedural_instruction = self.create_procedural_instruction()

        instruction = base_instruction
        instruction += "\n\n" + awareness_instruction
        instruction += "\n\n" + episodic_instruction
        if procedural_instruction:
            instruction += "\n\n" + procedural_instruction

        print(instruction)
        return instruction





def main():
    yuki = Yuki("Yuki", "Elias")
    yuki.chat()

if __name__ == "__main__":
    main()
