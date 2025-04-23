#https://github.com/ALucek/agentic-memory/blob/main/agentic_memory.ipynb

# TODO: add more conversations (more query outputs) for episodic memory.
# TODO: decay based long-term memory system (should check it out): https://arxiv.org/pdf/2305.10250
# TODO: add autonomous decision-making
# TODO: Fine-tuning...
# TODO: Does the AI have a good long-term memory of the user as of now?
# TODO: Better time awareness. Perhaps add timestamps to all messages. But don't show in vdb and previous conversations (except for the start and end times + dates perhaps)
    # Maybe add user (system) message right before the AI answers?
#TODO: Does the AI have semantic memory in all cases? I.e. auto messages don't have semantic context right?

import os
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
    UPDATE_TIME_COMMITMENTS,
    UPDATE_EPISODIC,
    UPDATE_PROCEDURAL,
    SHOULD_AUTO_MSG_PROMPT,
    GENERATE_AUTO_MSG_PROMPT,
)

DOCS_DIR = "./docs/"
MEM_DIR = "./mem/"
for dir in [DOCS_DIR, MEM_DIR]:
    os.makedirs(dir, exist_ok=True)

LOG_PATH = "yuki.log"
TIME_COMMITMENTS_PATH = MEM_DIR + "time_commitments"
PROCEDURAL_PATH = MEM_DIR + "procedural_memory"
PREVIOUS_CONVERSATIONS_PATH = MEM_DIR + "previous_conversations"
for file in [LOG_PATH, TIME_COMMITMENTS_PATH, PROCEDURAL_PATH, PREVIOUS_CONVERSATIONS_PATH]:
    if not os.path.exists(file):
        open(file, 'w').close()

IDLE_TIME = 2 * 60 # seconds
MEMORY_DECAY_INTERVAL = 1000000 # seconds
AUTO_MSG_INTERVAL = 3 * 60 # seconds (MUST BE LONGER THAN IDLE_TIME)


class Yuki:
    def __init__(self, ai_name, user_name, api_key):
        self.log("\n---------- Starting new Yui instance... ----------\n", timestamp=False)
        
        self.ai_name = ai_name
        self.user_name = user_name
 
        self.llm = GoogleChatAI(api_key=api_key)

        self.pending_update_after_conversation = False
        self.last_memory_decay_time = datetime.now()
        self.last_auto_msg_check_time = datetime.now()

        self.previous_messages = []
        self.messages = []
        self.system_instruction = ""
        self.previous_conversations = []
        self.time_commitments = []

        self.what_worked = set()
        self.what_to_avoid = set()

        self.vdb_client = chromadb.PersistentClient(path="./vdb")
        self.vdb_episodic = self.vdb_client.get_or_create_collection(name="episodic_memory")
        self.vdb_semantic = self.vdb_client.get_or_create_collection(name="semantic_memory")

        self.load_docs()
        self.load_previous_conversations()
        self.load_time_commitments()



    def log(self, text, timestamp=True, sub_log=False):
        with open(LOG_PATH, 'a', encoding='utf-8') as file:
            if sub_log:
                file.write(f"                    \t{text}\n")
            elif timestamp:
                file.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')}\t{text}\n")
            else:
                file.write(f"{text}\n")





    #
    # Awareness
    #

    def update_time_commitments(self):
        class time_commitments_schema(BaseModel):
            time_commitments: list[str]

        conversation = self.format_conversation(self.messages)
        prompt = UPDATE_TIME_COMMITMENTS.format(
            ai_name=self.ai_name,
            user_name=self.user_name,
            conversation=conversation,
            time=datetime.now().strftime('%H:%M on %A, %d %B'),
            time_commitments=self.time_commitments
        )

        mem: time_commitments_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=time_commitments_schema
        )

        self.time_commitments = []
        with open(TIME_COMMITMENTS_PATH, 'w', encoding='utf-8') as file:
            for commitment in mem.time_commitments:
                file.write(f"{commitment.strip()}\n")
                self.time_commitments.append(commitment)

        self.log(f"Updated time commitments and wrote to '{TIME_COMMITMENTS_PATH}'")
        self.log(f"Time commitments: {self.time_commitments}", sub_log=True)


    def load_time_commitments(self):
        with open(TIME_COMMITMENTS_PATH, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        self.time_commitments = []
        for commitment in lines:
            self.time_commitments.append(commitment.strip())

        self.log(f"Loaded time commitments from '{TIME_COMMITMENTS_PATH}'")
        self.log(f"Time commitments: {self.time_commitments}", sub_log=True)


    def create_awareness_instruction(self):
        time = datetime.now().strftime('%H:%M on %A, %d %B')
        time_commitments = "\n".join(self.time_commitments)

        if not time_commitments:
            return f"Current time: {time}"

        return AWARENESS_INSTRUCTION.format(
            user_name=self.user_name,
            time=time,
            time_commitments=time_commitments
        )



    #
    # Episodic Memory
    #

    def format_conversation(self, conversation, include_start_time=True):
        if not conversation:
            return ""

        formatted = []
        if include_start_time:
            formatted.append(f"[CONVERSATION START TIME: {conversation[0].time.strftime('%H:%M on %A, %d %B')}]")
        for message in conversation:
            role = self.user_name.upper() if message.role == "user" else self.ai_name.upper()
            formatted.append(f"{role}: {message.content}")
        formatted.append(f"[CONVERSATION END TIME: {conversation[-1].time.strftime('%H:%M on %A, %d %B')}]")

        return "\n".join(formatted)


    def store_conversations(self):
        convos = self.previous_conversations
        if len(convos) > 2:
            convos = convos[-2:]

        convos_escaped = [convo.replace('\n', '\\n') for convo in convos]
        with open(PREVIOUS_CONVERSATIONS_PATH, 'w', encoding='utf-8') as file:
            for convo in convos_escaped:
                file.write(f"{convo}\n")

        self.log(f"Stored conversations to '{PREVIOUS_CONVERSATIONS_PATH}'")


    def load_previous_conversations(self):
        with open(PREVIOUS_CONVERSATIONS_PATH, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for convo in lines:
            self.previous_conversations.append(convo.replace('\\n', '\n').strip())

        self.log(f"Loaded previous conversations from '{PREVIOUS_CONVERSATIONS_PATH}'")


    def update_episodic_memory(self):
        class episodic_schema(BaseModel):
            context_tags: list[str]
            conversation_summary: str
            what_worked: str
            what_to_avoid: str

        conversation = self.format_conversation(self.messages)
        self.previous_conversations.append(conversation)
        self.store_conversations()

        prompt = UPDATE_EPISODIC.format(
            ai_name=self.ai_name,
            user_name=self.user_name,
            conversation=conversation
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
                "what_to_avoid": mem.what_to_avoid,
            },
            ids=[f"{datetime.now().timestamp()}"]
        )

        self.log("Updated episodic memory")


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

        self.log(f"Loaded pdf from '{path}'")


    def load_text(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            document = file.read()
        self.load_chunks(self.get_chunks(document), path)

        self.log(f"Loaded text from '{path}'")


    def load_docs(self):
        docs = []
        for doc_name in os.listdir(DOCS_DIR):
            doc_path = os.path.join(DOCS_DIR, doc_name)
            docs.append(doc_path)

        self.log(f"Attempting to load documents in '{DOCS_DIR}'")
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

        self.log(f"Updated procedural memory and wrote to '{PROCEDURAL_PATH}'")


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

        return instruction



    #
    # Autonomous Decision Making
    #

    def should_auto_msg(self):
        self.log("Attempting to send autonomous message...")

        self.last_auto_msg_check_time = datetime.now()

        class should_auto_msg_schema(BaseModel):
            decision: bool
            reason: str

        prompt = SHOULD_AUTO_MSG_PROMPT.format(
            user_name=self.user_name,
            time=datetime.now().strftime('%H:%M (%B %d)'),
            time_commitments="\n".join(self.time_commitments),
            previous_messages=self.format_conversation(
                conversation=self.previous_messages,
                include_start_time=False
            )
        )

        instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )
        
        decision: should_auto_msg_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=should_auto_msg_schema,
            system_instruction=instruction
        )

        if decision.decision:
            self.log(f"Decided to send autonomous message ({decision.reason})", sub_log=True)
        else:
            self.log(f"Decided not to send autonomous message ({decision.reason})", sub_log=True)
        
        return decision.decision


    def generate_auto_msg(self):
        class auto_msg_schema(BaseModel):
            message: str

        prompt = GENERATE_AUTO_MSG_PROMPT.format(
            user_name=self.user_name,
            time=datetime.now().strftime('%H:%M'),
            time_commitments="\n".join(self.time_commitments),
            previous_messages=self.format_conversation(
                conversation=self.previous_messages,
                include_start_time=False
            )
        )

        instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )

        auto_msg: auto_msg_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=auto_msg_schema,
            system_instruction=instruction
        )

        return auto_msg.message



    #
    # Memory Decay
    #

    def seconds_since_memory_decay(self):
        return datetime.now().timestamp() - self.last_memory_decay_time.timestamp()


    def decay_working_memory(self):
        if len(self.messages) > 6:
            self.previous_messages = self.messages[-6:]
        else:
            self.previous_messages = self.messages
        self.messages = []

        self.log("Decayed working memory")


    def decay_memory(self):
        self.last_memory_decay_time = datetime.now()

        self.log("Decayed memory")



    #
    # Tick - Main update function
    #

    def seconds_since_last_message(self):
        if len(self.messages) > 0:
            return datetime.now().timestamp() - self.messages[-1].time.timestamp()
        elif len(self.previous_messages) > 0:
            return datetime.now().timestamp() - self.previous_messages[-1].time.timestamp()
        else:
            return 24 * 60 * 60

    def seconds_since_last_auto_msg_check(self):
        return datetime.now().timestamp() - self.last_auto_msg_check_time.timestamp()


    def tick(self, user_input=None):

        # Check if ongoing conversation
        if self.seconds_since_last_message() > IDLE_TIME:
            self.conversation_ongoing = False
        
        # Check if update needed after conversation ended
        if not self.conversation_ongoing and self.pending_update_after_conversation:
            self.pending_update_after_conversation = False
            self.update_time_commitments()
            self.update_episodic_memory()
            self.update_procedural_memory()
            self.decay_working_memory()

        # Decay memory
        if self.seconds_since_memory_decay() > MEMORY_DECAY_INTERVAL:
            self.decay_memory()



        msg = None
        image = None
        audio = None


        # User input
        if user_input:
            user_message = self.llm.user_message(user_input)
            system_instruction = self.create_system_instruction(user_message)
            semantic_context = self.llm.user_message(self.create_semantic_context(user_message))

            self.messages.append(user_message)

            response = self.llm.invoke(
                contents=[semantic_context, *self.messages],
                system_instruction=system_instruction
            )

            ai_message = self.llm.ai_message(response)
            self.messages.append(ai_message)

            msg = ai_message.content
            self.log(f"Responded to user input: '{msg}'")


        # AI-initiated message
        elif not self.conversation_ongoing and self.seconds_since_last_auto_msg_check() > AUTO_MSG_INTERVAL and self.should_auto_msg():
            ai_message = self.llm.ai_message(self.generate_auto_msg())
            self.messages.append(ai_message)

            msg = ai_message.content
            self.log(f"Autonomous message: '{msg}'")


        if msg:
            self.conversation_ongoing = True
            self.pending_update_after_conversation = True

        return {
            "msg": msg,
            "image": image,
            "audio": audio
        }
