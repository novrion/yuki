# ========== TODO ==========
# image + audio

from datetime import datetime
from llm import GoogleChatAI, Message, BaseModel
from memory import Memory
from episodic_memory import EpisodicMemory
from procedural_memory import ProceduralMemory
from semantic_memory import SemanticMemory
from temporal_memory import TemporalMemory
from working_memory import WorkingMemory
import chromadb
import os

from prompts import (
    BASE_INSTRUCTION,
    EPISODIC_INSTRUCTION,
    PROCEDURAL_INSTRUCTION,
    SEMANTIC_INSTRUCTION,
    TEMPORAL_INSTRUCTION,

    SHOULD_AUTO_MSG_PROMPT,
    SHOULD_RESPOND_PROMPT,
    AUTO_MSG_PROMPT
)

# Storage
DOCS_DIR = "./docs"
LOG_PATH = "./log"
PROCEDURAL_MEM_PATH = "./mem/notes"
TEMPORAL_MEM_PATH = "./mem/commitments"
WORKING_MEM_PATH = "./mem/messages"
os.makedirs("./mem", exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
for file in [LOG_PATH, PROCEDURAL_MEM_PATH, TEMPORAL_MEM_PATH, WORKING_MEM_PATH]:
    if not os.path.exists(file):
        open(file, 'w').close()

# Parameters
IDLE_TIME = 3 * 60
AUTO_MSG_INTERVAL = 3 * 60
QUERY_CONTEXT_LENGTH = 10

class Yuki:
    def __init__(self, ai_name, user_name, api_key):
        self.ai_name = ai_name
        self.user_name = user_name

        # Clients
        self.llm = GoogleChatAI(api_key=api_key)
        self.vdb_client = chromadb.PersistentClient(path="./vdb")

        # Log
        self.log("\n\n==================== Initialising Yuki ====================")

        # Memories
        self.episodic_memory = EpisodicMemory(LOG_PATH, ai_name, user_name, self.vdb_client)
        self.procedural_memory = ProceduralMemory(LOG_PATH, PROCEDURAL_MEM_PATH, ai_name, user_name, self.llm)
        self.semantic_memory = SemanticMemory(LOG_PATH, DOCS_DIR, ai_name, user_name, self.vdb_client)
        self.temporal_memory = TemporalMemory(LOG_PATH, TEMPORAL_MEM_PATH, ai_name, user_name, self.llm)
        self.working_memory = WorkingMemory(LOG_PATH, WORKING_MEM_PATH, ai_name, user_name)

        # Helper variables
        self.pending_memory_update = False
        self.last_memory_update = datetime.now().timestamp()
        self.last_auto_msg_attempt = datetime.now().timestamp()



    def log(self, text, timestamp=True, sub_log=False):
        with open(LOG_PATH, 'a', encoding='utf-8') as file:
            if sub_log:
                file.write(f"                    \t{text}\n")
            elif timestamp:
                file.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')}\t{text}\n")
            else:
                file.write(f"{text}\n")



    def make_instruction(self, query, context):
        instruction = []

        # Base Personality
        instruction.append(
            BASE_INSTRUCTION.format(
                ai_name=self.ai_name,
                user_name=self.user_name
            )
        )

        # Procedural Memory
        procedural_context = self.procedural_memory.retrieve()
        if procedural_context:
            formatted_notes = "\n".join(procedural_context)
            instruction.append(
                PROCEDURAL_INSTRUCTION.format(
                    user_name=self.user_name,
                    notes=formatted_notes
                )
            )

        # Temporal Memory
        temporal_context = self.temporal_memory.retrieve()
        time = temporal_context["time"]
        if temporal_context["commitments"]:
            commitments = temporal_context["commitments"]
            formatted_commitments = "\n".join([f"[{commitment['time']}]: {commitment['commitment']}" for commitment in commitments])
        else:
            formatted_commitments = "No commitments"
        instruction.append(
            TEMPORAL_INSTRUCTION.format(
                user_name=self.user_name,
                time=time,
                commitments=formatted_commitments
            )
        )

        # Episodic Memory
        episodic_context = self.episodic_memory.retrieve(query, context)
        if episodic_context:
            formatted_context = "\n\n".join(episodic_context)
            instruction.append(
                EPISODIC_INSTRUCTION.format(
                    user_name=self.user_name,
                    context=formatted_context
                )
            )

        return "\n".join(instruction)





    #
    # Autonomous Decision-Making
    #

    def should_auto_msg(self):
        self.last_auto_msg_attempt = datetime.now().timestamp()

        class should_auto_msg_schema(BaseModel):
            decision: bool
            reason: str

        temporal_context = self.temporal_memory.retrieve()
        time = temporal_context["time"]
        if temporal_context["commitments"]:
            commitments = temporal_context["commitments"]
            formatted_commitments = "\n".join([f"[{commitment['time']}]: {commitment['commitment']}" for commitment in commitments])
        else:
            formatted_commitments = "No commitments"

        prompt = SHOULD_AUTO_MSG_PROMPT.format(
            user_name=self.user_name,
            time=time,
            commitments=formatted_commitments,
            previous_messages=self.working_memory.format_conversation(messages=self.working_memory.mem, timestamps=True)
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


        self.log(f"Auto message attempt: {decision.decision} ({decision.reason})")
        return decision.decision


    def should_respond(self):
        class should_respond_schema(BaseModel):
            decision: bool
            reason: str

        prompt = SHOULD_RESPOND_PROMPT.format(
            user_name=self.user_name,
            message_history=self.working_memory.format_conversation(messages=self.working_memory.mem[:-1] , timestamps=True),
            latest_message=self.working_memory.format_message(self.working_memory.mem[-1])
        )

        instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )

        decision: should_respond_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=should_respond_schema,
            system_instruction=instruction
        )

        self.log(f"Response: {decision.decision} ({decision.reason})")
        return decision.decision





    #
    # Generation
    #

    def make_query_context(self, msg_history):
        if len(msg_history) > QUERY_CONTEXT_LENGTH:
            return self.working_memory.format_conversation(msg_history[-QUERY_CONTEXT_LENGTH:])
        else:
            return self.working_memory.format_conversation(msg_history)


    def remove_timestamp(self, content):
        import re
        pattern = r'^\[\d{2}:\d{2}\]\s*'
        return re.sub(pattern, '', content)


    def generate_auto_msg(self):
        class auto_msg_schema(BaseModel):
            message: str

        temporal_context = self.temporal_memory.retrieve()
        time = temporal_context["time"]
        if temporal_context["commitments"]:
            commitments = temporal_context["commitments"]
            formatted_commitments = "\n".join([f"[{commitment['time']}]: {commitment['commitment']}" for commitment in commitments])
        else:
            formatted_commitments = "No commitments"

        prompt = AUTO_MSG_PROMPT.format(
            user_name=self.user_name,
            time=time,
            commitments=formatted_commitments,
            previous_messages=self.working_memory.format_conversation(messages=self.working_memory.mem)
        )

        instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )

        result: auto_msg_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=auto_msg_schema,
            system_instruction=instruction
        )

        self.log(f"Generated auto message: {result.message}")

        ai_message = Message(self.remove_timestamp(result.message), "model")
        return ai_message


    def generate_response(self):
        msg_history = self.working_memory.retrieve()
        query = msg_history[-1].content
        query_context = self.make_query_context(msg_history)
        
        chunks = ""
        for i, chunk in enumerate(self.semantic_memory.retrieve(query, query_context)):
            chunks += f"\nCHUNK {i+1} (from {chunk['path']}):\n"
            chunks += chunk["content"]
        semantic_context = Message(
            SEMANTIC_INSTRUCTION.format(
                user_name=self.user_name,
                chunks=chunks
            ),
            "user"
        )
        msg_history = self.working_memory.retrieve()

        response = self.llm.invoke(
            contents=[semantic_context, *msg_history],
            system_instruction=self.make_instruction(query, query_context)
        )

        self.log(f"Generated response: {response}")

        ai_message = Message(self.remove_timestamp(response), "model")
        return ai_message





    #
    # Main Update Loop
    #

    def seconds_since_last_msg(self):
        if self.working_memory.mem:
            return datetime.now().timestamp() - self.working_memory.mem[-1].timestamp
        else:
            return 24 * 60 * 60

    def seconds_since_last_auto_msg(self):
        return datetime.now().timestamp() - self.last_auto_msg_attempt

    def tick(self, user_input=None):
        idle = self.seconds_since_last_msg() > IDLE_TIME
        pending_auto_msg = self.seconds_since_last_auto_msg() > AUTO_MSG_INTERVAL

        # Memory update once after each conversation
        if self.pending_memory_update and idle:
            self.pending_memory_update = False
            self.last_memory_update = datetime.now().timestamp()

            conversation = self.working_memory.get_most_recent_conversation(IDLE_TIME)
            self.working_memory.update()

            self.episodic_memory.update(conversation)
            self.procedural_memory.update(conversation)
            self.temporal_memory.update(conversation)


        msg = None
        image = None
        audio = None

        # User-initiated message
        if user_input:
            self.log(f"User input: {user_input}")
            self.pending_memory_update = True

            user_message = Message(user_input, "user")
            self.working_memory.add_message(user_message)

            if self.should_respond():
                ai_message = self.generate_response()
                self.working_memory.add_message(ai_message)
                msg = ai_message.content

        # AI-initiated message
        elif idle and pending_auto_msg and self.should_auto_msg():
            self.pending_memory_update = True # Remove? Only update memory if user involved in conversation?
            
            ai_message = self.generate_auto_msg()
            self.working_memory.add_message(ai_message)
            msg = ai_message.content

        return {
            "msg": msg,
            "image": image,
            "audio": audio
        }
