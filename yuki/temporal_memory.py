# ===== TODO =====
# Sort commitments in chronological order on update

from llm import GoogleChatAI, BaseModel, Message
from datetime import datetime
from memory import Memory
import os

from prompts import UPDATE_COMMITMENTS

class TemporalMemory(Memory):
    def __init__(self, log_path, storage_path, ai_name, user_name, llm):
        super().__init__(log_path, ai_name, user_name)
        self.storage_path = storage_path
        self.llm: GoogleChatAI = llm
        self.mem = []
        self.initialise()

    def initialise(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if os.path.exists(self.storage_path):
            data = self.load_json(self.storage_path)
            if "commitments" in data:
                self.mem = [commitment for commitment in data["commitments"]]
            self.log("Initialised temporal memory")
            self.log("".join([f"[{commitment['time']}] {commitment['commitment']}" for commitment in self.mem]) or "[]", sub_log=True)
        else:
            with open(self.storage_path, 'w') as f:
                f.write('{"commitments": []}')
            self.log("Initialised temporal memory")
            self.log("[]", sub_log=True)

    def update(self, messages: list[Message]):
        class commitmentItem(BaseModel):
            commitment: str
            time: str
        class commitments_schema(BaseModel):
            commitments: list[commitmentItem]

        prompt = UPDATE_COMMITMENTS.format(
            ai_name=self.ai_name,
            user_name=self.user_name,
            conversation=self.format_conversation(messages, timestamps=True),
            time=datetime.now().strftime('%H:%M on %A, %d %B'),
            commitments=self.mem
        )

        result: commitments_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=commitments_schema
        )

        self.mem = []
        for commitment in result.commitments:
            self.mem.append({
                "commitment": commitment.commitment,
                "time": commitment.time
            })

        data = {"commitments": self.mem}
        self.save_json(data, self.storage_path)
        self.log("Updated temporal memory")

    def retrieve(self):
        time = datetime.now().strftime('%H:%M on %A, %d %B')
        commitments = self.mem
        return {"time": time, "commitments": commitments}
