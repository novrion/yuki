from datetime import datetime
from memory import Memory
from llm import Message
import os

MAX_STORAGE_DURATION = 23 * 60 * 60
MAX_OLD_MESSAGES = 10 # Max number of messages of the most recent conversation to keep in mem

class WorkingMemory(Memory):
    def __init__(self, log_path, storage_path, ai_name, user_name):
        super().__init__(log_path, ai_name, user_name)
        self.storage_path = storage_path
        self.mem = []
        self.initialise()

    def initialise(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if os.path.exists(self.storage_path):
            data = self.load_json(self.storage_path)
            if "messages" in data:
                current_time = datetime.now().timestamp()
                self.mem = [
                    Message.from_dict(msg) for msg in data["messages"]
                    if current_time - msg["timestamp"] <= MAX_STORAGE_DURATION
                ]
                self.log("Initialised working memory")
                if (self.mem):
                    for msg in self.mem:
                        self.log(f"{msg.role}: {msg.content}", sub_log=True)
                else:
                    self.log("[]", sub_log=True)
            else:
                with open(self.storage_path, 'w') as f:
                    f.write('{"messages": []}')
                self.log("Empty or deformed working memory")
                self.log("[]", sub_log=True)
        else:
            with open(self.storage_path, 'w') as f:
                f.write('{"messages": []}')
            self.log("Initialised working memory")
            self.log("[]", sub_log=True)

    def update(self):
        current_time = datetime.now().timestamp()
        self.mem = [
                msg for msg in self.mem
                if current_time - msg.timestamp <= MAX_STORAGE_DURATION
        ]

        for i in range(len(self.mem)):
            self.mem[i].content = self.mem[i].content.strip()

        if len(self.mem) > MAX_OLD_MESSAGES:
            self.mem = self.mem[-MAX_OLD_MESSAGES:]

        data = {"messages": [msg.to_dict() for msg in self.mem]}
        self.save_json(data, self.storage_path)
        self.log("Updated working memory")

    def retrieve(self):
        return self.mem

    def add_message(self, message: Message):
        self.mem.append(message)

    def get_most_recent_conversation(self, idle_time):
        if len(self.mem) < 2:
            return self.mem.copy()

        for i in range(len(self.mem) - 2, -1, -1):
            time_diff = self.mem[i+1].timestamp - self.mem[i].timestamp
            if time_diff > idle_time:
                return self.mem[i+1:]
        
        return self.mem.copy()


