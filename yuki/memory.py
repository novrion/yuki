# ========== Memory Structure ==========
# Working Memory: current conversation + past conversations (json file)
# Temporal Memory: Time awareness and scheduling (json file for commitments)
# Semantic Memory: RAG of docs (vector database)
# Episodic Memory: RAG of past conversations + what_worked + what_to_avoid
# Procedural Memory: Dynamic behavioural guidelines (json file)

from datetime import datetime
from llm import Message
import json

class Memory:
    def __init__(self, log_path, ai_name, user_name):
        self.ai_name = ai_name
        self.user_name = user_name
        self.log_path = log_path

    @staticmethod
    def save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            return {}

    def log(self, text, timestamp=True, sub_log=False):
        with open(self.log_path, 'a', encoding='utf-8') as file:
            if sub_log:
                file.write(f"                    \t{text}\n")
            elif timestamp:
                file.write(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')}\t{text}\n")
            else:
                file.write(f"{text}\n")

    def format_conversation(self, messages, roles=True, timestamps=False):
        ret = []
        for msg in messages:
            ret.append(self.format_message(msg, roles, timestamps))
        return "\n".join(ret) 

    def format_message(self, message, role=False, timestamp=False):
        ret = ""
        if timestamp:
            time = datetime.fromtimestamp(message.timestamp).strftime('%H:%M')
            ret += f"[{time}] "
        if role:
            role = self.user_name if message.role == "user" else self.ai_name
            ret += f"{role}: "
        ret += message.content.replace('\n', '\\n').strip()
        return ret
