from llm import GoogleChatAI, BaseModel, Message
from memory import Memory
import os

from prompts import (
    UPDATE_NOTES,
    BASE_INSTRUCTION
)

class ProceduralMemory(Memory):
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
            if "notes" in data:
                self.mem = [note for note in data["notes"]]
                self.log("Initialised procedural memory")
                for note in self.mem:
                    self.log(note, sub_log=True)
            else:
                with open(self.storage_path, 'w') as f:
                    f.write('{"notes": []}')
                self.log("Empty or deformed procedural memory")
                self.log("[]", sub_log=true)

        else:
            with open(self.storage_path, 'w') as f:
                f.write('{"notes": []}')
            self.log("Initialised procedural memory")
            self.log("[]", sub_log=true)

    def update(self, messages: list[Message]):                      # What_worked and what_to_avoid missing
        class notes_schema(BaseModel):
            notes: list[str]

        prompt = UPDATE_NOTES.format(
            ai_name=self.ai_name,
            user_name=self.user_name,
            current_notes="\n".join(self.mem),
            conversation=self.format_conversation(messages)
        )

        instruction = BASE_INSTRUCTION.format(
            ai_name=self.ai_name,
            user_name=self.user_name
        )

        result: notes_schema = self.llm.invoke_json(
            prompt=prompt,
            schema=notes_schema,
            system_instruction=instruction
        )

        self.mem = []
        for note in result.notes:
            self.mem.append(note.strip())

        data = {"notes": self.mem}
        self.save_json(data, self.storage_path)

        self.log("Updated procedural memory")

    def retrieve(self):
        return self.mem
