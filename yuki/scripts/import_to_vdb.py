import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import Message
from episodic_memory import EpisodicMemory
import chromadb

def import_messages(message_file_path, vdb_path="./vdb"):
    vdb_client = chromadb.PersistentClient(path=vdb_path)
    episodic_memory = EpisodicMemory(
        log_path="./scripts/import_log",
        ai_name="Yuki",
        user_name="Elias",
        vdb_client=vdb_client
    )
    
    messages = []
    with open(message_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                messages.append(Message(content=line.strip(), role="user"))
    
    episodic_memory.update(messages, importance=0.8)
    print(f"Successfully imported {len(messages)} messages into episodic memory")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_messages.py <message_file_path>")
        sys.exit(1)
    
    import_messages(sys.argv[1])
