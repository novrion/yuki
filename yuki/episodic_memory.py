from datetime import datetime
from memory import Memory
from llm import Message
import chromadb
import math
import os

from prompts import UPDATE_COMMITMENTS

BASE_IMPORTANCE = 0.1
MAX_IMPORTANCE = 1.0
POWER_LAW_EXPONENT = 0.7   # Controls curve shape (0.5-0.8 works well)
BASE_FORGETTING_RATE = 2.0 # Base rate of forgetting
CUTOFF_FACTOR = 0.05       # Threshold for memory deletion
LIGHT_REINFORCEMENT = 0.05 # Minor reinforcement for retrieved but unused memories
STRONG_REINFORCEMENT = 0.2 # Stronger reinforcement for directly used memories
N_RAW_RESULTS = 4          # Number of results to retrieve per query
N_RESULTS = 2              # Number of results to return

# Weight factors for scoring (must sum to 1.0)
RELEVANCE_WEIGHT = 0.6     # Emphasis on query relevance 
RECENCY_WEIGHT = 0.2       # Emphasis on memory recency
IMPORTANCE_WEIGHT = 0.2    # Emphasis on memory importance

class EpisodicMemory(Memory):
    def __init__(self, log_path, ai_name, user_name, vdb_client):
        super().__init__(log_path, ai_name, user_name)
        self.vdb = vdb_client.get_or_create_collection(name="episodic_memory")

    def update(self, messages: list[Message], importance=BASE_IMPORTANCE):
        self.decay_memory()
        conversation = self.format_conversation(messages, roles=False)

        self.vdb.add(
            documents=[conversation],
            metadatas=[{
                "timestamp": datetime.now().timestamp(),
                "importance": importance,
                "resistance": 0.0
            }],
            ids=[f"{datetime.now().timestamp()}"]
        )

        self.log("Updated episodic memory - added conversation to vector database")


    def calculate_memory_weight(self, timestamp, resistance):
        time_diff = datetime.now().timestamp() - timestamp
        days_old = time_diff / (24 * 60 * 60)
        
        effective_power = POWER_LAW_EXPONENT / (1 + resistance * 3)
        
        # Power law formula: weight = 1 / (1 + (rate * time)^power)
        if days_old < 0.001:  # Avoid division issues with very new memories
            return 1.0
        return 1.0 / (1 + (BASE_FORGETTING_RATE * days_old) ** effective_power)

    def reinforce_memory(self, memory_id, reinforcement_factor):
        results = self.vdb.get(ids=[memory_id])
        if not results["metadatas"]:
            return
        
        metadata = results["metadatas"][0]
        
        current_importance = float(metadata.get("importance", BASE_IMPORTANCE))
        new_importance = current_importance + (MAX_IMPORTANCE - current_importance) * reinforcement_factor
        
        current_resistance = float(metadata.get("resistance", 0.0))
        new_resistance = min(current_resistance + reinforcement_factor, 1.0)
        
        self.vdb.update(
            ids=[memory_id],
            metadatas=[{
                "timestamp": metadata["timestamp"],
                "importance": new_importance,
                "resistance": new_resistance
            }]
        )

    def decay_memory(self):
        all_memories = self.vdb.get()
        if not all_memories["ids"]:
            return

        to_delete = []
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            timestamp = float(metadata["timestamp"])
            resistance = float(metadata.get("resistance", 0.0))
            
            memory_weight = self.calculate_memory_weight(timestamp, resistance)
            if memory_weight < CUTOFF_FACTOR:
                to_delete.append(memory_id)

        if to_delete:
            self.vdb.delete(ids=to_delete)

    def retrieve(self, query, context=None):
        if context:
            queries = [query, context]
        else:
            queries = [query]
            
        raw_results = self.vdb.query(
            query_texts=queries,
            n_results=N_RAW_RESULTS,
            include=["metadatas", "distances", "documents"]
        )
    
        if not raw_results["documents"] or not len(raw_results["documents"]):
            return []
            
        scored_results = []
        for j in range(len(raw_results["documents"])):
            for i, doc in enumerate(raw_results["documents"][j]):
                memory_id = raw_results["ids"][j][i]
                metadata = raw_results["metadatas"][j][i]
                timestamp = float(metadata["timestamp"])
                resistance = float(metadata.get("resistance", 0.0))
                importance = float(metadata.get("importance", BASE_IMPORTANCE))
                
                recency_score = self.calculate_memory_weight(timestamp, resistance)
                relevance_score = 1.0 - raw_results["distances"][j][i]
                importance_score = importance
                
                final_score = (
                    relevance_score * RELEVANCE_WEIGHT +
                    recency_score * RECENCY_WEIGHT +
                    importance_score * IMPORTANCE_WEIGHT
                )
                
                self.reinforce_memory(memory_id, LIGHT_REINFORCEMENT * relevance_score)
                
                # Check if result already in list
                if not any(r["id"] == memory_id for r in scored_results):
                    scored_results.append({
                        "document": doc, 
                        "id": memory_id, 
                        "score": final_score,
                        "query_index": j,
                        "result_index": i
                    })
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        if len(scored_results) > N_RESULTS:
            best_results = scored_results[:N_RESULTS]
        else:
            best_results = scored_results
        
        ret = []
        for result in best_results:
            j = result["query_index"]
            i = result["result_index"]
            relevance_score = 1.0 - raw_results["distances"][j][i]
            self.reinforce_memory(result["id"], STRONG_REINFORCEMENT * relevance_score)
            ret.append(result["document"])
            
        return ret
