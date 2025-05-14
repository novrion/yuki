from langchain_community.document_loaders import PyPDFLoader
from chunking_evaluation.chunking import RecursiveTokenChunker
from datetime import datetime
from memory import Memory
import chromadb
import os

N_RESULTS = 5

class SemanticMemory(Memory):
    def __init__(self, log_path,docs_dir, ai_name, user_name, vdb_client):
        super().__init__(log_path, ai_name, user_name)
        self.vdb = vdb_client.get_or_create_collection(name="semantic_memory")
        self.docs_dir = docs_dir
        self.initialise()


    def load_chunks(self, chunks, path):
        self.vdb.delete(where={"path": path})
        for chunk in chunks:
            self.vdb.add(
                documents=[chunk],
                metadatas=[{"path": path}],
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
        with open(path, 'r', encoding='utf-8') as f:
            document = f.read()
        self.load_chunks(self.get_chunks(document), path)

    def initialise(self):
        self.log("Initialising semantic memory...")
        os.makedirs(self.docs_dir, exist_ok=True)
        docs = []
        for doc_name in os.listdir(self.docs_dir):
            doc_path = os.path.join(self.docs_dir, doc_name)
            docs.append(doc_path)
        for doc in docs:
            if doc.endswith(".pdf"):
                self.load_pdf(doc)
            else:
                self.load_text(doc)
            self.log(f"Loaded {doc} in vector database", sub_log=True)

    def retrieve(self, query, context=None):
        if context:
            queries = [query, context]
        else:
            queries = [query]
        
        results = self.vdb.query(
            query_texts=queries,
            n_results=N_RESULTS,
            include=["metadatas", "documents", "distances"]
        )
        
        if not results["documents"] or not len(results["documents"]):
            return []
        
        all_results = []
        for query_idx in range(len(results["documents"])):
            for doc_idx in range(len(results["documents"][query_idx])):
                all_results.append({
                    "content": results["documents"][query_idx][doc_idx],
                    "path": results["metadatas"][query_idx][doc_idx]["path"],
                    "distance": results["distances"][query_idx][doc_idx]
                })
        
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:N_RESULTS]
