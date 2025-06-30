# memory.py
import chromadb
import uuid

class LongTermMemory:
    def __init__(self, db_path="./memoria_crew"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="aprendizados_de_tarefas")

    def store_learning(self, task_description: str, summary: str):
        learning_id = str(uuid.uuid4())
        self.collection.add(
            documents=[summary],
            metadatas=[{"task": task_description}],
            ids=[learning_id]
        )

    def retrieve_learnings(self, query: str, n_results: int = 3) -> list:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []