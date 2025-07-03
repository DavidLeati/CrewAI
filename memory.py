# memory.py
import chromadb
import uuid
import json
import google.generativeai as genai
from app_logger import logger

# 1. Crie uma classe de Embedding personalizada para o ChromaDB
class GoogleGeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Função de embedding personalizada que usa a API da Google (Gemini)
    para converter documentos em vetores para o ChromaDB.
    """
    def __call__(self, input_texts: chromadb.Documents) -> chromadb.Embeddings:
        try:
            # Usa o modelo de embedding da Google, que é otimizado para esta tarefa
            model = 'models/embedding-001'
            # Gera os embeddings para a lista de textos recebida
            return genai.embed_content(model=model,
                                       content=input_texts,
                                       task_type="retrieval_document")["embedding"]
        except Exception as e:
            logger.add_log_for_ui(f"Falha ao gerar embedding com a API da Google: {e}", "error")
            # Retorna uma lista de listas vazias para manter a consistência do tipo
            return [[] for _ in input_texts]

# 2. Modifique a classe LongTermMemory para usar a nova função de embedding
class LongTermMemory:
    """
    Gerencia a memória de longo prazo da Crew, agora configurada para usar
    os embeddings da Google, eliminando a dependência do onnxruntime.
    """
    def __init__(self, db_path="./memoria_crew"):
        self.client = chromadb.PersistentClient(path=db_path)
        # 3. Passe a nossa função de embedding personalizada ao criar/obter a coleção
        self.embedding_function = GoogleGeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="aprendizados_de_tarefas",
            embedding_function=self.embedding_function
        )
        logger.add_log_for_ui("Memória de Longo Prazo inicializada com embeddings da Google.")

    def store_learning(self, task_description: str, project_data: dict):
        """
        Armazena o aprendizado de uma tarefa. O ChromaDB usará automaticamente
        a GoogleGeminiEmbeddingFunction para vetorizar os dados.
        """
        learning_id = str(uuid.uuid4())
        
        if isinstance(project_data, str):
            project_data = {"summary.txt": project_data}

        try:
            # Serializa o dicionário do projeto para ser armazenado como um documento
            project_data_json = json.dumps(project_data, indent=2)
        except TypeError as e:
            logger.add_log_for_ui(f"Erro de serialização ao armazenar aprendizado: {e}", "error")
            project_data_json = json.dumps({"error": "unserializable_data", "content": str(project_data)})

        self.collection.add(
            documents=[project_data_json],
            metadatas=[{"task": task_description}],
            ids=[learning_id]
        )
        logger.add_log_for_ui(f"Aprendizado para a tarefa '{task_description[:30]}...' armazenado na memória.")

    def retrieve_learnings(self, query: str, n_results: int = 3) -> list:
        """
        Recupera os aprendizados mais relevantes para uma nova tarefa.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        learnings = []
        if not results.get('documents'):
            return learnings

        for i, doc_json in enumerate(results['documents'][0]):
            task_description = results['metadatas'][0][i].get('task', 'N/A')
            try:
                project_data = json.loads(doc_json)
                if not isinstance(project_data, dict):
                    project_data = {"retrieved_text": str(project_data)}
            except (json.JSONDecodeError, TypeError):
                project_data = {"summary.txt": doc_json}
            
            learnings.append({
                "task_description": task_description,
                "project_data": project_data
            })

        return learnings