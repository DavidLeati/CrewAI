# memory.py
import chromadb
import uuid
import json 
from app_logger import logger

class LongTermMemory:
    """
    Gerencia a memória de longo prazo da Crew, agora capaz de armazenar
    estruturas de projetos completos (arquivos e seus conteúdos).
    """
    def __init__(self, db_path="./memoria_crew"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="aprendizados_de_tarefas")

    def store_learning(self, task_description: str, project_data: dict):
        """
        Armazena o aprendizado de uma tarefa. Se project_data for uma string,
        ela é encapsulada em um dicionário para garantir consistência.
        """
        learning_id = str(uuid.uuid4())
        
        # GARANTE QUE O DADO SEJA SEMPRE UM DICIONÁRIO ANTES DE SERIALIZAR
        if isinstance(project_data, str):
            logger.add_log_for_ui("Dados de aprendizado recebidos como string. Encapsulando em dicionário.", "warning")
            project_data = {"summary.txt": project_data}

        try:
            project_data_json = json.dumps(project_data, indent=2)
        except TypeError as e:
            logger.add_log_for_ui(f"Erro de serialização ao armazenar aprendizado: {e}. Armazenando como string.", "error")
            project_data_json = json.dumps({"error": "unserializable_data", "content": str(project_data)})

        self.collection.add(
            documents=[project_data_json],
            metadatas=[{"task": task_description}],
            ids=[learning_id]
        )
        logger.add_log_for_ui(f"Aprendizado para a tarefa '{task_description[:30]}...' armazenado na memória.")

    def retrieve_learnings(self, query: str, n_results: int = 3) -> list:
        """
        Recupera os aprendizados mais relevantes para uma nova tarefa, garantindo
        que os dados do projeto sejam sempre retornados como um dicionário.
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
                # GARANTE QUE O DADO RETORNADO SEJA SEMPRE UM DICIONÁRIO
                if not isinstance(project_data, dict):
                    project_data = {"retrieved_text": str(project_data)}
            except (json.JSONDecodeError, TypeError):
                # Se a decodificação falhar, trata como texto simples para evitar quebras.
                project_data = {"summary.txt": doc_json}
            
            learnings.append({
                "task_description": task_description,
                "project_data": project_data
            })

        return learnings