# memory.py
import chromadb
import uuid
import json # Importe o JSON para serialização

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
        Armazena o aprendizado de uma tarefa, incluindo a descrição e
        os dados completos do projeto (arquivos e conteúdos).

        Args:
            task_description: A descrição da tarefa original.
            project_data: Um dicionário onde as chaves são nomes de arquivo e
                          os valores são o conteúdo desses arquivos.
        """
        learning_id = str(uuid.uuid4())
        
        # Converte o dicionário do projeto em uma string JSON para armazenamento
        project_data_json = json.dumps(project_data, indent=2)
        
        self.collection.add(
            documents=[project_data_json], # Armazena o JSON completo
            metadatas=[{"task": task_description}],
            ids=[learning_id]
        )
        print(f"Aprendizado e projeto completo para a tarefa '{task_description[:30]}...' armazenados na memória.")

    def retrieve_learnings(self, query: str, n_results: int = 3) -> list:
        """
        Recupera os aprendizados mais relevantes para uma nova tarefa.

        Returns:
            Uma lista de dicionários, onde cada um contém a tarefa e os
            dados do projeto recuperados.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        learnings = []
        if not results['documents']:
            return learnings

        for i, doc_json in enumerate(results['documents'][0]):
            try:
                # Tenta converter a string JSON de volta para um dicionário
                project_data = json.loads(doc_json)
                task_description = results['metadatas'][0][i].get('task', 'N/A')
                learnings.append({
                    "task_description": task_description,
                    "project_data": project_data
                })
            except json.JSONDecodeError:
                # Se falhar, trata como um aprendizado de texto simples (para retrocompatibilidade)
                learnings.append({
                    "task_description": "Legacy text-based learning",
                    "project_data": {"summary.txt": doc_json}
                })

        return learnings