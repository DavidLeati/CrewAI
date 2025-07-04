# shared_context.py
import os
import logging
from typing import List, Dict, Any, Optional
from app_logger import logger

class SharedContext:
    """
    Gerencia um estado compartilhado para uma sessão de crew, incluindo
    comunicação entre agentes e o conteúdo dos arquivos do projeto.
    """
    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._file_context: Dict[str, str] = {}
        logger.add_log_for_ui("Contexto Compartilhado (SharedContext) inicializado.")

    def add_message(self, sender: str, content: str, recipient: str = "all"):
        message = {"sender": sender, "recipient": recipient, "content": content}
        self._messages.append(message)
        logger.add_log_for_ui(f"Mensagem adicionada ao contexto por '{sender}' para '{recipient}': '{content[:80]}...'")

    def get_messages_for_agent(self, agent_role: str) -> List[Dict[str, Any]]:
        return [
            msg for msg in self._messages
            if msg['recipient'] == agent_role or msg['recipient'] == 'all'
        ]

    def get_full_context_for_prompt(self, agent_role: str) -> str:
        messages_for_agent = self.get_messages_for_agent(agent_role)
        if not messages_for_agent:
            return "Nenhuma mensagem no contexto compartilhado até agora."
        context_str = "A seguir estão as mensagens trocadas pela equipe:\n"
        for msg in messages_for_agent:
            context_str += f"- De: {msg['sender']} | Para: {msg['recipient']} | Mensagem: {msg['content']}\n"
        return context_str

    def load_files_to_context(self, project_files: Dict[str, str]):
        self._file_context = project_files
        logger.add_log_for_ui(f"{len(project_files)} arquivo(s) foram carregados no contexto compartilhado.")

    def rescan_and_update_context(self, workspace_dir: str):
        """
        Varre o diretório de trabalho, lê todos os arquivos de texto e atualiza
        a memória de arquivos da sessão com o estado mais recente, ignorando caches.
        """
        logger.add_log_for_ui("Sincronizando contexto: Re-escaneando o diretório de trabalho...")
        updated_files = 0
        for root, dirs, files in os.walk(workspace_dir):
            # Ignora explicitamente o diretório __pycache__
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            
            for file in files:
                # Ignora arquivos .pyc
                if file.endswith('.pyc'):
                    continue
                    
                try:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, workspace_dir)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Atualiza ou adiciona o arquivo no contexto se houver mudança
                        if self._file_context.get(relative_path) != content:
                            self._file_context[relative_path] = content
                            updated_files += 1
                except (UnicodeDecodeError, IOError) as e:
                    logger.add_log_for_ui(f"Aviso ao re-escanear: Não foi possível ler o arquivo '{file}': {e}", "warning")
        
        if updated_files > 0:
            logger.add_log_for_ui(f"Contexto atualizado. {updated_files} arquivo(s) foram modificados ou adicionados.")

    def get_file_content(self, filename: str) -> Optional[str]:
        normalized_requested_path = os.path.normpath(filename).replace("\\", "/")
        for stored_path, content in self._file_context.items():
            normalized_stored_path = os.path.normpath(stored_path).replace("\\", "/")
            if normalized_stored_path.lower() == normalized_requested_path.lower():
                return content
        requested_basename = os.path.basename(normalized_requested_path)
        for stored_path, content in self._file_context.items():
            stored_basename = os.path.basename(os.path.normpath(stored_path))
            if stored_basename.lower() == requested_basename.lower():
                logger.add_log_for_ui(f"Arquivo '{filename}' não encontrado no caminho exato, mas foi correspondido pelo nome base '{requested_basename}'.", "warning")
                return content
        return None

    def get_all_filenames(self) -> List[str]:
        return list(self._file_context.keys())