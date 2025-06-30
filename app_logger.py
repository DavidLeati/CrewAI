import logging
import time
import inspect
import os
from typing import List, Optional, Callable

class CentralLogger():
    """
    Uma classe Singleton para gerenciar o logging em todo o aplicativo,
    incluindo o envio de mensagens para uma interface de usuário (UI callback)
    e a captura do local de origem do log.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CentralLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.logs: List[str] = ["Aguardando conexão..."]
        self.ui_callback: Optional[Callable[[str], None]] = None
        self.max_log_size = 300
        self._initialized = True

    def setup(self, ui_callback: Optional[Callable[[str], None]] = None):
        """Define o callback da UI que será usado para enviar logs."""
        self.ui_callback = ui_callback
        self.logs.clear() # Limpa os logs iniciais ao configurar

    def add_log_for_ui(self, message: str, level: str = "info"):
        """
        Função principal de logging. Envia a mensagem para o console, armazena para a UI
        e envia para o callback da interface, incluindo o local de origem da chamada.
        """
        # --- Modificação para capturar o contexto do chamador ---
        # O argumento stacklevel=2 faz com que o módulo logging procure o chamador
        # dois níveis acima na pilha (ignora esta função e entra no código que a chamou).
        if level == "warning":
            logging.warning(message, stacklevel=2)
        elif level == "error":
            logging.error(message, stacklevel=2)
        elif level == "critical":
            logging.critical(message, stacklevel=2)
        else:
            logging.info(message, stacklevel=2)

        # --- Modificação para o log da UI ---
        # Captura o frame do chamador para adicionar ao log da UI
        try:
            caller_frame = inspect.stack()[1]
            filename = os.path.basename(caller_frame.filename) # Pega apenas o nome do arquivo
            lineno = caller_frame.lineno
            func_name = caller_frame.function
            context = f"[{filename}:{lineno} in {func_name}]"
        except IndexError:
            # Fallback caso a pilha de chamadas seja inesperada
            context = "[unknown context]"

        # Nova mensagem de log com o contexto
        log_entry = f"{context} {message}"
        self.logs.append(log_entry)

        # Callback para a UI (enviando a mensagem original ou a formatada, como preferir)
        if self.ui_callback:
            # Enviando a mensagem formatada completa para a UI
            self.ui_callback(log_entry)

    def get_ui_logs(self) -> List[str]:
        """Retorna a lista de logs para a UI."""
        return self.logs

# Cria a instância única que será importada em outros arquivos
logger = CentralLogger()