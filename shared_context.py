# shared_context.py
import logging
from typing import List, Dict, Any
from app_logger import logger

class SharedContext:
    """
    Gerencia um estado compartilhado para comunicação entre agentes dentro de uma crew.
    Funciona como um quadro branco onde os agentes podem postar e ler mensagens.
    """
    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        logger.add_log_for_ui("Contexto Compartilhado (SharedContext) inicializado.")

    def add_message(self, sender: str, content: str, recipient: str = "all"):
        """
        Adiciona uma nova mensagem ao contexto.

        Args:
            sender (str): O papel do agente que envia a mensagem.
            content (str): O conteúdo da mensagem.
            recipient (str): O destinatário da mensagem ('all' por padrão).
        """
        message = {
            "sender": sender,
            "recipient": recipient,
            "content": content
        }
        self._messages.append(message)
        logger.add_log_for_ui(f"Mensagem adicionada ao contexto por '{sender}' para '{recipient}': '{content[:80]}...'")

    def get_messages_for_agent(self, agent_role: str) -> List[Dict[str, Any]]:
        """
        Obtém todas as mensagens destinadas a um agente específico ou para todos.

        Args:
            agent_role (str): O papel do agente que está lendo as mensagens.

        Returns:
            List[Dict[str, Any]]: Uma lista de mensagens.
        """
        # Retorna mensagens onde o agente é o destinatário ou o destinatário é 'all'
        return [
            msg for msg in self._messages
            if msg['recipient'] == agent_role or msg['recipient'] == 'all'
        ]

    def get_full_context_for_prompt(self, agent_role: str) -> str:
        """
        Formata o contexto de mensagens para ser injetado no prompt de um agente.
        """
        messages_for_agent = self.get_messages_for_agent(agent_role)
        if not messages_for_agent:
            return "Nenhuma mensagem no contexto compartilhado até agora."

        context_str = "A seguir estão as mensagens trocadas pela equipe. Use-as para guiar sua próxima ação:\n"
        for msg in messages_for_agent:
            context_str += f"- De: {msg['sender']} | Para: {msg['recipient']} | Mensagem: {msg['content']}\n"
        return context_str