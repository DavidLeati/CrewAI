# services.py
import google.generativeai as genai
import time
import logging
from typing import Dict, Any, List
import os
import json
from datetime import datetime
import random

from config import config
from app_logger import logger
# Importa a sua nova função de busca diretamente
from search_util import search_startpage

class GeminiService:
    """Serviço para interagir com a API do Gemini e o sistema de busca web."""
    def __init__(self, model_name: str, fallback_model_name: str, chaos_mode: bool = False):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.chaos_mode = chaos_mode
        try:
            self.model = genai.GenerativeModel(self.model_name)
            self.model_for_json = genai.GenerativeModel(
                self.model_name,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            logger.add_log_for_ui(f"Modelo Gemini '{self.model_name}' inicializado.")
        except Exception as e:
            logging.warning(f"Erro ao inicializar '{self.model_name}': {e}. Tentando fallback para '{self.fallback_model_name}'.")
            self.model_name = self.fallback_model_name
            self.model = genai.GenerativeModel(self.model_name)
            self.model_for_json = self.model

    def generate_text(self, prompt: str, temperature: float, is_json_output: bool = False) -> Dict[str, Any]:
        """Gera texto e retorna um dicionário com o texto e o motivo da finalização."""
        if self.chaos_mode and random.random() < 0.1: # 10% de chance de falha
            chaos_type = random.choice(['api_error', 'bad_json', 'empty_response'])
            logger.add_log_for_ui(f"CHAOS MODE: Injetando erro do tipo '{chaos_type}'", "warning")

            if chaos_type == 'api_error':
                return {"text": "Erro 500: Erro interno do servidor (Simulado pelo Chaos Mode)", "finish_reason": "ERROR"}
            if chaos_type == 'bad_json':
                return {"text": '{"key": "value",,, "bad_syntax"}', "finish_reason": "STOP"} # JSON inválido
            if chaos_type == 'empty_response':
                return {"text": "", "finish_reason": "EMPTY"}
        
        current_model = self.model_for_json if is_json_output else self.model
        for attempt in range(config.MAX_RETRIES_API + 1):
            try:
                generation_config = genai.types.GenerationConfig(temperature=temperature)
                response = current_model.generate_content(prompt, generation_config=generation_config)
                
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason_message
                    logging.error(f"Geração de conteúdo bloqueada. Razão: {reason}")
                    return {"text": f"Conteúdo bloqueado: {reason}", "finish_reason": "SAFETY"}
                
                if not response.candidates:
                    logging.warning("A resposta da API não continha candidatos.")
                    return {"text": "Resposta da API vazia.", "finish_reason": "EMPTY"}

                candidate = response.candidates[0]
                text_output = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
                
                if is_json_output:
                    text_output = text_output.strip().removeprefix("```json").removesuffix("```").strip()

                self._save_log_to_file(prompt, "input")
                self._save_log_to_file(text_output, f"response_{'json' if is_json_output else 'txt'}")

                return {"text": text_output, "finish_reason": finish_reason}

            except Exception as e:
                error_message = str(e)
                delay = config.RETRY_DELAY_SECONDS * (attempt + 1)
                is_retriable = "500" in error_message or "429" in error_message

                if is_retriable and attempt < config.MAX_RETRIES_API:
                    logging.warning(f"Erro recuperável na API (Tentativa {attempt + 1}): {error_message}. Aguardando {delay}s...")
                    time.sleep(delay)
                else:
                    logging.error(f"Erro final na API Gemini após {attempt + 1} tentativas: {error_message}")
                    return {"text": f"Erro final na API Gemini: {error_message}", "finish_reason": "ERROR"}

        return {"text": "Erro: Número máximo de tentativas da API atingido sem sucesso.", "finish_reason": "MAX_RETRIES"}

    def perform_web_search(self, query: str) -> List[Dict[str, str]]:
        """
        Realiza uma pesquisa na web usando o scraper do Startpage.
        """
        logger.add_log_for_ui(f"Iniciando busca na web (via Startpage) para: '{query}'")
        try:
            # Chama diretamente a sua função de busca do search_util.py
            results_list = search_startpage(query=query, num_results=5)

            if results_list:
                logger.add_log_for_ui(f"Busca para '{query}' encontrou {len(results_list)} resultados com conteúdo.")
            else:
                logging.warning(f"Nenhum resultado encontrado ou extraído para '{query}'.")
            
            # A função já retorna a lista de dicionários no formato correto
            return results_list

        except Exception as e:
            logging.error(f"Erro inesperado ao invocar a função de busca: {e}", exc_info=True)
            return []

    def _save_log_to_file(self, content: str, log_type: str):
        """Salva o conteúdo em um arquivo com timestamp na pasta 'log'."""
        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("log", f"{timestamp}_{log_type}.log")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Erro ao salvar o log em arquivo: {e}")