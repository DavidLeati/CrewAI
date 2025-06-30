# services.py
import google.generativeai as genai
import time
import logging
from typing import Dict, Any, List
import os
import json
import time
from datetime import datetime

from config import config
from app_logger import logger
from search_scraper import DuckDuckGoSearchService

class GeminiService:
    """Serviço para interagir com a API do Gemini, gerenciando chamadas e configurações."""
    def __init__(self, model_name: str, fallback_model_name: str):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
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

        self.search_service = DuckDuckGoSearchService()
        logger.add_log_for_ui("Serviço de pesquisa web integrado ao GeminiService.")

    def generate_text(self, prompt: str, temperature: float, is_json_output: bool = False) -> Dict[str, Any]:
        """Gera texto e retorna um dicionário com o texto e o motivo da finalização, salvando o resultado em disco."""
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

                # Salvar o conteúdo gerado
                self._save_input_to_file(prompt)
                self._save_output_to_file(text_output, is_json_output)

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
        Realiza uma pesquisa web utilizando o serviço DuckDuckGoSearchService
        e retorna uma lista de URLs e títulos relevantes.

        Esta função encapsula a chamada ao serviço de pesquisa, garantindo que
        quaisquer falhas inesperadas do serviço de pesquisa sejam capturadas e logadas
        no nível do GeminiService, retornando uma lista vazia de forma graciosa.

        Args:
            query (str): A string de consulta para a pesquisa web.

        Returns:
            List[Dict[str, str]]: Uma lista de dicionários, onde cada dicionário
                                  contém 'title' e 'url' dos resultados da pesquisa.
                                  Retorna uma lista vazia em caso de falha ou nenhum resultado.
        """
        logging.info(f"Iniciando pesquisa web para a query: '{query}'")
        search_results = []
        try:
            # A lógica de retries e tratamento de erros detalhada (como falhas de rede,
            # bloqueios ou problemas de parsing) está encapsulada em DuckDuckGoSearchService.
            # Este bloco try-except serve como uma camada defensiva final para capturar
            # quaisquer exceções inesperadas que possam escapar do serviço de pesquisa,
            # garantindo que o GeminiService não falhe abruptamente.
            search_results = self.search_service.perform_search(query)
        except Exception as e:
            # Loga o erro inesperado que escapou do serviço de pesquisa
            logging.error(f"Erro inesperado ao invocar o serviço de pesquisa web para '{query}': {e}", exc_info=True)
            # Retorna uma lista vazia, conforme a assinatura, e o erro é logado.
            return []
        
        formatted_results = []
        if search_results:
            for result in search_results:
                # Extrai apenas o título e a URL conforme solicitado
                formatted_results.append({
                    "title": result.get("title", "Título não disponível"),
                    "url": result.get("url", "URL não disponível")
                })
            logging.info(f"Pesquisa web para '{query}' concluída. Encontrados {len(formatted_results)} resultados.")
        else:
            # Se search_results estiver vazio, pode ser por falta de resultados ou por um erro
            # que já foi logado detalhadamente pelo DuckDuckGoSearchService.
            # O log aqui indica a ausência de resultados para o GeminiService.
            logging.warning(f"Nenhum resultado de pesquisa web encontrado para '{query}'.")
            
        return formatted_results

    def _save_output_to_file(self, content: str, is_json: bool):
        """Salva o conteúdo em um arquivo com timestamp na pasta 'log'."""
        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "json" if is_json else "txt"
        filepath = os.path.join("log", f"reponse_{timestamp}.{extension}")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                if is_json:
                    try:
                        parsed = json.loads(content)
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        f.write(content)  # salva como string se JSON inválido
                else:
                    f.write(content)
            logging.info(f"Conteúdo salvo em {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar o conteúdo em arquivo: {e}")
    
    def _save_input_to_file(self, content: str):
        """Salva o conteúdo em um arquivo com timestamp na pasta 'log'."""
        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "txt"
        filepath = os.path.join("log", f"input_{timestamp}.{extension}")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Conteúdo salvo em {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar o conteúdo em arquivo: {e}")