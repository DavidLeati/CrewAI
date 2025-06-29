# services.py
import google.generativeai as genai
import time
import logging
from typing import Dict, Any

from config import config

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
            logging.info(f"Modelo Gemini '{self.model_name}' inicializado.")
        except Exception as e:
            logging.warning(f"Erro ao inicializar '{self.model_name}': {e}. Tentando fallback para '{self.fallback_model_name}'.")
            self.model_name = self.fallback_model_name
            self.model = genai.GenerativeModel(self.model_name)
            self.model_for_json = self.model

    def generate_text(self, prompt: str, temperature: float, is_json_output: bool = False) -> Dict[str, Any]:
        """Gera texto e retorna um dicionário com o texto e o motivo da finalização."""
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