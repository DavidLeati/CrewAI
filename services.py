# services.py
import google.generativeai as genai
import time
import logging
from typing import Dict, Any
import os
import json
import time
from datetime import datetime

from config import config
from app_logger import logger

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

                # ✅ Salvar o conteúdo gerado
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

    def _save_output_to_file(self, content: str, is_json: bool):
        """Salva o conteúdo em um arquivo com timestamp na pasta 'log'."""
        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "json" if is_json else "txt"
        filepath = os.path.join("log", f"{timestamp}.{extension}")
        
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