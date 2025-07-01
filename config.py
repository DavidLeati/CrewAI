# config.py
import logging
from dataclasses import dataclass
from app_logger import logger

@dataclass
class Config:
    """Centraliza as configurações do sistema."""
    MODEL_NAME: str = "gemini-2.5-flash"
    FALLBACK_MODEL_NAME: str = "gemini-pro"
    MAX_ITERATIONS: int = 10
    MAX_RETRIES_API: int = 3
    RETRY_DELAY_SECONDS: int = 5
    TEMPERATURE_PLANNING: float = 0.2
    TEMPERATURE_EXECUTION: float = 0.4
    TEMPERATURE_VALIDATION: float = 0.1
    OUTPUT_ROOT_DIR: str = "resultados"
    VERBOSE_LOGGING: bool = True

# Instancia a configuração para ser importada por outros módulos
config = Config()

def setup_logging():
    """Configura o sistema de logging com base nas configurações."""
    log_level = logging.DEBUG if config.VERBOSE_LOGGING else logger.add_log_for_ui
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )