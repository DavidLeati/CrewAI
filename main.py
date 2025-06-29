# main.py
import os
import logging
import google.generativeai as genai

from config import config, setup_logging
from services import GeminiService
from tasks import TaskManager

def main():
    """Ponto de entrada principal da aplicação."""
    # Configura o logging no início da execução
    setup_logging()

    # Configuração da API Key
    try:
        from keys import GOOGLE_API
        genai.configure(api_key=GOOGLE_API)
        logging.info("API do Google Gemini configurada com sucesso.")
    except ImportError:
        logging.critical("Arquivo 'keys.py' não encontrado. Crie-o com sua chave: GOOGLE_API = 'SUA_CHAVE_API'")
        return
    except Exception as e:
        logging.critical(f"Erro ao configurar a API Gemini: {e}")
        return

    # Instancia os serviços principais
    gemini_service = GeminiService(
        model_name=config.MODEL_NAME, 
        fallback_model_name=config.FALLBACK_MODEL_NAME
    )
    task_manager = TaskManager(
        llm_service=gemini_service, 
        output_dir=config.OUTPUT_ROOT_DIR
    )
    
    # Defina a tarefa a ser executada
    tarefa = """
    Crie uma aplicação de 'lista de tarefas' (to-do list) usando Python e Flask. Utilize React e CSS na aplicação para torna-la o mais funcional, intuitiva, e visualmente bonita.
    A aplicação deve ter as seguintes funcionalidades:
    1.  Interface web simples para visualizar, adicionar e remover tarefas.
    2.  As tarefas devem ser salvas em um arquivo JSON (`tasks.json`) para persistência.
    3.  A aplicação deve ser contida em um único arquivo python chamado `app.py`.
    4.  Crie um `templates/index.html` para a interface.
    5.  Crie um `README.md` explicando como instalar as dependências (Flask) e rodar a aplicação.
    """

    # Delega a tarefa para o TaskManager
    resultado_final = task_manager.delegate_task(tarefa)
    
    print(f"\n{'#'*20} RESULTADO FINAL DA EXECUÇÃO {'#'*20}\n")
    print(resultado_final)
    print(f"\n{'#'*60}\n")

if __name__ == "__main__":
    main()