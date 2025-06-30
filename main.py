# main.py
import os
import logging
import google.generativeai as genai
from flask import Flask, render_template_string, jsonify, request
from threading import Thread, Lock
import time
from app_logger import logger

from config import config, setup_logging
from services import GeminiService
from tasks import TaskManager

# --- Armazenamento de logs e estado da aplicação ---
app_logs = []
is_task_running = False
task_lock = Lock() # Garante que apenas uma tarefa rode por vez

def ui_callback(message: str):
    """Adiciona mensagens à lista de logs para a interface."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if len(app_logs) > 300:
        app_logs.pop(0)
    app_logs.append(f"[{timestamp}] {message}")

# --- Configuração da Aplicação Flask ---
app = Flask(__name__)

# --- TEMPLATE HTML ATUALIZADO ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>CrewAI - Painel de Controle</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        #logs { height: 50vh; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #fafafa; border-radius: 5px; line-height: 1.6; white-space: pre-wrap; font-family: "SF Mono", "Fira Code", "Roboto Mono", monospace; font-size: 14px; margin-top: 15px; }
        .log-line { padding: 2px 5px; margin-bottom: 2px; border-radius: 3px; }
        textarea { width: 98%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-size: 15px; min-height: 120px; }
        button { background-color: #007bff; color: white; padding: 12px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:disabled { background-color: #aaa; cursor: not-allowed; }
        #status { margin-top: 15px; font-weight: bold; }
    </style>
    <script>
        // Função para buscar e atualizar os logs
        async function fetchLogs() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                // Atualiza o status
                const statusDiv = document.getElementById('status');
                const runButton = document.getElementById('runButton');
                const taskText = document.getElementById('taskText');
                statusDiv.textContent = data.is_running ? 'Status: Missão em andamento...' : 'Status: Aguardando nova missão.';
                runButton.disabled = data.is_running;
                taskText.disabled = data.is_running;

                // Atualiza os logs
                const logDiv = document.getElementById('logs');
                const shouldScroll = logDiv.scrollTop + logDiv.clientHeight === logDiv.scrollHeight;
                logDiv.innerHTML = data.logs.map(log => `<div class="log-line">${escapeHtml(log)}</div>`).join('');
                if (shouldScroll) {
                    logDiv.scrollTop = logDiv.scrollHeight;
                }
            } catch (error) {
                console.error("Erro ao buscar status:", error);
            }
        }

        // Função para iniciar a tarefa
        async function startTask() {
            const taskDescription = document.getElementById('taskText').value;
            if (!taskDescription.trim()) {
                alert('Por favor, defina uma tarefa.');
                return;
            }
            
            try {
                const response = await fetch('/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tarefa: taskDescription })
                });
                const data = await response.json();
                if (!data.success) {
                    alert('Erro ao iniciar a tarefa: ' + data.message);
                }
                fetchLogs(); // Atualiza o status imediatamente
            } catch (error) {
                console.error("Erro ao iniciar a tarefa:", error);
            }
        }

        function escapeHtml(text) {
          var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
          return text.replace(/[&<>"']/g, function(m) { return map[m]; });
        }

        window.onload = () => {
            setInterval(fetchLogs, 2000); // Inicia o polling
            document.getElementById('runButton').addEventListener('click', startTask);
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>Painel de Controle da Missão</h1>
        <h2>Defina a Tarefa</h2>
        <textarea id="taskText" placeholder="Descreva aqui a tarefa complexa para a equipe de IAs..."></textarea>
        <button id="runButton">Iniciar Missão</button>
        
        <div id="status">Status: Carregando...</div>
        
        <h2>Logs da Missão</h2>
        <div id="logs"><div class="log-line">Aguardando conexão...</div></div>
    </div>
</body>
</html>
"""

# --- ROTAS DA API ---

@app.route('/')
def index():
    """Rota principal que exibe a página de controle."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def get_status():
    """Rota da API que fornece os logs e o status atual da tarefa."""
    return jsonify({"logs": app_logs, "is_running": is_task_running})

@app.route('/start', methods=['POST'])
def start_task_endpoint():
    """Rota para iniciar uma nova tarefa. Recebe a descrição da tarefa via JSON."""
    global is_task_running
    with task_lock:
        if is_task_running:
            return jsonify({"success": False, "message": "Uma tarefa já está em andamento."}), 400

        data = request.get_json()
        tarefa = data.get('tarefa')
        if not tarefa:
            return jsonify({"success": False, "message": "A descrição da tarefa não pode ser vazia."}), 400

        # Inicia a tarefa da CrewAI em uma thread
        thread = Thread(target=run_crewai_task_in_background, args=(app.task_manager, tarefa))
        thread.daemon = True
        thread.start()
        is_task_running = True
    
    return jsonify({"success": True, "message": "Tarefa iniciada."})

def run_crewai_task_in_background(task_manager: TaskManager, tarefa: str):
    """Função que executa a tarefa da CrewAI em segundo plano."""
    global is_task_running
    
    # Limpa os logs antigos e inicia a nova missão
    app_logs.clear()
    ui_callback("=> Missão Iniciada. O TaskManager está assumindo o controle.")
    
    try:
        resultado_final = task_manager.delegate_task(main_task_description=tarefa, status_callback=ui_callback)
        ui_callback(f"========= EXECUÇÃO FINALIZADA =========")
        for line in resultado_final.split('\n'):
            ui_callback(line)
    except Exception as e:
        ui_callback(f"Erro crítico na tarefa: {e}")
        logging.critical(f"Erro crítico não tratado na thread da tarefa: {e}", exc_info=True)
    finally:
        # Garante que o status seja atualizado mesmo se ocorrer um erro
        with task_lock:
            is_task_running = False

def main():
    """Ponto de entrada principal da aplicação."""
    setup_logging()

    logger.setup(ui_callback=ui_callback)

    try:
        from keys import GOOGLE_API
        genai.configure(api_key=GOOGLE_API)
        logging.info("API do Google Gemini configurada com sucesso.")
    except ImportError:
        msg = "Arquivo 'keys.py' não encontrado. Crie-o com sua chave: GOOGLE_API = 'SUA_CHAVE_API'"
        logging.critical(msg)
        print(msg)
        return
    except Exception as e:
        logging.critical(f"Erro ao configurar a API Gemini: {e}")
        return

    gemini_service = GeminiService(
        model_name=config.MODEL_NAME, 
        fallback_model_name=config.FALLBACK_MODEL_NAME
    )
    # Anexa o task_manager à instância do app Flask para que possa ser acessado na rota /start
    app.task_manager = TaskManager(
        llm_service=gemini_service, 
        output_dir=config.OUTPUT_ROOT_DIR
    )
    
    print("\n>>> Interface de controle disponível em http://127.0.0.1:5000 <<<\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()