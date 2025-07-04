# main.py
import os
import logging
import google.generativeai as genai
from flask import Flask, render_template_string, jsonify, request
from threading import Thread, Lock
import time
import pandas as pd
from werkzeug.utils import secure_filename
import shutil

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

logger.setup(ui_callback=ui_callback)

# --- Configuração da Aplicação Flask ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Lógica para Listar Projetos Existentes ---
def get_existing_projects():
    """Escaneia o diretório de resultados e retorna uma lista de todos os diretórios de projetos."""
    if not os.path.exists(config.OUTPUT_ROOT_DIR):
        return []
    
    projects = []
    for item in os.listdir(config.OUTPUT_ROOT_DIR):
        # A nova lógica agora lista QUALQUER diretório dentro da pasta de resultados.
        if os.path.isdir(os.path.join(config.OUTPUT_ROOT_DIR, item)):
            projects.append(item)
            
    return sorted(projects, key=lambda p: os.path.getmtime(os.path.join(config.OUTPUT_ROOT_DIR, p)), reverse=True)

# --- TEMPLATE HTML ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>CrewAI - Painel de Controle</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Roboto+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #1a1b26;
            --surface-color: #24283b;
            --text-color: #c0caf5;
            --text-secondary: #a9b1d6;
            --primary-color: #7aa2f7;
            --primary-hover: #9ece6a;
            --border-color: #414868;
            --error-color: #f7768e;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }
        .container {
            max-width: 1000px;
            margin: auto;
            background: var(--surface-color);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border: 1px solid var(--border-color);
        }
        h1, h2 {
            color: var(--text-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 0;
            font-weight: 700;
        }
        h1 {
            font-size: 2rem;
        }
        h2 {
            font-size: 1.25rem;
            margin-top: 1.5rem;
        }
        #logs {
            height: 60vh;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            padding: 1rem;
            background-color: var(--bg-color);
            border-radius: 8px;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.9rem;
            margin-top: 1rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        /* Custom Scrollbar */
        #logs::-webkit-scrollbar { width: 8px; }
        #logs::-webkit-scrollbar-track { background: var(--bg-color); }
        #logs::-webkit-scrollbar-thumb { background-color: var(--border-color); border-radius: 10px; }
        #logs::-webkit-scrollbar-thumb:hover { background-color: var(--primary-color); }

        input[type="text"], textarea, select, input[type="file"] {
            width: 100%;
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-color);
            color: var(--text-color);
            font-size: 1rem;
            margin-top: 0.5rem;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(122, 162, 247, 0.2);
        }
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        button {
            background-color: var(--primary-color);
            color: var(--bg-color);
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 1rem;
            transition: background-color 0.2s, transform 0.1s;
        }
        button:hover:not(:disabled) {
            background-color: var(--primary-hover);
        }
        button:active:not(:disabled) {
            transform: scale(0.98);
        }
        button:disabled {
            background-color: var(--border-color);
            color: var(--text-secondary);
            cursor: not-allowed;
        }
        #status {
            margin-top: 1.5rem;
            padding: 0.75rem;
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-weight: 500;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group p {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            margin-bottom: 0;
        }
        .log-line {
            padding: 2px 0;
        }
    </style>
    <script>
        // O JavaScript permanece o mesmo, pois a lógica não mudou.
        async function fetchLogs() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                const statusDiv = document.getElementById('status');
                const runButton = document.getElementById('runButton');
                const taskText = document.getElementById('taskText');
                const fileInput = document.getElementById('fileInput');
                const projectSelector = document.getElementById('projectSelector');
                const projectNameInput = document.getElementById('projectName');

                statusDiv.textContent = data.is_running ? 'Status: 🚀 Missão em andamento...' : 'Status:  idling... Aguardando nova missão.';
                runButton.disabled = data.is_running;
                taskText.disabled = data.is_running;
                fileInput.disabled = data.is_running;
                projectSelector.disabled = data.is_running;
                projectNameInput.disabled = data.is_running || projectSelector.value !== "";

                if (!data.is_running) {
                    const currentSelected = projectSelector.value;
                    projectSelector.innerHTML = '<option value="">✨ Criar novo projeto</option>';
                    data.projects.forEach(project => {
                        const option = document.createElement('option');
                        option.value = project;
                        option.textContent = project;
                        projectSelector.appendChild(option);
                    });
                    projectSelector.value = currentSelected;
                }
                
                const logDiv = document.getElementById('logs');
                const shouldScroll = logDiv.scrollTop + logDiv.clientHeight >= logDiv.scrollHeight - 20;
                logDiv.innerHTML = data.logs.map(log => `<div class="log-line">${escapeHtml(log)}</div>`).join('');
                if (shouldScroll) {
                    logDiv.scrollTop = logDiv.scrollHeight;
                }
            } catch (error) {
                console.error("Erro ao buscar status:", error);
            }
        }

        async function startTask() {
            const taskDescription = document.getElementById('taskText').value;
            const files = document.getElementById('fileInput').files;
            const selectedProject = document.getElementById('projectSelector').value;
            const projectName = document.getElementById('projectName').value;

            if (selectedProject === "" && !projectName.trim()) {
                alert('Por favor, defina um nome para o novo projeto.');
                return;
            }
            if (!taskDescription.trim()) {
                alert('Por favor, defina uma tarefa.');
                return;
            }
            
            const formData = new FormData();
            formData.append('tarefa', taskDescription);
            formData.append('projeto_selecionado', selectedProject);
            formData.append('nome_projeto', projectName);

            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }

            try {
                const response = await fetch('/start', { method: 'POST', body: formData });
                const data = await response.json();
                if (!data.success) {
                    alert('Erro ao iniciar a tarefa: ' + data.message);
                }
                fetchLogs();
            } catch (error) {
                console.error("Erro ao iniciar a tarefa:", error);
            }
        }

        function escapeHtml(text) {
            var map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
            return text.toString().replace(/[&<>"']/g, function(m) { return map[m]; });
        }

        window.onload = () => {
            fetchLogs();
            setInterval(fetchLogs, 2000);

            document.getElementById('projectSelector').addEventListener('change', function() {
                const projectNameInput = document.getElementById('projectName');
                projectNameInput.disabled = this.value !== "";
                if (this.value !== "") {
                    projectNameInput.value = "";
                }
            });

            document.getElementById('taskForm').addEventListener('submit', function(e) {
                e.preventDefault();
                startTask();
            });
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>🎮 CrewAI - Painel de Controle da Missão</h1>
        <form id="taskForm">
            <div class="form-group">
                <h2>1. Selecione o Projeto</h2>
                <select id="projectSelector"></select>
                <p>Escolha um projeto existente para continuar ou crie um novo.</p>
            </div>
            <div class="form-group">
                <h2>2. Defina o Nome do Novo Projeto</h2>
                <input type="text" id="projectName" placeholder="Ex: Jogo de Sobrevivência com JavaScript">
                <p>Obrigatório apenas ao criar um novo projeto.</p>
            </div>
            <div class="form-group">
                <h2>3. Defina a Tarefa</h2>
                <textarea id="taskText" placeholder="Descreva a missão para a equipe de IAs..."></textarea>
            </div>
            <div class="form-group">
                <h2>4. Anexar Arquivos (Opcional)</h2>
                <input type="file" id="fileInput" multiple>
            </div>
            <button id="runButton">🚀 Iniciar Missão</button>
        </form>
        <div id="status">Aguardando conexão...</div>
        <h2>🛰️ Logs da Missão</h2>
        <div id="logs"><div class="log-line">Aguardando conexão com a base...</div></div>
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
    """Rota da API que fornece os logs, o status da tarefa e a lista de projetos."""
    projects = get_existing_projects()
    return jsonify({
        "logs": app_logs, 
        "is_running": is_task_running,
        "projects": projects
    })

@app.route('/start', methods=['POST'])
def start_task_endpoint():
    """Rota para iniciar uma nova tarefa."""
    global is_task_running
    with task_lock:
        if is_task_running:
            return jsonify({"success": False, "message": "Uma tarefa já está em andamento."}), 400

        tarefa = request.form.get('tarefa')
        projeto_selecionado = request.form.get('projeto_selecionado')
        nome_projeto = request.form.get('nome_projeto')

        if not tarefa:
            return jsonify({"success": False, "message": "A descrição da tarefa não pode ser vazia."}), 400
        
        if not projeto_selecionado and not nome_projeto:
            return jsonify({"success": False, "message": "O nome do projeto é obrigatório para novos projetos."}), 400

        uploaded_files_content = {}
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    content = ""
                    try:
                        if filename.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file)
                            content = df.to_string()
                        else:
                            content = file.read().decode('utf-8')
                        
                        uploaded_files_content[filename] = content
                        logger.add_log_for_ui(f"Arquivo '{filename}' recebido e processado.")
                    except Exception as e:
                        logger.add_log_for_ui(f"Erro ao processar o arquivo '{filename}': {e}", "error")

        thread = Thread(target=run_crewai_task_in_background, args=(app.task_manager, tarefa, uploaded_files_content, projeto_selecionado, nome_projeto))
        thread.daemon = True
        thread.start()
        is_task_running = True
    
    return jsonify({"success": True, "message": "Tarefa iniciada."})

def run_crewai_task_in_background(task_manager: TaskManager, tarefa: str, uploaded_content: dict, projeto_selecionado: str, nome_projeto: str):
    """Função que executa a tarefa da CrewAI em segundo plano."""
    global is_task_running
    
    app_logs.clear()
    ui_callback("=> Missão Iniciada. O TaskManager está assumindo o controle.")
    
    try:
        resultado_final = task_manager.delegate_task(
                            main_task_description=tarefa,
                            project_name=nome_projeto,
                            status_callback=ui_callback,
                            uploaded_files_content=uploaded_content,
                            existing_project_dir=projeto_selecionado or None
                            )
        ui_callback(f"========= EXECUÇÃO FINALIZADA =========")
        for line in resultado_final.split('\n'):
            ui_callback(line)
    except Exception as e:
        ui_callback(f"Erro crítico na tarefa: {e}")
        logging.critical(f"Erro crítico não tratado na thread da tarefa: {e}", exc_info=True)
    finally:
        with task_lock:
            is_task_running = False

def main():
    """Ponto de entrada principal da aplicação."""
    setup_logging()

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

    app.task_manager = TaskManager(
        llm_service=gemini_service, 
        output_dir=config.OUTPUT_ROOT_DIR
    )
    
    print("\n>>> Interface de controle disponível em http://127.0.0.1:5000 <<<\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()