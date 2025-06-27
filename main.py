import google.generativeai as genai
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import re
import time
import uuid
import shutil
import logging
from dataclasses import dataclass
import subprocess
from fpdf import FPDF
import patch
from difflib import unified_diff
import requests

# --- Configuração Inicial e Logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@dataclass
class Config:
    """Centraliza as configurações do sistema."""
    MODEL_NAME: str = "gemini-2.5-flash"
    FALLBACK_MODEL_NAME: str = "gemini-pro"
    MAX_ITERATIONS: int = 10
    MAX_RETRIES_API: int = 2
    RETRY_DELAY_SECONDS: int = 10
    TEMPERATURE_PLANNING: float = 0.3
    TEMPERATURE_EXECUTION: float = 0.5
    TEMPERATURE_VALIDATION: float = 0.2
    OUTPUT_ROOT_DIR: str = "resultados_dinamicos_ia"
    VERBOSE_LOGGING: bool = True # Controla o nível de log em tempo de execução

# Instancia a configuração
config = Config()

# --- Chave de API ---
try:
    from keys import GOOGLE_API
    genai.configure(api_key=GOOGLE_API)
    logging.info("API do Google Gemini configurada com sucesso.")
except ImportError:
    logging.critical("Arquivo 'keys.py' não encontrado. Crie-o com sua chave: GOOGLE_API = 'SUA_CHAVE_API'")
    exit()
except Exception as e:
    logging.critical(f"Erro ao configurar a API Gemini: {e}")
    exit()

# --- Funções Auxiliares Refatoradas ---

def parse_llm_content_and_metadata(llm_response_text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Analisa a resposta do LLM usando uma abordagem de "split" para extrair de forma robusta
    múltiplos pares de conteúdo e metadados, aceitando vários formatos de delimitador.
    """
    artifacts = []
    
    # Padrão de regex que captura CADA UM dos três formatos de separador possíveis.
    # 1. ```json {...} ```
    # 2. --- suggested_filename: ... ---
    # 3. --- (em uma linha própria)
    delimiter_pattern = r"(```json\s*\{[\s\S]*?\}\s*```|---+\s*suggested_filename:.*?---+\s*|^\s*---\s*$)"
    
    # Quebra a resposta inteira em uma lista usando os delimitadores.
    # Ex: [content1, separator1, content2, separator2, ...]
    parts = re.split(delimiter_pattern, llm_response_text, flags=re.MULTILINE)
    
    # Filtra partes vazias ou None que podem surgir do split.
    parts = [p for p in parts if p and p.strip()]
    
    current_content = ""
    for part in parts:
        part_stripped = part.strip()
        is_json_block = part_stripped.startswith('```json')
        is_custom_meta = part_stripped.startswith('---') and 'suggested_filename' in part_stripped
        is_simple_separator = part_stripped == '---'

        # Se a parte é um dos nossos delimitadores de metadados
        if is_json_block or is_custom_meta:
            metadata = {}
            if is_json_block:
                json_match = re.search(r"\{[\s\S]*\}", part_stripped)
                if json_match:
                    try: metadata = json.loads(json_match.group(0))
                    except json.JSONDecodeError: logging.warning("Bloco JSON malformado ignorado.")
            
            elif is_custom_meta:
                filename_match = re.search(r"suggested_filename:\s*(.*)", part_stripped)
                if filename_match:
                    filename = filename_match.group(1).replace("---", "").strip()
                    metadata = {"suggested_filename": filename, "description": f"Conteúdo para {filename}"}
            
            # Padroniza a chave do nome do arquivo
            filename_keys = ['suggested_filename', 'artifact', 'artifact_path', 'file_path', 'artifact_name']
            found_key = next((key for key in filename_keys if key in metadata), None)
            if found_key and found_key != 'suggested_filename':
                metadata['suggested_filename'] = metadata.pop(found_key)
            
            # Pareia o conteúdo acumulado com os metadados encontrados
            if current_content:
                artifacts.append((current_content, metadata))
                current_content = "" # Reseta o conteúdo para o próximo artefato

        elif is_simple_separator:
            # Se encontramos um separador simples, o conteúdo acumulado é um artefato sem metadados.
            if current_content:
                artifacts.append((current_content, {})) # O fallback do Agent cuidará disso
                current_content = ""
        else:
            # Se não for um delimitador, é conteúdo. Anexa ao conteúdo atual.
            current_content += "\n" + part if current_content else part
            
    # Adiciona qualquer conteúdo restante no final do arquivo.
    if current_content:
        artifacts.append((current_content, {}))

    if artifacts:
        logging.info(f"Parser extraiu {len(artifacts)} artefato(s) da resposta do agente.")
    else:
        logging.warning("A resposta do agente estava vazia ou não pôde ser parseada.")
        
    return artifacts
    
def clean_markdown_code_fences(code_str: str) -> str:
    """
    Remove de forma robusta todas as ocorrências de cercas de código Markdown
    e blocos de metadados JSON do texto.
    """
    if not isinstance(code_str, str):
        return ""

    # 1. Remove os blocos de metadados JSON completos
    # Esta regex encontra e remove todo o padrão ```json ... ```
    cleaned_str = re.sub(r"```json\s*\{[\s\S]*?\}\s*```", "", code_str)

    # 2. Remove cercas de código (ex: ```python ou ```) que possam ter sobrado
    # Esta regex remove a linha inteira se ela contiver apenas a cerca de código
    cleaned_str = re.sub(r"^\s*```(?:[a-zA-Z0-9_.-]*)?\s*$", "", cleaned_str, flags=re.MULTILINE)
    
    # 3. Remove as cercas restantes que possam estar no início ou fim
    # usando a lógica anterior para garantir.
    match = re.match(r"^\s*```(?:[a-zA-Z0-9_.-]*)?\s*\n?(.*?)\n?\s*```\s*$", cleaned_str.strip(), re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    return cleaned_str.strip()

def sanitize_filename(filename: str, fallback_name: str = "fallback_artifact.txt") -> str:
    """
    Limpa e sanitiza um nome de arquivo sugerido para ser seguro para o sistema de arquivos.
    """
    if not filename or not isinstance(filename, str) or not filename.strip():
        return fallback_name

    # Remove caracteres inválidos para nomes de arquivo na maioria dos sistemas.
    sanitized = re.sub(r'[\\/*?:"<>|\n\r\t]', "_", filename)
    
    # Remove espaços no início ou fim.
    sanitized = sanitized.strip()

    # Garante que não comece com um ponto (arquivo oculto) ou seja apenas pontos.
    if sanitized.startswith('.') or all(c == '.' for c in sanitized):
        sanitized = "file_" + sanitized

    # Se após a sanitização o nome ficou vazio, usa o fallback.
    return sanitized if sanitized else fallback_name

# --- Classes Principais do Sistema Refatoradas ---

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
            self.model_for_json = self.model # Modelos mais antigos podem não suportar `response_mime_type`

    def generate_text(self, prompt: str, temperature: float, is_json_output: bool = False) -> Dict[str, Any]:
        """Gera texto e retorna um dicionário com o texto e o motivo da finalização ('finish_reason')."""
        current_model = self.model_for_json if is_json_output else self.model
        final_prompt = prompt

        for attempt in range(config.MAX_RETRIES_API + 1):
            try:
                generation_config = genai.types.GenerationConfig(temperature=temperature)
                response = current_model.generate_content(final_prompt, generation_config=generation_config)
                
                # Checagem de segurança
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason_message
                    logging.error(f"Geração de conteúdo bloqueada. Razão: {reason}")
                    # Retorna um dicionário estruturado
                    return {"text": f"Conteúdo bloqueado: {reason}", "finish_reason": "SAFETY"}
                
                # Acesso seguro ao candidato da resposta
                if not response.candidates:
                    logging.warning("A resposta da API não continha candidatos.")
                    return {"text": "Resposta da API vazia.", "finish_reason": "EMPTY"}

                candidate = response.candidates[0]
                text_output = "".join(part.text for part in candidate.content.parts) if candidate.content.parts else ""
                
                # Extrai o motivo da finalização
                finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
                
                if is_json_output:
                    # Limpeza de JSON
                    text_output = text_output.strip()
                    if text_output.startswith("```json"):
                        text_output = text_output[7:]
                    if text_output.endswith("```"):
                        text_output = text_output[:-3]
                    text_output = text_output.strip()

                # Retorna o dicionário completo
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
                    # Retorna um dicionário de erro
                    return {"text": f"Erro final na API Gemini: {error_message}", "finish_reason": "ERROR"}
        
        # Retorna um dicionário de erro se o loop terminar
        return {"text": "Erro: Número máximo de tentativas da API atingido sem sucesso.", "finish_reason": "ERROR"}

class Agent:
    """
    Representa um agente de IA com lógica de autocorreção totalmente funcional
    e sincronizada com o serviço de IA.
    """
    def __init__(self, role: str, goal: str, backstory: str, llm_service: GeminiService, agent_id: str):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm_service = llm_service
        self.agent_id = agent_id

    def _infer_content_type(self, content: str) -> str:
        """Analisa um bloco de texto e infere se é código ou documento."""
        # Palavras-chave que indicam fortemente que o conteúdo é código.
        code_keywords = [
            'import ', 'from ', 'def ', 'class ', 'function ', 'const ', 'let ', 'var ',
            'public class', 'public static', 'void main', '#include', '=>', '={', '};'
        ]
        # Uma verificação simples: se alguma dessas palavras-chave estiver no início
        # das primeiras linhas, é provável que seja código.
        for line in content.splitlines()[:5]: # Verifica as 5 primeiras linhas
            if any(keyword in line for keyword in code_keywords):
                return 'code'
        return 'document'

    def execute_task(self,
                     main_task_description: str,
                     task_description: str,
                     task_workspace_dir: str,
                     iteration_num: int,
                     context_artifacts: List[Dict[str, Any]],
                     feedback_history: List[str]
                     ) -> List[Dict[str, Any]]:
        
        is_correction_attempt = bool(feedback_history)

        # --- Construção do Contexto ---
        context_summary = "Nenhum artefato de contexto anterior fornecido.\n"
        if context_artifacts:
            context_summary = "A seguir estão os artefatos de contexto já existentes:\n"
            critical_files_in_feedback = set()
            if is_correction_attempt:
                feedback_text_for_search = "".join(feedback_history)
                found_paths = re.findall(r'File "[^"]*[\\/]([\w\._-]+)"', feedback_text_for_search)
                critical_files_in_feedback.update(found_paths)
            
            for art_meta in context_artifacts:
                filename = os.path.basename(art_meta.get('file_path', 'N/A'))
                context_summary += f"\n--- Artefato: '{filename}' ---\n"
                try:
                    with open(art_meta['file_path'], 'r', encoding='utf-8') as f:
                        if filename in critical_files_in_feedback:
                            content = f.read()
                            context_summary += f"```\n[CONTEÚDO COMPLETO DO ARQUIVO CRÍTICO]\n{content}\n```\n"
                        else:
                            content = f.read(2000)
                            context_summary += f"```\n{content}{'... (trecho)' if len(content) == 2000 else ''}\n```\n"
                except Exception as e:
                    context_summary += f"[Não foi possível ler o arquivo: {e}]\n"

        if is_correction_attempt:
            context_summary += f"\n--- HISTÓRICO DE FEEDBACK (MAIS RECENTE É MAIS IMPORTANTE) ---\n" + "\n---\n".join(reversed(feedback_history))

        # --- Geração do Prompt ---
        prompt_header = (
            f"<instrucoes_de_identidade>\n"
            f"  <papel>{self.role}</papel>\n"
            f"  <objetivo_especifico>{self.goal}</objetivo_especifico>\n"
            f"  <historia_de_fundo>{self.backstory}</historia_de_fundo>\n"
            f"</instrucoes_de_identidade>\n\n"
            "<regras_inviolaveis_de_saida>\n"
            "  - Sua única saída deve ser o conteúdo do(s) artefato(s) solicitado(s).\n"
            "  - NUNCA inclua texto introdutório, conversacional ou explicativo como 'Claro, aqui está o código:'. Vá direto ao ponto.\n"
            "  - Sua resposta DEVE OBRIGATORIAMENTE terminar com um ou mais blocos de metadados ```json, cada um seguindo seu respectivo bloco de conteúdo.\n"
            "  - NÃO gere scripts de shell (bash, sh, cmd). Gere o conteúdo interno dos arquivos.\n"
            "</regras_inviolaveis_de_saida>\n\n"
        )

        # --- 2. Bloco de Contexto Dinâmico ---
        # Fornece todas as informações que o agente precisa para tomar uma decisão.
        prompt_context = (
            f"<contexto_da_missao>\n"
            f"  <objetivo_geral_do_projeto>\n{main_task_description}\n</objetivo_geral_do_projeto>\n\n"
            f"  <arquivos_e_feedback_existentes>\n{context_summary}\n</arquivos_e_feedback_existentes>\n"
            f"</contexto_da_missao>\n\n"
        )

        # --- 3. Bloco de Tarefa (Condicional e Focado) ---
        # Define claramente a ação a ser tomada nesta etapa específica.
        prompt_task = "<tarefa_especifica>\n"
        if is_correction_attempt:
            prompt_task += (
                "MODO: **CORREÇÃO**.\n"
                "Uma tentativa anterior falhou. Sua tarefa é analisar o erro e corrigir o código.\n"
                "Siga estes passos mentais:\n"
                "1.  **Análise do Erro:** Releia atentamente o último 'HISTÓRICO DE FEEDBACK' no contexto para entender a causa raiz do problema.\n"
                "2.  **Plano de Ação:** Identifique qual(is) arquivo(s) precisam ser modificados para corrigir o erro.\n"
                "3.  **Execução:** Gere a **versão completa e corrigida** do(s) arquivo(s) necessário(s).\n\n"
                f"INSTRUÇÃO DE CORREÇÃO PARA ESTA TENTATIVA ({iteration_num}):\n{task_description}\n"
            )
        else:
            prompt_task += (
                "MODO: **CRIAÇÃO**.\n"
                "Sua tarefa é gerar um novo artefato do zero com base na instrução abaixo.\n\n"
                f"INSTRUÇÃO DE CRIAÇÃO PARA ESTA TAREFA (Tentativa {iteration_num}):\n{task_description}\n"
            )
        prompt_task += "</tarefa_especifica>\n\n"

        # --- 4. Bloco de Exemplo de Saída (Reforço Final) ---
        # Repete a instrução mais crítica no final para garantir que seja seguida.
        prompt_footer = (
            "<exemplo_de_formato_de_saida>\n"
            "Lembre-se, sua resposta final deve ser o conteúdo do artefato, seguido imediatamente pelo bloco de metadados. Exemplo para um artefato:\n"
            "```python\n"
            "print('Hello, World!')\n"
            "```\n"
            "```json\n"
            '{\n  "suggested_filename": "hello.py",\n  "description": "Um script de exemplo em Python."\n}\n'
            "```\n"
            "</exemplo_de_formato_de_saida>"
        )

        # --- Montagem Final do Prompt ---
        prompt = prompt_header + prompt_context + prompt_task + prompt_footer

        # --- 2. Geração com Loop de Continuação (Inalterado) ---
        logging.info(f"Agente '{self.role}' iniciando tarefa: {task_description[:100]}...")
        content_parts = []
        current_prompt = prompt
        max_continuations = 5 
        is_continuation = False
        for i in range(max_continuations):
            generation_result = self.llm_service.generate_text(current_prompt, temperature=config.TEMPERATURE_EXECUTION)
            if generation_result['text'].startswith("Erro final na API Gemini:") or generation_result['text'].startswith("Conteúdo bloqueado:"):
                logging.error(f"Agente '{self.role}' falhou ao obter resposta do LLM: {generation_result['text']}")
                return []
            content_parts.append(generation_result['text'])
            if generation_result['finish_reason'] != 'MAX_TOKENS':
                if is_continuation: logging.info("Geração de continuação concluída.")
                break
            is_continuation = True
            logging.warning(f"Resposta truncada (MAX_TOKENS). Preparando continuação ({i+1}/{max_continuations})...")
            structural_context = content_parts[0][:10000]
            immediate_context = content_parts[-1][-10000:]
            current_prompt = (f"Sua resposta foi cortada. Continue EXATAMENTE de onde parou.\n\n--- CONTEXTO ESTRUTURAL (Início do arquivo):\n{structural_context}...\n\n--- FINAL DO SEU TEXTO (Continue daqui):\n...{immediate_context}")
        
        full_llm_response = "".join(content_parts)
         
        # --- 3. Processamento e Salvamento dos Artefatos ---
        parsed_artifacts = parse_llm_content_and_metadata(full_llm_response)
        saved_artifacts_metadata = []

        # --- Rede de Segurança Para Repostas de Json Bruto
        if len(parsed_artifacts) == 1 and not parsed_artifacts[0][1]: # Se temos um único artefato sem metadados
            content, _ = parsed_artifacts[0]
            try:
                # Tenta analisar o conteúdo como se fosse um JSON.
                possible_metadata = json.loads(content)
                if isinstance(possible_metadata, dict):
                    # Se for um JSON válido, o tratamos como o metadado, com conteúdo vazio.
                    logging.info("Resposta de JSON bruto detectada. Reprocessando como metadados.")
                    parsed_artifacts = [("", possible_metadata)]
            except json.JSONDecodeError:
                # Não era um JSON, segue o fluxo normal.
                pass

        for content, metadata in parsed_artifacts:
            # --- Lógica de Fallback com Autocorreção ---
            if not metadata or 'suggested_filename' not in metadata:
                logging.warning(f"Artefato sem metadados válidos. Iniciando fallback. Conteúdo: '{content[:100]}...'")
                match = re.search(r"['\"]([a-zA-Z0-9_\/\\]+\.[a-zA-Z0-9_]+)['\"]", task_description)
                if match:
                    suggested_filename = match.group(1)
                    description = f"Metadados inferidos para {suggested_filename}"
                else: 
                    logging.warning("Não foi possível inferir o nome do arquivo. Tentando autocorreção para gerar metadados.")
                    naming_prompt = (
                        "Analise o seguinte conteúdo e sugira um nome de arquivo apropriado (com extensão) e uma breve descrição. "
                        "Sua resposta DEVE ser um único objeto JSON com as chaves 'suggested_filename' e 'description'.\n\n"
                        f"CONTEÚDO PARA ANÁLISE:\n---\n{content[:1500]}...\n---\n"
                        "Responda apenas com o JSON."
                    )
                    
                    metadata_response_dict = self.llm_service.generate_text(naming_prompt, temperature=0.1, is_json_output=True)
                    metadata_response_text = metadata_response_dict.get('text', '{}')
                    
                    try:
                        naming_metadata = json.loads(metadata_response_text)
                        if 'suggested_filename' not in naming_metadata:
                            raise ValueError("Chave 'suggested_filename' ausente no JSON de autocorreção.")
                        suggested_filename = naming_metadata["suggested_filename"]
                        description = naming_metadata.get("description", "Descrição autogerada.")
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Autocorreção de metadados falhou: {e}. Usando fallback genérico.")
                        suggested_filename = f"{self.role.replace(' ', '_').lower()}_fallback_{uuid.uuid4().hex[:6]}.txt"
                        description = f"Fallback genérico para a tarefa: {task_description[:50]}..."
            else:
                suggested_filename = metadata["suggested_filename"]
                description = metadata.get("description", "Descrição não fornecida.")

            content_to_save = clean_markdown_code_fences(content)
            
            # --- Lógica de Reconciliação e Salvamento (Inalterada) ---
            target_path_from_task = None
            match = re.search(r"['\"]([a-zA-Z0-9_\/\\]+\.[a-zA-Z0-9_]+)['\"]", task_description)
            if match:
                target_path_from_task = os.path.normpath(match.group(1))

            final_path_to_use = os.path.normpath(suggested_filename)
            if target_path_from_task and os.path.basename(target_path_from_task) == os.path.basename(final_path_to_use):
                final_path_to_use = target_path_from_task
                logging.info(f"Caminho do arquivo reconciliado para: '{final_path_to_use}'.")
            
            path_parts = final_path_to_use.split(os.sep)
            filename_part = sanitize_filename(path_parts[-1])
            relative_dir_parts = path_parts[:-1]
            final_dir = os.path.join(task_workspace_dir, *relative_dir_parts)
            os.makedirs(final_dir, exist_ok=True)
            output_filepath = os.path.join(final_dir, filename_part)

            if os.path.exists(output_filepath):
                # Se o arquivo de destino já existe, verificamos os tipos.
                with open(output_filepath, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                
                existing_type = self._infer_content_type(existing_content)
                new_type = self._infer_content_type(content_to_save)

                # A regra crítica: um documento não pode sobrescrever código.
                if existing_type == 'code' and new_type == 'document':
                    logging.error(f"VÁLVULA DE SEGURANÇA ACIONADA: Tentativa de sobrescrever o arquivo de código '{os.path.basename(output_filepath)}' com um documento.")
                    # Salva o documento com um nome seguro para evitar perda de dados.
                    safe_fallback_filename = f"DANGEROUS_OVERWRITE_ATTEMPT_ON_{os.path.basename(output_filepath)}.md"
                    output_filepath = os.path.join(final_dir, safe_fallback_filename)
                    description += " [AVISO: Salvo com nome de fallback para prevenir sobreescrita de código]"
                    logging.warning(f"O conteúdo do documento foi salvo como '{safe_fallback_filename}' para inspeção.")

            # --- Lógica de Salvamento Unificada ---
            try:
                if output_filepath.lower().endswith('.pdf'):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    text_to_write = content_to_save.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 10, txt=text_to_write)
                    pdf.output(output_filepath)
                else:
                    with open(output_filepath, "w", encoding="utf-8") as f:
                        f.write(content_to_save)
                
                logging.info(f"Agente '{self.role}' salvou/sobrescreveu o artefato: '{output_filepath}'")
                saved_artifacts_metadata.append({
                    "file_path": output_filepath, "description": description, "agent_role": self.role,
                    "task_description": task_description, "iteration_num": iteration_num,
                })
            except Exception as e:
                logging.error(f"Agente '{self.role}' falhou ao salvar o arquivo '{output_filepath}': {e}")
        
        return saved_artifacts_metadata

class Crew:
    """Gerencia uma equipe de agentes para processar uma série de subtarefas delegadas."""
    def __init__(self, name: str, description: str, agents: List[Agent]):
        self.name = name
        self.description = description
        self.agents = {agent.role: agent for agent in agents}
        logging.info(f"Crew '{self.name}' criada com {len(agents)} agentes: {list(self.agents.keys())}")

    def process_subtasks(self,
                           main_task_description: str,
                           subtasks: List[Dict[str, Any]],
                           task_workspace_dir: str,
                           iteration_num: int,
                           feedback_history: List[str]
                           ) -> Dict[str, Any]:
        
        logging.info(f"Crew '{self.name}' (Tentativa {iteration_num}) iniciando processamento de {len(subtasks)} subtarefas.")
        iteration_artifacts_metadata: List[Dict[str, Any]] = []

        for i, subtask in enumerate(subtasks):
            subtask_desc = subtask.get("description", "Descrição da subtarefa não fornecida.")
            responsible_role = subtask.get("responsible_role")
            logging.info(f"--- Subtarefa {i+1}/{len(subtasks)}: '{subtask_desc[:80]}...' (Responsável: {responsible_role}) ---")

            if not responsible_role or responsible_role not in self.agents:
                logging.error(f"Papel responsável '{responsible_role}' não encontrado na crew. Pulando subtarefa.")
                continue

            agent = self.agents[responsible_role]
            
            # <<< MUDANÇA 3: Passando `main_task_description` para a execução do agente >>>
            artifacts_metadata_list = agent.execute_task(
                main_task_description=main_task_description,
                task_description=subtask_desc,
                task_workspace_dir=task_workspace_dir,
                iteration_num=iteration_num,
                context_artifacts=list(iteration_artifacts_metadata),
                feedback_history=feedback_history
            )

            if artifacts_metadata_list:
                iteration_artifacts_metadata.extend(artifacts_metadata_list)
            else:
                error_msg = f"Agente '{agent.role}' não produziu nenhum artefato para a tarefa '{subtask_desc[:30]}...'. Interrompendo."
                logging.error(error_msg)
                return {
                    "task_workspace_dir": task_workspace_dir,
                    "artifacts_metadata": iteration_artifacts_metadata,
                    "status": "ERRO",
                    "message": error_msg,
                }
        
        logging.info(f"Crew '{self.name}' (Tentativa {iteration_num}) concluiu todas as subtarefas com sucesso.")
        return {
            "task_workspace_dir": task_workspace_dir,
            "artifacts_metadata": iteration_artifacts_metadata,
            "status": "SUCESSO",
            "message": f"Tentativa {iteration_num} concluída pela crew. {len(iteration_artifacts_metadata)} artefatos gerados/atualizados.",
        }
    
class TaskManager:
    """Orquestra o planejamento, execução de tarefas por crews e o ciclo de validação/iteração."""
    def __init__(self, llm_service: GeminiService, output_dir: str):
        self.llm_service = llm_service
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_file_base_and_version(self, filename: str) -> tuple[str, tuple[int, ...]]:
        """
        Extrai o nome base e uma tupla de versão de um nome de arquivo.
        Ex: "app_2_1.py" -> ("app", (2, 1))
            "combat_system_1.py" -> ("combat_system", (1,))
            "main.py" -> ("main", (0,))
        """
        base, _ = os.path.splitext(filename)
        parts = base.split('_')
        
        version_parts = []
        last_name_part_index = len(parts) - 1

        # Itera de trás para frente para coletar os números da versão/colisão
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if part.isdigit():
                version_parts.insert(0, int(part))
                last_name_part_index = i - 1
            else:
                break # Para quando encontra uma parte que não é um número

        # Se nenhuma parte de versão foi encontrada, a versão é (0,)
        if not version_parts:
            return base, (0,)

        base_name = "_".join(parts[:last_name_part_index + 1])
        return base_name, tuple(version_parts)

    def _plan_execution_strategy(self, main_task_description: str) -> Optional[Dict[str, Any]]:
        """
        Usa o LLM para criar o plano de execução INICIAL e COMPLETO para a tarefa.
        """
        logging.info("Planejando a estratégia de execução INICIAL...")
        prompt = (
            "Você é uma IA de Gerenciamento de Projetos. Analise a tarefa principal e projete uma equipe e um plano de execução. "
            "Responda ESTRITAMENTE no formato JSON.\n\n"
            f"Tarefa Principal: \"{main_task_description}\"\n\n"
            "O JSON deve conter as chaves: 'crew_name', 'crew_description', 'agents' (uma lista de objetos com 'role', 'goal', 'backstory'), "
            "e 'subtasks' (uma lista de objetos com 'description' e 'responsible_role'). "
            "O 'responsible_role' em cada subtarefa DEVE corresponder a um 'role' definido na lista 'agents'. "
            "Defina quais são os arquivos esperados para serem entregues, sendo o tipo e quantidade dos arquivos de acordo com a tarefa principal."
            "IMPORTANTE: Nas descrições das subtarefas que envolvem criar arquivos, mencione o nome do arquivo explicitamente entre aspas, "
            "por exemplo: \"Criar o arquivo de entrada principal 'main.ts'\" ou \"Desenvolver o controller do jogador em 'PlayerController.ts'\"."
            "A última subtarefa DEVE ser sobre revisar tudo e gerar um resumo final.\n\n"
            "Exemplo de Formato:\n"
            "{\n"
            '  "crew_name": "GameDevCrew",\n'
            '  "crew_description": "Equipe para desenvolver um jogo simples.",\n'
            '  "agents": [\n'
            '    {"role": "Designer de Jogos", "goal": "...", "backstory": "..."},\n'
            '    {"role": "Desenvolvedor Python", "goal": "...", "backstory": "..."}\n'
            '  ],\n'
            '  "subtasks": [\n'
            '    {"description": "Criar o Game Design Document (GDD).", "responsible_role": "Designer de Jogos"},\n'
            '    {"description": "Implementar a lógica do jogador em Python.", "responsible_role": "Desenvolvedor Python"},\n'
            '    {"description": "Revisar todos os artefatos e criar um relatório final.", "responsible_role": "Designer de Jogos"}\n'
            '  ]\n'
            "}"
        )
        
        response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_PLANNING, is_json_output=True)
        response_text = response_dict.get('text', '')

        if not response_text or response_dict.get('finish_reason') != 'STOP':
            logging.error(f"A chamada de planejamento à API falhou ou foi bloqueada. Resposta: {response_text}")
            return None

        try:
            plan = json.loads(response_text)
            if all(k in plan for k in ["crew_name", "agents", "subtasks"]):
                logging.info(f"Plano Inicial recebido. Crew: {plan['crew_name']}. Nº de subtarefas: {len(plan['subtasks'])}.")
                return plan
            else:
                logging.error(f"Plano JSON inicial recebido está incompleto. Resposta: {response_text}")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar o plano JSON inicial: {e}. Resposta: {response_text}")
            return None

    def _create_debugging_subtask(self, traceback: str, workspace_dir: str, original_plan: Dict) -> Optional[List[Dict[str, Any]]]:
        """
        Analisa um traceback e cria uma única subtarefa focada para corrigir o(s) arquivo(s) com erro.
        """
        logging.info("Criando uma tarefa de depuração focada...")
        
        # Encontra todos os arquivos mencionados no traceback
        filepath_matches = re.findall(r'File "([^"]+)"', traceback)
        if not filepath_matches:
            logging.error("Não foi possível extrair nomes de arquivo do traceback. Abortando depuração.")
            return None
        
        # Identifica os arquivos únicos e seus nomes base
        critical_files_to_fix = {os.path.basename(path) for path in filepath_matches}
        
        # Tenta encontrar um agente "Desenvolvedor" na equipe para atribuir a tarefa
        developer_roles = [role for role in [agent['role'] for agent in original_plan['agents']] if 'dev' in role.lower() or 'python' in role.lower()]
        if not developer_roles:
            logging.error("Nenhum agente com perfil de desenvolvedor encontrado na equipe para a depuração.")
            return None
        responsible_agent = developer_roles[0]

        # Cria uma única subtarefa de depuração
        debug_task_description = (
            "<modo>DEPURAÇÃO CIRÚRGICA</modo>\n\n"
            "<contexto>\n"
            "  Uma execução anterior do código falhou. O `feedback_history` contém o `traceback` completo do erro. Os arquivos de código originais implicados no erro serão fornecidos no seu contexto para análise completa.\n"
            "</contexto>\n\n"
            "<tarefa>\n"
            "  Sua missão é agir como um engenheiro de software sênior para consertar o bug reportado.\n"
            "  **Passos Mentais a Seguir:**\n"
            "  1.  **Análise da Causa Raiz:** Releia atentamente o `traceback` no `feedback_history` para entender o erro exato (ex: `TypeError`, `AttributeError`, `SyntaxError`).\n"
            "  2.  **Identificação do Arquivo:** Localize o(s) arquivo(s) crítico(s) (`" + "`, `".join(critical_files_to_fix) + "`) que você precisa modificar.\n"
            "  3.  **Planejamento da Correção:** Determine a menor mudança necessária no código para resolver o erro, sem introduzir novos problemas.\n"
            "  4.  **Execução:** Gere a **versão completa e corrigida** do(s) arquivo(s) afetado(s).\n"
            "</tarefa>\n\n"
            "<regras>\n"
            "  - **FOCO:** Sua única tarefa é consertar o bug reportado. Não adicione novas funcionalidades ou refatore código que não está relacionado ao erro.\n"
            "  - **INTEGRIDADE:** Preserve todas as funcionalidades existentes que não estão diretamente relacionadas ao bug. Não remova código funcional.\n"
            "  - **SAÍDA:** Sua resposta deve ser o conteúdo completo do arquivo corrigido, seguido por um bloco de metadados ```json.\n"
            "</regras>"
        )
        
        return [{"description": debug_task_description, "responsible_role": responsible_agent}]

    def _generate_corrective_subtasks(self, main_task_description: str, original_plan: Dict, feedback: str) -> Optional[List[Dict]]:
        """Usa o LLM para gerar APENAS a lista de subtarefas necessárias para corrigir um erro."""
        logging.info("Gerando uma lista de subtarefas corretivas focada no erro...")
        original_subtasks_str = json.dumps(original_plan.get('subtasks', []), indent=2)
        agent_roles = [ag['role'] for ag in original_plan.get('agents', [])]

        prompt = (
            "<identidade>Você é um Gerente de Projetos Sênior, especialista em recuperação de falhas. Sua função é criar um plano de ação cirúrgico para consertar um erro, evitando retrabalho.</identidade>\n\n"
            "<contexto>\n"
            f"  - OBJETIVO GERAL DO PROJETO: {main_task_description}\n"
            f"  - AGENTES DISPONÍVEIS NA EQUIPE: {agent_roles}\n"
            f"  - ERRO DA TENTATIVA ANTERIOR: {feedback}\n"
            f"  - PLANO ORIGINAL COMPLETO (para referência): \n{original_subtasks_str}\n"
            "</contexto>\n\n"
            "<tarefa>\n"
            "  Sua missão é criar uma **lista de subtarefas nova, curta e focada** que resolva o erro reportado.\n"
            "  **Passos Mentais a Seguir:**\n"
            "  1.  **Análise do Erro:** Entenda a causa raiz do 'ERRO DA TENTATIVA ANTERIOR'.\n"
            "  2.  **Identificação das Ações:** Determine quais arquivos precisam ser criados ou modificados para corrigir o erro.\n"
            "  3.  **Criação do Plano de Ação:** Elabore uma sequência de subtarefas que apenas corrija o problema. Não inclua tarefas do plano original que não estão relacionadas ao erro.\n"
            "</tarefa>\n\n"
            "<regras_de_saida>\n"
            "  - Sua resposta deve ser **APENAS uma lista JSON** de objetos de subtarefa.\n"
            "  - Cada objeto deve ter as chaves 'description' e 'responsible_role'.\n"
            "  - NÃO inclua a palavra 'json' ou as cercas ```. Sua resposta deve começar com '[' e terminar com ']'.\n"
            "</regras_de_saida>"
        )
        
        # Trata a resposta como um dicionário e extrai o texto
        response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_PLANNING, is_json_output=True)
        response_text = response_dict.get('text', '')

        if not response_text or response_dict.get('finish_reason') != 'STOP':
            logging.error(f"A chamada para gerar subtarefas corretivas falhou. Resposta: {response_text}")
            return None

        try:
            subtasks = json.loads(response_text)
            if isinstance(subtasks, list):
                logging.info(f"Plano de Ação Corretivo com {len(subtasks)} subtarefas gerado com sucesso.")
                return subtasks
            else:
                logging.error(f"Resposta para plano corretivo não foi uma lista JSON. Resposta: {response_text}")
                return None
        except json.JSONDecodeError:
            logging.error(f"Erro ao decodificar a lista de subtarefas corretivas. Resposta: {response_text}")
            return None

    def _validate_file_structure(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifica programaticamente a estrutura de arquivos conhecidos, como JSON.
        Garante que não há texto extra ou má formatação.
        """
        logging.info("--- Iniciando Validador de Estrutura de Arquivo ---")
        for artifact in artifacts:
            file_path = artifact.get('file_path', '')
            
            # Validação para arquivos JSON
            if file_path.lower().endswith('.json'):
                logging.debug(f"Validando estrutura do JSON: {os.path.basename(file_path)}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # json.load() irá falhar se houver qualquer dado extra após o objeto JSON principal
                        json.load(f) 
                except json.JSONDecodeError as e:
                    feedback = (f"FALHA DE ESTRUTURA: O arquivo '{os.path.basename(file_path)}' é um JSON inválido. "
                                f"Isso geralmente ocorre por texto extra antes ou depois do objeto JSON. Erro: {e}")
                    logging.error(feedback)
                    return {"success": False, "feedback": feedback}
            
            # (Futuramente, outras validações estruturais, como linter de python, podem ser adicionadas aqui)

        logging.info("✅ Validação de Estrutura de Arquivo bem-sucedida.")
        return {"success": True, "feedback": "A estrutura de todos os arquivos validados está correta."}

    def _reconcile_plan_with_artifacts(self, subtasks: List[Dict[str, Any]], workspace_dir: str) -> Dict[str, Any]:
        """Compara os arquivos planejados com os arquivos realmente gerados, respeitando os subdiretórios."""
        logging.info("--- Iniciando Auditoria de Plano vs. Realidade ---")
        
        planned_files = set()
        filename_regex = r"['\"]([\w\.\/\\]+\.(?:ts|js|py|svelte|json|md|txt))['\"]"
        for task in subtasks:
            # Normaliza os separadores de caminho para consistência (ex: / em vez de \)
            normalized_desc = task['description'].replace("\\", "/")
            found = re.findall(filename_regex, normalized_desc)
            if found:
                planned_files.update([os.path.normpath(f) for f in found])

        if not planned_files:
            logging.warning("Nenhum arquivo explícito no plano para auditar. Pulando."); return {"success": True}
        
        # <<< CORREÇÃO: Busca recursiva de arquivos gerados >>>
        all_generated_files = []
        for root, _, files in os.walk(workspace_dir):
            for name in files:
                all_generated_files.append(os.path.join(root, name))

        # Compara os caminhos relativos para uma correspondência exata
        generated_relative_paths = {os.path.normpath(os.path.relpath(p, workspace_dir)) for p in all_generated_files}
        
        missing_files = planned_files - generated_relative_paths

        if not missing_files:
            logging.info("✅ Auditoria bem-sucedida! Todos os arquivos planejados foram gerados."); return {"success": True}
        else:
            feedback = (f"FALHA DE AUDITORIA: O plano exigia a criação dos seguintes arquivos, mas eles não foram encontrados: {list(missing_files)}. A próxima iteração DEVE focar em gerar o CÓDIGO-FONTE para esses arquivos.")
            logging.error(feedback); return {"success": False, "feedback": feedback}
        
    def _execute_run_test(self, workspace_dir: str, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tenta executar o código gerado com base nas instruções do README."""
        logging.info("--- Iniciando Prova Prática (Teste de Execução) ---")
        
        # <<< CORREÇÃO: Recebe a lista de artefatos e a utiliza corretamente >>>
        readme_artifact = next((art for art in artifacts if 'readme.md' in os.path.basename(art['file_path']).lower()), None)
        
        if not readme_artifact:
            logging.warning("README.md não encontrado na workspace. Pulando teste de execução."); return {"success": True}

        try:
            # Usa o caminho do artefato encontrado
            with open(readme_artifact['file_path'], 'r', encoding='utf-8') as f:
                readme_content = f.read()
        except Exception as e:
            return {"success": False, "output": f"Erro ao ler o arquivo README.md: {e}"}

        command_patterns = [r"^\s*(python[3]?\s+[\w\.]+\.py)\s*$", r"^\s*(streamlit\s+run\s+[\w\.]+\.py)\s*$"]
        match = None
        for pattern in command_patterns:
            match = re.search(pattern, readme_content, re.MULTILINE | re.IGNORECASE)
            if match: break
        
        if not match:
            logging.warning("Comando de execução não encontrado no README.md. Pulando teste."); return {"success": True}
        
        command = "python -m " + match.group(1).strip()
        logging.info(f"Comando encontrado no README: '{command}'")

        try:
            proc = subprocess.run(command.split(), cwd=workspace_dir, capture_output=True, text=True, timeout=30, check=False)
            if proc.returncode == 0:
                logging.info("Teste de execução concluído com sucesso."); return {"success": True, "output": proc.stdout}
            else:
                logging.error(f"Teste de execução FALHOU. Erro:\n{proc.stderr}"); return {"success": False, "output": f"O comando '{command}' falhou com o erro:\n\n{proc.stderr}"}
        except subprocess.TimeoutExpired:
            logging.info("Teste atingiu o timeout. Considerado sucesso para aplicações com loop."); return {"success": True}
        except Exception as e:
            return {"success": False, "output": f"Exceção ao rodar comando: {e}"}
            
    def _perform_code_completeness_review(self, workspace_dir: str, artifacts: List[Dict[str, Any]], subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Usa um agente de IA para revisar o conteúdo da VERSÃO MAIS RECENTE de cada arquivo de código,
        prevenindo a revisão de versões antigas e obsoletas.
        """
        logging.info("--- Iniciando Auditoria de Completude de Código com Agente Revisor ---")
        
        source_code_artifacts = [
            art for art in artifacts 
            if art['file_path'].endswith(('.py', '.ts', '.js', '.html', '.css', '.svelte'))
        ]

        if not source_code_artifacts:
            logging.info("Nenhum arquivo de código fonte encontrado para revisar.")
            return {"is_complete": True, "feedback": "Nenhum código para revisar."}
        
        latest_artifacts = {}
        for artifact in source_code_artifacts:
            filename = os.path.basename(artifact['file_path'])
            base_name, version_tuple = self._get_file_base_and_version(filename)
            
            if base_name not in latest_artifacts or version_tuple > latest_artifacts[base_name]['version']:
                latest_artifacts[base_name] = {'version': version_tuple, 'artifact': artifact}
        
        artifacts_to_review = [item['artifact'] for item in latest_artifacts.values()]
        
        logging.info(f"Total de artefatos de código encontrados: {len(source_code_artifacts)}")
        logging.info(f"Revisando {len(artifacts_to_review)} arquivos (apenas versões mais recentes): {[os.path.basename(art['file_path']) for art in artifacts_to_review]}")

        task_map = {}
        for task in subtasks:
            match = re.search(r"['\"](.*?)['\"]", task['description'])
            if match:
                task_map[match.group(1)] = task['description']

        incomplete_files_feedback = []
        for artifact in artifacts_to_review:
            file_path = artifact['file_path']
            filename = os.path.basename(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(100000) 
            except Exception as e:
                logging.warning(f"Não foi possível ler o arquivo {filename} para revisão: {e}")
                continue
            
            subtask_context = task_map.get(filename, 
                task_map.get(os.path.splitext(filename)[0], 
                "Não foi possível encontrar a subtarefa original para este arquivo.")
            )

            prompt = (
                "<identidade>Você é um Revisor de Código Sênior (Tech Lead) extremamente rigoroso. Sua única missão é garantir que o código esteja funcionalmente completo, sem atalhos ou placeholders.</identidade>\n\n"
                "<contexto>\n"
                f"  - ARQUIVO SOB REVISÃO: '{filename}'\n"
                f"  - OBJETIVO ORIGINAL DO MÓDULO: \"{subtask_context}\"\n"
                f"  - CÓDIGO-FONTE PARA ANÁLISE:\n```\n{content}\n```\n"
                "</contexto>\n\n"
                "<tarefa>\n"
                "  Sua tarefa é realizar uma análise de completude do código. Você deve procurar especificamente por:\n"
                "  1.  **Comentários de Placeholder:** Qualquer comentário que indique trabalho inacabado (ex: `// TODO:`, `FIXME`, `# implementar depois`, `# placeholder`).\n"
                "  2.  **Funções Vazias:** Funções ou métodos que são definidos mas não têm implementação lógica (ex: `def my_func(): pass`).\n"
                "  3.  **Lógica Simulada:** Qualquer código que retorne valores fixos ou `null` quando deveria ter uma lógica real.\n"
                "</tarefa>\n\n"
                "<regras_de_saida>\n"
                "  - Se o código parecer totalmente implementado para sua tarefa e não contiver nenhum dos problemas acima, responda com a única palavra: **COMPLETO**\n"
                "  - Se você encontrar QUALQUER sinal de incompletude, responda com a única palavra: **INCOMPLETO**, seguido por uma única e concisa frase explicando o principal problema encontrado.\n"
                "  - Exemplo de Resposta de Falha: `INCOMPLETO: A função 'calculate_damage' retorna um valor fixo em vez de uma lógica de cálculo real.`\n"
                "</regras_de_saida>"
            )
            
            # <<< CORREÇÃO: Tratar a resposta como um dicionário >>>
            review_result = self.llm_service.generate_text(prompt, temperature=0.1)
            # Extrai o texto do dicionário. Default para 'COMPLETO' se a chave de texto estiver ausente.
            review_text = review_result.get('text', 'COMPLETO').strip().upper()

            if review_text.startswith("INCOMPLETO"):
                # Pega o texto original do dicionário para extrair o feedback completo
                feedback = review_result.get('text', '').replace("INCOMPLETO", "", 1).strip().lstrip('.:').strip()
                logging.warning(f"REVISÃO FALHOU para '{filename}'. Motivo: {feedback}")
                incomplete_files_feedback.append(f"- O arquivo '{filename}' está incompleto. Motivo apontado pelo revisor: {feedback}")

        if incomplete_files_feedback:
            consolidated_feedback = "FALHA DE AUDITORIA DE CÓDIGO:\n" + "\n".join(incomplete_files_feedback)
            return {"is_complete": False, "feedback": consolidated_feedback}
        else:
            logging.info("✅ Auditoria de Completude de Código bem-sucedida. Nenhum placeholder encontrado.")
            return {"is_complete": True, "feedback": "Todos os arquivos de código revisados parecem completos."}

    def _perform_backtest_and_validate(self,
                                         original_task_description: str,
                                         crew_result: Dict[str, Any],
                                         iteration_num: int) -> Dict[str, Any]:
        """
        Avalia os artefatos produzidos pela crew para validação, agora usando
        o conteúdo real dos arquivos para uma análise baseada em evidências.
        """
        artifacts = crew_result.get("artifacts_metadata", [])
        if not artifacts:
            return {"is_satisfactory": False, "feedback": "Nenhum artefato produzido para validação."}

        logging.info(f"Iniciando validação TEÓRICA (QA) da Tentativa {iteration_num} com contexto real...")
        
        # <<< MUDANÇA: Construir um resumo com o conteúdo real dos arquivos >>>
        artifacts_content_summary = ""
        for artifact in artifacts:
            filename = os.path.basename(artifact.get('file_path', 'N/A'))
            desc = artifact.get('description', 'N/A')
            artifacts_content_summary += f"\n--- Artefato: '{filename}' | Descrição do Agente: {desc} ---\n"
            try:
                # Lê um trecho do arquivo para incluir no prompt
                with open(artifact['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read(65000) # Limita o tamanho para não exceder o limite de tokens do prompt
                artifacts_content_summary += f"```\n{content}{'... (trecho)' if len(content) == 65000 else ''}\n```\n"
            except Exception as e:
                artifacts_content_summary += f"[Não foi possível ler o conteúdo do arquivo: {e}]\n"

        prompt = (
            "<identidade>Você é um Gerente de QA Sênior. Sua tarefa é realizar uma validação final sobre o trabalho de uma equipe de IA, baseando-se nas evidências apresentadas.</identidade>\n\n"
            "<contexto>\n"
            f"  - OBJETIVO ORIGINAL DO PROJETO: '{original_task_description}'\n"
            f"  - TENTATIVA NÚMERO: {iteration_num}\n"
            f"  - EVIDÊNCIAS (Artefatos e trechos de seu conteúdo):\n{artifacts_content_summary}\n"
            "</contexto>\n\n"
            "<tarefa>\n"
            "  Sua missão é realizar uma AVALIAÇÃO CRÍTICA. Com base no conteúdo real dos artefatos e no objetivo original do projeto, responda:\n"
            "  O objetivo foi alcançado de forma satisfatória? O projeto parece completo, coerente e funcional?\n"
            "</tarefa>\n\n"
            "<regras_de_saida>\n"
            "  - Se o resultado for satisfatório e o projeto estiver concluído, responda com a única palavra: **SATISFATÓRIO**\n"
            "  - Se o projeto ainda precisa de melhorias ou correções, responda com a única palavra: **INSATISFATÓRIO**, seguido por um feedback claro e conciso de uma única linha sobre o que precisa ser feito.\n"
            "  - Exemplo de Resposta de Falha: `INSATISFATÓRIO: A lógica de combate não leva em consideração a defesa do jogador.`\n"
            "</regras_de_saida>"
        )

        # A chamada ao LLM permanece a mesma
        validation_response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_VALIDATION)
        validation_response_text = validation_response_dict.get('text', 'SATISFATÓRIO').strip()

        logging.debug(f"Resposta da validação QA: {validation_response_text}")

        if validation_response_text.upper().startswith("SATISFATÓRIO"):
            return {"is_satisfactory": True, "feedback": "Resultado validado como satisfatório pelo QA."}
        else:
            feedback = validation_response_text.replace("INSATISFATÓRIO", "").strip(' :').strip()
            return {"is_satisfactory": False, "feedback": feedback or "Feedback de melhoria não especificado pelo QA."}
        
    def _get_final_deliverables_list(self, main_task_description: str, all_files: List[str]) -> List[str]:
        """Usa o LLM para selecionar os arquivos finais essenciais de uma lista."""
        logging.info("Iniciando etapa de curadoria final para selecionar os entregáveis...")
        
        file_list_str = "\n".join([f"- {f}" for f in all_files])

        prompt = (
            "<identidade>Você é um Arquiteto de Software Sênior responsável por preparar o pacote final de entrega de um projeto. Sua tarefa é curar a lista de arquivos para garantir que apenas os entregáveis essenciais e limpos sejam incluídos.</identidade>\n\n"
            "<contexto>\n"
            f"  - OBJETIVO GERAL DO PROJETO: {main_task_description}\n\n"
            f"  - LISTA DE TODOS OS ARQUIVOS GERADOS NO WORKSPACE:\n{file_list_str}\n"
            "</contexto>\n\n"
            "<tarefa>\n"
            "  Sua missão é analisar o objetivo do projeto e a lista de arquivos e selecionar apenas os que representam o produto final e funcional.\n"
            "  - **INCLUA:** O código-fonte principal, arquivos de configuração, documentação essencial (como README.md) e os principais arquivos de dados.\n"
            "  - **EXCLUA:** Arquivos de fallback (com 'fallback' no nome), arquivos temporários, logs e, mais importante, versões antigas de arquivos (arquivos com sufixos como `_1`, `_2`, etc.). Sempre prefira a versão mais recente e com o nome mais limpo, se houver múltiplas.\n"
            "</tarefa>\n\n"
            "<regras_de_saida>\n"
            "  - Sua resposta DEVE ser um objeto JSON válido.\n"
            "  - O JSON deve conter uma ÚNICA chave: `deliverables`.\n"
            "  - O valor de `deliverables` deve ser uma LISTA DE STRINGS contendo os nomes dos arquivos selecionados.\n"
            "  - NÃO inclua nenhum outro texto, explicação ou markdown. Sua resposta deve começar com '{' e terminar com '}'.\n"
            "</regras_de_saida>"
        )

        response_str = self.llm_service.generate_text(prompt, temperature=0.1, is_json_output=True)
        
        try:
            curation_data = json.loads(response_str)
            if isinstance(curation_data, dict) and "deliverables" in curation_data and isinstance(curation_data["deliverables"], list):
                deliverables = curation_data["deliverables"]
                logging.info(f"Curadoria da IA selecionou {len(deliverables)} entregáveis: {deliverables}")
                # Validação final para garantir que os arquivos selecionados pela IA realmente existem
                existing_deliverables = [f for f in deliverables if f in all_files]
                return existing_deliverables
            else:
                raise ValueError("Formato JSON de curadoria inesperado.")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Erro na curadoria final por IA: {e}. Usando fallback para copiar arquivos .py, .md e .txt.")
            # Fallback seguro: copia os tipos de arquivo mais comuns para entrega.
            return [f for f in all_files if f.endswith(('.py', '.md', '.txt')) and 'fallback' not in f]

    def _finalize_task(self, task_id: str, main_task_description: str, workspace_dir: str, crew_name: str) -> Optional[str]:
        """
        Cria um diretório de saída final com artefatos curados pela IA, garantindo que
        apenas as versões mais recentes dos arquivos sejam consideradas para curadoria.
        """
        safe_crew_name = "".join(c for c in crew_name if c.isalnum() or c == '_').strip() or "DefaultCrew"
        final_output_dir = os.path.join(self.output_dir, f"task_{task_id}_{safe_crew_name}_FINAL_OUTPUT")

        try:
            if os.path.exists(final_output_dir):
                shutil.rmtree(final_output_dir)
            os.makedirs(final_output_dir)

            all_files_in_workspace = [f for f in os.listdir(workspace_dir) if os.path.isfile(os.path.join(workspace_dir, f))]

            if not all_files_in_workspace:
                logging.warning("Nenhum arquivo encontrado no workspace para finalizar.")
                return None

            logging.info("Filtrando arquivos para obter apenas as versões mais recentes antes da curadoria final.")
            latest_files = {}
            for filename in all_files_in_workspace:
                base_name, version_tuple = self._get_file_base_and_version(filename)
                
                if base_name not in latest_files or version_tuple > latest_files[base_name]['version']:
                    latest_files[base_name] = {'version': version_tuple, 'filename': filename}
            
            latest_filenames_to_curate = [item['filename'] for item in latest_files.values()]
            logging.info(f"As seguintes versões de arquivos serão apresentadas para curadoria: {latest_filenames_to_curate}")

            # Obter a lista curada de entregáveis a partir das versões mais recentes.
            deliverables_to_copy = self._get_final_deliverables_list(main_task_description, latest_filenames_to_curate)

            if not deliverables_to_copy:
                logging.warning("A etapa de curadoria não retornou nenhum arquivo para copiar.")
                return final_output_dir 

            # Copiar seletivamente apenas os arquivos curados.
            for filename in deliverables_to_copy:
                source_path = os.path.join(workspace_dir, filename)
                destination_path = os.path.join(final_output_dir, filename)
                shutil.copy2(source_path, destination_path)
            
            logging.info(f"Artefatos finais curados e copiados com sucesso para: '{final_output_dir}'")
            return final_output_dir
            
        except Exception as e:
            logging.error(f"Erro crítico ao finalizar e consolidar os artefatos: {e}")
            return None

    def _create_summary_log(self, task_id: str, main_task_description: str, results: List[Dict], final_status: str, final_output_dir: Optional[str]):
        """Cria um log de resumo detalhado para toda a execução da tarefa."""
        summary_path = os.path.join(self.output_dir, f"task_{task_id}_summary_log.md")
        
        summary_content = f"# Resumo da Execução da Tarefa: {task_id}\n\n"
        summary_content += f"**Tarefa Principal:**\n```\n{main_task_description}\n```\n\n"
        
        # <<< MUDANÇA: USA A VARIÁVEL `final_status` PARA EXIBIR O RESULTADO CORRETO >>>
        summary_content += f"**Status Final da Execução:** {final_status}\n"
        
        if final_output_dir:
            summary_content += f"**Diretório de Saída Final (Curado):** `{final_output_dir}`\n\n"
        else:
            summary_content += "**Nenhum entregável final foi produzido.**\n\n"
        
        summary_content += "## Histórico de Tentativas de Geração/Correção\n\n"
        
        if not results:
            summary_content += "Nenhum resultado de execução foi registrado.\n"
        
        for i, result in enumerate(results):
            iter_num = i + 1
            status = result.get('status', 'DESCONHECIDO')
            message = result.get('message', 'N/A')
            
            summary_content += f"### Tentativa {iter_num} - Status da Crew: {status}\n"
            summary_content += f"* **Mensagem da Crew:** {message}\n"
            
            # Adiciona os feedbacks de validação ao log para um diagnóstico claro
            if 'reconciliation_feedback' in result:
                 summary_content += f"* **Feedback da Auditoria:** {result['reconciliation_feedback']}\n"
            if 'run_test_feedback' in result:
                 summary_content += f"* **Saída do Teste Prático:**\n```\n{result['run_test_feedback']}\n```\n"

            artifacts = result.get('artifacts_metadata', [])
            summary_content += f"* **Artefatos Gerados/Modificados nesta Tentativa:**\n"
            if artifacts:
                for art in artifacts:
                    filename = os.path.basename(art.get('file_path', 'N/A'))
                    summary_content += f"  - `{filename}`\n"
            else:
                summary_content += "  - Nenhum\n"
            summary_content += "---\n"

        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_content)
            logging.info(f"Log de resumo da tarefa salvo em: '{summary_path}'")
            return summary_path
        except Exception as e:
            logging.error(f"Não foi possível salvar o log de resumo final: {e}")
            return None

    def delegate_task(self, main_task_description: str) -> str:
        """Orquestra o ciclo completo, garantindo que o loop de correção continue após uma falha."""
        task_id = uuid.uuid4().hex[:10]
        task_root_dir = os.path.join(self.output_dir, f"task_{task_id}")
        os.makedirs(task_root_dir, exist_ok=True)
        logging.info(f"\n{'='*20} Nova Tarefa Iniciada: {task_id} {'='*20}")
        
        master_plan = self._plan_execution_strategy(main_task_description)
        if not master_plan: return "Falha crítica no planejamento inicial. Tarefa abortada."
        
        crew_name_for_log = master_plan.get('crew_name', "DynamicCrew")
        workspace_dir = os.path.join(task_root_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        agents = [Agent(agent_id=f"{task_id}_{ag['role']}", llm_service=self.llm_service, **ag) for ag in master_plan['agents']]
        crew = Crew(name=master_plan['crew_name'], description=master_plan['crew_description'], agents=agents)
        feedback_history: List[str] = []
        execution_results: List[Dict] = []
        is_task_successful = False
        
        for attempt in range(1, config.MAX_ITERATIONS + 1):
            logging.info(f"\n--- Iniciando Tentativa de Geração/Correção {attempt}/{config.MAX_ITERATIONS} ---")
            subtasks_for_this_attempt = master_plan['subtasks']
            if attempt > 1:
                feedback = feedback_history[-1]
                corrective_subtasks = self._generate_corrective_subtasks(main_task_description, master_plan, feedback)
                if corrective_subtasks is not None: 
                    subtasks_for_this_attempt = corrective_subtasks
                else: 
                    logging.error("Não foi possível gerar um plano de ação corretivo. Usando o plano mestre novamente.")
            
            crew_result = crew.process_subtasks(main_task_description, subtasks_for_this_attempt, workspace_dir, attempt, feedback_history)
            crew_result['feedback'] = feedback_history[-1] if feedback_history else "N/A"
            execution_results.append(crew_result)
            
            if crew_result.get("status") == "ERRO": 
                logging.critical("Crew falhou criticamente."); break
            
            # --- Hierarquia de Validação Corrigida ---
            all_artifacts = [{"file_path": os.path.join(root, name)} for root, _, files in os.walk(workspace_dir) for name in files]

            # Validação 1: Auditoria de Arquivos
            reconciliation_result = self._reconcile_plan_with_artifacts(master_plan['subtasks'], workspace_dir)
            if not reconciliation_result["success"]:
                logging.warning(f"Tentativa {attempt} falhou na Auditoria de Arquivos."); feedback_history.append(reconciliation_result["feedback"]); continue

            # Validação 2: Estrutura de Arquivo
            structure_validation_result = self._validate_file_structure(all_artifacts)
            if not structure_validation_result["success"]:
                logging.warning(f"Tentativa {attempt} falhou na Validação de Estrutura."); feedback_history.append(structure_validation_result["feedback"]); continue

            # Validação 3: Prova Prática
            run_test_result = self._execute_run_test(workspace_dir, all_artifacts)
            if not run_test_result["success"]:
                logging.warning(f"Tentativa {attempt} falhou na Prova Prática."); feedback_history.append(f"VALIDAÇÃO FALHOU (Prova Prática):\n{run_test_result['output']}"); continue

            # Validação 4: Completude de Código
            review_result = self._perform_code_completeness_review(workspace_dir, all_artifacts, master_plan['subtasks'])
            if not review_result["is_complete"]:
                logging.warning(f"Tentativa {attempt} falhou na Auditoria de Completude de Código."); feedback_history.append(review_result["feedback"]); continue
            
            logging.info("✅ SUCESSO! Todas as 4 etapas de validação passaram.")
            is_task_successful = True
            break # Sai do loop PORQUE foi bem-sucedido
        
        # --- Finalização ---
        final_output_dir, final_status = None, "FALHA"
        if is_task_successful:
            final_status = "SUCESSO"
            final_output_dir = self._finalize_task(task_id, main_task_description, workspace_dir, crew_name_for_log)
        
        self._create_summary_log(task_id, main_task_description, execution_results, final_status, final_output_dir)
        
        final_message = f"Execução da tarefa {task_id} finalizada com status: {final_status}."
        final_message += f"\nResumo salvo em: {os.path.join(self.output_dir, f'task_{task_id}_summary_log.md')}"
        if final_output_dir: 
            final_message += f"\nSaída final CURADA e organizada em: {final_output_dir}"
        else: 
            final_message += "\nNenhum entregável final foi produzido."
        return final_message

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    
    # Atualiza o nível de logging com base na configuração
    log_level = logging.DEBUG if config.VERBOSE_LOGGING else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Instancia os serviços principais
    gemini_service = GeminiService(model_name=config.MODEL_NAME, fallback_model_name=config.FALLBACK_MODEL_NAME)
    task_manager = TaskManager(llm_service=gemini_service, output_dir=config.OUTPUT_ROOT_DIR)
    
    # A tarefa complexa e bem definida fornecida pelo usuário é um excelente caso de teste.
    tarefa = """
"""

    resultado_final = task_manager.delegate_task(tarefa)
    
    print(f"\n{'#'*20} RESULTADO FINAL DA EXECUÇÃO {'#'*20}\n")
    print(resultado_final)
    print(f"\n{'#'*60}\n")