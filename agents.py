# agents.py
import os
import re
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable

from fpdf import FPDF

from config import config
from services import GeminiService
from utils import parse_llm_output, clean_markdown_code_fences, sanitize_filename
from shared_context import SharedContext
from app_logger import logger

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

    def _infer_content_type(self, content: str, analysis_depth_lines: int = 20) -> str:
        """
        Analisa um bloco de texto de forma mais robusta para inferir se é código ou documento,
        usando um sistema de pontuação baseado em heurísticas.
        """
        if not isinstance(content, str) or not content.strip():
            return 'document'

        code_score = 0
        lines = content.strip().splitlines()
        lines_to_analyze = lines[:analysis_depth_lines]

        # 1. Indicadores Fortes de Código (alta pontuação)
        strong_code_keywords = [
            'import ', 'from ', 'def ', 'class ', 'function ', 'const ', 'let ', 'var ',
            'public class', 'public static', 'void main', '#include', 'require(', 'using '
        ]
        # Shebangs no início do arquivo
        if lines and lines[0].startswith('#!'):
            code_score += 10

        for line in lines_to_analyze:
            # Palavras-chave de definição/importação
            if any(keyword in line for keyword in strong_code_keywords):
                code_score += 5
            
            # Sintaxe de comentários de código
            if line.strip().startswith(('//', '#', '/*')):
                code_score += 2

            # Finais de linha comuns em código
            if line.strip().endswith((';', '{', '}', '):', '=> {')):
                code_score += 2

        # 2. Densidade de Caracteres Especiais
        text_sample = "\n".join(lines_to_analyze)
        total_chars = len(text_sample)
        if total_chars > 0:
            special_chars = sum(text_sample.count(c) for c in '(){}[]<>;:=!&|+-*/%')
            density = special_chars / total_chars
            if density > 0.1:  # Mais de 10% de caracteres são símbolos de código
                code_score += 10
            elif density > 0.05: # Mais de 5%
                code_score += 5

        # 3. Indicadores de Documento (pontuação negativa)
        # Verifica se há sentenças longas e bem formadas (prosa)
        prose_indicators = 0
        for line in lines_to_analyze:
            line = line.strip()
            # Linhas longas com espaços e terminando com um ponto.
            if len(line) > 80 and line.endswith('.') and ' ' in line:
                prose_indicators += 1
        
        if prose_indicators >= 2: # Se encontrar 2 ou mais linhas que parecem prosa
            code_score -= 10
            
        # Títulos de Markdown
        if any(line.strip().startswith(('# ', '## ')) for line in lines_to_analyze):
             code_score -= 5

        # Decisão Final com base na pontuação
        # O limiar de 10 foi escolhido empiricamente. Pode ser ajustado.
        if code_score >= 10:
            return 'code'
        else:
            return 'document'

    def execute_task(self,
                     main_task_description: str,
                     task_description: str,
                     task_workspace_dir: str,
                     iteration_num: int,
                     context_artifacts: List[Dict[str, Any]],
                     feedback_history: List[str],
                     shared_context: SharedContext,
                     uploaded_files_content: dict = None
                     ) -> List[Dict[str, Any]]:
        
        is_correction_attempt = bool(feedback_history)
        web_search_results = ""
        max_search_attempts = 2

        for search_attempt in range(max_search_attempts):
            # --- Construção do Contexto ---
            context_summary = "Nenhum artefato de contexto anterior fornecido.\n"
            if context_artifacts:
                latest_artifacts = {}
                for artifact in context_artifacts:
                    file_path = artifact.get('file_path')
                    if file_path:
                        # A chave é o caminho do arquivo; o valor é o artefato mais recente encontrado.
                        # Como a lista é sequencial, o último artefato para um caminho será sempre o mais novo.
                        latest_artifacts[file_path] = artifact
                
                # Usa a lista filtrada de artefatos únicos e mais recentes.
                artifacts_to_process = list(latest_artifacts.values())

                context_summary = "A seguir estão os artefatos de contexto já existentes:\n"
                critical_files_in_feedback = set()
                if is_correction_attempt:
                    feedback_text_for_search = "".join(feedback_history)
                    found_paths = re.findall(r'File "[^"]*[\\/]([\w\._-]+)"', feedback_text_for_search)
                    critical_files_in_feedback.update(found_paths)
                
                for art_meta in artifacts_to_process:
                    filename = os.path.basename(art_meta.get('file_path', 'N/A'))
                    context_summary += f"\n--- Artefato: '{filename}' ---\n"
                    try:
                        with open(art_meta['file_path'], 'r', encoding='utf-8') as f:
                            if filename in critical_files_in_feedback:
                                content = f.read()
                                context_summary += f"```\n[CONTEÚDO COMPLETO DO ARQUIVO CRÍTICO]\n{content}\n```\n"
                            else:
                                content = f.read(20000)
                                context_summary += f"```\n{content}\n"
                    except Exception as e:
                        context_summary += f"[Não foi possível ler o arquivo: {e}]\n"

            if is_correction_attempt:
                context_summary += f"\n--- HISTÓRICO DE FEEDBACK (MAIS RECENTE É MAIS IMPORTANTE) ---\n" + "\n---\n".join(reversed(feedback_history))
            
            if uploaded_files_content:
                context_summary += "\n--- CONTEÚDO DOS ARQUIVOS ANEXADOS ---\n"
                for filename, content in uploaded_files_content.items():
                    context_summary += f"--- Arquivo: {filename} ---\n"
                    context_summary += f"```\n{content}\n```\n"
                context_summary += "-------------------------------------\n"
                
            communication_history = shared_context.get_full_context_for_prompt(self.role)
            context_summary += f"\n--- HISTÓRICO DE COMUNICAÇÃO DA EQUIPE ---\n{communication_history}\n"

            if web_search_results:
                    context_summary += f"\n--- RESULTADOS DA PESQUISA WEB (Use esta informação para sua tarefa) ---\n{web_search_results}\n"
                    
            # --- Geração do Prompt ---
            prompt_header = (
                f"<instrucoes_de_identidade>\n"
                f"  <papel>{self.role}</papel>\n"
                f"  <objetivo_especifico>{self.goal}</objetivo_especifico>\n"
                f"  <historia_de_fundo>{self.backstory}</historia_de_fundo>\n"
                f"</instrucoes_de_identidade>\n\n"
                "<regras_inviolaveis_de_saida>\n"
                "  - Sua resposta DEVE OBRIGATORIAMENTE ser uma das duas opções:\n"
                "  1.  **AÇÃO DE PESQUISA:** Se você precisar de mais informações, sua ÚNICA resposta deve ser um bloco de código JSON contendo uma ação de pesquisa. Exemplo:\n"
                "      ```json\n"
                '      {\n        "action": "search",\n        "query": "qual a sintaxe de uma arrow function em JavaScript?"\n      }\n'
                "      ```\n"
                "  2.  **CRIAÇÃO DE ARTEFATO:** Se você já tem informações suficientes, sua ÚNICA resposta deve ser o conteúdo do(s) artefato(s) solicitado(s), seguido por seu respectivo bloco de metadados ```json.\n"
                "  - NUNCA inclua texto introdutório, conversacional ou explicativo como 'Claro, aqui está o código:'. Vá direto ao ponto.\n"
                "  - Sua resposta DEVE OBRIGATORIAMENTE terminar com um ou mais blocos de metadados ```json, cada um seguindo seu respectivo bloco de conteúdo.\n"
                "  - NÃO gere scripts de shell (bash, sh, cmd). Gere o conteúdo interno dos arquivos.\n"
                "  - Para se comunicar com outros agentes, use um bloco ```message. O conteúdo DEVE ser um JSON com as chaves 'recipient' (o 'role' do destinatário) e 'content' (sua mensagem).\n"
                "  - NÃO subscreva arquivos sem escrever todo o conteúdo do artefato. Se o arquivo já existir, verifique se é seguro sobrescrever.\n"
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
            # Se já temos resultados de uma pesquisa anterior, a instrução deve ser para USÁ-LOS.
            if web_search_results:
                prompt_task += (
                    "MODO: **SÍNTESE DE ARTEFATO**.\n"
                    "Você acabou de receber os resultados de uma pesquisa na web (disponíveis no contexto acima).\n"
                    "Sua ÚNICA tarefa agora é analisar esses resultados e usá-los para gerar o conteúdo completo do artefato solicitado na instrução original.\n"
                    "NÃO peça para pesquisar novamente. Gere o artefato final.\n\n"
                    f"INSTRUÇÃO ORIGINAL (para referência): {task_description}\n"
                )
            # Se não, o agente pode decidir entre criar ou pesquisar.
            elif is_correction_attempt:
                prompt_task += (
                    "MODO: **CORREÇÃO**.\n"
                    "Uma tentativa anterior falhou. Sua tarefa é analisar o erro e decidir o próximo passo.\n"
                    "1.  **Análise:** Releia o 'HISTÓRICO DE FEEDBACK' para entender o erro.\n"
                    "2.  **Decisão:** Se tiver informações suficientes (incluindo o código dos arquivos críticos), gere a versão corrigida do artefato. Se precisar de informações externas (sobre uma API, biblioteca, etc.), gere uma AÇÃO DE PESQUISA em JSON.\n\n"
                    f"INSTRUÇÃO DE CORREÇÃO: {task_description}\n"
                )
            else: # Primeira tentativa de criação
                prompt_task += (
                    "MODO: **CRIAÇÃO**.\n"
                    "Sua tarefa é gerar um novo artefato. Analise a instrução abaixo.\n"
                    "Se tiver todas as informações para completá-la, gere o artefato. Se precisar de dados da internet, gere uma AÇÃO DE PESQUISA em JSON.\n\n"
                    f"INSTRUÇÃO DE CRIAÇÃO: {task_description}\n"
                )
            prompt_task += "</tarefa_especifica>\n\n"

            # --- 4. Bloco de Exemplo de Saída (Reforço Final) ---
            # Repete a instrução mais crítica no final para garantir que seja seguida.
            prompt_footer = (
                "<exemplo_de_formato_de_saida>\n"
                "O caminho do arquivo deve estar presente no suggested_filename.\n"
                "Lembre-se, sua resposta final deve ser o conteúdo do artefato, seguido imediatamente pelo bloco de metadados. Exemplo para um artefato:\n"
                "Opção 1 (Pesquisa):\n"
                "```json\n"
                '{\n  "action": "search",\n  "query": "qual a sintaxe de uma arrow function em JavaScript?"\n}\n'
                "```\n\n"
                "Opção 2 (Artefato):\n"
                "```python\n"
                "print('Hello, World!')\n"
                "```\n"
                "```json\n"
                '{\n  "suggested_filename": "script/hello.py",\n  "description": "Um script de exemplo."\n}\n'
                "```\n"
                "</exemplo_de_saida>"
            )

            # --- Montagem Final do Prompt ---
            prompt = prompt_header + prompt_context + prompt_task + prompt_footer

            # --- 2. Geração com Loop de Continuação ---
            logger.add_log_for_ui(f"Agente '{self.role}' (Iteração {iteration_num}, Pesquisa {search_attempt + 1}) iniciando tarefa: {task_description[:100]}...")
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
                    if is_continuation: logger.add_log_for_ui("Geração de continuação concluída.")
                    break
                is_continuation = True
                logging.warning(f"Resposta truncada (MAX_TOKENS). Preparando continuação ({i+1}/{max_continuations})...")
                structural_context = content_parts[0][:10000]
                immediate_context = content_parts[-1][-10000:]
                current_prompt = (f"Sua resposta foi cortada. Continue EXATAMENTE de onde parou.\n\n--- CONTEXTO ESTRUTURAL (Início do arquivo):\n{structural_context}...\n\n--- FINAL DO SEU TEXTO (Continue daqui):\n...{immediate_context}")
            
            full_llm_response = "".join(content_parts)

            try:
                action_match = re.search(r"```json\s*(\{[\s\S]*?\"action\"\s*:\s*\"search\"[\s\S]*?\})\s*```", full_llm_response, re.DOTALL)
                if action_match:
                    action_json = json.loads(action_match.group(1))
                    query = action_json.get("query")
                    if query:
                        logger.add_log_for_ui(f"Agente '{self.role}' decidiu pesquisar por: '{query}'")
                        search_results = self.llm_service.perform_web_search(query)
                        if search_results:
                            web_search_results = "Resultados da pesquisa:\n" + "\n".join([
                                f"- Título: {res['title']}\n  URL: {res['url']}\n  Trecho: {res['snippet']}"
                                for res in search_results
                            ])
                        else:
                            web_search_results = "Sua pesquisa não retornou resultados. Tente com outros termos ou prossiga com a informação que já possui."
                        continue
            except (json.JSONDecodeError, KeyError):
                pass

            # --- 3. Processamento e Salvamento dos Artefatos ---
            parsed_artifacts = parse_llm_output(full_llm_response)

            if not parsed_artifacts or not parsed_artifacts[0].get("content"):
                feedback_history.append("O agente não produziu um artefato ou ação de pesquisa válida. A resposta estava vazia ou mal formatada.")
                logger.add_log_for_ui(f"Agente '{self.role}' produziu uma resposta inválida/vazia. Forçando nova tentativa.", "warning")
                return []
            
            saved_artifacts_metadata = []

            # Iteramos sobre cada item (dicionário) que o parser encontrou.
            for output in parsed_artifacts:
                output_type = output.get("type")

                # --- Ramo 1: O agente enviou uma mensagem ---
                if output_type == "message":
                    recipient = output.get("recipient", "all")
                    content = output.get("content", "")
                    if shared_context: # Garante que o contexto exista
                        shared_context.add_message(sender=self.role, content=content, recipient=recipient)
                        logger.add_log_for_ui(f"Agente '{self.role}' enviou uma mensagem para '{recipient}'.")
                    continue # Pula para o próximo item da lista

                # --- Ramo 2: O agente produziu um artefato (código/documento) ---
                elif output_type == "artifact":
                    content = output.get("content", "")
                    metadata = output.get("metadata", {})

                    # --- Lógica de Fallback com Autocorreção (adaptada para a nova estrutura) ---
                    if not metadata or 'suggested_filename' not in metadata:
                        logger.add_log_for_ui(f"Artefato sem metadados válidos. Iniciando fallback. Conteúdo: '{content[:100]}...'", "warning")
                        # Sua lógica de fallback existente vai aqui, como o prompt para gerar metadados.
                        naming_prompt = (
                            "Analise o seguinte conteúdo e sugira um nome de arquivo apropriado (com extensão) e uma breve descrição. "
                            "Sua resposta DEVE ser um único objeto JSON com as chaves 'suggested_filename' e 'description'.\n\n"
                            f"CONTEÚDO PARA ANÁLISE:\n---\n{content[:1500]}...\n---\n"
                            "Responda apenas com o JSON."
                        )
                        metadata_response_dict = self.llm_service.generate_text(naming_prompt, temperature=0.1, is_json_output=True)
                        try:
                            # Usa o dicionário retornado diretamente
                            metadata = json.loads(metadata_response_dict.get('text', '{}'))
                            if 'suggested_filename' not in metadata:
                                raise ValueError("Chave 'suggested_filename' ausente no JSON de autocorreção.")
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.add_log_for_ui(f"Autocorreção de metadados falhou: {e}. Usando fallback genérico.", "error")
                            metadata = {
                                "suggested_filename": f"{self.role.replace(' ', '_').lower()}_fallback_{uuid.uuid4().hex[:6]}.txt",
                                "description": f"Fallback genérico para a tarefa: {task_description[:50]}..."
                            }

                    # Extrai as informações dos metadados
                    suggested_filename = metadata.get("suggested_filename")
                    description = metadata.get("description", "Descrição não fornecida.")

                    content_to_save = clean_markdown_code_fences(content)
                
                # --- Lógica de Reconciliação e Salvamento ---
                target_path_from_task = None
                match = re.search(r"['\"]([a-zA-Z0-9_\/\\]+\.[a-zA-Z0-9_]+)['\"]", task_description)
                if match:
                    target_path_from_task = os.path.normpath(match.group(1))

                final_path_to_use = os.path.normpath(suggested_filename)
                if target_path_from_task and os.path.basename(target_path_from_task) == os.path.basename(final_path_to_use):
                    final_path_to_use = target_path_from_task
                    logger.add_log_for_ui(f"Caminho do arquivo reconciliado para: '{final_path_to_use}'.")
                
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
                    
                    logger.add_log_for_ui(f"Agente '{self.role}' salvou/sobrescreveu o artefato: '{output_filepath}'")
                    saved_artifacts_metadata.append({
                        "file_path": output_filepath, "description": description, "agent_role": self.role,
                        "task_description": task_description, "iteration_num": iteration_num,
                    })
                except Exception as e:
                    logging.error(f"Agente '{self.role}' falhou ao salvar o arquivo '{output_filepath}': {e}")
            
            return saved_artifacts_metadata
        
        # Se o loop de pesquisa terminar sem retornar, significa que o agente falhou em produzir um artefato
        logger.add_log_for_ui(f"Agente '{self.role}' excedeu o número de tentativas de pesquisa sem produzir um artefato.", "error")
        return []

class Crew:
    """Gerencia uma equipe de agentes para processar uma série de subtarefas delegadas."""
    def __init__(self, name: str, description: str, agents: List[Agent]):
        self.name = name
        self.description = description
        self.agents = {agent.role: agent for agent in agents}
        self.shared_context = SharedContext()

        logger.add_log_for_ui(f"Crew '{self.name}' criada com {len(agents)} agentes: {list(self.agents.keys())}")

    def process_subtasks(self, main_task_description: str, subtasks: List[Dict[str, Any]], task_workspace_dir: str, iteration_num: int, feedback_history: List[str], status_callback: Optional[Callable[[str], None]] = None, uploaded_files_content: dict = None) -> Dict[str, Any]:
        logger.add_log_for_ui(f"Crew '{self.name}' (Tentativa {iteration_num}) iniciando processamento de {len(subtasks)} subtarefas.")
        iteration_artifacts_metadata: List[Dict[str, Any]] = []
        for i, subtask in enumerate(subtasks):
            subtask_desc = subtask.get("description", "N/A")
            responsible_role = subtask.get("responsible_role")
            if status_callback: status_callback(f"Etapa {i+1}/{len(subtasks)}: Agente '{responsible_role}' iniciando: {subtask_desc[:40]}...")
            
            agent = self.agents.get(responsible_role)
            if not agent:
                logger.add_log_for_ui(f"ERRO: Papel '{responsible_role}' não encontrado na crew. Pulando.", "error"); continue
            
            artifacts_metadata_list = agent.execute_task(main_task_description, subtask_desc, task_workspace_dir, iteration_num, list(iteration_artifacts_metadata), feedback_history, self.shared_context, uploaded_files_content)
            
            if artifacts_metadata_list:
                iteration_artifacts_metadata.extend(artifacts_metadata_list)
            else:
                error_msg = f"Agente '{agent.role}' não produziu nenhum artefato para a tarefa."
                logger.add_log_for_ui(f"ERRO: {error_msg}", "error")
                return {"status": "ERRO", "message": error_msg, "artifacts_metadata": iteration_artifacts_metadata}
        
        return {"status": "SUCESSO", "message": "Subtarefas da tentativa concluídas.", "artifacts_metadata": iteration_artifacts_metadata}