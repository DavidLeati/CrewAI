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
    Representa um agente de IA com um ciclo de execução em loop que lhe permite
    realizar ações intermediárias (como ler arquivos ou pesquisar na web) antes de
    produzir um artefato final.
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

        strong_code_keywords = [
            'import ', 'from ', 'def ', 'class ', 'function ', 'const ', 'let ', 'var ',
            'public class', 'public static', 'void main', '#include', 'require(', 'using '
        ]
        if lines and lines[0].startswith('#!'):
            code_score += 10

        for line in lines_to_analyze:
            if any(keyword in line for keyword in strong_code_keywords):
                code_score += 5
            if line.strip().startswith(('//', '#', '/*')):
                code_score += 2
            if line.strip().endswith((';', '{', '}', '):', '=> {')):
                code_score += 2

        text_sample = "\n".join(lines_to_analyze)
        total_chars = len(text_sample)
        if total_chars > 0:
            special_chars = sum(text_sample.count(c) for c in '(){}[]<>;:=!&|+-*/%')
            density = special_chars / total_chars
            if density > 0.1:
                code_score += 10
            elif density > 0.05:
                code_score += 5

        prose_indicators = 0
        for line in lines_to_analyze:
            line = line.strip()
            if len(line) > 80 and line.endswith('.') and ' ' in line:
                prose_indicators += 1
        if prose_indicators >= 2:
            code_score -= 10
        if any(line.strip().startswith(('# ', '## ')) for line in lines_to_analyze):
             code_score -= 5

        return 'code' if code_score >= 5 else 'document'

    def execute_task(self,
                     main_task_description: str,
                     task_description: str,
                     task_workspace_dir: str,
                     iteration_num: int,
                     context_artifacts: List[Dict[str, Any]],
                     feedback_history: List[str],
                     shared_context: SharedContext
                     ) -> List[Dict[str, Any]]:
        """
        Executa uma tarefa em um loop, usando o SharedContext para obter
        informações de arquivos (sejam de projetos existentes ou anexados).
        """
        max_attempts = 5
        read_files_context = {}
        web_search_results = ""

        for attempt in range(max_attempts):
            current_feedback = list(feedback_history)
            
            # O contexto agora é construído de forma unificada
            context_summary = self._build_prompt_context(
                context_artifacts, current_feedback, shared_context, read_files_context, web_search_results
            )
            prompt = self._build_agent_prompt(main_task_description, task_description, context_summary, shared_context)
            
            logger.add_log_for_ui(f"Agente '{self.role}' (Tentativa {attempt + 1}/{max_attempts}) ...")
            
            generation_result = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_EXECUTION)
            full_llm_response = generation_result.get('text', '').strip()

            action_executed = False
            try:
                # Procura por um bloco de ação JSON na resposta
                action_match = re.search(r"^```json\s*(\{[\s\S]*?\})\s*```$", full_llm_response)
                if action_match:
                    action_json = json.loads(action_match.group(1))
                    action_type = action_json.get("action")

                    if action_type == "read_file":
                        filename = action_json.get("filename")
                        if filename:
                            logger.add_log_for_ui(f"AÇÃO: Lendo o arquivo '{filename}'...")
                            content = shared_context.get_file_content(filename)
                            if content:
                                read_files_context[filename] = content
                                logger.add_log_for_ui(f"Conteúdo de '{filename}' carregado para a próxima iteração.")
                            else:
                                feedback_history.append(f"AVISO: Tentativa de ler '{filename}' falhou (não encontrado).")
                            action_executed = True
                    
                    elif action_type == "search":
                        query = action_json.get("query")
                        if query:
                            logger.add_log_for_ui(f"AÇÃO: Pesquisando na web por '{query}'...")
                            search_results_list = self.llm_service.perform_web_search(query)
                            if search_results_list:
                                web_search_results = "Resultados da pesquisa:\n" + "\n".join([f"- {res['title']}: {res['snippet']}" for res in search_results_list])
                            else:
                                web_search_results = "Sua pesquisa não retornou resultados."
                            action_executed = True

            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Resposta JSON não era uma ação válida. Tratando como artefato. Erro: {e}")

            # Se uma ação foi executada com sucesso, reinicia o loop para que o agente use o novo contexto.
            if action_executed:
                continue

            # Se não foi uma ação, então deve ser um artefato.
            parsed_artifacts = parse_llm_output(full_llm_response)
            if not parsed_artifacts or not any(p.get("content", "").strip() for p in parsed_artifacts):
                feedback_history.append("O agente não produziu um artefato ou ação válida. A resposta estava vazia ou mal formatada.")
                if attempt < max_attempts - 1:
                    logger.add_log_for_ui(f"Resposta inválida do agente. Tentando novamente...", "warning")
                    continue
                else: # Falha na última tentativa
                    break 

            # Se o agente produziu um artefato com sucesso, o salvamos e encerramos o loop.
            return self._save_artifacts(parsed_artifacts, task_workspace_dir, task_description, iteration_num, shared_context)

        logger.add_log_for_ui(f"Agente '{self.role}' não conseguiu produzir um artefato após {max_attempts} tentativas.", "error")
        return []

    def _build_agent_prompt(self, main_task: str, current_task: str, context: str, shared_context: SharedContext) -> str:
        available_files = shared_context.get_all_filenames()
        file_context_instruction = ""
        if available_files:
            file_list_str = "\n".join([f"- `{f}`" for f in available_files])
            file_context_instruction = (
                "<arquivos_disponiveis_no_projeto>\n"
                "Para ver o conteúdo de qualquer um dos arquivos abaixo, use a ação `read_file`.\n"
                f"{file_list_str}\n"
                "</arquivos_disponiveis_no_projeto>\n\n"
            )

        return (
            f"<identidade>\n<papel>{self.role}</papel>\n<objetivo_especifico>{self.goal}</objetivo_especifico>\n</identidade>\n\n"
            "<regras_de_acao_obrigatorias>\n"
            "  Sua resposta DEVE ser UMA das seguintes opções, em ordem de prioridade:\n"
            "  1.  **LER ARQUIVO:** Se você precisa ler ou modificar um arquivo existente, sua primeira ação DEVE ser usar `read_file` para carregar seu conteúdo. Gere um JSON com a ação.\n"
            "      ```json\n"
            '      {"action": "read_file", "filename": "workspace/js/player.js"}\n'
            "      ```\n"
            "  - **CASO** comece com workspace, não esqueça de coloca-lo... caso contrário, não coloque!\n"
            "  2.  **PESQUISAR NA WEB:** Se você precisa de informações externas (uma API, uma biblioteca, etc.), use a ação `search`.\n"
            "      ```json\n"
            '      {"action": "search", "query": "javascript detect collision between two divs"}\n'
            "      ```\n"
            "  3.  **CRIAR/MODIFICAR ARTEFATO:** APENAS se você já tem todas as informações necessárias (após ler os arquivos ou pesquisar), gere o conteúdo COMPLETO do novo arquivo ou da versão MODIFICADA do arquivo existente, seguido por seu bloco de metadados ```json.\n"
            "  - **NUNCA** inclua texto introdutório como 'Claro, aqui está o código'. Vá direto ao ponto.\n"
            "</regras_de_acao_obrigatorias>\n\n"
            f"{file_context_instruction}"
            f"<contexto_da_missao>\n{context}\n</contexto_da_missao>\n\n"
            f"<tarefa_especifica>\n{current_task}\n</tarefa_especifica>\n"
            f"Expecifique TODOS os arquivos dentro do current_task, **por exemplo:** ao invés de usar 'integre script.py ao jogo', use 'integre script.py ao game.py'.\n"
            "<exemplo_de_saida_de_artefato>\n"
            "```python\n"
            "# Conteúdo completo do arquivo\n"
            "print('Hello, World!')\n"
            "```\n"
            "```json\n"
            '{"suggested_filename": "src/hello.py", "description": "Um script de exemplo."}\n'
            "```\n"
            "</exemplo_de_saida_de_artefato>"
        )

    def _build_prompt_context(self, artifacts: List[Dict], feedback: List[str], shared_context: SharedContext, read_files: Dict, web_results: str) -> str:
        context_parts = []
        if artifacts:
            artifact_list = ", ".join([f"`{os.path.basename(a.get('file_path', 'N/A'))}`" for a in artifacts])
            context_parts.append(f"ARTEFATOS CRIADOS ANTERIORMENTE NESTA SESSÃO: {artifact_list}")
                
        if read_files:
            files_str = "\n\n".join([f"--- Conteúdo de `{fname}` (lido nesta tarefa) ---\n```\n{content}\n```" for fname, content in read_files.items()])
            context_parts.append(files_str)
        if web_results:
            context_parts.append(f"RESULTADOS DA PESQUISA WEB RECENTE:\n{web_results}")
        if feedback:
            context_parts.append("HISTÓRICO DE FEEDBACK (O mais recente é mais importante):\n- " + "\n- ".join(reversed(feedback)))
        comms = shared_context.get_full_context_for_prompt(self.role)
        if "Nenhuma mensagem" not in comms:
            context_parts.append("COMUNICAÇÃO DA EQUIPE:\n" + comms)
        return "\n\n".join(context_parts) if context_parts else "Nenhum contexto prévio disponível."
    
    def _save_artifacts(self, parsed_outputs: List[Dict], task_workspace_dir: str, task_description: str, iteration_num: int, shared_context: SharedContext) -> List[Dict[str, Any]]:
        saved_artifacts_metadata = []
        for output in parsed_outputs:
            output_type = output.get("type")

            if output_type == "message":
                recipient = output.get("recipient", "all")
                content = output.get("content", "")
                if shared_context:
                    shared_context.add_message(sender=self.role, content=content, recipient=recipient)
                continue

            elif output_type == "artifact":
                content = output.get("content", "")
                metadata = output.get("metadata", {})

                has_filename = 'suggested_filename' in metadata or 'filename' in metadata

                if not metadata or not has_filename:
                    logger.add_log_for_ui(f"Artefato sem metadados válidos. Iniciando autocorreção. Conteúdo: '{content[:100]}...'", "warning")
                    naming_prompt = (
                        "Analise o seguinte conteúdo e sugira um nome de arquivo apropriado (com extensão) e uma breve descrição. "
                        "Sua resposta DEVE ser um único objeto JSON com as chaves 'suggested_filename' e 'description'.\n\n"
                        f"CONTEÚDO PARA ANÁLISE:\n---\n{content[:1500]}...\n---\n"
                        "Responda apenas com o JSON."
                    )
                    metadata_response_dict = self.llm_service.generate_text(naming_prompt, temperature=0.1, is_json_output=True)
                    try:
                        metadata = json.loads(metadata_response_dict.get('text', '{}'))
                        if 'suggested_filename' not in metadata:
                            raise ValueError("Chave 'suggested_filename' ausente no JSON de autocorreção.")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.add_log_for_ui(f"Autocorreção de metadados falhou: {e}. Usando fallback.", "error")
                        metadata = {
                            "suggested_filename": f"{self.role.replace(' ', '_').lower()}_fallback_{uuid.uuid4().hex[:6]}.txt",
                            "description": f"Fallback para a tarefa: {task_description[:50]}..."
                        }

                suggested_filename = metadata.get("suggested_filename", "")
                description = metadata.get("description", "Descrição não fornecida.")
                content_to_save = clean_markdown_code_fences(content)

                match = re.search(r"['\"]([a-zA-Z0-9_\/\\]+\.[a-zA-Z0-9_]+)['\"]", task_description)
                target_path_from_task = os.path.normpath(match.group(1)) if match else None

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
                    with open(output_filepath, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    existing_type = self._infer_content_type(existing_content)
                    new_type = self._infer_content_type(content_to_save)

                    if existing_type == 'code' and new_type == 'document':
                        logging.error(f"VÁLVULA DE SEGURANÇA: Tentativa de sobrescrever o arquivo de código '{os.path.basename(output_filepath)}' com um documento.")
                        safe_fallback_filename = f"DANGEROUS_OVERWRITE_ATTEMPT_ON_{os.path.basename(output_filepath)}.md"
                        output_filepath = os.path.join(final_dir, safe_fallback_filename)
                        description += " [AVISO: Salvo com nome de fallback para prevenir sobreescrita de código]"
                        logging.warning(f"O conteúdo do documento foi salvo como '{safe_fallback_filename}'.")

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

class Crew:
    """Gerencia uma equipe de agentes para processar uma série de subtarefas delegadas."""
    def __init__(self, name: str, description: str, agents: List[Agent], shared_context: SharedContext):
        self.name = name
        self.description = description
        self.agents = {agent.role: agent for agent in agents}
        self.shared_context = shared_context
        logger.add_log_for_ui(f"Crew '{self.name}' criada com {len(agents)} agentes: {list(self.agents.keys())}")

    def process_subtasks(self, main_task_description: str, subtasks: List[Dict[str, Any]], task_workspace_dir: str, iteration_num: int, feedback_history: List[str], status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        logger.add_log_for_ui(f"Crew '{self.name}' (Tentativa {iteration_num}) processando {len(subtasks)} subtarefas.")
        iteration_artifacts_metadata: List[Dict[str, Any]] = []

        for i, subtask in enumerate(subtasks):
            original_subtask_desc = subtask.get("description", "N/A")
            responsible_role = subtask.get("responsible_role")
            if status_callback: status_callback(f"Etapa {i+1}/{len(subtasks)}: Agente '{responsible_role}'...")
            
            agent = self.agents.get(responsible_role)
            if not agent:
                logger.add_log_for_ui(f"ERRO: Papel '{responsible_role}' não encontrado. Pulando.", "error"); continue

            final_task_desc = self._inject_context_into_task(original_subtask_desc, self.shared_context)

            artifacts_metadata_list = agent.execute_task(
                main_task_description, final_task_desc, task_workspace_dir, iteration_num, 
                list(iteration_artifacts_metadata), feedback_history, self.shared_context
            )
            
            if artifacts_metadata_list:
                iteration_artifacts_metadata.extend(artifacts_metadata_list)
            
            self.shared_context.rescan_and_update_context(task_workspace_dir)
        
        return {"status": "SUCESSO", "message": "Subtarefas concluídas.", "artifacts_metadata": iteration_artifacts_metadata}

    def _inject_context_into_task(self, task_description: str, shared_context: SharedContext) -> str:
        """
        Injeta proativamente o conteúdo de arquivos relevantes na descrição da tarefa.
        Prioriza arquivos mencionados na tarefa e arquivos de contexto essenciais.
        """
        # Palavras-chave para identificar arquivos de contexto essenciais
        CORE_CONTEXT_KEYWORDS = ['arquitetura', 'architecture', 'design', 'concept', 'documento_conceito']
        
        # 1. Encontra arquivos mencionados explicitamente na tarefa
        explicit_filenames = set(re.findall(r"['\"`]([\w\.\/\\]+?)['\"`]", task_description))
        
        # 2. Encontra arquivos de contexto essenciais
        core_filenames = set()
        all_available_files = shared_context.get_all_filenames()
        for filename in all_available_files:
            if any(keyword in filename.lower() for keyword in CORE_CONTEXT_KEYWORDS):
                core_filenames.add(filename)
                
        # 3. Combina as listas, garantindo que não haja duplicatas
        files_to_inject = explicit_filenames.union(core_filenames)

        if not files_to_inject:
            return task_description

        injected_context = ""
        injected_files_count = 0
        for filename in files_to_inject:
            content = shared_context.get_file_content(filename)
            if content:
                injected_context += f"\n\n--- CONTEÚDO DE `{filename}` (fornecido para sua conveniência) ---\n```\n{content}\n```"
                injected_files_count += 1
        
        if injected_files_count > 0:
            logger.add_log_for_ui(f"Injetando conteúdo de {injected_files_count} arquivo(s) relevante(s) no prompt do agente.")
            directive_instruction = (
                "<instrucao_importante>\n"
                "O conteúdo dos arquivos cruciais para esta tarefa foi fornecido acima. "
                "USE ESTE CONTEÚDO DIRETAMENTE para completar sua tarefa. "
                "NÃO use a ação `read_file` para estes arquivos que já foram fornecidos. Comece a trabalhar na tarefa imediatamente.\n"
                "</instrucao_importante>\n\n"
            )
            return directive_instruction + injected_context + "\n\n" + task_description
        else:
            return task_description