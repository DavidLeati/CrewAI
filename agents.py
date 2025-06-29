# agents.py
import os
import re
import json
import uuid
import logging
from typing import List, Dict, Any

from fpdf import FPDF

from config import config
from services import GeminiService
from utils import parse_llm_content_and_metadata, clean_markdown_code_fences, sanitize_filename


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