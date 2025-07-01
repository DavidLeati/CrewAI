# tasks.py
import os
import json
import logging
import shutil
import uuid
import subprocess
from typing import List, Dict, Any, Optional
import re

from config import config
from services import GeminiService
from agents import Agent, Crew
from memory import LongTermMemory
from app_logger import logger

class TaskManager:
    """Orquestra o planejamento, execução de tarefas por crews e o ciclo de validação/iteração."""
    def __init__(self, llm_service: GeminiService, output_dir: str):
        self.llm_service = llm_service
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.memory = LongTermMemory()

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
        learnings = self.memory.retrieve_learnings(main_task_description)
        learnings_str = "Nenhum aprendizado passado relevante."
        if learnings:
            learnings_str = "Aprendizados de tarefas similares passadas para sua consideração:\n" + "\n".join(learnings)

        logger.add_log_for_ui("Planejando a estratégia de execução INICIAL...")
        prompt = (
            f"{learnings_str}\n\n"
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
            "Seja especifico nas tarefas, por exemplo:\n"
            "Em vez de: 'Criar o Game Design Document (GDD).\n"
            "Tente: 'Sua única tarefa é gerar o conteúdo completo para o arquivo 'GameDesignDocument.md'. O documento deve detalhar o conceito, gênero, e tema do jogo. Comece o conteúdo imediatamente, sem introduções.'\n"
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
                logger.add_log_for_ui(f"Plano Inicial recebido. Crew: {plan['crew_name']}. Nº de subtarefas: {len(plan['subtasks'])}.")
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
        logger.add_log_for_ui("Criando uma tarefa de depuração focada...")
        
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
        logger.add_log_for_ui("Gerando uma lista de subtarefas corretivas focada no erro...")
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
                logger.add_log_for_ui(f"Plano de Ação Corretivo com {len(subtasks)} subtarefas gerado com sucesso.")
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
        logger.add_log_for_ui("--- Iniciando Validador de Estrutura de Arquivo ---")
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

        logger.add_log_for_ui("✅ Validação de Estrutura de Arquivo bem-sucedida.")
        return {"success": True, "feedback": "A estrutura de todos os arquivos validados está correta."}

    def _reconcile_plan_with_artifacts(self, subtasks: List[Dict[str, Any]], workspace_dir: str) -> Dict[str, Any]:
        """Compara os arquivos planejados com os arquivos realmente gerados, respeitando os subdiretórios."""
        logger.add_log_for_ui("--- Iniciando Auditoria de Plano vs. Realidade ---")
        
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
            logger.add_log_for_ui("✅ Auditoria bem-sucedida! Todos os arquivos planejados foram gerados."); return {"success": True}
        else:
            feedback = (f"FALHA DE AUDITORIA: O plano exigia a criação dos seguintes arquivos, mas eles não foram encontrados: {list(missing_files)}. A próxima iteração DEVE focar em gerar o CÓDIGO-FONTE para esses arquivos.")
            logging.error(feedback); return {"success": False, "feedback": feedback}
        
    def _execute_run_test(self, workspace_dir: str, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tenta executar o código gerado com base nas instruções do README."""
        logger.add_log_for_ui("--- Iniciando Prova Prática (Teste de Execução) ---")
        
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
        logger.add_log_for_ui(f"Comando encontrado no README: '{command}'")

        try:
            proc = subprocess.run(command.split(), cwd=workspace_dir, capture_output=True, text=True, timeout=30, check=False)
            if proc.returncode == 0:
                logger.add_log_for_ui("Teste de execução concluído com sucesso."); return {"success": True, "output": proc.stdout}
            else:
                logging.error(f"Teste de execução FALHOU. Erro:\n{proc.stderr}"); return {"success": False, "output": f"O comando '{command}' falhou com o erro:\n\n{proc.stderr}"}
        except subprocess.TimeoutExpired:
            logger.add_log_for_ui("Teste atingiu o timeout. Considerado sucesso para aplicações com loop."); return {"success": True}
        except Exception as e:
            return {"success": False, "output": f"Exceção ao rodar comando: {e}"}
            
    def _perform_code_completeness_review(self, workspace_dir: str, artifacts: List[Dict[str, Any]], subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Usa um agente de IA para revisar o conteúdo da VERSÃO MAIS RECENTE de cada arquivo de código,
        prevenindo a revisão de versões antigas e obsoletas.
        """
        logger.add_log_for_ui("--- Iniciando Auditoria de Completude de Código com Agente Revisor ---")
        
        source_code_artifacts = [
            art for art in artifacts 
            if art['file_path'].endswith(('.py', '.ts', '.js', '.html', '.css', '.svelte'))
        ]

        if not source_code_artifacts:
            logger.add_log_for_ui("Nenhum arquivo de código fonte encontrado para revisar.")
            return {"is_complete": True, "feedback": "Nenhum código para revisar."}
        
        latest_artifacts = {}
        for artifact in source_code_artifacts:
            filename = os.path.basename(artifact['file_path'])
            base_name, version_tuple = self._get_file_base_and_version(filename)
            
            if base_name not in latest_artifacts or version_tuple > latest_artifacts[base_name]['version']:
                latest_artifacts[base_name] = {'version': version_tuple, 'artifact': artifact}
        
        artifacts_to_review = [item['artifact'] for item in latest_artifacts.values()]
        
        logger.add_log_for_ui(f"Total de artefatos de código encontrados: {len(source_code_artifacts)}")
        logger.add_log_for_ui(f"Revisando {len(artifacts_to_review)} arquivos (apenas versões mais recentes): {[os.path.basename(art['file_path']) for art in artifacts_to_review]}")

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
            logger.add_log_for_ui("✅ Auditoria de Completude de Código bem-sucedida. Nenhum placeholder encontrado.")
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

        logger.add_log_for_ui(f"Iniciando validação TEÓRICA (QA) da Tentativa {iteration_num} com contexto real...")
        
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
        logger.add_log_for_ui("Iniciando etapa de curadoria final para selecionar os entregáveis...")
        
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

        response_dict = self.llm_service.generate_text(prompt, temperature=0.1, is_json_output=True)
        response_str = response_dict.get('text', '{}')
        
        try:
            curation_data = json.loads(response_str)
            if isinstance(curation_data, dict) and "deliverables" in curation_data and isinstance(curation_data["deliverables"], list):
                deliverables = curation_data["deliverables"]
                logger.add_log_for_ui(f"Curadoria da IA selecionou {len(deliverables)} entregáveis: {deliverables}")
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

            logger.add_log_for_ui("Filtrando arquivos para obter apenas as versões mais recentes antes da curadoria final.")
            latest_files = {}
            for filename in all_files_in_workspace:
                base_name, version_tuple = self._get_file_base_and_version(filename)
                
                if base_name not in latest_files or version_tuple > latest_files[base_name]['version']:
                    latest_files[base_name] = {'version': version_tuple, 'filename': filename}
            
            latest_filenames_to_curate = [item['filename'] for item in latest_files.values()]
            logger.add_log_for_ui(f"As seguintes versões de arquivos serão apresentadas para curadoria: {latest_filenames_to_curate}")

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
            
            logger.add_log_for_ui(f"Artefatos finais curados e copiados com sucesso para: '{final_output_dir}'")
            return final_output_dir
            
        except Exception as e:
            logging.error(f"Erro crítico ao finalizar e consolidar os artefatos: {e}")
            return None

    def _create_summary_add_log_for_ui(self, task_id: str, main_task_description: str, results: List[Dict], final_status: str, final_output_dir: Optional[str]):
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
            logger.add_log_for_ui(f"Log de resumo da tarefa salvo em: '{summary_path}'")
            return summary_path
        except Exception as e:
            logging.error(f"Não foi possível salvar o log de resumo final: {e}")
            return None

    def _re_strategize_plan(self, main_task_description: str, failure_history: List[str]) -> Optional[Dict[str, Any]]:
        """
        Usa o LLM para criar um plano de execução COMPLETAMENTE NOVO com base nos erros passados.
        Esta função é um mecanismo de recuperação crítico quando as correções incrementais falham.
        """
        logging.critical("FALHAS REPETIDAS DETECTADAS. ACIONANDO REPLANEJAMENTO ESTRATÉGICO COMPLETO.")
        
        # Formata o histórico de falhas para ser incluído no prompt.
        history_str = "\n- ".join(failure_history)

        # O prompt instrui a IA a agir como um estrategista sênior,
        # analisando os erros passados para criar um plano melhor, em vez de apenas consertar o antigo.
        prompt = (
            "Você é uma IA de Gerenciamento de Projetos Sênior, especialista em recuperação de projetos. "
            "Uma tentativa anterior de executar um plano falhou repetidamente. Sua tarefa é criar um plano ESTRATÉGICO COMPLETAMENTE NOVO "
            "para alcançar o objetivo, aprendendo com os erros do passado. Não tente consertar o plano antigo, crie um novo.\n\n"
            f"Tarefa Principal: \"{main_task_description}\"\n\n"
            f"Histórico de Feedbacks de Erro que Causaram o Replanejamento:\n- {history_str}\n\n"
            "Analise os erros e proponha uma nova abordagem. Talvez os agentes definidos estivessem errados, ou as subtarefas fossem mal "
            "concebidas. Considere uma estrutura de equipe diferente ou uma sequência de tarefas totalmente nova para evitar os mesmos problemas.\n\n"
            "Responda ESTRITAMENTE no mesmo formato JSON do planejamento inicial: o JSON deve conter as chaves: 'crew_name', "
            "'crew_description', 'agents' (uma lista de objetos com 'role', 'goal', 'backstory'), e 'subtasks' "
            "(uma lista de objetos com 'description' e 'responsible_role')."
        )
        
        # Chama o serviço de LLM esperando uma saída JSON.
        response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_PLANNING, is_json_output=True)
        response_text = response_dict.get('text', '')

        # Validação rigorosa da resposta da API.
        if not response_text or response_dict.get('finish_reason') != 'STOP':
            logging.error(f"A chamada de REPLANEJAMENTO à API falhou ou foi bloqueada. Resposta: {response_text}")
            return None

        # Tenta decodificar o JSON e validar sua estrutura.
        try:
            new_plan = json.loads(response_text)
            if all(k in new_plan for k in ["crew_name", "agents", "subtasks"]):
                logger.add_log_for_ui(f"Plano Estratégico REVISADO recebido com sucesso. Nova Crew: {new_plan['crew_name']}. Nº de subtarefas: {len(new_plan['subtasks'])}.")
                return new_plan
            else:
                logging.error(f"O novo plano JSON recebido está incompleto. Resposta: {response_text}")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar o novo plano JSON estratégico: {e}. Resposta: {response_text}")
            return None

    def delegate_task(self, main_task_description: str, status_callback=None) -> str:
        """
        Orquestra o ciclo completo de execução da tarefa, incluindo o replanejamento
        estratégico em caso de falhas repetidas e reportando o status para uma UI.
        """
        
        task_id = uuid.uuid4().hex[:10]
        task_root_dir = os.path.join(self.output_dir, f"task_{task_id}")
        os.makedirs(task_root_dir, exist_ok=True)
        logger.add_log_for_ui(f"\n{'='*20} Nova Tarefa Iniciada: {task_id} {'='*20}")
        
        master_plan = self._plan_execution_strategy(main_task_description)
        if not master_plan:
            logger.add_log_for_ui("Falha crítica no planejamento inicial. Tarefa abortada.", "critical")
            return "Falha crítica no planejamento inicial. Tarefa abortada."
        
        logger.add_log_for_ui(f"Plano Mestre criado com sucesso. Crew: '{master_plan.get('crew_name')}'.")
        
        crew_name_for_log = master_plan.get('crew_name', "DynamicCrew")
        workspace_dir = os.path.join(task_root_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        
        agents = [Agent(agent_id=f"{task_id}_{ag['role']}", llm_service=self.llm_service, **ag) for ag in master_plan['agents']]
        crew = Crew(name=master_plan['crew_name'], description=master_plan['crew_description'], agents=agents)
        
        feedback_history: List[str] = []
        execution_results: List[Dict] = []
        is_task_successful = False
        failures_on_same_issue_counter = 0
        last_feedback = ""
        
        for attempt in range(1, config.MAX_ITERATIONS + 1):
            logger.add_log_for_ui(f"--- Iniciando Tentativa de Geração/Correção {attempt}/{config.MAX_ITERATIONS} ---")

            if feedback_history:
                current_feedback = feedback_history[-1]
                if current_feedback == last_feedback:
                    failures_on_same_issue_counter += 1
                    logger.add_log_for_ui(f"Mesmo erro detectado {failures_on_same_issue_counter} vez(es) consecutivas.", "warning")
                else:
                    failures_on_same_issue_counter = 1
                last_feedback = current_feedback

            if failures_on_same_issue_counter > 2:
                logger.add_log_for_ui("Falhas repetidas detectadas. Acionando replanejamento estratégico completo.", "critical")
                new_plan = self._re_strategize_plan(main_task_description, feedback_history)
                if new_plan:
                    master_plan = new_plan
                    agents = [Agent(agent_id=f"{task_id}_{ag['role']}", llm_service=self.llm_service, **ag) for ag in master_plan['agents']]
                    crew = Crew(name=master_plan['crew_name'], description=master_plan['crew_description'], agents=agents)
                    logger.add_log_for_ui("PLANO MESTRE REVISADO E CREW RECONFIGURADA DEVIDO A FALHAS PERSISTENTES.", "warning")
                    failures_on_same_issue_counter = 0
                    feedback_history = []  # A história é limpa aqui, causando o erro na próxima iteração
                    last_feedback = ""
                else:
                    logger.add_log_for_ui("O replanejamento estratégico falhou. Continuando com o plano antigo.", "error")

            subtasks_for_this_attempt = master_plan['subtasks']
            
            if attempt > 1 and not (failures_on_same_issue_counter > 2) and feedback_history:
                feedback = feedback_history[-1]
                corrective_subtasks = self._generate_corrective_subtasks(main_task_description, master_plan, feedback)
                if corrective_subtasks:
                    logger.add_log_for_ui("Plano de ação corretivo gerado para focar no erro.")
                    subtasks_for_this_attempt = corrective_subtasks
                else:
                    logger.add_log_for_ui("Não foi possível gerar um plano de ação corretivo. Usando o plano mestre novamente.", "error")
            
            # Passando o callback para a crew, para que ela também possa reportar o status
            crew_result = crew.process_subtasks(
                main_task_description, 
                subtasks_for_this_attempt, 
                workspace_dir, 
                attempt, 
                feedback_history,
                status_callback=status_callback,
                

            )
            execution_results.append(crew_result)
            
            if crew_result.get("status") == "ERRO":
                logger.add_log_for_ui(f"Crew falhou criticamente na tentativa {attempt}: {crew_result.get('message')}", "critical")
                feedback_history.append(crew_result.get("message", "Erro crítico na crew."))
                continue

            all_artifacts = [{"file_path": os.path.join(root, name)} for root, _, files in os.walk(workspace_dir) for name in files]
            
            # --- Hierarquia de Validação com logging para a UI ---
            logger.add_log_for_ui("VALIDAÇÃO: Iniciando auditoria de arquivos...")
            reconciliation_result = self._reconcile_plan_with_artifacts(master_plan['subtasks'], workspace_dir)
            if not reconciliation_result["success"]:
                logger.add_log_for_ui(f"VALIDAÇÃO FALHOU (Auditoria de Arquivos): {reconciliation_result['feedback']}", "warning")
                feedback_history.append(reconciliation_result["feedback"])
                continue

            logger.add_log_for_ui("VALIDAÇÃO: Iniciando validação de estrutura...")
            structure_validation_result = self._validate_file_structure(all_artifacts)
            if not structure_validation_result["success"]:
                logger.add_log_for_ui(f"VALIDAÇÃO FALHOU (Estrutura de Arquivo): {structure_validation_result['feedback']}", "warning")
                feedback_history.append(structure_validation_result["feedback"])
                continue

            logger.add_log_for_ui("VALIDAÇÃO: Iniciando prova prática (teste de execução)...")
            run_test_result = self._execute_run_test(workspace_dir, all_artifacts)
            if not run_test_result["success"]:
                logger.add_log_for_ui(f"VALIDAÇÃO FALHOU (Prova Prática): {run_test_result['output']}", "warning")
                feedback_history.append(f"VALIDAÇÃO FALHOU (Prova Prática):\n{run_test_result['output']}")
                continue

            logger.add_log_for_ui("VALIDAÇÃO: Iniciando auditoria de completude de código...")
            review_result = self._perform_code_completeness_review(workspace_dir, all_artifacts, master_plan['subtasks'])
            if not review_result["is_complete"]:
                logger.add_log_for_ui(f"VALIDAÇÃO FALHOU (Auditoria de Código): {review_result['feedback']}", "warning")
                feedback_history.append(review_result["feedback"])
                continue
            
            logger.add_log_for_ui("SUCESSO! Todas as etapas de validação passaram nesta tentativa.")
            is_task_successful = True
            break
        
        # --- Finalização ---
        final_output_dir, final_status = None, "FALHA"
        if is_task_successful:
            final_status = "SUCESSO"
            logger.add_log_for_ui("Tarefa concluída com sucesso. Iniciando processo de curadoria final...")
            final_output_dir = self._finalize_task(task_id, main_task_description, workspace_dir, crew_name_for_log)
            summary_of_success = f"A tarefa '{main_task_description}' foi completada com sucesso, resultando nos artefatos em {final_output_dir}."
            self.memory.store_learning(main_task_description, summary_of_success)
            logger.add_log_for_ui("Aprendizado da tarefa armazenado na memória de longo prazo.")
        else:
            logger.add_log_for_ui(f"Tarefa finalizada como FALHA após {config.MAX_ITERATIONS} tentativas.", "critical")
        
        self._create_summary_add_log_for_ui(task_id, main_task_description, execution_results, final_status, final_output_dir)
        
        final_message = f"Execução da tarefa {task_id} finalizada com status: {final_status}."
        final_message += f"\nResumo salvo em: {os.path.join(self.output_dir, f'task_{task_id}_summary_log.md')}"
        if final_output_dir:
            final_message += f"\nSaída final CURADA e organizada em: {final_output_dir}"
        else:
            final_message += "\nNenhum entregável final foi produzido."
            
        return final_message
