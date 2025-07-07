# tasks.py
import os
import json
import logging
import shutil
import uuid
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import re
import ast

from config import config
from services import GeminiService
from agents import Agent, Crew
from app_logger import logger
from shared_context import SharedContext

class TaskManager:
    """Orquestra o planejamento, execução de tarefas por crews e o ciclo de validação/iteração."""
    def __init__(self, llm_service: GeminiService, output_dir: str):
        self.llm_service = llm_service
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_project_name(self, name: str) -> str:
        """Limpa o nome do projeto para ser um nome de diretório válido."""
        if not name:
            return f"projeto_{uuid.uuid4().hex[:6]}"
        # Remove caracteres inválidos para nomes de pasta
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        # Substitui espaços e pontos por underscores
        name = re.sub(r'[\s\.]+', '_', name)
        return name.strip('_')
    
    def _get_file_base_and_version(self, filename: str) -> tuple[str, tuple[int, ...]]:
        base, _ = os.path.splitext(filename)
        parts = base.split('_')
        version_parts = []
        last_name_part_index = len(parts) - 1
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if part.isdigit():
                version_parts.insert(0, int(part))
                last_name_part_index = i - 1
            else:
                break
        if not version_parts:
            return base, (0,)
        base_name = "_".join(parts[:last_name_part_index + 1])
        return base_name, tuple(version_parts)

    def _plan_execution_strategy(self, main_task_description: str, existing_project_files: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Usa o LLM para criar o plano de execução. A lógica muda drasticamente
        se for um projeto novo ou a modificação de um existente.
        """
        if existing_project_files:
            return self._plan_modification_strategy(main_task_description, existing_project_files)
        else:
            return self._plan_creation_strategy(main_task_description)

    def _plan_creation_strategy(self, main_task_description: str) -> Optional[Dict[str, Any]]:
        """Usa o LLM para criar um plano de execução direto e focado em ação para um NOVO projeto."""
        logger.add_log_for_ui("Planejando a estratégia de execução para um NOVO projeto...")

        prompt = (
            "Você é uma IA de Gerenciamento de Projetos, focada em criar planos de execução eficientes e diretos. Analise a tarefa principal e projete uma equipe e um plano de ação para **construir um software funcional**. Responda ESTRITAMENTE no formato JSON.\n\n"
            f"Tarefa Principal: \"{main_task_description}\"\n\n"
            
            "<regras_de_planejamento_obrigatorias>\n"
            "1.  **Foco no Código**: O objetivo principal é gerar **código funcional**. Evite criar subtarefas excessivamente focadas em documentação de design. O plano deve levar à criação de arquivos de código executáveis.\n"
            "2.  **Equipe Enxuta ('agents')**: Defina o menor número de agentes necessários, com papéis claros e voltados para o desenvolvimento (ex: 'Desenvolvedor Full-Stack', 'Engenheiro de Software').\n"
            "3.  **Subtarefas Diretas e Acionáveis ('subtasks')**: Crie uma lista de subtarefas que representem passos de desenvolvimento concretos. Cada subtarefa deve, idealmente, resultar na criação ou modificação de um ou mais arquivos de código.\n"
            "4.  **Consolidação de Tarefas**: Agrupe tarefas relacionadas. Por exemplo, em vez de uma tarefa para HTML, uma para CSS e uma para JS, crie uma única tarefa como: 'Criar a estrutura inicial do projeto com os arquivos `index.html`, `css/style.css`, e `js/app.js`.'\n"
            "5.  **Arquitetura Inicial**: A primeira subtarefa DEVE ser a criação de um `ARCHITECTURE.md` para definir a estrutura de arquivos. O plano principal de implementação deve estar nas subtarefas subsequentes.\n"
            "6.  **Tarefa Final de Revisão**: A última subtarefa DEVE ser uma revisão completa do código gerado para criar um `README.md` final com instruções de uso e um resumo do projeto.\n"
            "7.  **Foco na Execução**: O plano deve ser direto e focado em ações que levam à criação de um software funcional. Evite discussões teóricas ou excessivamente detalhadas sobre design.\n" \
            "8.  **Organização:** TODA mudança necessária deve SEMPRE ser acompanhada com o nome do arquivo e a descrição do que deve ser feito. Exemplo: 'Criar o arquivo `index.html` com a estrutura básica de um documento HTML, incluindo as tags `<html>`, `<head>`, `<body>` e um título.'\n"
            "</regras_de_planejamento_obrigatorias>\n\n"
            
            "Exemplo de Formato de Saída para a criação de uma aplicação web simples:\n"
            "{\n"
            '  "crew_name": "WebAppBuilders",\n'
            '  "crew_description": "Equipe especializada na criação de aplicações web interativas.",\n'
            '  "agents": [\n'
            '    {"role": "Desenvolvedor Web Sênior", "goal": "Projetar e implementar a estrutura e a funcionalidade completa da aplicação web.", "backstory": "Especialista em HTML, CSS e JavaScript puro, focado em código limpo e funcional."},\n'
            '    {"role": "Engenheiro de QA", "goal": "Garantir a qualidade, funcionalidade e a documentação final do projeto.", "backstory": "Focado em testes e na experiência do usuário final."}\n'
            '  ],\n'
            '  "subtasks": [\n'
            '    {"description": "Criar a estrutura de diretórios e os arquivos base do projeto: `index.html` (com a estrutura semântica), `css/style.css` (com um reset básico e variáveis de cor), e `js/app.js` (com o ponto de entrada principal, como um `window.onload`).", "responsible_role": "Desenvolvedor Web Sênior"},\n'
            '    {"description": "Implementar a lógica principal da aplicação no arquivo `js/app.js`, incluindo as funções essenciais para a funcionalidade descrita na tarefa principal.", "responsible_role": "Desenvolvedor Web Sênior"},\n'
            '    {"description": "Estilizar a aplicação em `css/style.css` para garantir que seja visualmente agradável e funcional, seguindo um layout lógico.", "responsible_role": "Desenvolvedor Web Sênior"},\n'
            '    {"description": "Revisar todo o código gerado (`index.html`, `js/app.js`, `css/style.css`). Com base na aplicação final, criar um `README.md` detalhado com o resumo do projeto e as instruções de como executá-lo.", "responsible_role": "Engenheiro de QA"}\n'
            '  ]\n'
            "}"
        )
        
        response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_PLANNING, is_json_output=True)
        response_text = response_dict.get('text', '')

        if not response_text or response_dict.get('finish_reason') != 'STOP':
            logging.error(f"A chamada de planejamento (criação) à API falhou. Resposta: {response_text}")
            return None

        try:
            plan = json.loads(response_text)
            if all(k in plan for k in ["crew_name", "agents", "subtasks"]):
                logger.add_log_for_ui(f"Plano de CRIAÇÃO recebido. Crew: {plan['crew_name']}.")
                return plan
            else:
                logging.error(f"Plano JSON (criação) está incompleto. Resposta: {response_text}")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar o plano JSON (criação): {e}. Resposta: {response_text}")
            return None

    def _plan_modification_strategy(self, main_task_description: str, existing_project_files: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Usa o LLM para criar um plano de MODIFICAÇÃO para um projeto existente."""
        logger.add_log_for_ui("Planejando a estratégia de execução para MODIFICAR um projeto existente...")

        # Formata o conteúdo dos arquivos existentes para incluir no prompt
        files_context = "\n\n".join(
            [f"--- Arquivo: `{filename}` ---\n```\n{content}\n```"
             for filename, content in existing_project_files.items()]
        )

        prompt = (
            "Você é uma IA de Gerenciamento de Projetos Sênior, especialista em modificar código existente. Sua tarefa é analisar um projeto e uma nova tarefa, e criar um plano de ação para implementar as mudanças. Responda ESTRITAMENTE no formato JSON.\n\n"
            f"Nova Tarefa: \"{main_task_description}\"\n\n"
            "<contexto_do_projeto_existente>\n"
            "A seguir estão os arquivos e seus conteúdos atuais. Analise-os cuidadosamente para entender a estrutura e a lógica antes de planejar.\n"
            f"{files_context}\n"
            "</contexto_do_projeto_existente>\n\n"
            "<regras_de_planejamento_obrigatorias_para_MODIFICAÇÕES>\n"
            "1.  **Revisão da Arquitetura (Primeira Tarefa)**: A primeira subtarefa DEVE ser revisar todos os arquivos existentes e, com base na nova tarefa, atualizar ou criar o arquivo 'ARCHITECTURE.md' para refletir as mudanças planejadas (quais arquivos serão modificados, criados ou removidos).\n"
            "2.  **Análise de Impacto:** Determine quais arquivos existentes precisam ser modificados e quais novos arquivos precisam ser criados.\n"
            "3.  **Definição da Equipe ('agents'):** Defina uma equipe enxuta, focada nas habilidades necessárias para a modificação.\n"
            "4.  **Plano de Subtarefas ('subtasks'):** Crie uma lista de subtarefas que detalhe as MUDANÇAS. Seja específico.\n"
            "5.  **TAREFA FINAL:** A última subtarefa DEVE ser uma revisão das alterações e a atualização do `README.md` para refletir as novas funcionalidades.\n"
            "6.  **FOCO:** O plano deve focar apenas no necessário para completar a nova tarefa.\n"
            "</regras_de_planejamento_obrigatorias_para_MODIFICAÇÕES>\n\n"
            
            "Exemplo de Formato de Saída para uma tarefa de 'adicionar um sistema de mana e magias':\n"
            "{\n"
            '  "crew_name": "FeatureUpdateCrew",\n'
            '  "crew_description": "Equipe focada em adicionar novas funcionalidades a um jogo existente.",\n'
            '  "agents": [\n'
            '    {"role": "Arquiteto de Software", "goal": "Analisar e documentar as mudanças arquiteturais necessárias.", "backstory": "Especialista em design de sistemas e documentação técnica."},\n'
            '    {"role": "Desenvolvedor de Jogos Sênior (JS)", "goal": "Implementar novas mecânicas de jogo e refatorar código existente.", "backstory": "Especialista em lógica de jogos com JavaScript."}\n'
            '  ],\n'
            '  "subtasks": [\n'
            '    {"description": "Revisar o código existente em `js/player.js` e `js/game.js`. Atualizar o arquivo `ARQUITETURA.md` para incluir o novo sistema de mana, o novo arquivo `js/magic_spells.js` e as mudanças na UI.", "responsible_role": "Arquiteto de Software"},\n'
            '    {"description": "No arquivo `js/player.js`, adicione as novas propriedades `mana` e `maxMana` ao objeto do jogador e crie um método `useMana(cost)`.", "responsible_role": "Desenvolvedor de Jogos Sênior (JS)"},\n'
            '    {"description": "Crie um novo arquivo `js/magic_spells.js` que exporta um array de objetos de magia, cada um com `name`, `manaCost` e `effect`.", "responsible_role": "Desenvolvedor de Jogos Sênior (JS)"},\n'
            '    {"description": "No arquivo `js/game.js`, importe as magias de `magic_spells.js` e adicione a lógica para lançar magias.", "responsible_role": "Desenvolvedor de Jogos Sênior (JS)"},\n'
            '    {"description": "Atualize o `index.html` para incluir uma barra de mana para o jogador na UI.", "responsible_role": "Desenvolvedor de Jogos Sênior (JS)"},\n'
            '    {"description": "Revise todas as alterações e atualize o `README.md` para documentar o novo sistema de magia.", "responsible_role": "Desenvolvedor de Jogos Sênior (JS)"}\n'
            '  ]\n'
            "}"
        )

        response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_PLANNING, is_json_output=True)
        response_text = response_dict.get('text', '')

        if not response_text or response_dict.get('finish_reason') != 'STOP':
            logging.error(f"A chamada de planejamento (modificação) à API falhou. Resposta: {response_text}")
            return None

        try:
            plan = json.loads(response_text)
            if all(k in plan for k in ["crew_name", "agents", "subtasks"]):
                logger.add_log_for_ui(f"Plano de MODIFICAÇÃO recebido. Crew: {plan['crew_name']}.")
                return plan
            else:
                logging.error(f"Plano JSON (modificação) está incompleto. Resposta: {response_text}")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar o plano JSON (modificação): {e}. Resposta: {response_text}")
            return None

    def _validate_code_integration(self, workspace_dir: str) -> Dict[str, Any]:
        """
        Verifica se todos os arquivos de código são referenciados, descobrindo
        dinamicamente os pontos de entrada do projeto (ex: index.html).
        """
        logger.add_log_for_ui("--- Iniciando Validador de Integração de Código (Dinâmico) ---")
        
        CODE_EXTENSIONS = ('.js', '.css', '.py', '.ts')
        file_contents = {}
        all_code_files = set()
        entry_points = set()

        # 1. Mapeia todos os arquivos e seus conteúdos
        index_html_path = None
        for root, _, files in os.walk(workspace_dir):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), workspace_dir)
                if file.lower() == 'index.html':
                    index_html_path = relative_path
                if file.endswith(CODE_EXTENSIONS):
                    all_code_files.add(relative_path)
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        file_contents[relative_path] = f.read()
                except Exception:
                    pass
        
        # 2. Descobre os pontos de entrada dinamicamente
        if index_html_path and index_html_path in file_contents:
            entry_points.add(index_html_path)
            html_content = file_contents[index_html_path]
            # Encontra scripts e links de css
            script_srcs = re.findall(r'<script.*?src=["\'](.+?)["\']', html_content)
            link_hrefs = re.findall(r'<link.*?href=["\'](.+?)["\']', html_content)
            for path in script_srcs + link_hrefs:
                entry_points.add(os.path.normpath(path))
            logger.add_log_for_ui(f"Pontos de entrada descobertos a partir do 'index.html': {entry_points}")
        else:
            # Fallback para convenções padrão se não houver index.html
            CONVENTIONAL_ENTRIES = {'main.py', 'app.py', 'main.js', 'styles.css'}
            for file in all_code_files:
                if os.path.basename(file) in CONVENTIONAL_ENTRIES:
                    entry_points.add(file)
            logger.add_log_for_ui(f"Nenhum 'index.html' encontrado. Usando pontos de entrada convencionais: {entry_points}")
        
        # 3. Verifica por arquivos órfãos
        files_to_check = all_code_files - entry_points
        orphaned_files = set()

        for file_to_check in files_to_check:
            is_integrated = False
            check_string = os.path.splitext(os.path.basename(file_to_check))[0]
            
            for other_file_path, content in file_contents.items():
                if other_file_path == file_to_check: continue
                if check_string in content:
                    is_integrated = True
                    break
            
            if not is_integrated:
                orphaned_files.add(file_to_check)

        if not orphaned_files:
            logger.add_log_for_ui("Validação de Integração de Código bem-sucedida.")
            return {"success": True}
        else:
            feedback = (f"FALHA DE INTEGRAÇÃO: Os seguintes arquivos foram criados mas não parecem estar sendo importados ou referenciados: {list(orphaned_files)}. "
                        "A próxima iteração DEVE focar em garantir que esses arquivos sejam corretamente integrados no fluxo da aplicação.")
            logging.error(feedback)
            return {"success": False, "feedback": feedback}

    def _rewrite_task_with_prompt_engineering(self, main_task_description: str, existing_project_files: Optional[List[str]] = None) -> str:
        """Usa o LLM para reescrever a tarefa do usuário, com contexto se for uma modificação."""
        logger.add_log_for_ui("Reescrevendo a tarefa com engenharia de prompt...")

        context_header = "<identidade>Você é um Engenheiro de Prompts especialista. Sua missão é reescrever a tarefa de um usuário para ser o mais clara, detalhada e otimizada possível para uma equipe de agentes de IA.</identidade>\n\n"
        
        # Adiciona contexto específico se for uma modificação
        if existing_project_files:
            file_list_str = ", ".join([f"`{f}`" for f in existing_project_files])
            task_type_context = (
                "<contexto_da_tarefa>\n"
                f"  - **TIPO DE TAREFA:** MODIFICAÇÃO DE PROJETO EXISTENTE.\n"
                f"  - **ARQUIVOS NO PROJETO:** {file_list_str}\n"
                f"  - **TAREFA ORIGINAL DO USUÁRIO:** \"{main_task_description}\"\n"
                "  - Sua reescrita deve focar em como esta tarefa se aplica e altera os arquivos existentes.\n"
                "</contexto_da_tarefa>\n\n"
            )
        else:
            task_type_context = (
                "<contexto_da_tarefa>\n"
                f"  - **TIPO DE TAREFA:** CRIAÇÃO DE NOVO PROJETO.\n"
                f"  - **TAREFA ORIGINAL DO USUÁRIO:** \"{main_task_description}\"\n"
                "</contexto_da_tarefa>\n\n"
            )

        prompt = (
            context_header +
            task_type_context +
            "<tarefa_de_reescrita>\n"
            "  Reescreva a tarefa acima. Siga estas diretrizes:\n"
            "  1.  **Clarifique o Objetivo:** Expanda a descrição para não haver ambiguidade.\n"
            "  2.  **Defina o Escopo:** Adicione detalhes sobre funcionalidades esperadas (e o que está fora do escopo).\n"
            "  3.  **Adicione Critérios de Qualidade:** Inclua pontos sobre o que tornaria o resultado excelente (código limpo, documentação, etc.).\n"
            "  4.  **Mantenha a Intenção Original:** A nova tarefa deve ser uma versão aprimorada da original.\n"
            "  5.  **Limitações:** Deve ser claro o não uso de imagens e sons na tarefa, a IA não é capaz de gerar nada além de texto.\n"
            "</tarefa_de_reescrita>\n\n"
            "<regras_de_saida>\n"
            "  - Sua resposta deve ser APENAS o texto da nova tarefa reescrita.\n"
            "  - Não inclua saudações ou explicações.\n"
            "</regras_de_saida>"
        )

        response_dict = self.llm_service.generate_text(prompt, temperature=0.7)
        rewritten_task = response_dict.get('text', main_task_description).strip()

        if rewritten_task and rewritten_task != main_task_description:
            logger.add_log_for_ui("Tarefa reescrita com sucesso.")
            logging.info(f"--- Tarefa Original ---\n{main_task_description}\n\n--- Tarefa Aprimorada ---\n{rewritten_task}")
            return rewritten_task
        else:
            logger.add_log_for_ui("Não foi possível reescrever a tarefa, usando a original.", "warning")
            return main_task_description

    def _create_debugging_subtask(self, traceback: str, workspace_dir: str, original_plan: Dict) -> Optional[List[Dict[str, Any]]]:
        """
        Analisa um traceback e cria uma única subtarefa focada para corrigir o(s) arquivo(s) com erro.
        """
        logger.add_log_for_ui("Criando uma tarefa de depuração focada...")
        
        filepath_matches = re.findall(r'File "([^"]+)"', traceback)
        if not filepath_matches:
            logging.error("Não foi possível extrair nomes de arquivo do traceback. Abortando depuração.")
            return None
        
        critical_files_to_fix = {os.path.basename(path) for path in filepath_matches}
        
        developer_roles = [role for role in [agent['role'] for agent in original_plan['agents']] if 'dev' in role.lower() or 'python' in role.lower()]
        if not developer_roles:
            logging.error("Nenhum agente com perfil de desenvolvedor encontrado na equipe para a depuração.")
            return None
        responsible_agent = developer_roles[0]

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
        """Usa o LLM para gerar um plano de correção CONSOLIDADO e ESTRATÉGICO."""
        logger.add_log_for_ui("Gerando uma lista de subtarefas corretivas focada no erro...")
        original_subtasks_str = json.dumps(original_plan.get('subtasks', []), indent=2)
        agent_roles = [ag['role'] for ag in original_plan.get('agents', [])]

        prompt = (
            "<identidade>Você é um Gerente de Projetos Sênior, especialista em recuperação de falhas. Sua missão é criar um plano de ação enxuto e inteligente para consertar um erro, evitando retrabalho.</identidade>\n\n"
            "<contexto>\n"
            f"  - OBJETIVO GERAL DO PROJETO: {main_task_description}\n"
            f"  - AGENTES DISPONÍVEIS NA EQUIPE: {agent_roles}\n"
            f"  - PLANO ORIGINAL (para referência): \n{original_subtasks_str}\n"
            f"  - FEEDBACK DETALHADO DA FALHA ANTERIOR (contém os arquivos problemáticos): {feedback}\n"
            "</contexto>\n\n"
            "<tarefa>\n"
            "  Sua missão é criar uma **lista de subtarefas NOVA, CURTA e ESTRATÉGICA** que resolva a causa raiz do erro reportado.\n"
            "  **Passos Mentais a Seguir:**\n"
            "  1.  **Análise do Erro:** Analise o 'FEEDBACK DETALHADO' para identificar os arquivos específicos que falharam e o motivo da falha.\n"
            "  2.  **Foco nos Arquivos Problemáticos:** Seu plano de correção deve se concentrar **exclusivamente** nos arquivos mencionados no feedback.\n"
            "  3.  **Consolidação:** Agrupe correções relacionadas para o mesmo arquivo em uma única subtarefa abrangente. O objetivo é a eficiência.\n"
            "  4.  **Criação do Plano de Ação:** Elabore uma sequência de subtarefas que resolva os problemas apontados de forma definitiva.\n"
            "</tarefa>\n\n"
            "<regras_de_saida>\n"
            "  - Sua resposta deve ser **APENAS uma lista JSON** de objetos de subtarefa.\n"
            "  - Cada objeto deve ter 'description' e 'responsible_role'.\n"
            "  - Exemplo para um feedback sobre 'placeholders de integração em game.js':\n"
            "    [\n"
            "      {\n"
            '        "description": "Com base no feedback, refatore o arquivo `js/game.js` para integrar completamente todos os módulos necessários (map, inventory, ui, etc.), removendo todos os comentários de lógica incompleta e ativando as funcionalidades.",\n'
            '        "responsible_role": "Desenvolvedor de Jogo Web"\n'
            "      },\n"
            "      {\n"
            '        "description": "Após a refatoração do `game.js`, revise o arquivo `js/modules/events.js` e implemente a lógica real para os efeitos dos eventos, garantindo que eles modifiquem o estado do jogo corretamente.",\n'
            '        "responsible_role": "Desenvolvedor de Jogo Web"\n'
            "      }\n"
            "    ]\n"
            "  - NÃO inclua a palavra 'json' ou as cercas ```. Sua resposta deve começar com '[' e terminar com ']'.\n"
            "</regras_de_saida>"
        )
        
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
        """Verifica programaticamente a estrutura de arquivos conhecidos, como JSON."""
        logger.add_log_for_ui("--- Iniciando Validador de Estrutura de Arquivo ---")
        for artifact in artifacts:
            file_path = artifact.get('file_path', '')
            
            if file_path.lower().endswith('.json'):
                logging.debug(f"Validando estrutura do JSON: {os.path.basename(file_path)}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f) 
                except json.JSONDecodeError as e:
                    feedback = (f"FALHA DE ESTRUTURA: O arquivo '{os.path.basename(file_path)}' é um JSON inválido. "
                                f"Isso geralmente ocorre por texto extra antes ou depois do objeto JSON. Erro: {e}")
                    logging.error(feedback)
                    return {"success": False, "feedback": feedback}

        logger.add_log_for_ui("Validação de Estrutura de Arquivo bem-sucedida.")
        return {"success": True, "feedback": "A estrutura de todos os arquivos validados está correta."}

    def _reconcile_plan_with_artifacts(self, subtasks: List[Dict[str, Any]], workspace_dir: str) -> Dict[str, Any]:
        """Compara os arquivos planejados com os arquivos realmente gerados de forma flexível."""
        logger.add_log_for_ui("--- Iniciando Auditoria de Plano vs. Realidade (Flexível) ---")
        
        # 1. Extrai os caminhos dos arquivos do plano
        planned_files = set()
        filename_regex = r"['\"`]([\w\.\/\\]+\.(?:ts|js|py|svelte|json|md|txt))['\"`]"
        for task in subtasks:
            normalized_desc = task['description'].replace("\\", "/")
            found = re.findall(filename_regex, normalized_desc)
            if found:
                planned_files.update([os.path.normpath(f) for f in found])

        if not planned_files:
            logging.warning("Nenhum arquivo explícito no plano para auditar. Pulando.")
            return {"success": True}
        
        # 2. Obtém todos os arquivos gerados no workspace, incluindo subdiretórios
        all_generated_files = []
        for root, _, files in os.walk(workspace_dir):
            for name in files:
                full_path = os.path.join(root, name)
                all_generated_files.append(os.path.relpath(full_path, workspace_dir))
        
        generated_relative_paths = {os.path.normpath(p) for p in all_generated_files}
        
        # 3. Primeira verificação: correspondência exata de caminho
        missing_files_pass1 = planned_files - generated_relative_paths
        
        if not missing_files_pass1:
            logger.add_log_for_ui("Auditoria bem-sucedida! Todos os arquivos planejados foram gerados nos caminhos exatos.")
            return {"success": True}

        # 4. Segunda verificação (fallback): correspondência de nome de arquivo
        # Se a primeira verificação encontrou arquivos "faltando", verificamos se eles
        # foram criados em outro lugar.
        generated_basenames = {os.path.basename(p) for p in generated_relative_paths}
        truly_missing_files = set()
        
        for missing_file in missing_files_pass1:
            if os.path.basename(missing_file) not in generated_basenames:
                truly_missing_files.add(missing_file)
        
        if not truly_missing_files:
            logger.add_log_for_ui("Auditoria Sucedida (com flexibilidade): Todos os arquivos planejados foram gerados, embora alguns em caminhos diferentes do plano original.", "warning")
            return {"success": True}
        else:
            feedback = (f"FALHA DE AUDITORIA: O plano exigia a criação dos seguintes arquivos, mas eles não foram encontrados em nenhum lugar do projeto: {list(truly_missing_files)}. A próxima iteração DEVE focar em gerar o CÓDIGO-FONTE para esses arquivos.")
            logging.error(feedback)
            return {"success": False, "feedback": feedback}
        
    def _execute_run_test(self, workspace_dir: str, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tenta executar o código gerado com base nas instruções do README."""
        logger.add_log_for_ui("--- Iniciando Prova Prática (Teste de Execução) ---")
        
        readme_artifact = next((art for art in artifacts if 'readme.md' in os.path.basename(art['file_path']).lower()), None)
        
        if not readme_artifact:
            logging.warning("README.md não encontrado na workspace. Pulando teste de execução."); return {"success": True}

        try:
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
        
        command = match.group(1).strip()
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
        Usa um agente de IA para revisar o código, agora com o contexto do
        documento de arquitetura para uma análise mais inteligente.
        """
        logger.add_log_for_ui("--- Iniciando Auditoria de Completude de Código (com Contexto) ---")
        
        source_code_artifacts = [art for art in artifacts if art['file_path'].endswith(('.py', '.ts', '.js', '.html', '.css', '.svelte'))]
        if not source_code_artifacts:
            logger.add_log_for_ui("Nenhum arquivo de código fonte encontrado para revisar.")
            return {"is_complete": True, "feedback": "Nenhum código para revisar."}
        
        # Encontra o documento de arquitetura para usar como contexto
        architecture_doc_content = ""
        arch_artifact = next((art for art in artifacts if 'arquitetura.md' in os.path.basename(art['file_path']).lower()), None)
        if arch_artifact:
            try:
                with open(arch_artifact['file_path'], 'r', encoding='utf-8') as f:
                    architecture_doc_content = f.read()
            except Exception as e:
                logger.add_log_for_ui(f"Aviso: não foi possível ler o documento de arquitetura: {e}", "warning")

        incomplete_files_feedback = []
        for artifact in source_code_artifacts:
            file_path = artifact['file_path']
            filename = os.path.basename(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(100000) 
            except Exception as e:
                continue
            
            prompt = (
                "<identidade>Você é um Revisor de Código Sênior (Tech Lead) extremamente rigoroso. Sua missão é garantir que o código esteja funcionalmente completo e cumpra seu papel na arquitetura do projeto.</identidade>\n\n"
                "<contexto_do_projeto>\n"
                "  A seguir está o documento de arquitetura que descreve o propósito de cada arquivo no projeto. Use-o como sua fonte da verdade.\n"
                f"  <documento_de_arquitetura>\n{architecture_doc_content}\n</documento_de_arquitetura>\n"
                "</contexto_do_projeto>\n\n"
                "<tarefa_de_revisao>\n"
                f"  - ARQUIVO SOB REVISÃO: '{filename}'\n"
                f"  - CÓDIGO-FONTE PARA ANÁLISE:\n```\n{content}\n```\n"
                "  Sua tarefa é responder a seguinte pergunta: **Este arquivo cumpre seu propósito conforme descrito na arquitetura, ou ele contém lógica inacabada (placeholders, TODOs, funções vazias)?**\n"
                "  - **IMPORTANTE**: Não considere um arquivo 'incompleto' apenas por ter elementos que serão preenchidos dinamicamente por JavaScript (ex: um `<body>` vazio, um `div` container). Isso é esperado. Foque em lógica de programação inacabada.\n"
                "</tarefa_de_revisao>\n\n"
                "<regras_de_saida>\n"
                "  - Se o arquivo cumpre seu papel arquitetural e não tem lógica inacabada, responda com a única palavra: **COMPLETO**\n"
                "  - Se você encontrar QUALQUER sinal de incompletude LÓGICA, responda com: **INCOMPLETO**, seguido por uma única e concisa frase explicando o principal problema encontrado.\n"
                "  - Exemplo de Resposta de Falha: `INCOMPLETO: A função 'calculate_damage' retorna um valor fixo em vez de uma lógica de cálculo real.`\n"
                "</regras_de_saida>"
            )
            
            review_result = self.llm_service.generate_text(prompt, temperature=0.1)
            review_text = review_result.get('text', 'COMPLETO').strip().upper()

            if review_text.startswith("INCOMPLETO"):
                feedback = review_text.replace("INCOMPLETO", "", 1).strip().lstrip('.:').strip()
                logging.warning(f"REVISÃO FALHOU para '{filename}'. Motivo: {feedback}")
                incomplete_files_feedback.append(f"- O arquivo '{filename}' está incompleto. Motivo apontado pelo revisor: {feedback}")

        if incomplete_files_feedback:
            consolidated_feedback = "FALHA DE AUDITORIA DE CÓDIGO:\n" + "\n".join(incomplete_files_feedback)
            return {"is_complete": False, "feedback": consolidated_feedback}
        else:
            logger.add_log_for_ui("Auditoria de Completude de Código (com Contexto) bem-sucedida.")
            return {"is_complete": True}

    def _perform_backtest_and_validate(self,
                                         original_task_description: str,
                                         crew_result: Dict[str, Any],
                                         iteration_num: int) -> Dict[str, Any]:
        """Avalia os artefatos produzidos pela crew para validação."""
        artifacts = crew_result.get("artifacts_metadata", [])
        if not artifacts:
            return {"is_satisfactory": False, "feedback": "Nenhum artefato produzido para validação."}

        logger.add_log_for_ui(f"Iniciando validação TEÓRICA (QA) da Tentativa {iteration_num} com contexto real...")
        
        artifacts_content_summary = ""
        for artifact in artifacts:
            filename = os.path.basename(artifact.get('file_path', 'N/A'))
            desc = artifact.get('description', 'N/A')
            artifacts_content_summary += f"\n--- Artefato: '{filename}' | Descrição do Agente: {desc} ---\n"
            try:
                with open(artifact['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read(65000)
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
                existing_deliverables = [f for f in deliverables if f in all_files]
                return existing_deliverables
            else:
                raise ValueError("Formato JSON de curadoria inesperado.")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Erro na curadoria final por IA: {e}. Usando fallback para copiar arquivos .py, .md e .txt.")
            return [f for f in all_files if f.endswith(('.py', '.md', '.txt', '.json')) and 'fallback' not in f]

    def _finalize_task(self, project_name: str, main_task_description: str, workspace_dir: str) -> Optional[str]:
        """Cria um diretório de saída final com artefatos curados, preservando a estrutura de pastas."""
        final_output_dir = os.path.join(self.output_dir, f"{project_name}_FINAL_OUTPUT")
        try:
            if os.path.exists(final_output_dir):
                shutil.rmtree(final_output_dir)
            os.makedirs(final_output_dir)

            # 1. Lista todos os arquivos no workspace, incluindo subdiretórios
            all_files_in_workspace = []
            for root, _, files in os.walk(workspace_dir):
                for name in files:
                    # Adiciona o caminho relativo para a lista
                    relative_path = os.path.relpath(os.path.join(root, name), workspace_dir)
                    all_files_in_workspace.append(relative_path)
            
            if not all_files_in_workspace:
                logging.warning("Nenhum arquivo encontrado no workspace para finalizar.")
                return None
            
            # 2. Filtra para obter as versões mais recentes (lógica existente)
            logger.add_log_for_ui("Filtrando arquivos para obter apenas as versões mais recentes antes da curadoria final.")
            latest_files = {}
            for filename in all_files_in_workspace:
                base_name, version_tuple = self._get_file_base_and_version(os.path.basename(filename))
                if base_name not in latest_files or version_tuple > latest_files[base_name]['version']:
                    latest_files[base_name] = {'version': version_tuple, 'filename': filename}
            latest_filenames_to_curate = [item['filename'] for item in latest_files.values()]
            logger.add_log_for_ui(f"As seguintes versões de arquivos serão apresentadas para curadoria: {latest_filenames_to_curate}")

            # 3. Obtém a lista curada de entregáveis
            deliverables_to_copy = self._get_final_deliverables_list(main_task_description, latest_filenames_to_curate)
            if not deliverables_to_copy:
                logging.warning("A etapa de curadoria não retornou nenhum arquivo para copiar.")
                return final_output_dir

            # 4. Copia os arquivos curados, criando subdiretórios conforme necessário
            for filename in deliverables_to_copy:
                source_path = os.path.join(workspace_dir, filename)
                destination_path = os.path.join(final_output_dir, filename)
                
                # Garante que o diretório de destino exista
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                
                shutil.copy2(source_path, destination_path)
            
            # 5. Armazena o aprendizado (lógica existente)
            final_project_data = {}
            if deliverables_to_copy:
                for filename in deliverables_to_copy:
                    try:
                        with open(os.path.join(final_output_dir, filename), 'r', encoding='utf-8') as f:
                            final_project_data[filename] = f.read()
                    except Exception as e:
                        logger.add_log_for_ui(f"Não foi possível ler o arquivo final '{filename}' para armazenar na memória: {e}", "warning")

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
        """Usa o LLM para criar um plano de execução COMPLETAMENTE NOVO com base nos erros passados."""
        logging.critical("FALHAS REPETIDAS DETECTADAS. ACIONANDO REPLANEJAMENTO ESTRATÉGICO COMPLETO.")
        
        history_str = "\n- ".join(failure_history)

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
        
        response_dict = self.llm_service.generate_text(prompt, temperature=config.TEMPERATURE_PLANNING, is_json_output=True)
        response_text = response_dict.get('text', '')

        if not response_text or response_dict.get('finish_reason') != 'STOP':
            logging.error(f"A chamada de REPLANEJAMENTO à API falhou ou foi bloqueada. Resposta: {response_text}")
            return None

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

    def delegate_task(self, main_task_description: str, project_name: str, status_callback=None, uploaded_files_content: dict = None, existing_project_dir: Optional[str] = None):
        """
        Orquestra o ciclo completo de execução da tarefa, unificando o tratamento de
        arquivos de projetos existentes e arquivos anexados.
        """
        sanitized_project_name = self._sanitize_project_name(project_name) if not existing_project_dir else existing_project_dir.replace("_FINAL_OUTPUT", "")
        task_id, workspace_dir = self._initialize_task(sanitized_project_name, existing_project_dir)
        
        logger.add_log_for_ui(f"\n{'='*20} Iniciando Missão: {sanitized_project_name} {'='*20}")

        # --- LÓGICA DE CONTEXTO UNIFICADO ---
        shared_context = SharedContext()
        initial_context_files = {}

        # 1. Carrega arquivos do projeto existente, se houver.
        if existing_project_dir:
            logger.add_log_for_ui(f"MODO DE MODIFICAÇÃO: Carregando contexto de '{existing_project_dir}'")
            project_files = self._load_project_files(workspace_dir)
            initial_context_files.update(project_files)
        else:
            logger.add_log_for_ui("MODO DE CRIAÇÃO: Iniciando um novo projeto.")

        # 2. Carrega arquivos anexados, sobrescrevendo se houver conflito de nome.
        if uploaded_files_content:
            logger.add_log_for_ui(f"Carregando {len(uploaded_files_content)} arquivo(s) anexado(s) para o contexto.")
            initial_context_files.update(uploaded_files_content)

        # 3. Popula o SharedContext com o contexto unificado.
        if initial_context_files:
            shared_context.load_files_to_context(initial_context_files)

        # 4. Continua o fluxo, agora sem precisar passar 'uploaded_files_content' separadamente.
        enhanced_task_description = self._rewrite_task_with_prompt_engineering(
            main_task_description,
            list(initial_context_files.keys())
        )
        master_plan = self._plan_execution_strategy(enhanced_task_description, initial_context_files)
        if not master_plan:
            return "Falha crítica no planejamento inicial. Tarefa abortada."

        crew, agents = self._setup_crew(master_plan, task_id, shared_context)

        is_task_successful, execution_results, feedback_history = self._execution_loop(
            crew=crew,
            master_plan=master_plan,
            task_id=task_id,
            enhanced_task_description=enhanced_task_description,
            workspace_dir=workspace_dir,
            status_callback=status_callback,
        )
        final_message = self._finalize_and_summarize(
            task_id=task_id,
            project_name=sanitized_project_name,
            is_task_successful=is_task_successful,
            enhanced_task_description=enhanced_task_description,
            workspace_dir=workspace_dir,
            execution_results=execution_results
        )
        return final_message


    # --- MÉTODOS DE APOIO ---
    def _load_project_files(self, workspace_dir: str) -> Dict[str, str]:
        """Lê o conteúdo de todos os arquivos de texto em um diretório, ignorando caches."""
        project_files = {}
        for root, dirs, files in os.walk(workspace_dir):
            # Ignora o diretório __pycache__ na própria iteração
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            
            for file in files:
                # Ignora arquivos .pyc
                if file.endswith('.pyc'):
                    continue
                try:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, workspace_dir)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        project_files[relative_path] = f.read()
                except (UnicodeDecodeError, IOError) as e:
                    logger.add_log_for_ui(f"Aviso ao carregar: Não foi possível ler o arquivo '{file}': {e}", "warning")
        return project_files

    def _initialize_task(self, project_name: str, existing_project_dir: Optional[str] = None) -> Tuple[str, str]:
        """Cria a estrutura de diretórios para a tarefa, copiando um projeto existente corretamente."""
        task_id = uuid.uuid4().hex[:10]
        task_root_dir = os.path.join(self.output_dir, f"{project_name}_{task_id}")
        workspace_dir = os.path.join(task_root_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        if existing_project_dir:
            # Aponta para a pasta 'workspace' DENTRO do projeto selecionado
            source_workspace = os.path.join(self.output_dir, existing_project_dir, "workspace")
            
            if os.path.isdir(source_workspace):
                logger.add_log_for_ui(f"Copiando conteúdo de '{source_workspace}' para o novo workspace...")
                try:
                    # Itera sobre o conteúdo do DIRETÓRIO DE TRABALHO de origem
                    for item in os.listdir(source_workspace):
                        source_item_path = os.path.join(source_workspace, item)
                        destination_item_path = os.path.join(workspace_dir, item)
                        
                        if os.path.isdir(source_item_path):
                            shutil.copytree(source_item_path, destination_item_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(source_item_path, destination_item_path)
                except Exception as e:
                    logger.add_log_for_ui(f"Erro ao copiar projeto existente: {e}", "error")
        
        return task_id, workspace_dir
    
    def _finalize_and_summarize(self, task_id: str, project_name: str, is_task_successful: bool, enhanced_task_description: str, workspace_dir: str, execution_results: List[Dict]) -> str:
        """Finaliza a tarefa, cria a saída e gera o resumo."""
        final_output_dir = None
        final_status = "FALHA"

        if is_task_successful:
            final_status = "SUCESSO"
            logger.add_log_for_ui("Tarefa concluída. Iniciando processo de curadoria final...")
            final_output_dir = self._finalize_task(project_name, enhanced_task_description, workspace_dir)

        self._create_summary_add_log_for_ui(project_name, enhanced_task_description, execution_results, final_status, final_output_dir)

        final_message = f"Execução do projeto '{project_name}' (ID: {task_id}) finalizada com status: {final_status}."
        final_message += f"\nResumo salvo em: {os.path.join(self.output_dir, f'{project_name}_summary_log.md')}"
        if final_output_dir:
            final_message += f"\nSaída final CURADA e organizada em: {final_output_dir}"
        else:
            final_message += "\nNenhum entregável final foi produzido."
        
        return final_message

    def _setup_crew(self, master_plan: Dict, task_id: str, shared_context: SharedContext) -> Tuple[Crew, List[Agent]]:
        """Cria e configura a equipe (crew) e os agentes, passando o contexto compartilhado."""
        logger.add_log_for_ui(f"Plano Mestre criado. Crew: '{master_plan.get('crew_name')}'.")
        agents = [Agent(agent_id=f"{task_id}_{ag['role']}", llm_service=self.llm_service, **ag) for ag in master_plan['agents']]
        crew = Crew(name=master_plan['crew_name'], description=master_plan['crew_description'], agents=agents, shared_context=shared_context)
        return crew, agents

    def _execution_loop(self, crew: Crew, master_plan: Dict, task_id: str, enhanced_task_description: str, workspace_dir: str, status_callback):
        """Executa o ciclo principal de tentativas, correções e validações."""
        current_crew = crew
        current_master_plan = master_plan
        feedback_history: List[str] = []
        execution_results: List[Dict] = []
        failures_on_same_issue_counter = 0
        last_feedback = ""

        for attempt in range(1, config.MAX_ITERATIONS + 1):
            logger.add_log_for_ui(f"--- Iniciando Tentativa de Geração/Correção {attempt}/{config.MAX_ITERATIONS} ---")

            current_master_plan, current_crew, feedback_history, failures_on_same_issue_counter, last_feedback = self._handle_replan_if_needed(
                failures_on_same_issue_counter, last_feedback, feedback_history, enhanced_task_description, current_master_plan, task_id, current_crew
            )
            
            subtasks_for_this_attempt = self._get_subtasks_for_current_attempt(
                attempt, failures_on_same_issue_counter, feedback_history, enhanced_task_description, current_master_plan
            )
            
            crew_result = current_crew.process_subtasks(
                main_task_description=enhanced_task_description,
                subtasks=subtasks_for_this_attempt,
                task_workspace_dir=workspace_dir,
                iteration_num=attempt,
                feedback_history=feedback_history,
                status_callback=status_callback
            )
            execution_results.append(crew_result)
            
            if crew_result.get("status") == "ERRO":
                logger.add_log_for_ui(f"Crew falhou criticamente: {crew_result.get('message')}", "critical")
                feedback_history.append(crew_result.get("message", "Erro crítico na crew."))
                continue
                
            validation_passed, feedback = self._run_validation_pipeline(workspace_dir, current_master_plan['subtasks'])
            if not validation_passed:
                feedback_history.append(feedback)
                continue
            
            logger.add_log_for_ui("SUCESSO! Todas as etapas de validação passaram.")
            return True, execution_results, feedback_history
        
        logger.add_log_for_ui(f"Tarefa finalizada como FALHA após {config.MAX_ITERATIONS} tentativas.", "critical")
        return False, execution_results, feedback_history

    def _map_dependencies(self, workspace_dir: str) -> Dict[str, Dict]:
        """
        Mapeia todas as definições e dependências de símbolos em arquivos Python
        usando a árvore de sintaxe abstrata (AST).
        """
        dependencies = {}
        py_files = [os.path.join(r, f) for r, d, fs in os.walk(workspace_dir) for f in fs if f.endswith('.py')]

        for file_path in py_files:
            relative_path = os.path.relpath(file_path, workspace_dir)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            symbol_name = node.name
                            if symbol_name not in dependencies:
                                dependencies[symbol_name] = {'defined_in': relative_path, 'depends_on': set()}
                            for sub_node in ast.walk(node):
                                if isinstance(sub_node, ast.Name) and isinstance(sub_node.ctx, ast.Load):
                                    dependencies[symbol_name]['depends_on'].add(sub_node.id)
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    var_name = target.id
                                    if var_name not in dependencies:
                                        dependencies[var_name] = {'defined_in': relative_path, 'depends_on': set()}
                                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                                        dependencies[var_name]['depends_on'].add(node.value.func.id)
            except Exception as e:
                logger.add_log_for_ui(f"Erro ao mapear dependências em '{relative_path}': {e}", "warning")
        
        return dependencies

    def _detect_cycles(self, dependencies: Dict[str, Dict]) -> Optional[str]:
        """
        Detecta ciclos em um grafo de dependências usando busca em profundidade.
        Retorna uma string de feedback se um ciclo for encontrado, senão None.
        """
        for start_node, data in dependencies.items():
            path = [start_node]
            q = list(data['depends_on'])
            
            visited_in_path = {start_node}
            
            while q:
                current_node = q.pop(0)
                
                if current_node in visited_in_path:
                    path.append(current_node)
                    cycle_str = " -> ".join(path)
                    feedback = (f"FALHA DE LÓGICA: Detectada uma dependência circular/recursiva: {cycle_str}. "
                                f"Isso provavelmente causará um loop infinito. O problema origina-se em '{data['defined_in']}'. "
                                "A próxima iteração deve focar em quebrar este ciclo.")
                    logging.error(feedback)
                    return feedback
                
                visited_in_path.add(current_node)
                path.append(current_node)
                
                if current_node in dependencies:
                    q.extend(dependencies[current_node]['depends_on'])
                
                # Para evitar loops infinitos na própria análise, limitamos a profundidade da busca
                if len(path) > len(dependencies) * 2: break
                
                # Backtrack
                path.pop()
                visited_in_path.remove(current_node)

        return None

    def _validate_code_logic_patterns(self, workspace_dir: str) -> Dict[str, Any]:
        """
        Orquestra a análise de código: mapeia dependências e depois detecta ciclos.
        """
        logger.add_log_for_ui("--- Iniciando Análise de Lógica de Código (Programática) ---")
        
        dependencies = self._map_dependencies(workspace_dir)
        cycle_feedback = self._detect_cycles(dependencies)
        
        if cycle_feedback:
            return {"success": False, "feedback": cycle_feedback}
            
        logger.add_log_for_ui("Análise de Lógica de Código (Programática) bem-sucedida.")
        return {"success": True}
    
    def _run_validation_pipeline(self, workspace_dir: str, original_subtasks: List[Dict]) -> Tuple[bool, str]:
        """Executa a sequência de validações, agora com a análise de lógica programática."""
        all_artifacts = [{"file_path": os.path.join(root, name)} for root, _, files in os.walk(workspace_dir) for name in files]

        validations = [
            ("Auditoria de Arquivos", self._reconcile_plan_with_artifacts, (original_subtasks, workspace_dir)),
            ("Validação de Estrutura", self._validate_file_structure, (all_artifacts,)),
            ("Validação de Integração", self._validate_code_integration, (workspace_dir,)),
            ("Análise de Lógica de Código", self._validate_code_logic_patterns, (workspace_dir,)),
            ("Auditoria de Completude", self._perform_code_completeness_review, (workspace_dir, all_artifacts, original_subtasks)),
            ("Prova Prática (Execução)", self._execute_run_test, (workspace_dir, all_artifacts))
        ]

        for name, func, args in validations:
            logger.add_log_for_ui(f"VALIDAÇÃO: Iniciando {name}...")
            result = func(*args)
            is_successful = result.get("success", result.get("is_complete", False))
            if not is_successful:
                feedback = result.get("feedback", result.get("output", "Erro de validação não especificado."))
                logger.add_log_for_ui(f"VALIDAÇÃO FALHOU ({name}): {feedback}", "warning")
                return False, feedback

        return True, "Todas as validações foram bem-sucedidas."

    def _handle_replan_if_needed(self, failures_on_same_issue_counter: int, last_feedback: str, feedback_history: List[str], enhanced_task_description: str, master_plan: Dict, task_id: str, crew: Crew):
        if feedback_history:
            if feedback_history[-1] == last_feedback:
                failures_on_same_issue_counter += 1
                logger.add_log_for_ui(f"Mesmo erro detectado {failures_on_same_issue_counter} vez(es).", "warning")
            else:
                failures_on_same_issue_counter = 1
            last_feedback = feedback_history[-1]

        if failures_on_same_issue_counter > 2:
            logger.add_log_for_ui("Falhas repetidas detectadas. Acionando replanejamento estratégico.", "critical")
            new_plan = self._re_strategize_plan(enhanced_task_description, feedback_history)
            if new_plan:
                master_plan = new_plan
                # A nova crew é criada com o mesmo shared_context da crew antiga para não perder o estado dos arquivos
                crew, _ = self._setup_crew(master_plan, task_id, crew.shared_context)
                logger.add_log_for_ui("PLANO MESTRE REVISADO E CREW RECONFIGURADA.", "warning")
                failures_on_same_issue_counter = 0
                feedback_history.clear()
                last_feedback = ""
            else:
                logger.add_log_for_ui("Replanejamento estratégico falhou. Continuando com o plano antigo.", "error")
        
        return master_plan, crew, feedback_history, failures_on_same_issue_counter, last_feedback

    def _get_subtasks_for_current_attempt(self, attempt, failures_on_same_issue_counter, feedback_history, enhanced_task_description, master_plan):
        if attempt > 1 and not (failures_on_same_issue_counter > 2) and feedback_history:
            feedback = feedback_history[-1]
            corrective_subtasks = self._generate_corrective_subtasks(enhanced_task_description, master_plan, feedback)
            if corrective_subtasks:
                logger.add_log_for_ui("Plano de ação corretivo gerado para focar no erro.")
                return corrective_subtasks
            else:
                logger.add_log_for_ui("Não foi possível gerar plano corretivo. Usando o plano mestre.", "error")
        return master_plan['subtasks']