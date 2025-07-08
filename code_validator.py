import ast
import os
from app_logger import logger
from typing import Dict, List, Set, Optional, Any, Tuple

class SymbolVisitor(ast.NodeVisitor):
    """
    Um NodeVisitor que rastreia símbolos definidos em escopos para encontrar variáveis órfãs.
    Ele visita os nós de um arquivo para garantir que uma variável seja definida antes de ser usada.
    """
    def __init__(self):
        # Uma pilha de escopos. Cada escopo é um set de nomes definidos.
        # Começa com os built-ins do Python no escopo global.
        self.scopes: List[Set[str]] = [set(dir(__builtins__))]
        # Lista de tuplas (nome_orfao, numero_linha)
        self.orphans: List[Tuple[str, int]] = []

    def _is_defined(self, name: str) -> bool:
        """Verifica se um nome foi definido em qualquer escopo, do mais interno ao mais externo."""
        for scope in reversed(self.scopes):
            if name in scope:
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visita uma definição de função."""
        # O nome da função é adicionado ao escopo externo.
        self.scopes[-1].add(node.name)
        
        # Cria um novo escopo para o corpo da função.
        function_scope = {arg.arg for arg in node.args.args}
        if node.args.vararg:
            function_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            function_scope.add(node.args.kwarg.arg)
        
        self.scopes.append(function_scope)
        self.generic_visit(node)
        self.scopes.pop() # Sai do escopo da função

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visita uma definição de classe."""
        # O nome da classe é adicionado ao escopo externo.
        self.scopes[-1].add(node.name)
        
        # Cria um novo escopo para o corpo da classe.
        self.scopes.append(set())
        self.generic_visit(node)
        self.scopes.pop() # Sai do escopo da classe

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visita uma atribuição. Visita o valor primeiro."""
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Adiciona a variável ao escopo atual.
                self.scopes[-1].add(target.id)
            else:
                self.visit(target)

    def visit_Name(self, node: ast.Name) -> None:
        """Visita um nome (variável, função, etc.)."""
        # Verifica se o nome está sendo lido/usado.
        if isinstance(node.ctx, ast.Load) and not self._is_defined(node.id):
            self.orphans.append((node.id, node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Visita uma declaração de import."""
        for alias in node.names:
            # `import a.b.c` define `a` no escopo local.
            self.scopes[-1].add(alias.asname or alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visita uma declaração `from ... import ...`."""
        for alias in node.names:
            self.scopes[-1].add(alias.asname or alias.name)
        self.generic_visit(node)


class CodeValidator:
    def _map_dependencies(self, workspace_dir: str) -> Dict[str, Dict]:
        """
        Mapeia todas as definições e dependências de símbolos em arquivos Python
        usando a árvore de sintaxe abstrata (AST).
        (O código original foi mantido, pois é necessário para a detecção de ciclos).
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

    def _dfs_cycle_check(self, node: str, dependencies: Dict, visiting: Set, visited: Set, path: List) -> Optional[List]:
        """Função auxiliar recursiva para a busca em profundidade (DFS)."""
        visiting.add(node)
        if node in dependencies:
            for neighbour in dependencies[node].get('depends_on', []):
                if neighbour in visiting:
                    return path + [neighbour]
                if neighbour not in visited:
                    result = self._dfs_cycle_check(neighbour, dependencies, visiting, visited, path + [neighbour])
                    if result:
                        return result
        visiting.remove(node)
        visited.add(node)
        return None

    def _detect_cycles(self, dependencies: Dict[str, Dict]) -> Optional[str]:
        """
        Detecta ciclos em um grafo de dependências usando busca em profundidade (DFS).
        Retorna uma string de feedback se um ciclo for encontrado, senão None.
        """
        visiting = set()  # Nós no caminho de recursão atual (cinza)
        visited = set()   # Nós completamente visitados (preto)

        for node in dependencies:
            if node not in visited:
                cycle_path = self._dfs_cycle_check(node, dependencies, visiting, visited, [node])
                if cycle_path:
                    cycle_str = " -> ".join(cycle_path)
                    origin_file = dependencies.get(cycle_path[0], {}).get('defined_in', 'arquivo desconhecido')
                    feedback = (f"FALHA DE LÓGICA: Detectada uma dependência circular/recursiva: {cycle_str}. "
                                f"Isso pode causar um loop infinito. O problema parece originar-se em '{origin_file}'. "
                                "A próxima iteração deve focar em quebrar este ciclo.")
                    logger.add_log_for_ui(feedback)
                    return feedback
        return None

    def _find_orphans(self, workspace_dir: str) -> List[str]:
        """
        Encontra variáveis órfãs (usadas antes de serem definidas) usando o SymbolVisitor.
        """
        orphan_reports = []
        py_files = [os.path.join(r, f) for r, d, fs in os.walk(workspace_dir) for f in fs if f.endswith('.py')]

        for file_path in py_files:
            relative_path = os.path.relpath(file_path, workspace_dir)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=file_path)
                    visitor = SymbolVisitor()
                    visitor.visit(tree)
                    # Remove duplicados e formata a saída
                    unique_orphans = sorted(list(set(visitor.orphans)), key=lambda x: x[1])
                    for name, line in unique_orphans:
                        report = f"FALHA DE LÓGICA: Variável órfã '{name}' usada na linha {line} do arquivo '{relative_path}' sem ser definida ou importada."
                        orphan_reports.append(report)
            except Exception as e:
                logger.add_log_for_ui(f"Erro ao analisar variáveis órfãs em '{relative_path}': {e}", "warning")
        
        return orphan_reports

    def _validate_code_logic_patterns(self, workspace_dir: str) -> Dict[str, Any]:
        """
        Orquestra a análise de código: mapeia dependências, detecta ciclos e encontra variáveis órfãs.
        """
        logger.add_log_for_ui("--- Iniciando Análise de Lógica de Código (Programática) ---")
        
        # 1. Mapear dependências para detecção de ciclo
        dependencies = self._map_dependencies(workspace_dir)
        
        # 2. Detectar ciclos
        cycle_feedback = self._detect_cycles(dependencies)
        if cycle_feedback:
            return {"success": False, "feedback": cycle_feedback}

        # 3. Encontrar variáveis órfãs
        orphan_feedback = self._find_orphans(workspace_dir)
        if orphan_feedback:
            # Junta todos os reports de variáveis órfãs em um único feedback
            full_feedback = "\n".join(orphan_feedback)
            return {"success": False, "feedback": full_feedback}
            
        logger.add_log_for_ui("Análise de Lógica de Código (Programática) bem-sucedida.")
        return {"success": True, "feedback": "Nenhum problema de lógica detectado."}