import tokenize
import io
import json
import re
import ast

class ReductionMap:
    """
    Stores information about removed or modified code segments for exact reconstruction.
    Each segment is a dictionary with:
    - 'type': 'comment', 'docstring', 'whitespace_normalization', 'string_compaction'
    - 'start_idx': Original starting character index in the code.
    - 'end_idx': Original ending character index in the code.
    - 'content': The original content of the segment.
    - 'new_content': (Optional) The content after transformation, if applicable.
    """
    def __init__(self):
        self.removed_segments = []

    def to_json(self) -> str:
        """Serializes the reduction map to a JSON string."""
        return json.dumps(self.removed_segments, indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        """Deserializes a JSON string back into a ReductionMap object."""
        instance = cls()
        instance.removed_segments = json.loads(json_str)
        return instance

def reduce_code(code: str) -> tuple[str, str]:
    """
    Reduces Python code to optimize token usage for LLMs by removing comments,
    docstrings, and normalizing excessive whitespace. It generates a reduction map
    containing information necessary for exact reconstruction of comments and docstrings.

    Args:
        code: The original Python code string.

    Returns:
        A tuple containing:
        - The reduced code string.
        - A JSON string representing the reduction map.
    """
    reduction_map = ReductionMap()
    original_lines = code.splitlines(keepends=True)
    
    # --- Step 1: Identify and remove comments using tokenize ---
    # Tokenize is used for robust identification of comments, avoiding issues with '#' inside strings.
    code_after_comments_parts = []
    
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError as e:
        # If tokenization fails (e.g., due to syntax errors), return original code
        # and an empty map, as we cannot safely proceed.
        print(f"Syntax error during tokenization: {e}. Returning original code.")
        return code, reduction_map.to_json()

    for token_type, token_string, (srow, scol), (erow, ecol), _ in tokens:
        # Calculate original start/end indices based on lines and columns.
        # This is robust as it uses the original line content and offsets.
        original_start_idx = sum(len(l) for l in original_lines[:srow-1]) + scol
        original_end_idx = sum(len(l) for l in original_lines[:erow-1]) + ecol

        if token_type == tokenize.COMMENT:
            reduction_map.removed_segments.append({
                'type': 'comment',
                'start_idx': original_start_idx,
                'end_idx': original_end_idx,
                'content': token_string
            })
            code_after_comments_parts.append("") # Remove the comment by replacing with empty string
        else:
            code_after_comments_parts.append(token_string)
    
    code_without_comments = "".join(code_after_comments_parts)

    # --- Step 2: Identify and remove docstrings using AST ---
    # AST is the most reliable way to identify actual docstrings (module, class, function).
    code_after_docstrings = code_without_comments
    try:
        tree = ast.parse(code_without_comments)
        
        # Traverse AST to find docstrings
        for node in ast.walk(tree):
            # Check for nodes that can have docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    # The docstring is typically the first statement in the body of these nodes.
                    # We need to find the exact string node to get its precise location.
                    if (isinstance(node, ast.Module) and node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)) or \
                       (not isinstance(node, ast.Module) and node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
                        
                        docstring_node = node.body[0]
                        
                        # Requires Python 3.8+ for end_lineno and end_col_offset
                        if hasattr(docstring_node, 'end_lineno') and hasattr(docstring_node, 'end_col_offset'):
                            docstring_start_line = docstring_node.lineno
                            docstring_start_col = docstring_node.col_offset
                            docstring_end_line = docstring_node.end_lineno
                            docstring_end_col = docstring_node.end_col_offset

                            # Calculate absolute character indices in `code_without_comments`
                            lines_after_comments = code_without_comments.splitlines(keepends=True)
                            doc_abs_start_idx = sum(len(l) for l in lines_after_comments[:docstring_start_line-1]) + docstring_start_col
                            doc_abs_end_idx = sum(len(l) for l in lines_after_comments[:docstring_end_line-1]) + docstring_end_col

                            # Extract the original docstring content from the code string
                            original_docstring_content = code_without_comments[doc_abs_start_idx:doc_abs_end_idx]

                            reduction_map.removed_segments.append({
                                'type': 'docstring',
                                'start_idx': doc_abs_start_idx,
                                'end_idx': doc_abs_end_idx,
                                'content': original_docstring_content
                            })
                            
                            # Replace the docstring in the current code string with spaces
                            # to preserve character offsets for subsequent operations.
                            code_after_docstrings = (
                                code_after_docstrings[:doc_abs_start_idx] +
                                ' ' * (doc_abs_end_idx - doc_abs_start_idx) +
                                code_after_docstrings[doc_abs_end_idx:]
                            )
                        else:
                            print("Warning: Python version older than 3.8. Docstring exact position tracking limited.")
                            # Fallback for older Python: record docstring content but not exact char indices.
                            # This would make exact reconstruction harder. For this task, we assume 3.8+.
    except SyntaxError as e:
        print(f"Syntax error during AST parsing for docstrings: {e}. Docstring removal skipped.")
        # If AST parsing fails, we cannot reliably remove docstrings.
        # `code_after_docstrings` remains `code_without_comments`.

    # --- Step 3: Normalize excessive whitespace ---
    # This step aims to reduce token count by standardizing whitespace.
    # It is inherently a lossy transformation for character-for-character exact whitespace
    # reconstruction, but preserves semantic equivalence.

    # Collapse multiple newlines to at most two (one blank line between code blocks).
    normalized_whitespace_code = re.sub(r'\n{3,}', '\n\n', code_after_docstrings)

    # Collapse multiple spaces/tabs within a line to a single space, preserving leading indentation.
    final_reduced_lines = []
    for line in normalized_whitespace_code.splitlines(keepends=True):
        leading_whitespace_match = re.match(r'^(\s*)', line)
        leading_whitespace = leading_whitespace_match.group(1) if leading_whitespace_match else ''
        content = line[len(leading_whitespace):].strip()
        if content:
            # Collapse internal whitespace (spaces/tabs) to a single space
            content = re.sub(r'[ \t]+', ' ', content)
            final_reduced_lines.append(leading_whitespace + content + "\n")
        elif line.strip() == '': # Keep single blank lines or indented blank lines
            final_reduced_lines.append(line)
        # Completely empty lines (no leading whitespace) are implicitly handled by previous steps
        # or by not appending if `content` is empty and no leading whitespace.

    final_reduced_code = "".join(final_reduced_lines).strip() # Final strip to remove trailing newlines

    # --- Step 4: Compact string literals ---
    # This step is omitted in this implementation due to the complexity of ensuring
    # "exact reconstruction" of the original string literal form (e.g., single vs. double quotes,
    # raw strings, f-strings, implicit string concatenation) while also performing compaction.
    # Robustly handling this would require more advanced AST manipulation or a custom parser
    # to preserve semantic equivalence and allow precise reversal, which is beyond the scope
    # of simple string/token-level transformations for this task.

    return final_reduced_code, reduction_map.to_json()