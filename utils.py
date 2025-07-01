# utils.py
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from app_logger import logger

ParsedOutput = Dict[str, Any]

def parse_llm_output(llm_response_text: str) -> List[Dict[str, Any]]:
    """
    Analisa a saída completa do LLM para extrair artefatos de código/documento
    e mensagens de comunicação para a equipe. Retorna dicionários com 'type', 'content' e 'metadata'.
    """
    artifacts = []
    last_end_index = 0
    delimiter_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"

    for match in re.finditer(delimiter_pattern, llm_response_text, re.DOTALL):
        try:
            metadata_str = match.group(1).strip()
            metadata = json.loads(metadata_str)
            filename_keys = ['suggested_filename', 'artifact', 'artifact_path', 'file_path', 'artifact_name']
            found_key = next((key for key in filename_keys if key in metadata), None)

            if isinstance(metadata, dict) and found_key:
                content = llm_response_text[last_end_index:match.start()].strip()
                if found_key != 'suggested_filename':
                    metadata['suggested_filename'] = metadata.pop(found_key)
                if content:
                    artifacts.append({
                        "type": "artifact",
                        "content": content,
                        "metadata": metadata
                    })
                last_end_index = match.end()
        except json.JSONDecodeError:
            continue

    # Qualquer conteúdo restante será tratado como um artefato genérico
    remaining_content = llm_response_text[last_end_index:].strip()
    if remaining_content:
        artifacts.append({
            "type": "artifact",
            "content": remaining_content,
            "metadata": {}
        })

    # Se não encontrou nada mas ainda há texto, adiciona como fallback
    if not artifacts and llm_response_text.strip():
        artifacts.append({
            "type": "artifact",
            "content": llm_response_text.strip(),
            "metadata": {}
        })

    logger.add_log_for_ui(f"Parser extraiu {len(artifacts)} artefato(s) da resposta do agente.")
    return artifacts

def clean_markdown_code_fences(code_str: str) -> str:
    """Remove de forma robusta cercas de código Markdown e blocos JSON."""
    if not isinstance(code_str, str): return ""
    cleaned_str = re.sub(r"^\s*```[a-zA-Z]*\n?|\n?\s*```\s*$", "", code_str, flags=re.MULTILINE)

    return cleaned_str.strip()

def sanitize_filename(filename: str, fallback_name: str = "fallback_artifact.txt") -> str:
    """Limpa e sanitiza um nome de arquivo para ser seguro."""
    if not filename or not isinstance(filename, str) or not filename.strip():
        return fallback_name
    sanitized = re.sub(r'[\\/*?:"<>|\n\r\t]', "_", filename)
    sanitized = sanitized.strip()
    if sanitized.startswith('.') or all(c == '.' for c in sanitized):
        sanitized = "file_" + sanitized
    return sanitized if sanitized else fallback_name