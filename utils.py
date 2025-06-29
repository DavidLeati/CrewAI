# utils.py
import re
import json
import logging
from typing import List, Dict, Any, Tuple

ParsedOutput = Dict[str, Any]

def parse_llm_output(llm_response_text: str) -> List[ParsedOutput]:
    """
    Analisa a saída completa do LLM para extrair artefatos de código/documento
    e mensagens de comunicação para a equipe.
    """
    # Regex para encontrar blocos de conteúdo (```), metadados (```json) e mensagens (```message)
    pattern = r"(```(?:[a-zA-Z]*\n[\s\S]*?\n)```|```json\n[\s\S]*?\n```|```message\n[\s\S]*?\n```)"
    parts = re.split(pattern, llm_response_text)
    
    # Filtra partes vazias ou que são apenas espaços
    parts = [p.strip() for p in parts if p and p.strip()]

    outputs: List[ParsedOutput] = []
    
    for i, part in enumerate(parts):
        if part.startswith("```json"):
            # Ignora, pois será associado ao bloco de código anterior
            continue
        elif part.startswith("```message"):
            content_match = re.search(r"```message\n([\s\S]*?)\n```", part)
            if content_match:
                try:
                    # O conteúdo da mensagem deve ser um JSON
                    message_content = json.loads(content_match.group(1).strip())
                    outputs.append({
                        "type": "message",
                        "content": message_content.get("content", ""),
                        "recipient": message_content.get("recipient", "all")
                    })
                    logging.info("Mensagem de comunicação parseada da saída do agente.")
                except json.JSONDecodeError:
                    logging.warning("Bloco de mensagem malformado (não é JSON válido) ignorado.")
            
        elif part.startswith("```"): # Bloco de código/documento
            content = re.sub(r"^```[a-zA-Z]*\n|```$", "", part)
            metadata = {}
            
            # Procura por um bloco JSON imediatamente a seguir
            if i + 1 < len(parts) and parts[i + 1].startswith("```json"):
                json_part = parts[i + 1]
                metadata_match = re.search(r"```json\n([\s\S]*?)\n```", json_part)
                if metadata_match:
                    try:
                        metadata = json.loads(metadata_match.group(1).strip())
                    except json.JSONDecodeError:
                        logging.warning("Bloco de metadados JSON malformado ignorado.")
            
            outputs.append({
                "type": "artifact",
                "content": content,
                "metadata": metadata
            })
            logging.info("Artefato de arquivo parseado da saída do agente.")
    
    return outputs

def clean_markdown_code_fences(code_str: str) -> str:
    """Remove de forma robusta cercas de código Markdown e blocos JSON."""
    if not isinstance(code_str, str): return ""
    cleaned_str = re.sub(r"```json\s*\{[\s\S]*?\}\s*```", "", code_str, flags=re.DOTALL)
    cleaned_str = re.sub(r"^\s*```[a-zA-Z]*\n?|\n?\s*```\s*$", "", cleaned_str, flags=re.MULTILINE)
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