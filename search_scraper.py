import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus, unquote

# Assumindo que 'config' e 'app_logger' estão disponíveis no ambiente de execução
from config import config
from app_logger import logger

class DuckDuckGoSearchService:
    """
    Serviço para realizar pesquisas na web usando DuckDuckGo Lite.
    Esta implementação foca na gratuidade total através de web scraping,
    com tratamento de erros e retries para maior robustez.
    """
    
    BASE_URL = "https://lite.duckduckgo.com/lite/"
    
    def __init__(self):
        logger.add_log_for_ui("Serviço de Pesquisa DuckDuckGo Lite inicializado.")
        # Configurações de retries e timeout, com valores padrão se não definidos em config
        self.max_retries = getattr(config, 'MAX_RETRIES_API', 3)
        self.retry_delay = getattr(config, 'RETRY_DELAY_SECONDS', 2)
        self.request_timeout = getattr(config, 'REQUEST_TIMEOUT_SECONDS', 10)
        self.max_search_results = getattr(config, 'MAX_SEARCH_RESULTS', 5)

    def perform_search(self, query: str) -> List[Dict[str, str]]:
        """
        Executa uma pesquisa no DuckDuckGo Lite e retorna uma lista de resultados.
        Cada resultado é um dicionário com as chaves 'title', 'url' e 'snippet'.
        
        Args:
            query (str): A string de consulta para a pesquisa.

        Returns:
            List[Dict[str, str]]: Uma lista de dicionários, onde cada dicionário
                                  representa um resultado de pesquisa.
                                  Retorna uma lista vazia em caso de falha.
        """
        encoded_query = quote_plus(query)
        search_url = f"{self.BASE_URL}?q={encoded_query}"
        
        headers = {
            # Usar um User-Agent comum para simular um navegador real
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        results = []
        for attempt in range(self.max_retries + 1):
            try:
                # Introduzir um pequeno atraso para simular comportamento humano e evitar bloqueios
                if attempt > 0: # Atrasar apenas em retries
                    delay = self.retry_delay * (attempt + 1)
                    logging.info(f"Aguardando {delay}s antes da tentativa {attempt + 1} de pesquisa.")
                    time.sleep(delay) 
                
                response = requests.get(search_url, headers=headers, timeout=self.request_timeout)
                response.raise_for_status() # Levanta um HTTPError para respostas 4xx/5xx
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # A estrutura do DuckDuckGo Lite é geralmente simples, com resultados em tabelas.
                # Tentamos encontrar a tabela principal de resultados.
                main_table = soup.find('table', class_='results')
                if not main_table:
                    # Fallback para uma tabela genérica se a classe 'results' não for encontrada
                    main_table = soup.find('table') 

                if main_table:
                    # Iterar sobre as linhas da tabela, onde cada linha pode ser um resultado
                    for row in main_table.find_all('tr'):
                        # O link do título geralmente está no primeiro <td> da linha
                        link_td = row.find('td')
                        if not link_td:
                            continue

                        link_tag = link_td.find('a')
                        if not link_tag or not link_tag.get('href'):
                            continue

                        title = link_tag.get_text(strip=True)
                        url = link_tag.get('href').strip()
                        
                        # DuckDuckGo Lite usa URLs de redirecionamento (/l/?kh=...).
                        # Precisamos extrair a URL real do parâmetro 'uddg'.
                        if url.startswith('/l/?kh='):
                            try:
                                # Exemplo: /l/?kh=123&uddg=https%3A%2F%2Fexample.com%2Fpage&...
                                redirect_params = url.split('uddg=')
                                if len(redirect_params) > 1:
                                    # Pega a parte após 'uddg=', e então a parte antes do próximo '&'
                                    actual_url_encoded = redirect_params[1].split('&')[0]
                                    url = unquote(actual_url_encoded) # Decodifica a URL
                                else:
                                    logging.warning(f"Parâmetro 'uddg' não encontrado na URL de redirecionamento DDG: {url}")
                                    continue # Pula este resultado se a URL não puder ser parseada
                            except Exception as e:
                                logging.warning(f"Erro ao parsear URL de redirecionamento DDG {url}: {e}")
                                continue # Pula este resultado em caso de erro

                        # O snippet geralmente está no <td> irmão seguinte ao <td> do link
                        snippet = ""
                        snippet_td = link_td.find_next_sibling('td')
                        if snippet_td:
                            # Tenta extrair o texto diretamente ou de um span comum de snippet
                            snippet_span = snippet_td.find('span', class_='result-snippet')
                            if snippet_span:
                                snippet = snippet_span.get_text(separator=" ", strip=True)
                            else:
                                # Fallback: pega todo o texto do <td>
                                snippet = snippet_td.get_text(separator=" ", strip=True)

                        if title and url:
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet
                            })
                            # Limitar o número de resultados para evitar processamento excessivo
                            if len(results) >= self.max_search_results:
                                break 

                if results:
                    logger.info(f"Pesquisa para '{query}' concluída. Encontrados {len(results)} resultados.")
                    return results
                else:
                    logging.warning(f"Nenhum resultado de pesquisa encontrado para '{query}'. HTML: {response.text[:500]}...")
                    return []

            except requests.exceptions.RequestException as e:
                error_message = str(e)
                delay = self.retry_delay * (attempt + 1)
                # Verifica se o erro é recuperável (ex: 429 Too Many Requests, 5xx Server Error, Timeout)
                is_retriable = any(status_code in error_message for status_code in ["429", "500"]) or "timeout" in error_message.lower()

                if is_retriable and attempt < self.max_retries:
                    logging.warning(f"Erro recuperável na pesquisa DuckDuckGo (Tentativa {attempt + 1}/{self.max_retries}): {error_message}. Aguardando {delay}s...")
                    time.sleep(delay)
                else:
                    logging.error(f"Erro final na pesquisa DuckDuckGo após {attempt + 1} tentativas: {error_message}")
                    return []
            except Exception as e:
                # Captura outros erros inesperados, como problemas de parsing
                logging.error(f"Erro inesperado ao parsear resultados da pesquisa para '{query}': {e}", exc_info=True)
                return []
        
        logging.warning(f"Número máximo de tentativas de pesquisa atingido para '{query}'.")
        return []