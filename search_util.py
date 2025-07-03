import time
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from app_logger import logger

def get_random_user_agent():
    """Retorna um User-Agent de um navegador moderno para parecer um utilizador real."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/109.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15',
    ]
    return random.choice(user_agents)

def _get_page_content(url: str, driver: webdriver.Chrome = None) -> str:
    """
    Acede a uma URL com Selenium, extrai o HTML e retorna apenas o texto limpo.
    """
    # Se não for fornecido um driver, cria um temporariamente
    own_driver = False
    if driver is None:
        own_driver = True
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument(f'user-agent={get_random_user_agent()}')
        driver = webdriver.Chrome(options=options)

    try:
        driver.set_page_load_timeout(20)
        driver.get(url)
        time.sleep(random.uniform(2, 4))  # Espera o conteúdo carregar

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Remove scripts e styles
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()

        # Extrai e limpa texto
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        logger.add_log_for_ui(f"Aviso: Erro ao acessar {url} via Selenium: {e}")
        return ""

    finally:
        if own_driver and driver:
            driver.quit()

def search_startpage(query: str, num_results: int = 10) -> list[dict]:
    """
    Pesquisa no Startpage usando Selenium e extrai os resultados.
    Retorna uma lista de dicionários, cada um contendo 'title', 'url' e 'snippet'.
    """
    results = []
    driver = None
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'user-agent={get_random_user_agent()}')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation']) 
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--log-level=3")
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)

        driver.get("https://www.startpage.com/")
        
        time.sleep(random.uniform(2, 4)) 

        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#q"))
        )
        
        logger.add_log_for_ui(f"Digitando '{query}' no campo de busca...")
        search_box.send_keys(query)
        time.sleep(random.uniform(0, 1))
        search_box.send_keys(Keys.RETURN)

        WebDriverWait(driver, 15).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='result']"))
)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        divs = soup.select('div[class^="result"]')

        if not divs:
            logger.add_log_for_ui("Nenhum resultado encontrado — pode ter havido bloqueio, seletor errado ou problemas de carregamento.")
            return []

        logger.add_log_for_ui(f"Encontrados {len(divs)} resultados brutos. Extraindo informações...")
        for div in divs[:num_results]:
            title_tag = div.select('a[class^="result-title"]')
            snippet_tag = div.select('p[class^="description"]')
            
            if title_tag and snippet_tag:
                url = title_tag['href']
                content = _get_page_content(url, driver)
                
                results.append({
                    "title": title_tag.get_text(strip=True),
                    "url": url,
                    "snippet": snippet_tag.get_text(strip=True),
                    "content": content
                })
            else:
                logger.add_log_for_ui(f"Aviso: Algum elemento (título/snippet) não foi encontrado para um div de resultado.")
                
            time.sleep(random.uniform(0.5, 1.5))

    except TimeoutException:
        logger.add_log_for_ui("[search_util] Erro: Tempo limite excedido ao carregar a página ou encontrar elemento.")
    except WebDriverException as e:
        logger.add_log_for_ui(f"[search_util] Erro do WebDriver (verifique o driver ou configuração): {e}")
    except Exception as e:
        logger.add_log_for_ui(f"[search_util] Erro inesperado: {e}")
    finally:
        if driver:
            driver.quit()
    if results:
        return results
    else:
        logger.add_log_for_ui("Nenhum resultado encontrado ou retornado.")
        return []

if __name__ == "__main__":
    print("A executar teste de busca para 'O que é a biblioteca pandas em Python?'...")
    query = "O que é a biblioteca pandas em Python?"
    search_results = search_startpage(query, num_results=3) # Pedir 3 resultados para um teste rápido
    
    if search_results:
        print(f"\n--- SUCESSO! Encontrados {len(search_results)} resultados. ---\n")
        for i, result in enumerate(search_results, 1):
            print(f"--- Resultado {i} ---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Content: {result['content'][:400]}...\n")