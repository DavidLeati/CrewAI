# search_util.py
import logging
import sqlite3
import requests
from bs4 import BeautifulSoup
import time
import re
from collections import defaultdict, deque
import json
import threading
import os
import sys # For directing error output to stderr
import cloudscraper
from newspaper import Article, Config as NewspaperConfig

# --- Configuration ---
# Consolidated configuration for the entire system, including data acquisition,
# indexing, and search parameters.
CONFIG = {
    "DATABASE_NAME": "db/search_index.db",
    "SCRAPE_INTERVAL_SECONDS": 3600, # How often to re-scrape sources (1 hour)
    "MAX_CONCURRENT_SCRAPES": 5, # Limits parallel HTTP requests for politeness
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "MAX_SNIPPET_LENGTH": 300, # Max characters for the snippet returned to LLM
    "MAX_RESULTS_PER_QUERY": 5, # Max search results to return
    "SCRAPE_TIMEOUT_SECONDS": 10, # Timeout for individual HTTP requests
    "SCRAPE_DELAY_SECONDS": 0.5, # Delay between individual HTTP requests to be polite
    "PAGINATION_HINTS": {
        "text": ["next", "próxima", "seguinte", "next page", "próxima página", ">", ">>"],
        "css_class": ["next", "next-page", "pagination-next", "proxima", "page-next", "next-posts-link"],
        "rel_attribute": ["next"]
    },
    "TRUSTED_SOURCES": [
        # --- Fontes de Economia (Brasil) ---
        {"name": "B3", "url_pattern": r"^https://www\.b3\.com\.br/pt_br/noticias/", "seed_urls": ["https://www.b3.com.br/pt_br/noticias/"]},
        {"name": "ANBIMA", "url_pattern": r"^https://www\.anbima\.com\.br/pt_br/noticias/", "seed_urls": ["https://www.anbima.com.br/pt_br/noticias/noticias.htm"]},
        {"name": "G1 Economia", "url_pattern": r"^https://g1\.globo\.com/economia/", "seed_urls": ["https://g1.globo.com/economia/"]},
        {"name": "UOL Economia", "url_pattern": r"^https://economia\.uol\.com\.br/", "seed_urls": ["https://economia.uol.com.br/noticias/", "https://economia.uol.com.br/cotacoes/"]},
    ],
    "CACHE_DOCUMENTS_LIMIT": 10000 # Max number of documents to keep in memory cache
}

# --- Data Acquisition Module ---
class DataAcquisitionModule:
    """
    Módulo dedicado para aquisição de dados web de forma mais robusta e "humana".
    Utiliza uma sessão para gerenciar cookies e headers realistas para evitar bloqueios.
    """
    def __init__(self, config: dict):
        """
        Inicializa o módulo com a configuração e cria uma sessão persistente.
        """
        self.config = config
        self.session = cloudscraper.create_scraper()
        self.newspaper_config = NewspaperConfig()
        self.newspaper_config.browser_user_agent = self.config.get("USER_AGENT")
        self.newspaper_config.request_timeout = self.config.get("SCRAPE_TIMEOUT_SECONDS", 10)
        self.newspaper_config.memoize_articles = False

        # Headers que imitam um navegador Chrome em um sistema Windows.
        # Incluir Accept-Language e outros headers torna a requisição muito mais credível.
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1" # "Do Not Track" header
        })
        
        # Desabilita os avisos sobre a verificação SSL ser ignorada
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


    def _fetch_page(self, url: str) -> str | None:
        """
        Executa um HTTP GET usando a sessão persistente do módulo.
        """
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            print(f"Error: Invalid URL format: '{url}'", file=sys.stderr)
            return None
        
        try:
            response = self.session.get(url, timeout=self.config["SCRAPE_TIMEOUT_SECONDS"])
            
            response.raise_for_status()
            time.sleep(self.config["SCRAPE_DELAY_SECONDS"])
            return response.text
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error fetching {url}: {e.response.status_code} - {e.response.reason}", file=sys.stderr)
            return None
        except cloudscraper.exceptions.CloudflareChallengeError as e:
            print(f"Cloudflare challenge failed for {url}: {e}", file=sys.stderr)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error fetching {url}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching {url}: {e}", file=sys.stderr)
            return None

    def _parse_html(self, html_content: str, base_url: str) -> tuple[str, str, list[str], list[str]]:
        """
        Analisa o conteúdo HTML para extrair título, texto, links de artigos
        e links de paginação (próxima página).
        
        Retorna:
            tuple[str, str, list[str], list[str]]: Título, conteúdo, lista de links de artigos,
                                                   lista de links de paginação.
        """
        if not isinstance(html_content, str):
            print("Error: html_content must be a string for parsing.", file=sys.stderr)
            return "Parsing Error", "", [], []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.string if soup.title and soup.title.string else "No Title"
            
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            
            text_content = soup.get_text(separator=' ', strip=True)

            article_links = []
            pagination_links = []
            
            hints = self.config.get("PAGINATION_HINTS", {})
            pagination_text = set(hints.get("text", []))
            pagination_classes = set(hints.get("css_class", []))
            pagination_rels = set(hints.get("rel_attribute", []))

            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href or href.startswith('#'):
                    continue
                
                full_url = requests.compat.urljoin(base_url, href)
                
                # Heurística para identificar se é um link de paginação
                link_text = link.get_text(strip=True).lower()
                link_class = " ".join(link.get('class', [])).lower()
                link_rel = " ".join(link.get('rel', [])).lower()

                is_pagination = False
                if link_text in pagination_text:
                    is_pagination = True
                elif any(cls in link_class for cls in pagination_classes):
                    is_pagination = True
                elif any(rel in link_rel for rel in pagination_rels):
                    is_pagination = True
                
                # Valida se o link pertence a uma fonte confiável
                is_trusted = False
                for source in self.config["TRUSTED_SOURCES"]:
                    if re.match(source["url_pattern"], full_url):
                        is_trusted = True
                        break

                if is_trusted:
                    if is_pagination:
                        pagination_links.append(full_url)
                    else:
                        article_links.append(full_url)

            # Retorna ambas as listas de links
            return title, text_content, article_links, list(set(pagination_links)) # Remove duplicatas
            
        except Exception as e:
            print(f"Error parsing HTML for {base_url}: {e}", file=sys.stderr)
            return "Parsing Error", "", [], []


    def acquire_data(self, url: str) -> dict | None:
        """
        Usa newspaper3k para baixar e analisar a URL.
        Retorna um dicionário com o conteúdo limpo.
        """
        if not isinstance(url, str):
            print(f"Error: acquire_data received non-string URL: {type(url)}", file=sys.stderr)
            return None

        try:
            # Cria um objeto Article com a URL e nossa configuração
            article = Article(url, config=self.newspaper_config)
            
            # Baixa o conteúdo da página
            article.download()
            
            # Analisa o conteúdo para extrair título, texto, etc.
            article.parse()
            
            # --- O "Pulo do Gato" está aqui ---
            # article.text contém APENAS o texto do artigo principal, sem menus, anúncios, etc.
            clean_text = article.text
            
            if not clean_text:
                print(f"Warning: newspaper3k could not extract main content from {url}. Skipping.", file=sys.stderr)
                return None
            
            # Usamos o BeautifulSoup apenas para encontrar os links na página bruta
            # para continuar a navegação do nosso scraper.
            soup = BeautifulSoup(article.html, 'html.parser')
            article_links = []
            pagination_links = []
            
            hints = self.config.get("PAGINATION_HINTS", {})
            pagination_text = set(hints.get("text", []))
            pagination_classes = set(hints.get("css_class", []))
            pagination_rels = set(hints.get("rel_attribute", []))

            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href or href.startswith('#'):
                    continue
                
                full_url = requests.compat.urljoin(url, href)
                
                link_text = link.get_text(strip=True).lower()
                link_class = " ".join(link.get('class', [])).lower()
                link_rel = " ".join(link.get('rel', [])).lower()

                is_pagination = False
                if link_text in pagination_text: is_pagination = True
                elif any(cls in link_class for cls in pagination_classes): is_pagination = True
                elif any(rel in link_rel for rel in pagination_rels): is_pagination = True
                
                is_trusted = any(re.match(source["url_pattern"], full_url) for source in self.config["TRUSTED_SOURCES"])

                if is_trusted:
                    if is_pagination: pagination_links.append(full_url)
                    else: article_links.append(full_url)

            return {
                "url": url,
                "title": article.title,
                "clean_text": clean_text,
                "article_links": article_links,
                "pagination_links": list(set(pagination_links))
            }
        except Exception as e:
            logging.warning(f"Newspaper3k failed for {url}: {e}.")
            return None

# --- Core Search System Class ---
class LLMSearchSystem:
    """
    Implements a self-contained, real-time search system. It manages data acquisition,
    indexing (using SQLite for persistence and in-memory caches for speed),
    and information retrieval. Designed to operate within a single file and
    without reliance on external search APIs.
    """
    def __init__(self):
        self.db_path = CONFIG["DATABASE_NAME"]
        self.conn = None
        self._connect_db()
        self._create_tables()
        
        # In-memory caches for performance
        # inverted_index_cache: term -> {doc_id: [positions]} - populated on demand during search
        self.inverted_index_cache = {} 
        # document_cache: doc_id -> {url, title, content} - LRU cache for recent documents
        self.document_cache = {}       
        
        # Sets for efficient O(1) lookup of URLs already processed or in queue
        self.indexed_urls_cache = set() # URLs of documents currently in DB
        self.queued_urls_cache = set()  # URLs currently in scrape_queue

        self._load_caches_from_db() # Load initial document cache and indexed_urls_cache

        self.scrape_queue = deque()
        self.scraping_active = threading.Event() # Flag to control scraper thread
        self.scraper_threads = []
        self.lock = threading.Lock() # For thread-safe operations on shared resources

        # Instantiate the DataAcquisitionModule
        self.data_acquisition_module = DataAcquisitionModule(CONFIG)

    def _connect_db(self):
        """Establishes connection to SQLite database."""
        try:
            db_dir = os.path.dirname(self.db_path)
            
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            # Using a timeout for busy database to improve robustness
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30) 
            self.conn.row_factory = sqlite3.Row # Access columns by name
            # Optimize SQLite for speed (journal mode, synchronous, cache size)
            self.conn.execute("PRAGMA journal_mode = WAL;") # Write-Ahead Logging for better concurrency
            self.conn.execute("PRAGMA synchronous = NORMAL;") # Reduce syncs for speed, still safe
            self.conn.execute(f"PRAGMA cache_size = -{1024 * 64};") # 64MB cache
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}", file=sys.stderr)
            # It's critical to raise here as the system cannot function without a DB connection
            raise

    def _create_tables(self):
        """Creates necessary database tables if they don't exist."""
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    content TEXT,
                    last_scraped INTEGER
                )
            """)
            # Add index to URL for faster lookup
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_url ON documents(url);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inverted_index (
                    term TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    positions TEXT, -- JSON serialized list of integers
                    PRIMARY KEY (term, doc_id),
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            # Add index to term for faster search
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inverted_index_term ON inverted_index(term);")
            self.conn.commit()
            print("Database tables checked/created.")
        except sqlite3.Error as e:
            print(f"Error creating database tables: {e}", file=sys.stderr)
            if self.conn:
                self.conn.rollback() # Rollback any partial changes
            raise # Re-raise to indicate a critical setup failure
        except Exception as e:
            print(f"An unexpected error occurred during table creation: {e}", file=sys.stderr)
            raise
        finally:
            if cursor:
                cursor.close() # Ensure cursor is closed

    def _load_caches_from_db(self):
        """Loads a subset of data into in-memory caches for faster search."""
        cursor = None
        try:
            cursor = self.conn.cursor()

            # Load recent documents into cache and populate indexed_urls_cache
            cursor.execute(f"SELECT id, url, title, content FROM documents ORDER BY last_scraped DESC LIMIT {CONFIG['CACHE_DOCUMENTS_LIMIT']}")
            for row in cursor.fetchall():
                self.document_cache[row['id']] = {"url": row['url'], "title": row['title'], "content": row['content']}
                self.indexed_urls_cache.add(row['url'])
            
            # Inverted index cache is NOT fully loaded at startup for scalability.
            # It will be populated on demand during searches.
            print(f"Loaded {len(self.document_cache)} documents into cache and {len(self.indexed_urls_cache)} URLs into indexed_urls_cache.")
        except sqlite3.Error as e:
            print(f"Error loading caches from database: {e}", file=sys.stderr)
            # Caches might be incomplete, but the system can still operate by falling back to DB
        except Exception as e:
            print(f"An unexpected error occurred during cache loading: {e}", file=sys.stderr)
        finally:
            if cursor:
                cursor.close()

    def _tokenize(self, text):
        """Simple tokenization: lowercase, split by non-alphanumeric, filter short words."""
        if not isinstance(text, str):
            print(f"Warning: _tokenize received non-string input: {type(text)}. Returning empty list.", file=sys.stderr)
            return []
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 1] # Filter out single-character tokens

    def _index_document(self, url, title, content):
        """Indexes a document into the SQLite database and updates in-memory caches."""
        if not all(isinstance(arg, str) for arg in [url, title, content]):
            print(f"Error: _index_document received non-string arguments. URL type: {type(url)}, Title type: {type(title)}, Content type: {type(content)}", file=sys.stderr)
            return None # Indicate failure
            
        cursor = None
        doc_id = None
        try:
            with self.lock: # Ensure thread-safe DB write operations
                cursor = self.conn.cursor()
                timestamp = int(time.time())

                if not self.conn:
                    print("Warning: Database connection is closed. Cannot index document.", file=sys.stderr)
                    return None

                cursor = self.conn.cursor()
                timestamp = int(time.time())

                # Check if document already exists
                cursor.execute("SELECT id FROM documents WHERE url = ?", (url,))
                doc_id_row = cursor.fetchone()

                if doc_id_row:
                    doc_id = doc_id_row['id']
                    # Update existing document
                    cursor.execute("UPDATE documents SET title=?, content=?, last_scraped=? WHERE id=?",
                                   (title, content, timestamp, doc_id))
                    
                    # Delete old index entries for this doc_id before re-indexing
                    cursor.execute("DELETE FROM inverted_index WHERE doc_id=?", (doc_id,))
                    
                    # Remove from document cache if it was there (to ensure fresh data)
                    if doc_id in self.document_cache:
                        del self.document_cache[doc_id]
                    # Remove URL from indexed_urls_cache if it was there (will be re-added)
                    if url in self.indexed_urls_cache:
                        self.indexed_urls_cache.remove(url)

                    # Invalidate specific term entries in inverted_index_cache for this doc_id
                    # This is an expensive operation for large caches, but ensures consistency.
                    # For a truly scalable system, this might be handled by an LRU cache or
                    # by accepting eventual consistency for cache entries.
                    terms_to_clean = []
                    for term, doc_ids_map in self.inverted_index_cache.items():
                        if doc_id in doc_ids_map:
                            del doc_ids_map[doc_id]
                            if not doc_ids_map: # If no more docs for this term, mark term for deletion
                                terms_to_clean.append(term)
                    for term in terms_to_clean:
                        del self.inverted_index_cache[term]

                else:
                    # Insert new document
                    cursor.execute("INSERT INTO documents (url, title, content, last_scraped) VALUES (?, ?, ?, ?)",
                                   (url, title, content, timestamp))
                    doc_id = cursor.lastrowid
                self.conn.commit()

                tokens = self._tokenize(content)
                term_positions = defaultdict(list)
                for i, token in enumerate(tokens):
                    term_positions[token].append(i)

                # Prepare data for batch insert
                inverted_index_data = []
                for term, positions in term_positions.items():
                    inverted_index_data.append((term, doc_id, json.dumps(positions)))
                    
                    # Update in-memory inverted_index_cache
                    if term not in self.inverted_index_cache:
                        self.inverted_index_cache[term] = {} # Initialize as dict
                    self.inverted_index_cache[term][doc_id] = positions # Assign (overwrite if exists)

                # Use executemany for batch insertion into inverted_index
                if inverted_index_data:
                    cursor.executemany("INSERT INTO inverted_index (term, doc_id, positions) VALUES (?, ?, ?)",
                                       inverted_index_data)
                self.conn.commit()

                # Update in-memory document cache and indexed_urls_cache
                self.document_cache[doc_id] = {"url": url, "title": title, "content": content}
                self.indexed_urls_cache.add(url)
                
                # Simple LRU for document cache: if over limit, remove oldest
                if len(self.document_cache) > CONFIG["CACHE_DOCUMENTS_LIMIT"]:
                    # Find and remove the oldest entry (least recently scraped)
                    oldest_doc_id_in_cache = None
                    # Iterate through items to find one to remove (simple FIFO for dict)
                    # For a true LRU, collections.OrderedDict would be better
                    for d_id in self.document_cache:
                        if d_id != doc_id: # Don't remove the one we just added
                            oldest_doc_id_in_cache = d_id
                            break
                    if oldest_doc_id_in_cache:
                        del self.document_cache[oldest_doc_id_in_cache]

            print(f"Indexed: {title} ({url}) [Doc ID: {doc_id}]")
            return doc_id
        except sqlite3.Error as e:
            print(f"Database error during indexing {url}: {e}", file=sys.stderr)
            if self.conn:
                self.conn.rollback() # Rollback changes if an error occurred
            return None
        except Exception as e:
            print(f"An unexpected error occurred during indexing {url}: {e}", file=sys.stderr)
            if self.conn:
                self.conn.rollback()
            return None
        finally:
            if cursor:
                cursor.close()

    def _scrape_and_index_url(self, url):
        """Busca, analisa e indexa uma única URL, usando o conteúdo limpo do newspaper3k."""
        if not isinstance(url, str):
            print(f"Error: _scrape_and_index_url received non-string URL: {type(url)}", file=sys.stderr)
            return

        with self.lock:
            if url in self.queued_urls_cache:
                self.queued_urls_cache.remove(url)

        print(f"Attempting to scrape: {url}")
        acquired_data = self.data_acquisition_module.acquire_data(url)

        if acquired_data:
            # Usa as novas chaves retornadas pelo acquire_data
            title = acquired_data["title"]
            clean_text = acquired_data["clean_text"]
            article_links = acquired_data["article_links"]
            pagination_links = acquired_data["pagination_links"]
            
            # Passa o texto limpo para ser indexado
            indexed_doc_id = self._index_document(url, title, clean_text)

            if indexed_doc_id is not None:
                with self.lock:
                    for page_url in reversed(pagination_links):
                        if page_url != url and page_url not in self.queued_urls_cache:
                            self.scrape_queue.appendleft(page_url)
                            self.queued_urls_cache.add(page_url)
                            print(f"PAGINATION link prioritized in queue: {page_url}")

                    for article_url in article_links:
                        if article_url != url and article_url not in self.indexed_urls_cache and article_url not in self.queued_urls_cache:
                            self.scrape_queue.append(article_url)
                            self.queued_urls_cache.add(article_url)
        else:
            print(f"Failed to acquire or process data for {url}. Not indexing.", file=sys.stderr)

    def _scraper_worker(self):
        """Worker function for background scraping threads."""
        while self.scraping_active.is_set() or self.scrape_queue:
            url_to_scrape = None
            with self.lock: # Protect queue access
                if self.scrape_queue:
                    url_to_scrape = self.scrape_queue.popleft()
            
            if url_to_scrape:
                self._scrape_and_index_url(url_to_scrape)
                # The delay is handled by DataAcquisitionModule._fetch_page
            else:
                # If queue is empty and scraping is active, wait a bit before checking again
                if self.scraping_active.is_set():
                    time.sleep(1) 
                else: # If scraping is no longer active and queue is empty, worker can exit
                    break

    def start_background_scraper(self):
        """Starts background threads to continuously scrape and index content."""
        if self.scraper_threads and any(t.is_alive() for t in self.scraper_threads):
            print("Scraper already running.")
            return

        print("Starting background scraper...")
        # Add initial seed URLs to the queue
        with self.lock:
            cursor = self.conn.cursor()
            for source in CONFIG["TRUSTED_SOURCES"]:
                for url in source["seed_urls"]:
                    if url in self.queued_urls_cache:
                        continue # Já está na fila, pular.

                    # Verifica a data da última coleta no banco de dados
                    cursor.execute("SELECT last_scraped FROM documents WHERE url = ?", (url,))
                    result = cursor.fetchone()
                    
                    should_scrape = True
                    if result:
                        last_scraped_time = result['last_scraped']
                        time_since_scrape = time.time() - last_scraped_time
                        if time_since_scrape < CONFIG["SCRAPE_INTERVAL_SECONDS"]:
                            should_scrape = False # Ainda não está "velha" o suficiente
                    
                    if should_scrape:
                        self.scrape_queue.append(url)
                        self.queued_urls_cache.add(url)
                        status = "Re-queuing stale source" if result else "Queuing new seed URL"
                        print(f"{status}: {url}")
            cursor.close()

        self.scraping_active.set()
        self.scraper_threads = []
        for i in range(CONFIG["MAX_CONCURRENT_SCRAPES"]):
            thread = threading.Thread(target=self._scraper_worker, name=f"Scraper-{i+1}")
            thread.daemon = True # Allow main program to exit even if scraper is running
            self.scraper_threads.append(thread)
            thread.start()
        print(f"Background scraper started with {CONFIG['MAX_CONCURRENT_SCRAPES']} threads.")

    def stop_background_scraper(self):
        """Stops the background scraper threads."""
        print("Stopping background scraper...")
        self.scraping_active.clear() # Signal threads to stop
        for thread in self.scraper_threads:
            print(f"Waiting for {thread.name} to finish...")
            thread.join(timeout=30) # Wait for threads to finish gracefully
            if thread.is_alive():
                print(f"Warning: Scraper thread {thread.name} did not terminate gracefully within 30 seconds. It might be stuck.", file=sys.stderr)
        print("Background scraper stopped.")

    # --- Search/Retrieval ---
    def _get_document_content(self, doc_id):
        """Retrieves document content from cache or database."""
        if not isinstance(doc_id, int):
            print(f"Error: _get_document_content received non-integer doc_id: {type(doc_id)}", file=sys.stderr)
            return None

        if doc_id in self.document_cache:
            return self.document_cache[doc_id]
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT url, title, content FROM documents WHERE id = ?", (doc_id,))
            result = cursor.fetchone()
            if result:
                doc_data = {"url": result['url'], "title": result['title'], "content": result['content']}
                # Add to cache, and if cache exceeds limit, remove oldest
                with self.lock: # Protect cache modifications
                    self.document_cache[doc_id] = doc_data 
                    if len(self.document_cache) > CONFIG["CACHE_DOCUMENTS_LIMIT"]:
                        # Find and remove the oldest entry (simple FIFO for dict)
                        oldest_doc_id_in_cache = None
                        for d_id in self.document_cache:
                            if d_id != doc_id: # Don't remove the one we just added
                                oldest_doc_id_in_cache = d_id
                                break
                        if oldest_doc_id_in_cache:
                            del self.document_cache[oldest_doc_id_in_cache]
                return doc_data
            return None
        except sqlite3.Error as e:
            print(f"Database error retrieving document {doc_id}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"An unexpected error occurred retrieving document {doc_id}: {e}", file=sys.stderr)
            return None
        finally:
            if cursor:
                cursor.close()

    def _generate_snippet(self, content, query_tokens):
        """Generates a concise snippet from document content relevant to query."""
        if not isinstance(content, str):
            print(f"Warning: _generate_snippet received non-string content: {type(content)}. Returning empty snippet.", file=sys.stderr)
            return ""
        if not isinstance(query_tokens, list):
            print(f"Warning: _generate_snippet received non-list query_tokens: {type(query_tokens)}. Treating as empty.", file=sys.stderr)
            query_tokens = []

        sentences = re.split(r'(?<=[.!?])\s+', content)
        relevant_sentences = []
        for sentence in sentences:
            # Check for tokens in a case-insensitive manner
            if any(token in sentence.lower() for token in query_tokens):
                relevant_sentences.append(sentence)
        
        snippet = " ".join(relevant_sentences)
        
        # Fallback if no relevant sentences found or snippet is too short
        if not snippet or len(snippet) < 50: # Arbitrary threshold for "too short"
            snippet = content[:min(len(content), CONFIG["MAX_SNIPPET_LENGTH"])]
            if len(content) > CONFIG["MAX_SNIPPET_LENGTH"]:
                snippet += "..."

        if len(snippet) > CONFIG["MAX_SNIPPET_LENGTH"]:
            # Trim to max length, trying to end on a sentence boundary
            snippet = snippet[:CONFIG["MAX_SNIPPET_LENGTH"]]
            last_punctuation = max(snippet.rfind('.'), snippet.rfind('!'), snippet.rfind('?'))
            if last_punctuation != -1:
                snippet = snippet[:last_punctuation + 1]
            else:
                # If no punctuation, trim at last whole word
                snippet = snippet.rsplit(' ', 1)[0] + "..." 
        
        return snippet.strip() # Ensure no leading/trailing whitespace

    def search(self, query: str) -> list:
        """
        Performs a search query against the local index.
        Returns a list of dictionaries, each representing a search result,
        optimized for LLM consumption.
        """
        if not isinstance(query, str):
            print(f"Error: Search query must be a string. Received: {type(query)}", file=sys.stderr)
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            print(f"Warning: Query '{query}' resulted in no valid tokens. Returning empty results.", file=sys.stderr)
            return []

        doc_scores = defaultdict(float)
        
        # Aggregate scores from all query tokens
        for token in query_tokens:
            # Prioritize in-memory cache
            if token in self.inverted_index_cache:
                for doc_id, positions in self.inverted_index_cache[token].items():
                    doc_scores[doc_id] += len(positions) # Simple term frequency scoring
            else:
                # Fallback to DB if not in cache (less efficient, but ensures completeness)
                cursor = None
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT doc_id, positions FROM inverted_index WHERE term = ?", (token,))
                    
                    # Initialize term's entry in cache if it doesn't exist
                    with self.lock: # Protect cache access
                        if token not in self.inverted_index_cache:
                            self.inverted_index_cache[token] = {} # Initialize as dict

                    for row in cursor.fetchall():
                        doc_id = row['doc_id']
                        positions_str = row['positions']
                        try:
                            positions = json.loads(positions_str)
                            if not isinstance(positions, list):
                                print(f"Warning: Invalid positions JSON from DB for term '{token}', doc_id '{doc_id}'. Expected list, got {type(positions)}. Skipping.", file=sys.stderr)
                                continue
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON positions from DB for term '{token}', doc_id '{doc_id}': {e}. Skipping.", file=sys.stderr)
                            continue
                        
                        doc_scores[doc_id] += len(positions)
                        # Add to cache for future queries
                        with self.lock: # Protect cache access
                            self.inverted_index_cache[token][doc_id] = positions # Assign (overwrite if exists)
                except sqlite3.Error as e:
                    print(f"Database error during search for term '{token}': {e}", file=sys.stderr)
                    # Continue with what we have, or return empty if critical
                except Exception as e:
                    print(f"An unexpected error occurred during search for term '{token}': {e}", file=sys.stderr)
                finally:
                    if cursor:
                        cursor.close()

        # Sort documents by score (descending)
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:CONFIG["MAX_RESULTS_PER_QUERY"]]:
            doc_data = self._get_document_content(doc_id)
            if doc_data:
                snippet = self._generate_snippet(doc_data["content"], query_tokens)
                results.append({
                    "title": doc_data["title"],
                    "url": doc_data["url"],
                    "snippet": snippet,
                    # "score": score # Can include score for debugging/transparency if LLM needs it
                })
            else:
                print(f"Warning: Could not retrieve content for doc_id {doc_id} during search results compilation.", file=sys.stderr)
        return results

    def llm_search_interface(self, query: str) -> str:
        """
        Provides a structured search interface for LLMs.
        Performs a search and formats the results into a JSON string,
        optimized for easy parsing and consumption by Large Language Models.

        Args:
            query (str): The search query provided by the LLM.

        Returns:
            str: A JSON string containing the search results,
                 including title, URL, and a relevant snippet for each.
                 Example format:
                 {
                   "query": "your search query",
                   "results": [
                     {"title": "...", "url": "...", "snippet": "..."},
                     {"title": "...", "url": "...", "snippet": "..."}
                   ],
                   "message": "Search completed successfully."
                 }
                 If no results are found, the 'results' list will be empty.
        """
        if not isinstance(query, str):
            error_message = f"Invalid query type. Expected string, got {type(query)}."
            output_data = {
                "query": str(query), # Convert to string for output to avoid serialization errors
                "results": [],
                "message": error_message
            }
            print(f"Error: {error_message}", file=sys.stderr)
            return json.dumps(output_data, indent=2)

        try:
            search_results = self.search(query)
            
            output_data = {
                "query": query,
                "results": search_results,
                "message": "Search completed successfully." if search_results else "No results found for your query."
            }
            
            return json.dumps(output_data, indent=2)
        except Exception as e:
            error_message = f"An unexpected error occurred during LLM search interface call: {e}"
            output_data = {
                "query": query,
                "results": [],
                "message": error_message
            }
            print(f"Critical Error: {error_message}", file=sys.stderr)
            return json.dumps(output_data, indent=2)


    def close(self):
        """Closes the database connection and stops background threads."""
        print("Initiating system shutdown...")
        self.stop_background_scraper()
        if self.conn:
            try:
                self.conn.close()
                print("Search system database connection closed.")
            except sqlite3.Error as e:
                print(f"Error closing database connection: {e}", file=sys.stderr)
        print("System shutdown complete.")

# --- Main execution block for demonstration/LLM interaction ---
if __name__ == "__main__":
    search_system = None
    try:
        search_system = LLMSearchSystem()
        
        # Start the background scraper to populate the index
        search_system.start_background_scraper()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting gracefully due to user interrupt...")
    except Exception as main_e:
        print(f"\nAn unhandled error occurred in the main execution block: {main_e}", file=sys.stderr)
    finally:
        if search_system:
            search_system.close()
        else:
            print("Search system was not initialized. No cleanup needed.", file=sys.stderr)