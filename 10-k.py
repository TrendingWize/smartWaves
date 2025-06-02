import os
import time
import logging
import pickle
import json
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup # For HTML parsing
import lxml # Parser for BeautifulSoup
import numpy as np
import faiss
from openai import OpenAI, RateLimitError, APIError # Import specific errors
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
import html # For escaping HTML characters in the output and recursive formatting
 # <-- Added for token counting
import streamlit as st
import sys # <-- Added for sys.stderr
import nltk
import warnings
from bs4 import XMLParsedAsHTMLWarning

proc = None

# Suppress XML parsing warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Ensure nltk 'punkt' tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# --- Configuration (Keep variable names and values) ---
symbol = os.environ.get("TICKER_SYMBOL")
if not symbol:
    logging.error("TICKER_SYMBOL environment variable not set. Exiting.")
    sys.stderr.write("TICKER_SYMBOL environment variable not set. Exiting.\n")
    exit(1)
    
# Use environment variables or set to None initially
openAI_api_key = None
financial_api_key = None

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIMENSION = 3072 # Dimension for text-embedding-3-large
EMBEDDING_BATCH_SIZE = 16        # OpenAI allows larger batches (check limits, up to 2048 inputs)
# NEW: Token limit for embedding model (slightly less than max 8191 for safety)
EMBEDDING_MODEL_TOKEN_LIMIT = 1500

# LLM Configuration
OPENAI_LLM_MODEL = "gpt-4o-mini" # Faster and cheaper than gpt-4

# Search/Context Configuration
FAISS_SEARCH_RESULTS = 1 # Number of context results for LLM (Adjust as needed)
OPENAI_LLM_MAX_WORKERS = 4       # Increase concurrent OpenAI LLM calls (watch rate limits)
OPENAI_LLM_MAX_TOKENS = 5000      # Max tokens for OpenAI LLM response (Adjust as needed - note: gpt-4o-mini has a large context window, but this limits the *output*)

# Other Configuration
REQUESTS_TIMEOUT = 30         # Timeout for HTTP requests in seconds
USER_AGENT = "MyCompanyName MyAppName/1.0 (Contact: myemail@example.com)" # Be polite to APIs

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()
# Use environment variables primarily
openAI_api_key  = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

financial_api_key = "Aw0rlddPHSnxmi3VmZ6jN4u3b2vvUvxn" # Example Key

# Check necessary keys now
if not all([openAI_api_key , financial_api_key ]):
    logging.error("Required API keys (OpenAI, FMP) not found. Please check your .env file or environment variables.")
    sys.stderr.write("Required API keys (OpenAI, FMP) not found. Please check your .env file or environment variables.\n")
    exit(1)

# --- Initialize API Clients ---
try:
    # Removed: client_voyageai = voyageai.Client(api_key=VOYAGE_API_KEY)
    client_openai = OpenAI(api_key=openAI_api_key )
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    sys.stderr.write(f"Failed to initialize OpenAI client: {e}\n")
    exit(1)



# Suppress XMLParsedAsHTMLWarning from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Ensure 'punkt' is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download("punkt")
    logging.info("NLTK 'punkt' downloaded successfully.")


# --- Helper Functions ---

def fetch_fmp_data(url: str, params: Dict = None) -> Dict | List | None:
    """Fetches data from Financial Modeling Prep API with error handling."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT, headers=headers)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response from {url}: {e}")
        # Check if response exists before accessing text
        response_text = "N/A"
        if 'response' in locals() and hasattr(response, 'text'):
            response_text = response.text[:500]
        logging.error(f"Response text (first 500 chars): {response_text}...")
        return None


# --- MODIFIED: Token-Based Chunking Function ---
def extract_text_and_chunk_by_tokens(html_content: str,
                                     token_limit: int = EMBEDDING_MODEL_TOKEN_LIMIT,
                                     chunk_overlap_sentences: int = 2) -> List[str]:
    """
    Extracts text using BeautifulSoup, tokenizes into sentences using NLTK,
    and then chunks based on tiktoken token counts to respect model limits.
    Includes sentence splitting for sentences that exceed token limits.
    """
    logging.error("Tiktoken encoder not available. Cannot perform token-based chunking.")
        return []

    logging.info(f"Parsing HTML and chunking text by token limit ({token_limit}) using tiktoken...")
    start_time = time.time()
    soup = BeautifulSoup(html_content, 'lxml')
    chunks = []
    min_meaningful_sentence_len = 15 # Keep sentences longer than this chars

    body = soup.find('body')
    if not body:
        logging.error("HTML body tag not found.")
        return []

    # 1. Extract text and clean up
    all_text = body.get_text(separator='\n', strip=True)
    all_text = '\n'.join(line for line in all_text.splitlines() if line.strip())

    # 2. Split into sentences
    try:
        sentences = nltk.sent_tokenize(all_text)
        logging.info(f"Tokenized into {len(sentences)} sentences.")
    except Exception as e:
        logging.error(f"NLTK sentence tokenization failed: {e}")
        return [] # Cannot proceed without sentences

    # Function to split long sentences into smaller parts
    def split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
        """Split a long sentence into smaller parts that fit within token limit."""
        # Try splitting by various punctuation marks first
        split_candidates = []
        
        # First try to split by semicolons, em dashes, or parentheses
        for delimiter in [';', '—', ')', '(', ':', '-']:
            if delimiter in sentence:
                parts = sentence.split(delimiter)
                reconstructed = []
                current_part = ""
                
                for i, part in enumerate(parts):
                    # Add delimiter back except for opening brackets
                    if i > 0 and delimiter not in ['(']:
                        test_part = current_part + delimiter + part
                    else:
                        test_part = current_part + part
                        
                    try:
                        part_tokens = len(.encode(test_part))
                    except Exception:
                        part_tokens = len(test_part) // 3  # Rough estimate if encoding fails
                        
                    if part_tokens <= max_tokens:
                        current_part = test_part
                    else:
                        if current_part:
                            split_candidates.append(current_part)
                        current_part = part
                        
                if current_part:
                    split_candidates.append(current_part)
                    
                # If we found valid splits, return them
                if len(split_candidates) > 1:
                    return split_candidates
        
        # If no good punctuation splits, try splitting by commas
        if ',' in sentence:
            parts = sentence.split(',')
            reconstructed = []
            current_part = ""
            
            for i, part in enumerate(parts):
                if i > 0:
                    test_part = current_part + ", " + part
                else:
                    test_part = part
                    
                try:
                    part_tokens = len(.encode(test_part))
                except Exception:
                    part_tokens = len(test_part) // 3  # Rough estimate if encoding fails
                    
                if part_tokens <= max_tokens:
                    current_part = test_part
                else:
                    if current_part:
                        split_candidates.append(current_part)
                    current_part = part
                    
            if current_part:
                split_candidates.append(current_part)
                
            # If we found valid splits, return them
            if len(split_candidates) > 1:
                return split_candidates
        
        # If still no good splits, fall back to word-based splitting
        words = sentence.split()
        current_chunk = []
        current_chunk_tokens = 0
        
        for word in words:
            try:
                word_tokens = len(.encode(word))
            except Exception:
                word_tokens = len(word) // 3  # Rough estimate if encoding fails
                
            # Handle single words that are too long (rare case)
            if word_tokens > max_tokens:
                if current_chunk:
                    split_candidates.append(" ".join(current_chunk))
                    current_chunk = []
                    current_chunk_tokens = 0
                
                # Split the word into characters chunks as a last resort
                for i in range(0, len(word), max_tokens // 4):  # Approximate char to token ratio
                    word_part = word[i:i + max_tokens // 4]
                    split_candidates.append(word_part)
                continue
                
            if current_chunk_tokens + word_tokens + 1 > max_tokens and current_chunk:
                split_candidates.append(" ".join(current_chunk))
                current_chunk = [word]
                current_chunk_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_chunk_tokens += word_tokens + (1 if current_chunk_tokens > 0 else 0)
                
        if current_chunk:
            split_candidates.append(" ".join(current_chunk))
            
        return split_candidates

    # 3. Group sentences into chunks based on token count
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    # Store token counts per sentence to avoid recalculation
    sentence_token_cache = {}

    for i, sentence in enumerate(sentences):
        trimmed_sentence = sentence.strip()
        if len(trimmed_sentence) < min_meaningful_sentence_len:
            continue # Skip very short sentences/fragments

        # Get token count, using cache if available
        if trimmed_sentence in sentence_token_cache:
            sentence_tokens = sentence_token_cache[trimmed_sentence]
        else:
            try:
                sentence_tokens = len(.encode(trimmed_sentence))
                sentence_token_cache[trimmed_sentence] = sentence_tokens
            except Exception as e:
                logging.warning(f"Could not encode sentence for token count: {e}. Skipping sentence: '{trimmed_sentence[:50]}...'")
                continue

        # Handle sentences that are individually too long by splitting them
        if sentence_tokens > token_limit:
            logging.info(f"Sentence {i+1} exceeds token limit ({sentence_tokens} > {token_limit}). Splitting sentence.")
            
            # Split the long sentence into smaller parts
            split_parts = split_long_sentence(trimmed_sentence, token_limit)
            
            if not split_parts:
                logging.warning(f"Failed to split long sentence {i+1}. Skipping: '{trimmed_sentence[:100]}...'")
                continue
                
            logging.info(f"Successfully split sentence {i+1} into {len(split_parts)} parts.")
            
            # Process each split part as if it were a regular sentence
            for part in split_parts:
                part_tokens = len(.encode(part))
                
                # Check if adding this part would exceed token limit
                if current_chunk_tokens > 0 and (current_chunk_tokens + part_tokens + 1) > token_limit:
                    # Finish the current chunk
                    if current_chunk_sentences:
                        chunk_prefix = f"[Chunk {len(chunks) + 1}] "
                        chunk_text = chunk_prefix + " ".join(current_chunk_sentences)
                        # Double check the final chunk's token count
                        try:
                            final_chunk_tokens = len(.encode(chunk_text))
                            if final_chunk_tokens <= token_limit:
                                chunks.append(chunk_text)
                            else:
                                logging.warning(f"Final constructed chunk {len(chunks)+1} exceeded token limit ({final_chunk_tokens} > {token_limit}) despite checks. Skipping this chunk. Text starts: '{chunk_text[:100]}...'")
                        except Exception as e:
                            logging.warning(f"Could not encode final chunk {len(chunks)+1} for token count check: {e}. Skipping.")

                        # Start new chunk with overlap
                        overlap_start_index = max(0, len(current_chunk_sentences) - chunk_overlap_sentences)
                        current_chunk_sentences = current_chunk_sentences[overlap_start_index:]

                        # Recalculate token count for the overlapping sentences using the cache
                        current_chunk_tokens = 0
                        if current_chunk_sentences:
                            try:
                                overlap_text = " ".join(current_chunk_sentences)
                                current_chunk_tokens = len(.encode(overlap_text))
                            except Exception as e:
                                logging.warning(f"Could not encode overlapping sentences for token count: {e}. Resetting overlap.")
                                current_chunk_sentences = []
                                current_chunk_tokens = 0

                # Add the current part to the chunk
                if (current_chunk_tokens + part_tokens + (1 if current_chunk_tokens > 0 else 0)) <= token_limit:
                    current_chunk_sentences.append(part)
                    current_chunk_tokens += part_tokens + (1 if current_chunk_tokens > 0 else 0)
                    # Cache the token count for this part
                    sentence_token_cache[part] = part_tokens
            
            # Continue to the next sentence after processing all parts
            continue

        # Check if adding the *next* sentence would exceed the token limit
        # Add 1 token to account for potential separator/space token between sentences
        if current_chunk_tokens > 0 and (current_chunk_tokens + sentence_tokens + 1) > token_limit:
            # Finish the current chunk
            if current_chunk_sentences:
                chunk_prefix = f"[Chunk {len(chunks) + 1}] "
                chunk_text = chunk_prefix + " ".join(current_chunk_sentences)
                # Double check the final chunk's token count just in case
                try:
                    final_chunk_tokens = len(.encode(chunk_text))
                    if final_chunk_tokens <= token_limit:
                         chunks.append(chunk_text)
                    else:
                          logging.warning(f"Final constructed chunk {len(chunks)+1} exceeded token limit ({final_chunk_tokens} > {token_limit}) despite checks. Skipping this chunk. Text starts: '{chunk_text[:100]}...'")
                except Exception as e:
                    logging.warning(f"Could not encode final chunk {len(chunks)+1} for token count check: {e}. Skipping.")


                # Start new chunk with overlap
                overlap_start_index = max(0, len(current_chunk_sentences) - chunk_overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start_index:]

                # Recalculate token count for the overlapping sentences using the cache when possible
                current_chunk_tokens = 0
                if current_chunk_sentences:
                    try:
                        overlap_text = " ".join(current_chunk_sentences)
                        current_chunk_tokens = len(.encode(overlap_text))
                    except Exception as e:
                         logging.warning(f"Could not encode overlapping sentences for token count: {e}. Resetting overlap.")
                         current_chunk_sentences = []
                         current_chunk_tokens = 0
            else:
                current_chunk_sentences = []
                current_chunk_tokens = 0


        # Add the current sentence to the *new* or *ongoing* chunk
        # Check if adding the current sentence (even after potentially starting a new chunk) fits
        if (current_chunk_tokens + sentence_tokens + (1 if current_chunk_tokens > 0 else 0)) <= token_limit:
            current_chunk_sentences.append(trimmed_sentence)
            current_chunk_tokens += sentence_tokens + (1 if current_chunk_tokens > 0 else 0)
        elif not current_chunk_sentences:
             # This should not happen now that we split long sentences, but keeping as failsafe
             logging.warning(f"Single sentence {i+1} cannot be added as it exceeds token limit ({sentence_tokens} > {token_limit}) even in an empty chunk. Skipping: '{trimmed_sentence[:100]}...'")


    # Add the last remaining chunk
    if current_chunk_sentences:
        chunk_prefix = f"[Chunk {len(chunks) + 1}] "
        chunk_text = chunk_prefix + " ".join(current_chunk_sentences)
        try:
            final_chunk_tokens = len(.encode(chunk_text))
            if final_chunk_tokens <= token_limit:
                chunks.append(chunk_text)
            else:
                logging.warning(f"Final trailing chunk exceeded token limit ({final_chunk_tokens} > {token_limit}) on assembly. Skipping. Text starts: '{chunk_text[:100]}...'")
        except Exception as e:
             logging.warning(f"Could not encode final trailing chunk for token count check: {e}. Skipping.")


    end_time = time.time()
    logging.info(f"Token-based HTML parsing and chunking finished in {end_time - start_time:.2f} seconds. Extracted {len(chunks)} chunks.")
    if not chunks:
        logging.warning("No text chunks could be extracted or all chunks were too long/skipped.")
    return chunks


# --- OpenAI Embedding Generation ---
def generate_openai_embeddings(chunks: List[str], openai_model: str, batch_size: int) -> List[Dict]:
    """Generates embeddings using OpenAI API with batching and error handling."""
    embeddings = []
    logging.info(f"Generating OpenAI embeddings for {len(chunks)} chunks using model {openai_model} in batches of {batch_size}...")
    start_time = time.time()
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = chunks[i:i + batch_size]
        logging.info(f"Processing batch {batch_num}/{total_batches}...")

        batch_cleaned = [text.replace("\n", " ") for text in batch]

        try:
            max_retries = 3
            retry_delay = 10 # seconds
            for attempt in range(max_retries):
                try:
                    response = client_openai.embeddings.create(input=batch_cleaned, model=openai_model)
                    # Success, exit retry loop
                    break
                except RateLimitError:
                    if attempt < max_retries - 1:
                        logging.warning(f"OpenAI Rate limit hit on batch {batch_num} (attempt {attempt+1}/{max_retries}). Retrying after {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2 # Exponential backoff
                    else:
                         logging.error(f"OpenAI Rate limit hit on batch {batch_num} after {max_retries} attempts. Aborting this batch.")
                         raise # Re-raise the last RateLimitError
                except APIError as api_e:
                    if api_e.status_code == 400 and 'context length' in str(api_e.body):
                         logging.error(f"OpenAI API Error on batch {batch_num} (attempt {attempt+1}): Context length error despite chunking! {api_e}")
                         logging.error(f"Problematic batch contents (first 100 chars each): {[item[:100] for item in batch_cleaned]}")
                         raise api_e # Re-raise critical error
                    else:
                        logging.error(f"OpenAI API Error on batch {batch_num} (attempt {attempt+1}): {api_e}")
                        if attempt < max_retries - 1:
                            logging.info(f"Retrying API error after 5s...")
                            time.sleep(5)
                        else:
                            raise # Re-raise if max retries reached for APIError too
                # Check if break was called (success)
                if 'response' in locals(): # Check if response variable was assigned
                    break
            else: # This else block executes if the loop finished without a break (i.e., all retries failed)
                logging.error(f"Skipping batch {batch_num} due to persistent API errors after {max_retries} attempts.")
                continue # Skip to the next batch

            if response and response.data:
                 if len(response.data) == len(batch):
                    for j, embedding_data in enumerate(response.data):
                        embeddings.append({'text': batch[j], 'embedding': embedding_data.embedding})
                 else:
                     logging.warning(f"Mismatch in returned embeddings count for batch {batch_num}. Expected {len(batch)}, got {len(response.data)}")
            else:
                logging.warning(f"No embeddings data returned for batch {batch_num}. Response: {response}")

        except RateLimitError:
            logging.error(f"Skipping batch {batch_num} due to persistent RateLimitError.")
        except APIError as e:
             logging.error(f"Skipping batch {batch_num} due to persistent OpenAI API Error: {e}")
             logging.error(f"First item in failing batch (first 100 chars): {batch_cleaned[0][:100] if batch_cleaned else 'N/A'}")
        except Exception as e:
            logging.error(f"Skipping batch {batch_num} due to unexpected error during OpenAI embedding: {e}")

    end_time = time.time()
    logging.info(f"OpenAI Embedding generation finished in {end_time - start_time:.2f} seconds. Got {len(embeddings)} embeddings.")
    return embeddings

# --- store_embeddings ---
def store_embeddings(embeddings: List[Dict], dimension: int,
                     index_file: str, embeddings_file: str) -> Tuple[faiss.Index | None, List[str]]:
    """Stores embeddings in FAISS and saves index/texts."""
    if not embeddings:
        logging.error("No embeddings provided to store.")
        return None, []

    logging.info(f"Storing {len(embeddings)} embeddings with dimension {dimension}...")
    start_time = time.time()
    try:
        actual_dim = len(embeddings[0]['embedding'])
        if actual_dim != dimension:
             logging.warning(f"Provided dimension ({dimension}) does not match actual embedding dimension ({actual_dim}). Using actual dimension {actual_dim} for FAISS index.")
             dimension = actual_dim

        index = faiss.IndexFlatL2(dimension)
        texts = [item['text'] for item in embeddings]
        vectors = [item['embedding'] for item in embeddings]

        vectors_np = np.array(vectors).astype('float32')

        if vectors_np.ndim != 2 or vectors_np.shape[1] != dimension:
             logging.error(f"Critical dimension mismatch or incorrect shape! Expected (n, {dimension}), got {vectors_np.shape}. Aborting storage.")
             return None, []

        index.add(vectors_np)

        faiss.write_index(index, index_file)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(texts, f)

        end_time = time.time()
        logging.info(f"FAISS index ({index_file}) and texts ({embeddings_file}) saved in {end_time - start_time:.2f} seconds.")
        return index, texts
    except Exception as e:
        logging.error(f"Failed to store embeddings: {e}")
        return None, []

# --- load_embeddings_and_texts ---
def load_embeddings_and_texts(index_file: str, embeddings_file: str) -> Tuple[faiss.Index | None, List[str]]:
    """Loads FAISS index and corresponding texts from disk."""
    logging.info(f"Loading embeddings from {index_file} and {embeddings_file}...")
    try:
        index = faiss.read_index(index_file)
        with open(embeddings_file, 'rb') as f:
            texts = pickle.load(f)
        logging.info(f"Successfully loaded FAISS index with {index.ntotal} vectors and {len(texts)} texts.")
        # Use the index's dimension as the source of truth after loading
        loaded_dimension = index.d
        if loaded_dimension != OPENAI_EMBEDDING_DIMENSION:
             logging.warning(f"Loaded FAISS index dimension ({loaded_dimension}) doesn't match configured dimension ({OPENAI_EMBEDDING_DIMENSION}). Using loaded index dimension {loaded_dimension}.")
             # Consider updating a global or returning the loaded dimension if needed elsewhere
        return index, texts
    except FileNotFoundError:
        logging.warning(f"Embeddings files ({index_file}, {embeddings_file}) not found. Need to process document first.")
        return None, []
    except Exception as e:
        logging.error(f"Failed to load embeddings: {e}")
        return None, []

# --- search_faiss ---
def search_faiss(queries: List[str], index: faiss.Index, texts: List[str],
                 openai_embedding_model: str, num_results: int) -> Dict[str, List[str]]:
    """Embeds queries using OpenAI and searches FAISS index."""
    results_dict = {}
    if not queries or index is None or not texts:
        logging.warning("Search cannot proceed. Missing queries, index, or texts.")
        return results_dict

    logging.info(f"Embedding {len(queries)} queries using {openai_embedding_model} and searching top {num_results} results...")
    start_time = time.time()
    try:
        queries_cleaned = [q.replace("\n", " ") for q in queries]

        # --- Add Retry Logic for Query Embedding ---
        max_retries = 3
        retry_delay = 5 # Shorter delay for potentially smaller query batches
        query_embeddings = None
        for attempt in range(max_retries):
            try:
                response = client_openai.embeddings.create(input=queries_cleaned, model=openai_embedding_model)
                query_embeddings = [item.embedding for item in response.data]
                break # Success
            except RateLimitError:
                if attempt < max_retries - 1:
                    logging.warning(f"OpenAI Rate limit hit during query embedding (attempt {attempt+1}/{max_retries}). Retrying after {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.error(f"OpenAI Rate limit hit on query embedding after {max_retries} attempts.")
                    raise # Re-raise to be caught by the outer try-except
            except APIError as api_e:
                 logging.error(f"OpenAI API Error during query embedding (attempt {attempt+1}): {api_e}")
                 if attempt < max_retries - 1:
                     logging.info(f"Retrying API error after 5s...")
                     time.sleep(5)
                 else:
                     raise # Re-raise
        # --- End Retry Logic ---

        if not query_embeddings or len(query_embeddings) != len(queries):
             logging.error(f"Failed to generate embeddings for all search queries after retries. Got {len(query_embeddings) if query_embeddings else 0}, expected {len(queries)}.")
             return results_dict # Return empty if embedding failed

        query_vectors = np.array(query_embeddings).astype('float32')

        # Check dimension compatibility with index (use index.d as source of truth)
        if query_vectors.shape[1] != index.d:
            logging.error(f"Query embedding dimension ({query_vectors.shape[1]}) does not match FAISS index dimension ({index.d}). Cannot search.")
            return results_dict

        # Perform batch search
        distances, indices = index.search(query_vectors, num_results)

        # Process results for each query
        for i, query in enumerate(queries):
            query_indices = indices[i]
            valid_indices = [idx for idx in query_indices if 0 <= idx < len(texts)]
            unique_texts = []
            seen_texts = set()
            for idx in valid_indices:
                text_content = texts[idx]
                if text_content not in seen_texts:
                    unique_texts.append(text_content)
                    seen_texts.add(text_content)

            results_dict[query] = unique_texts
            if len(unique_texts) < len(valid_indices):
                 logging.debug(f"Removed duplicate context chunks for query: '{query}'")

    except RateLimitError:
        logging.error(f"FAISS search failed due to persistent OpenAI Rate limit during query embedding.")
    except APIError as e:
        logging.error(f"FAISS search failed due to persistent OpenAI API Error during query embedding: {e}")
    except Exception as e:
        logging.error(f"FAISS search failed with unexpected error: {e}")


    end_time = time.time()
    logging.info(f"FAISS search completed in {end_time - start_time:.2f} seconds.")
    return results_dict

# --- ask_llm_batch (Using Standardized Prompt) ---
def ask_llm_batch(queries_with_context: List[Dict[str, str]], max_workers: int) -> List[Dict]:
    """
    Processes multiple queries in parallel using OpenAI LLM API via ThreadPoolExecutor.
    Expects input like: [{"Question": "question1", "context": "context1"}, ...]
    Returns results including the raw JSON string from the LLM.
    """
    results = []
    total_queries = len(queries_with_context)
    logging.info(f"Sending {total_queries} queries to OpenAI LLM {OPENAI_LLM_MODEL} using {max_workers} workers...")
    start_time = time.time()

    def process_single_query(query_dict):
        query = query_dict['Question']
        context = query_dict['context']

        max_context_char_warn_limit = 100000 # Rough proxy for token limit warning
        if len(context) > max_context_char_warn_limit:
             logging.warning(f"Context for query '{query}' is very long ({len(context)} chars). Ensure it fits within the LLM's ({OPENAI_LLM_MODEL}) context window.")

        # --- Standardized Prompt ---
        prompt = f"""
        You are an AI financial analyst. Analyze the provided context from a company's financial filing (like a 10-K or 20-F)
        to answer the specific question asked. Be concise and base your answer ONLY on the provided context.
        Structure your response STRICTLY as a JSON object. This object MUST contain:
        1. An 'analysis' key: The value should be your answer. If the answer is naturally a list (e.g., listing items), provide it as a JSON list of strings/objects under this key. If it's naturally a dictionary, provide a JSON object. Otherwise, provide the answer as a single JSON string.
        2. A 'classification' key: The value must be one of 'Positive', 'Negative', 'Neutral', or 'Severe Negative', reflecting the sentiment/implication for the company, as a JSON string.

        Context:
        {context}

        Question:
        {query}

        Answer (JSON format ONLY, using 'analysis' and 'classification' keys):
        """
        # --- End of Standardized Prompt ---

        try:
            response = client_openai.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=OPENAI_LLM_MAX_TOKENS,
                temperature=0,
                response_format={"type": "json_object"}
            )
            answer_content_string = response.choices[0].message.content.strip()
            #print(answer_content_string)

            parsed_answer_for_validation = None
            validation_passed = False
            try:
                 parsed_answer_for_validation = json.loads(answer_content_string)
                 if isinstance(parsed_answer_for_validation, dict) and \
                    'analysis' in parsed_answer_for_validation and \
                    'classification' in parsed_answer_for_validation:
                     validation_passed = True
                 else:
                     logging.warning(f"OpenAI LLM response JSON for query '{query}' lacks required keys ('analysis', 'classification'): {answer_content_string}")
            except json.JSONDecodeError:
                 logging.warning(f"OpenAI LLM response for query '{query}' was not valid JSON: {answer_content_string}")

            return {
                "Question": query,
                "answer_json_string": answer_content_string,
                "parsed_answer_for_validation": parsed_answer_for_validation, # Store potentially invalid structure too
                "validation_passed": validation_passed, # Flag if structure is correct
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "status": "Success"
            }
        except RateLimitError:
            logging.error(f"OpenAI LLM Rate limit hit for query: {query}. Try reducing max_workers or waiting.")
            return {"Question": query, "answer_json_string": None, "validation_passed": False, "error": "RateLimitError", "total_tokens": 0, "prompt_tokens": 0, "status": "Failed"}
        except APIError as e:
             if e.status_code == 400 and 'context_length_exceeded' in str(e.code):
                 logging.error(f"OpenAI LLM API Error for query '{query}': Context length exceeded. {e}")
                 logging.error(f"Context length (chars): {len(context)}, Prompt length (chars): {len(prompt)}")
                 return {"Question": query, "answer_json_string": None, "validation_passed": False, "error": f"APIError - Context Length Exceeded: {e.code}", "total_tokens": 0, "prompt_tokens": 0, "status": "Failed"}
             else:
                logging.error(f"OpenAI LLM API Error for query '{query}': {e}")
                error_details = str(e)
                error_code = "APIError"
                if hasattr(e, 'body') and e.body and isinstance(e.body, dict):
                    error_details = e.body.get('message', error_details)
                    # Check for specific error types if available in the body
                    if 'type' in e.body and e.body['type'] == 'invalid_request_error' and 'param' in e.body:
                         error_code = f"InvalidRequestError ({e.body['param']})"
                    elif 'code' in e.body:
                         error_code = f"APIError ({e.body['code']})"

                # Handle content filter specifically
                if "content management policy" in error_details.lower() or (hasattr(e, 'code') and e.code == 'content_filter'):
                     logging.warning(f"Query '{query}' blocked by content policy. Error: {error_details}")
                     return {"Question": query, "answer_json_string": None, "validation_passed": False, "error": f"Content Policy Error: {error_details}", "total_tokens": 0, "prompt_tokens": 0, "status": "Failed"}
                else:
                     return {"Question": query, "answer_json_string": None, "validation_passed": False, "error": f"{error_code}: {error_details}", "total_tokens": 0, "prompt_tokens": 0, "status": "Failed"}

        except Exception as e:
            logging.error(f"Error processing query '{query}' with OpenAI LLM: {e}")
            return {"Question": query, "answer_json_string": None, "validation_passed": False, "error": str(e), "total_tokens": 0, "prompt_tokens": 0, "status": "Failed"}


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {executor.submit(process_single_query, query_dict): query_dict
                           for query_dict in queries_with_context}

        processed_count = 0
        for future in as_completed(future_to_query):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                original_query_dict = future_to_query[future]
                query_text = original_query_dict.get("Question", "Unknown query")
                logging.error(f"LLM Query '{query_text}' generated an unexpected exception in thread: {exc}")
                results.append({"Question": query_text, "answer_json_string": None, "validation_passed": False, "error": f"Unhandled Thread Exception: {exc}", "total_tokens": 0, "prompt_tokens": 0, "status": "Failed"})

            processed_count += 1
            logging.info(f"Processed OpenAI LLM query {processed_count}/{total_queries}...")

    end_time = time.time()
    logging.info(f"OpenAI LLM processing finished in {end_time - start_time:.2f} seconds.")
    return results


# --- NEW: Recursive HTML Formatting Function ---
def format_analysis_data_recursively(data, level=0):
    """
    Recursively formats Python data structures (strings, lists, dicts) into HTML.
    Handles nested lists and dictionaries.
    """
    indent = "    " * (level + 2) # For pretty HTML source indentation
    html_output = ""

    if isinstance(data, str):
        formatted_str = html.escape(data).replace('\n', '<br>\n' + indent)
        # Wrap top-level or long strings in <p>, otherwise just return the string
        if level == 0 or len(data) > 80:
             html_output += f"<p>{formatted_str}</p>"
        else:
             html_output += formatted_str
    elif isinstance(data, list):
        if not data:
            html_output += "<p><em>(Empty list)</em></p>"
        else:
            html_output += "<ul>\n"
            for item in data:
                formatted_item = format_analysis_data_recursively(item, level + 1)
                # Avoid adding empty list items if recursion returned nothing
                if formatted_item:
                    html_output += f"{indent}<li>{formatted_item}</li>\n"
            html_output += "    " * (level + 1) + "</ul>"
    elif isinstance(data, dict):
        if not data:
             html_output += "<p><em>(Empty dictionary)</em></p>"
        else:
            html_output += "<dl>\n"
            for key, value in data.items():
                formatted_key = html.escape(str(key))
                html_output += f"{indent}<dt><strong>{formatted_key}:</strong></dt>\n"
                formatted_value = format_analysis_data_recursively(value, level + 1)
                html_output += f"{indent}<dd>{formatted_value if formatted_value else '<em>N/A</em>'}</dd>\n" # Add fallback for empty value
            html_output += "    " * (level + 1) + "</dl>"
    elif isinstance(data, (int, float, bool)) or data is None:
        html_output += html.escape(str(data))
    else:
        logging.warning(f"Unexpected data type in analysis formatting: {type(data)}. Converting to string.")
        html_output += html.escape(str(data))

    # Return None if the result is effectively empty (e.g., empty list/dict rendered as empty tags)
    # Check against common empty representations
    stripped_output = html_output.strip()
    if not stripped_output or stripped_output in ["<ul></ul>", "<dl></dl>", "<p><em>(Empty list)</em></p>", "<p><em>(Empty dictionary)</em></p>"]:
        return None
    return stripped_output


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"--- Starting Analysis for Ticker: {symbol} using OpenAI Embeddings ({OPENAI_EMBEDDING_MODEL}) ---")
    overall_start_time = time.time()

    # Check if tiktoken loaded correctly before proceeding
    logging.error("Tiktoken failed to load. Cannot proceed with accurate tokenization. Exiting.")
        sys.stderr.write("Tiktoken failed to load. Cannot proceed with accurate tokenization. Exiting.\n")
        exit(1)

    # Check if the company is an ADR by examining the profile data
    # == Step 1: Fetch Company Profile ==
    logging.info("Fetching company profile...")
    profile_url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={financial_api_key}'
    profile_data = fetch_fmp_data(profile_url)
    isAdr = False
    if profile_data and isinstance(profile_data, list) and len(profile_data) > 0:
        company_profile = profile_data[0]
        isAdr = company_profile.get('isAdr', False)
        logging.info(f"Company ADR status: {isAdr}")
    else:
        logging.warning("Could not determine ADR status due to missing profile data. Proceeding with default (False).")
    
    # == CHECK ADDED HERE: If isAdr is True, print message and exit ==
    # For 10-K, we process 20-F for ADRs. This check is more for 10-Q.
    # However, the original code had an exit commented out.
    # For 10-K, if it's an ADR, we should fetch '20-f'.
    if isAdr:
        filing_type = '20-f'
        logging.info(f"Company is an ADR. Will look for 20-F filings instead of 10-K.")
    else:
        filing_type= '10-k'
        
    logging.info(f"Searching for latest '{filing_type}' filing...")
    filings_url = f'https://financialmodelingprep.com/api/v3/sec_filings/{symbol}'
    params = {
        "type": filing_type,
        "page": 0, 
        "apikey": financial_api_key
    }
    filings_data = fetch_fmp_data(filings_url, params=params)
    final_link = None
    filing_date = "unknown_date" 
    if filings_data and isinstance(filings_data, list) and len(filings_data) > 0:
         first_filing = filings_data[0]
         if isinstance(first_filing, dict):
            final_link = first_filing.get("finalLink")
            filing_date = first_filing.get("fillingDate", "unknown_date").split(" ")[0]
            if final_link:
                logging.info(f"Found filing link for date {filing_date}: {final_link}")
            else:
                logging.error("No 'finalLink' found in the latest filing data. Exiting.")
                sys.stderr.write("No 'finalLink' found in the latest filing data. Exiting.\n")
                exit(1)
         else:
             logging.error(f"First filing data item is not a dictionary: {first_filing}. Exiting.")
             sys.stderr.write(f"First filing data item is not a dictionary: {first_filing}. Exiting.\n")
             exit(1)
    else:
        logging.error(f"No {filing_type} filings found for {symbol}. Response: {filings_data}. Exiting.")
        sys.stderr.write(f"No {filing_type} filings found for {symbol}. Response: {filings_data}. Exiting.\n")
        exit(1)


    # == Step 3: Process Document (Download, Extract, Embed, Store) ==
    safe_model_name = OPENAI_EMBEDDING_MODEL.replace("/", "_").replace("-", "_")
    base_filename = f"{symbol}_{filing_type}_{filing_date}_{safe_model_name}"
    output_dir = "test_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    embeddings_pickle_file = os.path.join(output_dir, f"{base_filename}_texts.pkl")
    faiss_index_file = os.path.join(output_dir, f"{base_filename}_faiss.index")


    index = None
    texts = []
    force_reprocess = False 

    if not force_reprocess and os.path.exists(faiss_index_file) and os.path.exists(embeddings_pickle_file):
        logging.info("Found existing embeddings files. Loading them...")
        index, texts = load_embeddings_and_texts(faiss_index_file, embeddings_pickle_file)
    else:
        if force_reprocess:
             logging.warning("Forcing document reprocessing...")
        else:
            logging.info(f"Embeddings files not found or incomplete. Processing document '{final_link}' from scratch...")

        logging.info("Downloading HTML content...")
        html_content_text = None # Renamed to avoid clash with html module
        try:
            headers = {"User-Agent": USER_AGENT}
            html_response = requests.get(final_link, headers=headers, timeout=REQUESTS_TIMEOUT)
            html_response.raise_for_status()
            html_response.encoding = html_response.apparent_encoding if html_response.apparent_encoding else 'utf-8'
            html_content_text = html_response.text
            logging.info(f"HTML content downloaded successfully ({len(html_content_text)} bytes). Encoding: {html_response.encoding}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download HTML content from {final_link}: {e}. Exiting.")
            sys.stderr.write(f"Failed to download HTML content from {final_link}: {e}. Exiting.\n")
            exit(1)

        if html_content_text:
            chunks = extract_text_and_chunk_by_tokens(html_content_text, token_limit=EMBEDDING_MODEL_TOKEN_LIMIT)

            if chunks:
                embedding_dimension_to_use = OPENAI_EMBEDDING_DIMENSION # Default, will be updated if loaded from index
                if index and hasattr(index, 'd'): # Check if index was loaded and has dimension
                    embedding_dimension_to_use = index.d

                embeddings_list = generate_openai_embeddings(chunks, OPENAI_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE)

                if embeddings_list:
                    # Pass the dimension confirmed/used by generation if first time, or actual dim from embedding
                    actual_embedding_dim = len(embeddings_list[0]['embedding'])
                    index, texts = store_embeddings(embeddings_list, actual_embedding_dim,
                                                    faiss_index_file, embeddings_pickle_file)
                else:
                    logging.error("OpenAI Embedding generation failed or produced no embeddings. Cannot proceed.")
                    sys.stderr.write("OpenAI Embedding generation failed or produced no embeddings. Cannot proceed.\n")
                    exit(1)
            else:
                logging.error("Text extraction and chunking failed. Cannot proceed.")
                sys.stderr.write("Text extraction and chunking failed. Cannot proceed.\n")
                exit(1)
        else:
             logging.error("HTML content is empty after download. Cannot proceed.")
             sys.stderr.write("HTML content is empty after download. Cannot proceed.\n")
             exit(1)


    if index is None or not texts:
        logging.error("Failed to load or generate embeddings and texts. Exiting.")
        sys.stderr.write("Failed to load or generate embeddings and texts. Exiting.\n")
        exit(1)

    # == Step 4: Define Queries and Search Context ==
    queries = [
        # General Insights
        "What are the main growth opportunities highlighted in the filing?",
        "Summarize the company's competitive advantages mentioned.",
        "What are the key risks mentioned in the filing?",
        "What are the company's stated strategic priorities and future plans?",
        # Business Overview (Item 1)
        "Provide a summary of the company’s business model, operations, and primary revenue sources.",
        "What markets or geographies does the company primarily operate in?",
        "What new products, services, or initiatives are mentioned in the business overview?",
        # Risk Factors (Item 1A)
        "Summarize the main risk factors described in the filing.",
        "Highlight any emerging risks or industry-specific concerns noted in the document.",
        # Legal Proceedings (Item 3)
        "What legal proceedings are disclosed in the filing? Are there any significant lawsuits or regulatory actions mentioned?",
        # Management's Discussion and Analysis (MD&A - Item 7)
        "What insights can be drawn from the Management’s Discussion and Analysis (MD&A)?",
        "What are the key financial trends and operational highlights noted by management?",
        "How has the company performed compared to its stated goals or projections?",
        "What concerns or challenges does management foresee for the next fiscal year?",
        # Financial Statements and Supplementary Data (Item 8)
        "Provide a summary of the financial performance metrics and key takeaways from the filing's financial statements.",
        "What changes are noticeable in revenue, profitability, or cash flows compared to prior years?",
        "Are there any unusual or noteworthy accounting items mentioned in the financial statements?",
        # Notes to Financial Statements
        "Summarize the significant accounting policies and estimates used by the company.",
        "Highlight any major changes in accounting methods or adjustments disclosed in the notes.",
        "What contingent liabilities or off-balance-sheet arrangements are disclosed in the notes?",
        # Executive Compensation (Item 11)
        "What details are provided about executive compensation structures and performance incentives?",
        "Are there any controversies or shareholder concerns related to executive pay?",
        # Corporate Governance (Item 12)
        "What insights can be gathered about the company’s governance practices and board composition?",
        "Are there any notable shareholder proposals or governance concerns mentioned?",
        # Forward-Looking Statements
        "What forward-looking statements are included in the filing, and what assumptions do they rely on?",
        "Are there any specific warnings or disclaimers about forward-looking information?"
    ]


    search_results = search_faiss(queries, index, texts, OPENAI_EMBEDDING_MODEL, FAISS_SEARCH_RESULTS)

    query_contexts_for_llm = []
    for query, contexts in search_results.items():
        if contexts:
            combined_context = "\n\n---\n\n".join(contexts)
            query_contexts_for_llm.append({"Question": query, "context": combined_context})
        else:
            logging.warning(f"No context found for query: '{query}'. Skipping LLM call for this query.")

    if not query_contexts_for_llm:
         logging.error("No queries have sufficient context after search. Cannot proceed with LLM analysis.")
         sys.stderr.write("No queries have sufficient context after search. Cannot proceed with LLM analysis.\n")
         exit(1)

    llm_results_list = ask_llm_batch(query_contexts_for_llm, OPENAI_LLM_MAX_WORKERS)
    
    logging.info("Re-ordering LLM results to match original query sequence...")
    query_order_map = {query: index for index, query in enumerate(queries)}
    
    def sort_key(result_item):
        question = result_item.get("Question")
        return query_order_map.get(question, float('inf'))
    
    llm_results_list.sort(key=sort_key)
    logging.info("LLM results re-ordered successfully.")
    
    logging.info("Transforming LLM results list into a dictionary...")
    llm_results_dict = {}
    llm_results_dict['filing_link'] = final_link
    llm_results_dict['company_symbol'] = symbol
    llm_results_dict['filing_date'] = filing_date
    llm_results_dict['analysis_results'] = llm_results_list 

    json_output_file = os.path.join(output_dir, f"{symbol}_{filing_date}_{filing_type}.json") # Added filing_type for uniqueness
    logging.info(f"Saving detailed LLM analysis results to {json_output_file}...")
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(llm_results_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"LLM results dictionary saved successfully to {json_output_file}")
    except IOError as e:
        logging.error(f"Failed to write results dictionary to JSON file {json_output_file}: {e}")
    except TypeError as e:
        logging.error(f"Failed to serialize results dictionary to JSON: {e}")


    output_file = os.path.join(output_dir, f"{symbol}_{filing_date}_{filing_type}.html") # Added filing_type for uniqueness
    logging.info(f"Generating HTML analysis report to {output_file}...")

    html_content_head = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Analysis Report - {symbol} ({filing_type_upper})</title>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 1000px; margin: auto; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
            .meta-info {{ background-color: #f8f9f9; border: 1px solid #e1e5e8; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .meta-info p {{ margin: 5px 0; }}
            .analysis-item {{ border: 1px solid #ddd; margin-bottom: 20px; padding: 15px; border-radius: 5px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .analysis-item h3 {{ margin-top: 0; color: #2980b9; }}
            .analysis-item .status-failed {{ color: #c0392b; font-weight: bold; }}
            .analysis-item .status-warning {{ color: #f39c12; font-weight: bold; }}
            .analysis-item .error-message {{ background-color: #fcebea; border: 1px solid #e74c3c; color: #c0392b; padding: 10px; margin-top: 10px; border-radius: 3px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word;}}
            .analysis-item .warning-message {{ background-color: #fff9e6; border: 1px solid #f39c12; color: #b07d2a; padding: 10px; margin-top: 10px; border-radius: 3px; font-size: 0.9em; white-space: pre-wrap; word-wrap: break-word;}}
            .classification {{ font-weight: bold; padding: 3px 8px; border-radius: 4px; color: white; display: inline-block; margin-left: 10px; font-size: 0.9em;}}
            .classification-positive {{ background-color: #27ae60; }}
            .classification-negative {{ background-color: #e74c3c; }}
            .classification-severe-negative {{ background-color: #c0392b; }}
            .classification-neutral {{ background-color: #7f8c8d; }}
            .classification-unknown, .classification-classification-not-found,
            .classification-parsing-error, .classification-processing-error,
            .classification-missing-content, .classification-invalid-structure, 
            .classification-missing-analysis {{ background-color: #f39c12; color: #333; }}
            .analysis-text {{ margin-top: 10px; background-color: #fdfefe; padding: 10px; border-left: 3px solid #3498db; }}
            .analysis-text p {{ margin: 5px 0 10px 0; }}
            .analysis-text ul {{ margin: 10px 0 10px 0; padding-left: 20px; list-style-type: disc; }}
            .analysis-text li {{ margin-bottom: 5px; }}
            .analysis-text dl {{ margin: 10px 0 10px 0; }}
            .analysis-text dt {{ font-weight: bold; margin-top: 8px; }}
            .analysis-text dd {{ margin-left: 20px; margin-bottom: 8px; }}
        </style>
    </head>
    <body>
        <h1>Financial Analysis Report: {symbol} ({filing_type_upper})</h1>
        <div class="meta-info">
            <p><strong>Company Ticker:</strong> {symbol}</p>
            <p><strong>Filing Type:</strong> {filing_type_upper}</p>
            <p><strong>Filing Date:</strong> {filing_date_val}</p>
            <p><strong>Source Document:</strong> <a href="{filing_link_val}" target="_blank" rel="noopener noreferrer">Link to Filing</a></p>
        </div>

        <h2>Analysis Results</h2>
    """.format(
        symbol=html.escape(llm_results_dict.get('company_symbol', 'N/A')),
        filing_type_upper=html.escape(filing_type.upper()),
        filing_date_val=html.escape(llm_results_dict.get('filing_date', 'N/A')),
        filing_link_val=html.escape(llm_results_dict.get('filing_link') or '#')
    )

    html_body_content = "" 
    for result in llm_results_dict.get('analysis_results', []):
        question = result.get("Question", "Unknown Question")
        status = result.get("status", "Unknown Status")
        validation_passed = result.get("validation_passed", False)

        html_body_content += '<div class="analysis-item">\n'
        html_body_content += f'    <h3>{html.escape(question)}</h3>\n'

        if status == "Success":
            answer_json_string = result.get("answer_json_string")
            classification = "Unknown"
            classification_css_class = "classification-unknown"
            analysis_html_output = "<p><em>Analysis data not available or could not be parsed.</em></p>"
            warning_message = None

            if not validation_passed:
                logging.warning(f"LLM response for '{question}' passed status 'Success' but failed structure validation (missing 'analysis' or 'classification'). Raw: {str(answer_json_string)[:200]}...")
                warning_message = "LLM response structure was invalid (missing 'analysis' or 'classification' key)."
                classification = "Invalid Structure" 
                classification_css_class = "classification-invalid-structure"
                if answer_json_string:
                    analysis_html_output = f"<p>Raw LLM response:</p><pre><code>{html.escape(str(answer_json_string))}</code></pre>"
                else:
                    analysis_html_output = "<p><em>No answer content received from LLM.</em></p>"

            elif answer_json_string: 
                try:
                    parsed_answer = json.loads(answer_json_string) 
                    analysis_value = parsed_answer.get("analysis")
                    classification = parsed_answer.get("classification", "Classification not found")

                    safe_classification = html.escape(str(classification)).lower().replace(" ", "-")
                    classification_css_class = f"classification-{safe_classification}"

                    if analysis_value is not None:
                        formatted_html = format_analysis_data_recursively(analysis_value)
                        if formatted_html:
                             analysis_html_output = formatted_html
                        else:
                             analysis_html_output = "<p><em>Analysis field present but contains no displayable content after formatting.</em></p>"
                             warning_message = "Analysis field was present but resulted in empty content (e.g., empty list/dict)."
                    else:
                         analysis_html_output = "<p><em>'analysis' field was null in the JSON response.</em></p>"
                         warning_message = "'analysis' key was present but its value was null."
                         if classification == "Classification not found": 
                              classification = "Missing Analysis"
                              classification_css_class = "classification-missing-analysis"

                except json.JSONDecodeError: 
                    logging.error(f"JSONDecodeError during final HTML generation for '{question}' despite initial validation! Raw: {str(answer_json_string)[:200]}...")
                    analysis_html_output = f"<p><strong><em>Could not parse LLM JSON response during HTML generation.</em></strong></p><p>Raw response (truncated):<br><pre><code>{html.escape(str(answer_json_string)[:300])}...</code></pre></p>"
                    classification = "Parsing Error"
                    classification_css_class = "classification-parsing-error"
                except Exception as e: 
                    logging.error(f"Error formatting analysis data for '{question}': {e}", exc_info=True)
                    analysis_html_output = f"<p><strong><em>Error formatting analysis data for HTML:</strong> {html.escape(str(e))}</em></p>"
                    classification = "Processing Error"
                    classification_css_class = "classification-processing-error"
            else: 
                 analysis_html_output = "<p><em>No answer content received from LLM (empty JSON string).</em></p>"
                 classification = "Missing Content"
                 classification_css_class = "classification-missing-content"

            status_html = 'Success'
            if warning_message:
                status_html = '<span class="status-warning">Success (with warnings)</span>'

            html_body_content += f'    <p><span class="classification {classification_css_class}">{html.escape(str(classification))}</span></p>\n'

            if warning_message:
                 html_body_content += f'    <div class="warning-message">{html.escape(warning_message)}</div>\n'

            html_body_content += f'    <div class="analysis-text">\n{analysis_html_output}\n    </div>\n'

        else: 
            error_message = result.get("error", "No error details provided.")
            html_body_content += f'    <p><strong class="status-failed">Status: Failed</strong></p>\n'
            html_body_content += f'    <div class="error-message"><pre>{html.escape(str(error_message))}</pre></div>\n'

        html_body_content += '</div>\n'

    html_content_foot = """
    </body>
    </html>
    """

    final_html_content = html_content_head + html_body_content + html_content_foot

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_html_content)
        logging.info(f"HTML report saved successfully to {output_file}")
    except IOError as e:
        logging.error(f"Failed to write results to HTML file {output_file}: {e}")

    logging.info("\n--- Analysis Summary ---")
    total_llm_tokens_used = 0
    successful_queries = 0
    failed_queries = 0
    validation_failures = 0

    for result in llm_results_list:
        if result.get("status") == "Success":
            successful_queries += 1
            total_llm_tokens_used += result.get("total_tokens", 0)
            if not result.get("validation_passed", False):
                validation_failures += 1
                logging.warning(f"LLM Query SUCCEEDED but FAILED VALIDATION: {result.get('Question', 'Unknown Query')} - Check LLM output format.")
        else:
            failed_queries += 1
            logging.warning(f"LLM Query FAILED: {result.get('Question', 'Unknown Query')} - Error: {result.get('error', 'Unknown')}")

    logging.info(f"\nProcessed {successful_queries + failed_queries} LLM queries.")
    logging.info(f"- {successful_queries} succeeded ({validation_failures} had validation issues).")
    if failed_queries > 0:
        logging.warning(f"- {failed_queries} failed.")
    logging.info(f"Total OpenAI LLM tokens used (approximate): {total_llm_tokens_used}")

    overall_end_time = time.time()
    logging.info(f"--- Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds ---")
