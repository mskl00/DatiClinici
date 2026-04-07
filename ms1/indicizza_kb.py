"""
Script di indicizzazione della Knowledge Base su Azure AI Search V1.

Supporta due tipi di sorgente:
  1. File Excel (.xlsx / .xls)  — struttura tabellare (come mostrato nell'immagine)
  2. File PDF tabellari         — estrazione SOLO delle colonne:
     PATOLOGIA, LIVELLO, PRESTAZ_AMB_V_DESC, TIPOLOGIA PRESTAZIONE

Come usarlo:
    python indicizza_kb.py --excel kb_patologie.xlsx
    python indicizza_kb.py --pdf   linee_guida.pdf
    python indicizza_kb.py --excel kb_patologie.xlsx --pdf linee_guida.pdf
    python indicizza_kb.py --clear   # cancella e ricrea l'indice

Variabili d'ambiente richieste:
    AZURE_SEARCH_ENDPOINT
    AZURE_SEARCH_KEY
    AZURE_SEARCH_INDEX_NAME      (default: kb-monitoraggio)
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_KEY
    AZURE_OPENAI_API_VERSION     (default: 2024-02-01)
    AZURE_OPENAI_EMBED_MODEL     (default: text-embedding-3-small)
"""

import os
import sys
import uuid
import logging
import argparse
import time
import re
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------------------------------------------------------
# Dipendenze — installare con:
#   pip install azure-search-documents openai pandas openpyxl PyMuPDF
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    logging.warning("⚠️ pandas non installato: pip install pandas openpyxl")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF non installato: il parsing PDF non sarà disponibile (pip install PyMuPDF)")

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AzureOpenAI = None
    OPENAI_AVAILABLE = False
    logging.warning("⚠️ openai non installato: pip install openai")

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile,
        SemanticConfiguration,
        SemanticSearch,
        SemanticPrioritizedFields,
        SemanticField,
    )
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False
    logging.warning("⚠️ azure-search-documents non installato: pip install azure-search-documents")


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
SEARCH_KEY      = os.environ.get("AZURE_SEARCH_KEY", "")
INDEX_NAME      = os.environ.get("AZURE_SEARCH_INDEX_NAME", "kb-monitoraggio")

OPENAI_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
OPENAI_KEY         = os.environ.get("AZURE_OPENAI_KEY", "")
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBED_MODEL        = os.environ.get("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS   = 1536  # text-embedding-3-small → 1536, ada-002 → 1536

BATCH_SIZE        = 50    # Documenti per batch di upload
EMBED_BATCH_SIZE  = 32    # Testi per chiamata embedding (riduce numero API call)
CHUNK_SIZE        = 500   # Caratteri per chunk (PDF)
CHUNK_OVERLAP     = 100   # Overlap tra chunk consecutivi (PDF)


# =============================================================================
# CLIENT
# =============================================================================

def _get_openai_client():
    """
    Restituisce un client AzureOpenAI autenticato.
    Se AZURE_OPENAI_KEY è presente usa API Key (legacy),
    altrimenti usa Managed Identity (DefaultAzureCredential).
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai non installato. Aggiungi 'openai' a requirements.txt.")

    if OPENAI_KEY:
        # Legacy: API Key
        return AzureOpenAI(
            azure_endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_KEY,
            api_version=OPENAI_API_VERSION
        )

    # Managed Identity
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError:
        raise ImportError(
            "AZURE_OPENAI_KEY non configurata e azure-identity non installato. "
            "Aggiungi azure-identity a requirements.txt oppure imposta AZURE_OPENAI_KEY."
        )
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    logging.info("🔐 OpenAI client via Managed Identity")
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=OPENAI_API_VERSION
    )


def _get_search_client():
    """
    Restituisce un SearchClient autenticato.
    Se AZURE_SEARCH_KEY è presente usa API Key (legacy),
    altrimenti usa Managed Identity (DefaultAzureCredential).
    """
    if not AZURE_SEARCH_AVAILABLE:
        raise ImportError("azure-search-documents non installato.")

    if SEARCH_KEY:
        credential = AzureKeyCredential(SEARCH_KEY)
    else:
        try:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            logging.info("🔐 SearchClient via Managed Identity")
        except ImportError:
            raise ImportError(
                "AZURE_SEARCH_KEY non configurata e azure-identity non installato. "
                "Aggiungi azure-identity a requirements.txt oppure imposta AZURE_SEARCH_KEY."
            )

    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential
    )


def _get_index_client():
    """
    Restituisce un SearchIndexClient autenticato.
    Se AZURE_SEARCH_KEY è presente usa API Key (legacy),
    altrimenti usa Managed Identity (DefaultAzureCredential).
    """
    if not AZURE_SEARCH_AVAILABLE:
        raise ImportError("azure-search-documents non installato.")

    if SEARCH_KEY:
        credential = AzureKeyCredential(SEARCH_KEY)
    else:
        try:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            logging.info("🔐 SearchIndexClient via Managed Identity")
        except ImportError:
            raise ImportError(
                "AZURE_SEARCH_KEY non configurata e azure-identity non installato. "
                "Aggiungi azure-identity a requirements.txt oppure imposta AZURE_SEARCH_KEY."
            )

    return SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=credential
    )


# =============================================================================
# GESTIONE INDICE
# =============================================================================

def crea_o_aggiorna_indice():
    """
    Crea (o aggiorna se già esiste) l'indice Azure AI Search.

    Schema dell'indice (minimale):
        id                  — chiave univoca tecnica
        patologia           — chiave logica di ricerca
        livello             — livello di complessità (1/2/3)
        prestaz_amb_v_desc  — descrizione prestazione
        tipologia_prestazione — tipologia prestazione (LABORATORIO / AMBULATORIALE)
        chunk_testo         — testo usato per embedding
        embedding           — vettore 1536 dimensioni
    """
    index_client = _get_index_client()

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SimpleField(
            name="livello",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SearchableField(
            name="patologia",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            analyzer_name="it.microsoft"  # analizzatore italiano
        ),
        SearchableField(
            name="prestaz_amb_v_desc",
            type=SearchFieldDataType.String,
            analyzer_name="it.microsoft"
        ),
        SearchableField(
            name="tipologia_prestazione",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            analyzer_name="it.microsoft"
        ),
        SearchableField(
            name="chunk_testo",
            type=SearchFieldDataType.String,
            analyzer_name="it.microsoft"
        ),
        SimpleField(
            name="source_document_name",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIMENSIONS,
            vector_search_profile_name="myHnswProfile"
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw"
            )
        ]
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="patologia"),
                    content_fields=[
                        SemanticField(field_name="prestaz_amb_v_desc"),
                        SemanticField(field_name="chunk_testo")
                    ]
                )
            )
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    try:
        index_client.create_or_update_index(index)
        logging.info(f"✅ Indice '{INDEX_NAME}' creato/aggiornato con successo.")
    except Exception as e:
        logging.error(f"❌ Errore creazione indice: {e}")
        raise


def cancella_indice():
    """Cancella l'indice (utile per ricrearlo da zero)."""
    index_client = _get_index_client()
    try:
        index_client.delete_index(INDEX_NAME)
        logging.info(f"🗑️  Indice '{INDEX_NAME}' cancellato.")
    except Exception as e:
        logging.warning(f"⚠️ Errore cancellazione indice (potrebbe non esistere): {e}")


# =============================================================================
# GENERAZIONE EMBEDDING
# =============================================================================

def genera_embedding(testo: str, openai_client: AzureOpenAI) -> List[float]:
    """
    Genera il vettore embedding per il testo fornito.

    Args:
        testo: Testo da vettorizzare
        openai_client: Client Azure OpenAI

    Returns:
        Lista di float (dimensione = EMBED_DIMENSIONS)
    """
    testo = testo.strip()
    if not testo:
        return [0.0] * EMBED_DIMENSIONS

    # Ritenta in caso di throttling
    for tentativo in range(3):
        try:
            response = openai_client.embeddings.create(
                input=testo,
                model=EMBED_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            if tentativo < 2:
                wait = (tentativo + 1) * 5
                logging.warning(f"⚠️ Errore embedding (tentativo {tentativo+1}), attendo {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def genera_embeddings_batch(testi: List[str], openai_client: AzureOpenAI) -> List[List[float]]:
    """
    Genera embedding per una lista di testi in batch.

    Questo approccio riduce drasticamente il numero di chiamate HTTP verso
    Azure OpenAI e velocizza l'indicizzazione in Azure Functions, mitigando
    timeout su documenti voluminosi.

    Args:
        testi: Lista di testi da vettorizzare
        openai_client: Client Azure OpenAI

    Returns:
        Lista di embedding nello stesso ordine dei testi in ingresso.
    """
    if not testi:
        return []

    embeddings: List[List[float]] = []

    for i in range(0, len(testi), EMBED_BATCH_SIZE):
        batch_originale = testi[i:i + EMBED_BATCH_SIZE]
        # Azure OpenAI richiede stringhe non vuote.
        batch_normalizzato = [t.strip() if t and t.strip() else "[vuoto]" for t in batch_originale]

        for tentativo in range(3):
            try:
                response = openai_client.embeddings.create(
                    input=batch_normalizzato,
                    model=EMBED_MODEL
                )
                batch_embeddings = [item.embedding for item in response.data]
                if len(batch_embeddings) != len(batch_originale):
                    raise RuntimeError(
                        "Numero embedding restituiti non coerente con il batch richiesto"
                    )
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if tentativo < 2:
                    wait = (tentativo + 1) * 5
                    logging.warning(
                        f"⚠️ Errore embedding batch {i // EMBED_BATCH_SIZE + 1} "
                        f"(tentativo {tentativo+1}), attendo {wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    raise

    return embeddings


# =============================================================================
# UPLOAD DOCUMENTI
# =============================================================================

def carica_documenti(docs: List[Dict[str, Any]], search_client: SearchClient):
    """
    Carica i documenti nell'indice Azure AI Search a batch.

    Args:
        docs: Lista di documenti da caricare
        search_client: Client Azure AI Search
    """
    totale = len(docs)
    caricati = 0

    for i in range(0, totale, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        try:
            result = search_client.upload_documents(documents=batch)
            ok = sum(1 for r in result if r.succeeded)
            ko = len(batch) - ok
            caricati += ok
            if ko > 0:
                logging.warning(f"⚠️ Batch {i//BATCH_SIZE + 1}: {ok} ok, {ko} falliti")
            else:
                logging.info(f"✅ Batch {i//BATCH_SIZE + 1}/{(totale + BATCH_SIZE - 1)//BATCH_SIZE}: {ok} documenti caricati")
        except Exception as e:
            logging.error(f"❌ Errore upload batch {i//BATCH_SIZE + 1}: {e}")

    logging.info(f"📊 Totale caricati: {caricati}/{totale}")


# =============================================================================
# INDICIZZAZIONE DA EXCEL
# =============================================================================

def indicizza_excel(percorso_file: str, source_document_name: Optional[str] = None):
    """
    Legge un file Excel con la struttura mostrata (Tipologia, ID_PATOLOGIA, PATOLOGIA,
    LIVELLO, PRESTAZ_AMB_V_ID, PRESTAZ_AMB_V_DESC, Pop. riferimento, Usufruenti, %)
    e crea un documento nell'indice per ogni riga.

    La colonna PATOLOGIA viene usata come chiave di raggruppamento per la ricerca.
    Ogni riga del foglio diventa un chunk separato con il proprio embedding.

    Args:
        percorso_file: Percorso al file Excel
    """
    logging.info(f"📂 Inizio indicizzazione Excel: {percorso_file}")
    nome_file = source_document_name or os.path.basename(percorso_file)

    openai_client = _get_openai_client()
    search_client = _get_search_client()

    # Lettura Excel
    try:
        df = pd.read_excel(percorso_file)
        logging.info(f"📊 Lette {len(df)} righe dal file Excel")
        logging.info(f"   Colonne: {list(df.columns)}")
    except Exception as e:
        logging.error(f"❌ Errore lettura Excel: {e}")
        raise

    # Normalizzazione nomi colonne
    # Mappa i nomi attesi → nomi effettivi nel file Excel
    # ADATTARE se il file ha nomi colonne diversi
    mappa_colonne = {
        "Tipologia prestazione": "tipologia",
        "ID_PATOLOGIA":          "id_patologia",
        "PATOLOGIA":             "patologia",
        "LIVELLO":               "livello",
        "PRESTAZ_AMB_V_ID":      "prestaz_id",
        "PRESTAZ_AMB_V_DESC":    "prestaz_desc",
        "Pop. di riferimento 2023":                          "pop_riferimento",
        "Numero usufruenti 2023":                            "num_usufruenti",
        "% Usufruenti su popolazione di riferimento 2023":   "percentuale_utilizzo",
    }

    df = df.rename(columns=mappa_colonne)

    # Rimuovi righe con dati essenziali mancanti
    df = df.dropna(subset=["patologia", "prestaz_desc"])
    logging.info(f"📊 Righe valide dopo pulizia: {len(df)}")

    # Mappatura testo → numero per la colonna LIVELLO
    # (alcuni file usano "PRIMO"/"SECONDO"/"TERZO" invece di 1/2/3)
    LIVELLO_MAP = {
        "primo": 1, "1": 1, "1.0": 1,
        "secondo": 2, "2": 2, "2.0": 2,
        "terzo": 3, "3": 3, "3.0": 3,
    }

    def _parse_livello(val) -> Optional[int]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return LIVELLO_MAP.get(str(val).strip().lower())

    rows_data = []
    testi_da_embeddare: List[str] = []

    for _, row in df.iterrows():
        # Costruisci il testo del chunk: testo ricco e specifico
        chunk = (
            f"Patologia: {row.get('patologia', '')}. "
            f"Tipologia: {row.get('tipologia', '')}. "
            f"Parametro/Prestazione: {row.get('prestaz_desc', '')} "
            f"(ID: {row.get('prestaz_id', '')}). "
            f"Percentuale di utilizzo nella popolazione di riferimento: "
            f"{row.get('percentuale_utilizzo', 'N/D')}%. "
            f"Anno di riferimento: 2023."
        )

        rows_data.append(row)
        testi_da_embeddare.append(chunk)

    try:
        embeddings = genera_embeddings_batch(testi_da_embeddare, openai_client)
    except Exception as e:
        logging.error(f"❌ Errore durante generazione embedding batch Excel: {e}")
        raise

    docs = []
    for row, chunk, embedding in zip(rows_data, testi_da_embeddare, embeddings):
        docs.append({
            "id":                 str(uuid.uuid4()),
            "livello":            _parse_livello(row.get("livello")),
            "patologia":          str(row.get("patologia", "")),
            "prestaz_amb_v_desc": str(row.get("prestaz_desc", "")),
            "tipologia_prestazione": str(row.get("tipologia", "")).strip(),
            "chunk_testo":        chunk,
            "source_document_name": nome_file,
            "embedding":          embedding,
        })

    logging.info(f"📄 Documenti Excel preparati: {len(docs)}")
    carica_documenti(docs, search_client)
    logging.info(f"✅ Indicizzazione Excel completata: {percorso_file}")


# =============================================================================
# INDICIZZAZIONE DA PDF
# =============================================================================

def _estrai_testo_pdf(percorso_file: str) -> str:
    """Estrae il testo grezzo da un PDF (usato come fallback)."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF non installato: pip install PyMuPDF")
    doc = fitz.open(percorso_file)
    testo_completo = []
    for num_pagina, pagina in enumerate(doc, 1):
        testo = pagina.get_text("text")
        if testo.strip():
            testo_completo.append(f"[Pagina {num_pagina}]\n{testo}")
    doc.close()
    return "\n\n".join(testo_completo)


def _estrai_righe_tabella_pdf_nativa(percorso_file: str) -> List[Dict[str, Any]]:
    """
    Estrae righe tabellari dal PDF usando PyMuPDF find_tables() — approccio nativo
    che rileva le tabelle visive (con bordi) indipendentemente dal layout del testo.

    Supporta il formato del documento regionale con colonne:
        Tipologia prestazione | ID_PATOLOGIA | PATOLOGIA | LIVELLO |
        PRESTAZ_AMB_V_ID | PRESTAZ_AMB_V_DESC |
        Pop. di riferimento | Numero usufruenti | % Usufruenti

    Returns:
        Lista di dizionari con chiavi: patologia, livello, prestaz_desc,
        tipologia_prestazione, id_patologia, percentuale_utilizzo
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF non installato: pip install PyMuPDF")

    ALIAS_PATOLOGIA = {"patologia"}
    ALIAS_LIVELLO   = {"livello"}
    ALIAS_DESC      = {"prestaz_amb_v_desc", "prestaz_amb_v desc", "descrizione"}
    ALIAS_TIPOLOGIA = {"tipologia prestazione", "tipologia", "tipo prestazione"}
    ALIAS_ID_PAT    = {"id_patologia", "id patologia", "id_pat"}
    ALIAS_PERC      = {"% usufruenti su popolazione di riferimento 2023",
                       "% usufruenti", "percentuale", "usufruenti %"}

    def _trova_col(cols, alias_set, exclude_set=None):
        """Trova la colonna migliore per un insieme di alias.

        Priorità:
        1) match esatto;
        2) match per token (es. "prestaz amb v desc" ~ "prestaz_amb_v_desc");
        3) fallback per sottostringa, evitando colonne escluse.
        """
        def _norm(v: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(v).strip().lower()).strip()

        exclude_set = {_norm(e) for e in (exclude_set or set())}
        alias_norm = {_norm(a) for a in alias_set}

        # 1) esatto
        for c in cols:
            c_norm = _norm(c)
            if c_norm in alias_norm and c_norm not in exclude_set:
                return c

        # 2) token set uguale
        for c in cols:
            c_norm = _norm(c)
            if c_norm in exclude_set:
                continue
            c_tokens = set(c_norm.split())
            for a in alias_norm:
                if c_tokens == set(a.split()):
                    return c

        # 3) sottostringa (più permissivo)
        for c in cols:
            c_norm = _norm(c)
            if c_norm in exclude_set:
                continue
            for a in alias_norm:
                if a and (a in c_norm or c_norm in a):
                    return c
        return None

    doc = fitz.open(percorso_file)
    rows_totali: List[Dict[str, Any]] = []

    for num_pagina, pagina in enumerate(doc, 1):
        try:
            tabs = pagina.find_tables()
        except Exception:
            continue

        for tab in tabs:
            try:
                df_raw = tab.to_pandas()
            except Exception:
                continue

            if df_raw.empty or len(df_raw.columns) < 3:
                continue

            # Normalizza nomi colonne
            df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]

            # Se la prima riga sembra un'intestazione, usala come header
            prima_riga = [str(v).strip().lower() for v in df_raw.iloc[0]]
            if any(a in " ".join(prima_riga) for a in ("patologia", "livello", "prestaz")):
                df_raw.columns = prima_riga
                df_raw = df_raw.iloc[1:].reset_index(drop=True)

            cols = list(df_raw.columns)
            col_patologia = _trova_col(cols, ALIAS_PATOLOGIA, exclude_set=ALIAS_ID_PAT)
            col_livello   = _trova_col(cols, ALIAS_LIVELLO)
            col_desc      = _trova_col(cols, ALIAS_DESC)
            col_tipologia = _trova_col(cols, ALIAS_TIPOLOGIA)
            col_id_pat    = _trova_col(cols, ALIAS_ID_PAT)
            col_perc      = _trova_col(cols, ALIAS_PERC)

            if not col_patologia or not col_livello or not col_desc:
                logging.debug(
                    f"[PDF p.{num_pagina}] Colonne chiave non trovate. "
                    f"Colonne presenti: {cols}"
                )
                continue

            logging.info(
                f"[PDF p.{num_pagina}] Tabella rilevata — "
                f"PATOLOGIA='{col_patologia}', LIVELLO='{col_livello}', "
                f"DESC='{col_desc}', TIPOLOGIA='{col_tipologia}', righe={len(df_raw)}"
            )

            for _, riga in df_raw.iterrows():
                patologia = str(riga.get(col_patologia, "")).strip()
                livello_s = str(riga.get(col_livello,   "")).strip()
                desc      = str(riga.get(col_desc,      "")).strip()
                tipologia = str(riga.get(col_tipologia, "")).strip() if col_tipologia else ""
                id_pat    = str(riga.get(col_id_pat,    "")).strip() if col_id_pat else ""
                perc_s    = str(riga.get(col_perc,      "")).strip() if col_perc  else ""

                if not patologia or not desc:
                    continue
                if patologia.upper() in ("PATOLOGIA", "NAN", "NONE", ""):
                    continue
                if not re.fullmatch(r"[123]", livello_s):
                    continue

                perc = None
                if perc_s:
                    try:
                        perc = float(perc_s.replace(",", ".").replace("%", "").strip())
                    except ValueError:
                        pass

                rows_totali.append({
                    "patologia":            patologia,
                    "livello":              int(livello_s),
                    "prestaz_desc":         desc,
                    "tipologia_prestazione": tipologia,
                    "id_patologia":         id_pat or f"pdf_{patologia[:20]}",
                    "percentuale_utilizzo": perc,
                })

    doc.close()
    logging.info(f"[PDF] Totale righe estratte con metodo nativo: {len(rows_totali)}")
    return rows_totali


def _split_in_chunk(testo: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Suddivide il testo in chunk sovrapposti.

    Strategia: suddivide per paragrafi (doppio newline), poi aggrega
    finché non si supera chunk_size. In questo modo si tende a preservare
    i paragrafi interi all'interno dei chunk.

    Args:
        testo: Testo da suddividere
        chunk_size: Dimensione target (caratteri)
        overlap: Overlap tra chunk consecutivi (caratteri)

    Returns:
        Lista di chunk testuali
    """
    paragrafi = [p.strip() for p in testo.split("\n\n") if p.strip()]

    chunks = []
    chunk_corrente = ""

    for paragrafo in paragrafi:
        if len(chunk_corrente) + len(paragrafo) + 2 <= chunk_size:
            chunk_corrente += ("" if not chunk_corrente else "\n\n") + paragrafo
        else:
            if chunk_corrente:
                chunks.append(chunk_corrente.strip())
            # Inizia un nuovo chunk con overlap dal precedente
            if chunks and overlap > 0:
                overlap_text = chunks[-1][-overlap:]
                chunk_corrente = overlap_text + "\n\n" + paragrafo
            else:
                chunk_corrente = paragrafo

    if chunk_corrente.strip():
        chunks.append(chunk_corrente.strip())

    return chunks


def _estrai_righe_tabella_pdf(testo: str) -> List[Dict[str, Any]]:
    """
    Estrae righe tabellari da PDF usando SOLO le colonne:
    - PATOLOGIA
    - LIVELLO
    - PRESTAZ_AMB_V_DESC
    """
    lines = [ln.strip() for ln in testo.splitlines() if ln.strip()]
    header_idx = None

    for i, line in enumerate(lines):
        upper = line.upper()
        if "PATOLOGIA" in upper and "LIVELLO" in upper and "PRESTAZ_AMB_V_DESC" in upper:
            header_idx = i
            break

    if header_idx is None:
        return []

    rows: List[Dict[str, Any]] = []
    for raw in lines[header_idx + 1:]:
        if "PATOLOGIA" in raw.upper() and "LIVELLO" in raw.upper():
            continue

        # split robusto: tab oppure >=2 spazi
        parts = [p.strip() for p in re.split(r"\t+|\s{2,}", raw) if p.strip()]
        if len(parts) < 3:
            continue

        # euristica: in molte tabelle le ultime 3 colonne utili sono [patologia, livello, descrizione]
        # se è presente anche un codice prestazione, la descrizione è la parte successiva.
        livello_idx = None
        for idx, part in enumerate(parts):
            if re.fullmatch(r"[123]", part):
                livello_idx = idx
                break

        if livello_idx is None or livello_idx == 0 or livello_idx >= len(parts) - 1:
            continue

        patologia = parts[livello_idx - 1]
        livello = parts[livello_idx]
        descrizione = parts[livello_idx + 1]

        if not patologia or not descrizione:
            continue

        rows.append({
            "patologia": patologia,
            "livello": int(livello),
            "prestaz_desc": descrizione,
        })

    return rows


def indicizza_pdf(
    percorso_file: str,
    patologia_default: Optional[str] = None,
    source_document_name: Optional[str] = None
):
    """
    Legge un file PDF, estrae le tabelle con PyMuPDF find_tables() e le indicizza
    su Azure AI Search. Supporta il formato regionale con colonne:
    PATOLOGIA, LIVELLO, PRESTAZ_AMB_V_DESC, ID_PATOLOGIA, % Usufruenti.

    Args:
        percorso_file: Percorso al file PDF
        patologia_default: Patologia da assegnare a tutti i chunk (override automatico)
    """
    logging.info(f"📂 Inizio indicizzazione PDF: {percorso_file}")
    nome_file = source_document_name or os.path.basename(percorso_file)

    openai_client = _get_openai_client()
    search_client = _get_search_client()

    # Estrazione tabellare nativa con PyMuPDF find_tables()
    try:
        rows = _estrai_righe_tabella_pdf_nativa(percorso_file)
        logging.info(f"📊 Righe tabellari PDF estratte: {len(rows)}")
    except Exception as e:
        logging.error(f"❌ Errore estrazione tabelle PDF: {e}")
        raise

    if not rows:
        logging.warning(
            "⚠️ Nessuna riga tabellare valida trovata con find_tables(); provo fallback su testo PDF."
        )
        try:
            testo_pdf = _estrai_testo_pdf(percorso_file)
            rows = _estrai_righe_tabella_pdf(testo_pdf)
            logging.info(f"📊 Righe tabellari PDF estratte con fallback testuale: {len(rows)}")
        except Exception as e:
            logging.error(f"❌ Errore fallback estrazione testo PDF: {e}")
            raise

    if not rows:
        logging.warning(
            "⚠️ Nessuna riga tabellare valida trovata nel PDF anche con fallback testuale. "
            "Verifica che il PDF contenga tabelle con le colonne "
            "PATOLOGIA, LIVELLO, PRESTAZ_AMB_V_DESC."
        )
        return

    testi_da_embeddare: List[str] = []
    metadata_rows = []

    for row in rows:
        patologia    = patologia_default or row["patologia"]
        livello      = row["livello"]
        prestaz_desc = row["prestaz_desc"]
        tipologia_prestazione = str(row.get("tipologia_prestazione", "")).strip()
        chunk = (
            f"Patologia: {patologia}. "
            f"Livello: {livello}. "
            f"Tipologia prestazione: {tipologia_prestazione}. "
            f"Parametro/Prestazione: {prestaz_desc}."
        )

        metadata_rows.append({
            "livello": livello,
            "patologia": patologia,
            "prestaz_desc": prestaz_desc,
            "tipologia_prestazione": tipologia_prestazione,
        })
        testi_da_embeddare.append(chunk)

    try:
        embeddings = genera_embeddings_batch(testi_da_embeddare, openai_client)
    except Exception as e:
        logging.error(f"❌ Errore durante generazione embedding batch PDF: {e}")
        raise

    docs = []
    for meta, chunk, embedding in zip(metadata_rows, testi_da_embeddare, embeddings):
        docs.append({
            "id":                 str(uuid.uuid4()),
            "livello":            meta["livello"],
            "patologia":          meta["patologia"],
            "prestaz_amb_v_desc": meta["prestaz_desc"],
            "tipologia_prestazione": meta["tipologia_prestazione"],
            "chunk_testo":        chunk,
            "source_document_name": nome_file,
            "embedding":          embedding,
        })

    logging.info(f"📄 Documenti PDF preparati: {len(docs)}")
    carica_documenti(docs, search_client)
    logging.info(f"✅ Indicizzazione PDF completata: {percorso_file}")


# =============================================================================
# MAIN
# =============================================================================

def _valida_env_o_lancia():
    """
    Valida le variabili d'ambiente minime richieste per l'indicizzazione.
    Gli endpoint sono obbligatori; le chiavi sono opzionali se si usa Managed Identity.
    """
    variabili_endpoint = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT"
    ]
    mancanti = [v for v in variabili_endpoint if not os.environ.get(v)]
    if mancanti:
        raise ValueError(f"Variabili d'ambiente mancanti: {mancanti}")

    # Avvisa se né chiave né Managed Identity sono disponibili
    if not os.environ.get("AZURE_SEARCH_KEY"):
        logging.info("ℹ️ AZURE_SEARCH_KEY non impostata: verrà usata Managed Identity per AI Search")
    if not os.environ.get("AZURE_OPENAI_KEY"):
        logging.info("ℹ️ AZURE_OPENAI_KEY non impostata: verrà usata Managed Identity per OpenAI")


def esegui_indicizzazione(
    excel: Optional[str] = None,
    pdf: Optional[str] = None,
    patologia: Optional[str] = None,
    clear: bool = False,
    source_document_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Esegue il flusso completo di indicizzazione in modalità riusabile
    (CLI o Azure Function).

    Returns:
        dict con riepilogo dell'operazione.
    """
    _valida_env_o_lancia()

    if not excel and not pdf and not clear:
        raise ValueError("Specificare almeno uno tra excel, pdf o clear=True")

    risultato = {
        "indice_reset": False,
        "excel_indicizzato": bool(excel),
        "pdf_indicizzato": bool(pdf),
    }

    # Cancella indice se richiesto
    if clear:
        cancella_indice()
        risultato["indice_reset"] = True

    # Crea/aggiorna indice
    crea_o_aggiorna_indice()

    # Indicizza Excel
    if excel:
        indicizza_excel(excel, source_document_name=source_document_name)

    # Indicizza PDF
    if pdf:
        indicizza_pdf(pdf, patologia_default=patologia, source_document_name=source_document_name)

    logging.info("🏁 Indicizzazione completata.")
    return risultato


def main():
    parser = argparse.ArgumentParser(
        description="Indicizzatore KB per Azure AI Search (Excel + PDF)"
    )
    parser.add_argument("--excel",     type=str, help="Percorso file Excel da indicizzare")
    parser.add_argument("--pdf",       type=str, help="Percorso file PDF da indicizzare")
    parser.add_argument("--patologia", type=str, help="Patologia da assegnare al PDF (override automatico)")
    parser.add_argument("--clear",     action="store_true", help="Cancella e ricrea l'indice prima di indicizzare")

    args = parser.parse_args()

    if not args.excel and not args.pdf and not args.clear:
        parser.print_help()
        sys.exit(0)

    try:
        esegui_indicizzazione(
            excel=args.excel,
            pdf=args.pdf,
            patologia=args.patologia,
            clear=args.clear,
        )
    except ValueError as e:
        logging.error(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()