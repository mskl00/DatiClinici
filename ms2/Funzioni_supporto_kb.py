import azure.functions as func
import logging
import json
import os
import uuid
import re
import math
import sys
from datetime import datetime
from openai import AzureOpenAI
from pymongo import MongoClient
from typing import Dict, Any, Union, Optional

#18:01
# --- FUNZIONI DI UTILITÀ ---

# Usa la versione "definitiva" della chiave dal modulo condiviso (root repo).
# Fallback alla logica locale solo se il modulo esterno non è disponibile.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from funzioni_supporto import genera_chiave as genera_chiave_definitiva
except Exception:
    genera_chiave_definitiva = None
 
def genera_chiave(input_data: Union[Dict[str, Any], int]) -> str:
    """
    Genera una chiave leggibile concatenando i dati del paziente e del professionista.
    Questa chiave sarà usata come 'Shard Key' (Partition Key) su Cosmos DB.
    """
    if genera_chiave_definitiva:
        return genera_chiave_definitiva(input_data)

    if not isinstance(input_data, dict):
        return "DATI_INVALIDI"

    paziente = input_data.get("paziente", {})
    professionista = input_data.get("professionista", {})

    if not paziente or not professionista:
        return "DATI_INVALIDI"

    def clean(s):
        return str(s).strip().replace(" ", "").replace("(", "").replace(")", "").upper()

    diagnosi = clean(paziente.get("diagnosi", "NA"))
    eta = clean(paziente.get("fascia_eta", "NA"))
    sesso = clean(paziente.get("sesso", "NA"))
    unita = clean(professionista.get("unita_operativa", "NA"))
    comorbidita_raw = paziente.get("comorbidita", [])

    if isinstance(comorbidita_raw, list):
        comorbidita = [clean(c) for c in comorbidita_raw if str(c).strip()]
    elif comorbidita_raw is None or str(comorbidita_raw).strip() == "":
        comorbidita = []
    else:
        parti = [p for p in re.split(r"[,;|]", str(comorbidita_raw)) if str(p).strip()]
        comorbidita = [clean(c) for c in parti] if parti else [clean(comorbidita_raw)]

    comorbidita_str = "+".join(sorted(comorbidita)) if comorbidita else "NOCOMORBIDITA"

    return f"{diagnosi}_{eta}_{sesso}_{unita}_{comorbidita_str}"

#===========================================#
#             Interazione con DB
#===========================================#

def get_mongo_client():
    """Crea e restituisce il client MongoDB usando la stringa di connessione."""
    try:
        conn_str = os.environ["MONGODB_CONNECTION_STRING"]
        # Connessione al cluster
        client = MongoClient(conn_str)
        return client
    except Exception as e:
        logging.error(f"Errore critico connessione MongoDB: {e}")
        raise e
    
# Estrazione parametri da risposta modello 
def estrai_parametri(testo: str) -> list:
    """
    Estrae tutti i parametri racchiusi tra virgolette doppie
    e li restituisce come lista.
    """
    return re.findall(r'"(.*?)"', testo)


# Preparazione feedback vuoto dal primo giro di PROMPT
def prepara_feedback(chiave_identificativa: str, lista_parametri: list, timestamp_now, get_FHIR) -> dict:
    """
    Genera il feedback da salvare nel DB.
    
    Args:
        nome_voce (str): Chiave principale del feedback (es. profilo clinico).
        lista_parametri (list): Lista dei parametri modello.

    Returns:
        dict: struttura feedback conforme.
    """
    feedback = {
        "chiave_input":chiave_identificativa,
            "output_parametri": {param: 0 for param in lista_parametri},
            "parametri_aggiunti": {},
            "Flag_qualità": 0,
            "GET_FHIR" : get_FHIR,
            "timestamp": timestamp_now
    }

    return feedback

# Controllo esistenza feedback sul DB

def check_feedback_exists(mongo_db_name: str, mongo_coll_name: str, chiave_input: str) -> int:
    """
    Verifica se esiste un documento nella collection passata
    con chiave_input uguale a quello passato.

    Ritorna:
        1 -> se esiste
        0 -> se non esiste
    """
    with get_mongo_client() as mongo_client:
        db = mongo_client[mongo_db_name]
        coll = db[mongo_coll_name]

        result = coll.find_one({"chiave_input": chiave_input})

        return 1 if result else 0
        
def leggi_parametrimodello_e_query(mongo_db_name: str, mongo_coll: str, chiave_input: str):
    """
    Legge dal DB MongoDB il documento associato a chiave_input
    e restituisce:
        - GET_FHIR
        - parametri_modello

    Se la chiave non esiste → ritorna (None, None).
    """

    with get_mongo_client() as mongo_client:
        db = mongo_client[mongo_db_name]
        coll = db[mongo_coll]

        # Cerca il documento con chiave_input esatto
        result = coll.find_one({"chiave_input": chiave_input})

        if not result:
            return None, None  # evitiamo errori sul return

        # NUOVA STRUTTURA FLAT
        get_fhir = result.get("GET_FHIR")
        parametri_modello = result.get("output_parametri", {})

        return get_fhir, parametri_modello

    
def leggi_flag(mongo_db_name: str, mongo_coll: str, chiave_input: str):
    """
    Legge dal DB MongoDB il documento associato a chiave_input
    e restituisce il Flag_qualità.

    Se la chiave non esiste → ritorna None.
    """

    with get_mongo_client() as mongo_client:
        db = mongo_client[mongo_db_name]
        coll = db[mongo_coll]

        result = coll.find_one({"chiave_input": chiave_input})

        if not result:
            return None

        # NUOVA STRUTTURA FLAT
        return result.get("Flag_qualità")
    
def leggi_kb_completa(mongo_db_name: str, mongo_coll: str, chiave_input: str) -> Optional[Dict[str, Any]]:
    """
    Legge dal DB MongoDB il documento associato a chiave_input
    e restituisce *l'intera KB con struttura corretta*, pronta
    per essere passata a `genera_parametri_monitoraggio_CON_kb()`.
    
    Se la chiave non esiste → ritorna None.
    """

    with get_mongo_client() as mongo_client:
        db = mongo_client[mongo_db_name]
        coll = db[mongo_coll]

        # Cerca il documento con chiave_input esatto
        result = coll.find_one({"chiave_input": chiave_input})

        if not result:
            return None  # 🔴 KB non trovata

        # ------------------------------------------------------
        #  VALIDAZIONE / NORMALIZZAZIONE DELLA STRUTTURA
        # ------------------------------------------------------

        kb = {
            # campo obbligatorio per la query successiva
            "chiave_input": result.get("chiave_input"),

            # elenco parametri validati (modello)
            "output_parametri": result.get("output_parametri", {}),

            # parametri aggiunti con valore aggregato
            "parametri_aggiunti": result.get("parametri_aggiunti", {}),

            # numero di feedback (DEFAULT = 1)
            "Feedback_number": result.get("Feedback_number", 1),

            # info utili
            "GET_FHIR": result.get("GET_FHIR"),
            "timestamp": result.get("timestamp"),

            # altri campi opzionali ma utili
            "transaction_id": result.get("transaction_id"),
            "Flag_qualità": result.get("Flag_qualità", 0),
            "id": result.get("id"),
        }

        return kb

#===========================================================
#
#               CALCOLO METRICHE DI QUALITA'
#
#============================================================

def calcola_QV_da_parametri_modello(doc: dict) -> float:
    """
    Calcola il valore s a partire da 'output_parametri' e 'Feedback_number'.

    Struttura attesa di `doc`:
        {
            "id": ...,
            "transaction_id": ...,
            "chiave_input": ...,
            "temestamp": ...,
            "output_parametri": {parametro: n, ...},
            "parametri_aggiunti": {...},
            "Flag_qualità": 1,
            "GET_FHIR": "la-query-completa",
            "Feedback_number": N
        }

    Formula:
        r = n / Feedback_number
        s = (6 * r - 1) / 5

    Se Feedback_number è 0 o i dati non sono validi, ritorna 0.0.
    """

    if not isinstance(doc, dict):
        return 0.0

    output_parametri = doc.get("output_parametri", {})
    if not isinstance(output_parametri, dict) or not output_parametri:
        return 0.0

    # Tutti gli n sono uguali: prendo il primo valore
    primo_valore = next(iter(output_parametri.values()))

    try:
        n = float(primo_valore)
    except (TypeError, ValueError):
        return 0.0

    feedback_number = doc.get("Feedback_number", 0)

    try:
        feedback_number = float(feedback_number)
    except (TypeError, ValueError):
        return 0.0

    if feedback_number == 0:
        return 0.0

    r = n / feedback_number
    r = r / 6
    s = (6 * r - 1) / 5
    return s

from typing import Dict, Any, Union

def calcola_intensita_suggerimento_feedback(doc: dict) -> float:
    """
    Calcola la somma totale dei valori in 'parametri_aggiunti'
    divisa per due volte il valore di 'Feedback_number'.

    Nuova struttura attesa del documento:
        {
            "id": ...,
            "transaction_id": ...,
            "chiave_input": ...,
            "temestamp": ...,
            "output_parametri": {parametro: n, ...},
            "parametri_aggiunti": {parametro_agg1: l1, ...},
            "Flag_qualità": ...,
            "GET_FHIR": "la-query-completa",
            "Feedback_number": N
        }

    Formula:
        risultato = somma_parametri_aggiunti / (2 * Feedback_number)

    Se i dati non sono validi o Feedback_number è 0, ritorna 0.0.
    """
    # Deve essere un dizionario
    if not isinstance(doc, dict):
        return 0.0

    # Leggiamo parametri_aggiunti e verifichiamo che sia un dizionario non vuoto
    parametri_aggiunti = doc.get("parametri_aggiunti", {})
    if not isinstance(parametri_aggiunti, dict) or not parametri_aggiunti:
        return 0.0

    # Leggiamo il numero totale di feedback
    feedback_number = doc.get("Feedback_number", 0)
    try:
        feedback_number = float(feedback_number)
    except (TypeError, ValueError):
        return 0.0

    if feedback_number == 0:
        return 0.0

    # Somma dei "counter" (ora sono direttamente i valori del dict)
    somma_counter = 0.0
    for valore in parametri_aggiunti.values():
        try:
            somma_counter += float(valore)
        except (TypeError, ValueError):
            # Se un valore non è convertibile a numero lo saltiamo
            continue

    return somma_counter / (2 * feedback_number)


def calcola_tasso_dispersione(doc: dict) -> float:
    """
    Calcola il rapporto tra il numero di parametri unici in 'parametri_aggiunti'
    e la somma totale dei loro valori (che rappresentano i counter).

    Struttura attesa del documento:
        {
            "id": ...,
            "transaction_id": ...,
            "chiave_input": ...,
            "temestamp": ...,
            "output_parametri": {parametro: n, ...},
            "parametri_aggiunti": {parametro_agg1: l1, ...},
            "Flag_qualità": ...,
            "GET_FHIR": "la-query-completa",
            "Feedback_number": N
        }

    Formula:
        risultato = numero_parametri_unici / somma_counter_totale

    Returns:
        float: valore del rapporto calcolato, oppure 0.0 se non valido o se la somma è 0.
    """
    if not isinstance(doc, dict):
        return 0.0

    parametri_aggiunti = doc.get("parametri_aggiunti", {})
    if not isinstance(parametri_aggiunti, dict) or not parametri_aggiunti:
        return 0.0

    num_parametri = len(parametri_aggiunti)
    somma_counter = 0.0

    for valore in parametri_aggiunti.values():
        try:
            somma_counter += float(valore)
        except (TypeError, ValueError):
            # Se un valore non è numerico lo ignoriamo
            continue

    if somma_counter == 0:
        return 0.0

    return num_parametri / somma_counter


def entropia_counter_parametri_aggiunti(doc: dict) -> float:
    """
    Calcola l'entropia Shannon (base e) dei valori in 'parametri_aggiunti'.
    
    Nuova struttura attesa:
        {
            "id": ...,
            "transaction_id": ...,
            "chiave_input": ...,
            "temestamp": ...,
            "output_parametri": {parametro: n, ...},
            "parametri_aggiunti": {parametro_agg1: l1, ...},   # <-- i counter sono i valori!
            "Flag_qualità": ...,
            "GET_FHIR": "la-query-completa",
            "Feedback_number": N
        }
        
    Formula calcolo entropia:
        H = - Σ( p_i * log(p_i) ), con p_i = valore_i / somma_totale
    
    Se non ci sono valori positivi → ritorna 0.0
    """
    
    # Verifica che 'doc' sia un dizionario
    if not isinstance(doc, dict):
        return 0.0

    parametri_aggiunti = doc.get("parametri_aggiunti", {})
    if not isinstance(parametri_aggiunti, dict) or not parametri_aggiunti:
        return 0.0

    # Estraiamo i CONTATORI (sono i VALORI del dict)
    counters = []
    for valore in parametri_aggiunti.values():
        try:
            num = float(valore)
            if num > 0:
                counters.append(num)
        except (TypeError, ValueError):
            continue

    if not counters:
        return 0.0

    total = sum(counters)

    # Evitiamo divisioni per zero
    if total == 0:
        return 0.0

    # Probabilità di ogni parametro
    probs = [c / total for c in counters]

    # Calcolo entropia Shannon
    entropy = 0.0
    for p in probs:
        entropy -= p * math.log(p)  # log base e
    if len(counters) == 1:
      entropy = 0
    else:
      entropy = entropy / math.log(len(counters))
    return entropy

def calcola_NAS_da_parametri_modello( QV, Intensity_suggerimento, entropy, Tasso_dispersione) -> float:
  w1 = 0.5
  w2 = 0.25
  w3 = 0.125
  w4 = 0.125
  NAS=w1*QV+w2*(1-Intensity_suggerimento)+w3*(1-entropy)+w4*(1-Tasso_dispersione)
  return NAS