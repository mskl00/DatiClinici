import ast
import logging
import os
import uuid
from datetime import datetime

import azure.functions as func

from Funzioni_supporto_kb import (
    genera_chiave,
    get_mongo_client,
    check_feedback_exists,
    leggi_flag,
    calcola_QV_da_parametri_modello,
    calcola_intensita_suggerimento_feedback,
    entropia_counter_parametri_aggiunti,
    calcola_tasso_dispersione,
    calcola_NAS_da_parametri_modello,
)


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


SOGLIA_NAS = 0.3
MIN_FEEDBACK_PER_FLAG = 10


def _normalize_list(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
        return [x.strip() for x in txt.split(",") if x.strip()]
    if value is None:
        return []
    try:
        return [str(x).strip() for x in list(value) if str(x).strip()]
    except Exception:
        return []


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _calcola_nas_documento(doc_corrente):
    qv = calcola_QV_da_parametri_modello(doc_corrente)
    intensita = calcola_intensita_suggerimento_feedback(doc_corrente)
    dispersione = calcola_tasso_dispersione(doc_corrente)
    entropia = entropia_counter_parametri_aggiunti(doc_corrente)
    nas = calcola_NAS_da_parametri_modello(qv, intensita, entropia, dispersione)

    logging.info(f"Calcolo metriche KB: QV={qv}")
    logging.info(f"Calcolo metriche KB: Intensità suggerimento={intensita}")
    logging.info(f"Calcolo metriche KB: Tasso dispersione={dispersione}")
    logging.info(f"Calcolo metriche KB: Entropia={entropia}")
    logging.info(f"Calcolo metriche KB: NAS={nas}")

    return nas


@app.route(route="Elaborafeedback", methods=["POST"])
def Feedback(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Richiesta ricevuta per elaborazione feedback e aggiornamento KB.")

    try:
        mongo_db_name = os.environ["MONGODB_DB_NAME"]
        mongo_coll_params = os.environ["MONGODB_COLLECTION_PARAMETRI"]
    except KeyError as e:
        return func.HttpResponse(f"Errore Server: Manca variabile d'ambiente {str(e)}", status_code=500)

    try:
        doc = req.get_json()
        if not doc:
            raise ValueError("Input vuoto")
    except ValueError:
        return func.HttpResponse("JSON invalido o incompleto.", status_code=400)

    chiave_input = doc.get("chiave_input")
    if not chiave_input:
        chiave_input = genera_chiave(doc)

    if not chiave_input or chiave_input == "DATI_INVALIDI":
        return func.HttpResponse("chiave_input mancante o non generabile.", status_code=400)

    transaction_id = str(uuid.uuid4())
    timestamp_now = datetime.utcnow().isoformat()

    query_fhir = doc.get("GET_FHIR")
    feedback_value = _to_float(doc.get("feedback_value"), default=0.0)
    parametri_aggiunti = _normalize_list(doc.get("parametri_aggiunti", []))
    lista_parametri = _normalize_list(doc.get("lista_parametri", []))
    parametri_con_dettaglio = doc.get("parametri_con_dettaglio")
    proposta_visita = doc.get("proposta_visita_prestazioni_icd9cm")
    rag_source = doc.get("rag_source")
    kb_patologia = doc.get("kb_patologia")
    kb_fonti = doc.get("kb_fonti")
    livello_complessita = doc.get("livello_complessita")

    exists = check_feedback_exists(mongo_db_name, mongo_coll_params, chiave_input)
    logging.info(f"Esistenza record feedback per chiave {chiave_input}: {exists}")

    try:
        with get_mongo_client() as mongo_client:
            db = mongo_client[mongo_db_name]
            coll = db[mongo_coll_params]

            doc_corrente = coll.find_one({"chiave_input": chiave_input})

            # Caso nuovo: inizializza contatori su struttura DB definitiva
            if not doc_corrente:
                nuovo_doc = {
                    "transaction_id": transaction_id,
                    "chiave_input": chiave_input,
                    "timestamp": timestamp_now,
                    "output_parametri": {param: feedback_value for param in lista_parametri},
                    "parametri_con_dettaglio": doc.get("parametri_con_dettaglio", []),
                    "proposta_visita_prestazioni_icd9cm": doc.get("proposta_visita_prestazioni_icd9cm"),
                    "rag_source": doc.get("rag_source"),
                    "kb_patologia": doc.get("kb_patologia"),
                    "kb_fonti": doc.get("kb_fonti", []),
                    "livello_complessita": doc.get("livello_complessita"),
                    "parametri_aggiunti": {p: 1 for p in parametri_aggiunti},
                    "Flag_qualità": 0,
                    "GET_FHIR": query_fhir,
                    "Feedback_number": 1,
                }
                coll.insert_one(nuovo_doc)
                return func.HttpResponse("Feedback inserito.", status_code=200)

            flag = leggi_flag(mongo_db_name, mongo_coll_params, chiave_input)
            logging.info(f"Flag qualità letto da DB per chiave {chiave_input}: {flag}")

            # Se il record è già marcato come qualità alta, non alterare la KB consolidata
            if flag == 1:
                return func.HttpResponse("Nessun aggiornamento: Flag_qualità != 0.", status_code=200)

            output_parametri_db = doc_corrente.get("output_parametri", {})
            if not isinstance(output_parametri_db, dict):
                output_parametri_db = {}

            nuovo_output_parametri = dict(output_parametri_db)
            for param in lista_parametri:
                nuovo_output_parametri[param] = _to_float(nuovo_output_parametri.get(param, 0.0)) + feedback_value

            parametri_aggiunti_db = doc_corrente.get("parametri_aggiunti", {})
            if not isinstance(parametri_aggiunti_db, dict):
                parametri_aggiunti_db = {}

            nuovo_parametri_aggiunti = dict(parametri_aggiunti_db)
            for p in parametri_aggiunti:
                nuovo_parametri_aggiunti[p] = _to_int(nuovo_parametri_aggiunti.get(p, 0)) + 1

            nuovo_feedback_number = _to_int(doc_corrente.get("Feedback_number", 0)) + 1

            doc_valutazione = dict(doc_corrente)
            doc_valutazione["output_parametri"] = nuovo_output_parametri
            doc_valutazione["parametri_aggiunti"] = nuovo_parametri_aggiunti
            doc_valutazione["Feedback_number"] = nuovo_feedback_number

            nas = _calcola_nas_documento(doc_valutazione)
            update_payload = {
                "transaction_id": transaction_id,
                "output_parametri": nuovo_output_parametri,
                "parametri_aggiunti": nuovo_parametri_aggiunti,
                "Feedback_number": nuovo_feedback_number,
                "timestamp": timestamp_now,
            }

            # Mantiene allineati i metadati principali con l'ultima esecuzione
            # della pipeline (coerenza con payload completo generato da MS_1).
            if parametri_con_dettaglio is not None:
                update_payload["parametri_con_dettaglio"] = parametri_con_dettaglio
            if proposta_visita is not None:
                update_payload["proposta_visita_prestazioni_icd9cm"] = proposta_visita
            if rag_source is not None:
                update_payload["rag_source"] = rag_source
            if kb_patologia is not None:
                update_payload["kb_patologia"] = kb_patologia
            if kb_fonti is not None:
                update_payload["kb_fonti"] = kb_fonti
            if livello_complessita is not None:
                update_payload["livello_complessita"] = livello_complessita

            if query_fhir is not None:
                update_payload["GET_FHIR"] = query_fhir

            if nas <= SOGLIA_NAS and nuovo_feedback_number >= MIN_FEEDBACK_PER_FLAG:
                update_payload["Flag_qualità"] = 1

            coll.update_one({"chiave_input": chiave_input}, {"$set": update_payload})
            return func.HttpResponse("Feedback aggiornato.", status_code=200)

    except Exception as e:
        logging.error(f"Errore aggiornamento MongoDB feedback: {e}")
        return func.HttpResponse("Errore server in aggiornamento feedback.", status_code=500)