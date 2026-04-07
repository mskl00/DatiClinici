import azure.functions as func
import logging
import json
import os
import uuid
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime
#Deploy 12:01
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from indicizza_kb import esegui_indicizzazione

from funzioni_supporto import (
    # Utility
    genera_chiave,
    estrai_parametri,
    # MongoDB
    get_mongo_client,
    check_feedback_exists,
    leggi_parametrimodello_e_query,
    leggi_flag,
    leggi_rag_source,
    leggi_kb_completa,
    # RAG
    cerca_patologia_in_kb,
    determina_livello_complessita,
    RAGError,
    # Prompt builders
    build_prompt_1_no_KB,
    build_prompt_1_con_rag,
    normalizza_fonti_parametri_rag,
    normalizza_parametri_llm_no_rag,
    normalizza_parametri_dettaglio_output,
    separa_parametri_da_esami,
    filtra_parametri_clinici_con_loinc,
    completa_parametri_a_sei,
    prepara_parametri_per_prompt_2,
    build_prompt_2,
    build_prompt_judge,
    build_prompt1_CON_kb,
    build_prompt1_CON_kb_con_rag_suggeriti,
    normalizza_query_fhir_lastn_unica,
    # AI Client
    AIFoundryClient,
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# =============================================================================
# HELPER — parse JSON da risposta LLM (gestisce blocchi ```json ... ```)
# =============================================================================

def _parse_llm_json(testo: str) -> dict:
    testo = testo.strip()
    if testo.startswith("```"):
        lines = testo.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        testo = "\n".join(lines).strip()
    return json.loads(testo)


# =============================================================================
# BLOB TRIGGER INDICIZZAZIONE KB
# =============================================================================

def _sposta_blob_in_elaborati(blob_name: str):
    """Sposta il blob da kb-documenti a kb-elaborati e poi elimina l'origine.

    Supporta sia connection string (legacy) sia Managed Identity tramite
    STORAGE_KB_ACCOUNT_URL. Se entrambe le variabili sono presenti,
    la Managed Identity (STORAGE_KB_ACCOUNT_URL) ha la precedenza.
    """
    source_container    = os.environ.get("KB_SOURCE_CONTAINER", "kb-documenti")
    processed_container = os.environ.get("KB_PROCESSED_CONTAINER", "kb-elaborati")

    storage_account_url = os.environ.get("STORAGE_KB_ACCOUNT_URL", "")
    connection_string   = os.environ.get("STORAGE_KB_CONNECTION", "")

    if storage_account_url:
        # Managed Identity
        from azure.identity import DefaultAzureCredential
        logging.info("🔐 BlobServiceClient via Managed Identity")
        service = BlobServiceClient(
            account_url=storage_account_url,
            credential=DefaultAzureCredential()
        )
    elif connection_string:
        # Legacy connection string
        logging.info("🔑 BlobServiceClient via connection string")
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        raise ValueError(
            "Configurare STORAGE_KB_ACCOUNT_URL (Managed Identity) "
            "oppure STORAGE_KB_CONNECTION (legacy connection string)"
        )

    source_client = service.get_blob_client(container=source_container, blob=blob_name)
    dest_client   = service.get_blob_client(container=processed_container, blob=blob_name)

    # Garantisce che il container di destinazione esista.
    processed_client = service.get_container_client(processed_container)
    try:
        processed_client.create_container()
    except ResourceExistsError:
        pass

    # Copia applicativa autenticata (download + upload), robusta su container privati.
    payload = source_client.download_blob().readall()
    dest_client.upload_blob(payload, overwrite=True)
    source_client.delete_blob()


@app.blob_trigger(
    arg_name="blob",
    path="kb-documenti/{name}",
    connection="STORAGE_KB"
)
def IndicizzaKBDaBlob(blob: func.InputStream):
    """
    Trigger automatico quando arriva un file in kb-documenti (storage: stmonitoraggio).
    Indicizza su Azure AI Search e sposta il file in kb-elaborati.
    """
    nome_blob  = blob.name.split("/", 1)[-1]
    estensione = os.path.splitext(nome_blob)[1].lower()

    logging.info(f"📂 Blob ricevuto: {nome_blob} | Estensione: {estensione}")

    if estensione not in {".xlsx", ".xls", ".pdf"}:
        logging.warning(f"⚠️ File non supportato per indicizzazione: {nome_blob}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=estensione) as tmp:
        tmp.write(blob.read())
        percorso_temp = tmp.name

    try:
        logging.info(f"🔄 Avvio indicizzazione: {nome_blob}")
        if estensione in {".xlsx", ".xls"}:
            esegui_indicizzazione(excel=percorso_temp, source_document_name=nome_blob)
        else:
            esegui_indicizzazione(pdf=percorso_temp, source_document_name=nome_blob)

        _sposta_blob_in_elaborati(nome_blob)
        logging.info(f"✅ Indicizzazione completata e blob spostato in kb-elaborati: {nome_blob}")

    except Exception:
        logging.exception(f"❌ Errore nel processamento blob: {nome_blob}")
        raise
    finally:
        if os.path.exists(percorso_temp):
            os.remove(percorso_temp)


# =============================================================================
# HTTP TRIGGER — GENERA PIANO MONITORAGGIO
# =============================================================================

@app.route(route="GeneraPianoMonitoraggio")
def GeneraPianoMonitoraggio(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Richiesta ricevuta per GeneraPianoMonitoraggio.")

    # ------------------------------------------------------------------
    # 1. CARICAMENTO CONFIGURAZIONE
    # ------------------------------------------------------------------
    try:
        ai_foundry_endpoint = os.environ["AZURE_AI_FOUNDRY_ENDPOINT"]
        ai_foundry_model    = os.environ["AI_FOUNDRY_MODEL_NAME"]
        model_judge         = os.environ["AI_FOUNDRY_MODEL_NAME_JUDGE"]
        mongo_db_name       = os.environ["MONGODB_DB_NAME"]
        mongo_coll_params   = os.environ["MONGODB_COLLECTION_PARAMETRI"]
        mongo_coll_fhir     = os.environ["MONGODB_COLLECTION_FHIR"]

        # Chiave opzionale: se assente si usa Managed Identity
        ai_foundry_key = os.environ.get("AZURE_AI_FOUNDRY_KEY", "")

        logging.info(f"🔧 Endpoint: {ai_foundry_endpoint} | Modello: {ai_foundry_model}")
        if ai_foundry_key:
            logging.info("🔑 Autenticazione AI Foundry: API Key")
        else:
            logging.info("🔐 Autenticazione AI Foundry: Managed Identity")

    except KeyError as e:
        logging.error(f"❌ Variabile d'ambiente mancante: {e}")
        return func.HttpResponse(
            f"Errore Server: Manca variabile d'ambiente {e}",
            status_code=500
        )

    # ------------------------------------------------------------------
    # 2. PARSING INPUT
    # ------------------------------------------------------------------
    try:
        req_body       = req.get_json()
        paziente       = req_body.get("paziente")
        professionista = req_body.get("professionista")

        if not paziente or not professionista:
            raise ValueError("Campi 'paziente' e 'professionista' obbligatori.")

    except (ValueError, Exception) as e:
        return func.HttpResponse(
            f"JSON invalido o incompleto: {e}",
            status_code=400
        )

    chiave_input   = genera_chiave(req_body)
    transaction_id = str(uuid.uuid4())
    timestamp_now  = datetime.utcnow().isoformat()

    # ------------------------------------------------------------------
    # 3. INIZIALIZZAZIONE CLIENT AI
    # ------------------------------------------------------------------
    try:
        logging.info("🔧 Inizializzazione client AI Foundry...")
        client_ai = AIFoundryClient(
            endpoint=ai_foundry_endpoint,
            api_key=ai_foundry_key  # stringa vuota → Managed Identity
        )
        logging.info(f"✅ Client AI inizializzato | Modello: {ai_foundry_model}")

    except Exception as e:
        logging.error(f"❌ Errore inizializzazione AI client: {e}")
        return func.HttpResponse(f"Errore inizializzazione AI: {e}", status_code=500)

    # ------------------------------------------------------------------
    # 4. CHECK FEEDBACK
    # ------------------------------------------------------------------
    check_feedback = check_feedback_exists(mongo_db_name, mongo_coll_params, chiave_input)
    logging.info(f"✅ Check feedback: {check_feedback}")

    # ==================================================================
    # PATH A — CASO NUOVO
    # ==================================================================
    if check_feedback == 0:
        logging.info("🔄 PATH A: Caso nuovo")

        diagnosi            = paziente.get("diagnosi", "")
        livello_complessita = determina_livello_complessita(comorbidita=paziente.get("comorbidita"))

        kb_result  = None
        rag_source = False

        logging.info(f"🔍 [RAG] Ricerca KB per diagnosi: '{diagnosi}' | livello={livello_complessita}")
        try:
            kb_result = cerca_patologia_in_kb(diagnosi, livello=livello_complessita)
            if kb_result:
                rag_source = True
                logging.info(f"✅ [RAG] Trovata in KB: '{kb_result['patologia']}'")
            else:
                logging.info("⚠️ [RAG] Diagnosi non presente in KB → path LLM puro")
        except RAGError as e:
            logging.warning(f"⚠️ [RAG] Errore ricerca KB, fallback a LLM: {e}")

        try:
            if rag_source and kb_result:
                logging.info("📝 Prompt 1 — PATH A1: con dati RAG")
                prompt_1_text = build_prompt_1_con_rag(paziente, professionista, kb_result)
            else:
                logging.info("📝 Prompt 1 — PATH A2: LLM puro (nessuna KB)")
                prompt_1_text = build_prompt_1_no_KB(paziente, professionista)

            risposta_prompt_1 = client_ai.chat_completion(
                model=ai_foundry_model,
                messages=[
                    {"role": "system", "content": "Sei un medico esperto. Rispondi SEMPRE e SOLO con un JSON valido, senza testo aggiuntivo."},
                    {"role": "user",   "content": prompt_1_text}
                ],
                temperature=0.0
            )
            logging.info(f"✅ Prompt 1 completato: {len(risposta_prompt_1)} caratteri")

        except Exception as e:
            logging.error(f"❌ Errore Prompt 1: {e}")
            return func.HttpResponse(f"Errore AI Prompt 1: {e}", status_code=500)

        try:
            dati_prompt_1         = _parse_llm_json(risposta_prompt_1)
            proposta_visita_prestazioni_icd9cm = dati_prompt_1.get("proposta_visita_prestazioni_icd9cm")
            parametri_dettagliati = dati_prompt_1.get("parametri", [])
            if rag_source and kb_result:
                parametri_dettagliati = normalizza_fonti_parametri_rag(parametri_dettagliati, kb_result)
            else:
                parametri_dettagliati = normalizza_parametri_llm_no_rag(parametri_dettagliati)
            parametri_dettagliati = normalizza_parametri_dettaglio_output(parametri_dettagliati)
            parametri_dettagliati, proposta_visita_prestazioni_icd9cm, esami_spostati = separa_parametri_da_esami(
                parametri_dettagliati,
                proposta_visita_prestazioni_icd9cm
            )
            parametri_dettagliati, parametri_scartati_no_loinc = filtra_parametri_clinici_con_loinc(parametri_dettagliati)
            if parametri_scartati_no_loinc:
                logging.info(
                    "🧹 Parametri scartati senza codice LOINC: %s",
                    ", ".join(parametri_scartati_no_loinc)
                )
            parametri_dettagliati, parametri_aggiunti = completa_parametri_a_sei(
                parametri_dettagliati,
                kb_result=kb_result if rag_source else None
            )
            if parametri_aggiunti:
                logging.info("➕ Parametri aggiunti automaticamente per arrivare a 6: %d", parametri_aggiunti)
            if esami_spostati:
                logging.info(
                    "🧹 Parametri filtrati: spostati %d esami in proposta visita: %s",
                    len(esami_spostati),
                    ", ".join(esami_spostati)
                )
            lista_paramerti = [p.get("nome", "") for p in parametri_dettagliati if p.get("nome")]
            logging.info(
                "📊 Prompt 1 normalizzato | parametri_finali=%d | proposta_visita_presente=%s",
                len(lista_paramerti),
                bool(proposta_visita_prestazioni_icd9cm)
            )
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"⚠️ Parse JSON Prompt 1 fallito ({e}), uso estrai_parametri")
            dati_prompt_1         = {}
            proposta_visita_prestazioni_icd9cm = None
            parametri_dettagliati = []
            lista_paramerti       = estrai_parametri(risposta_prompt_1)

        parametri_testo = prepara_parametri_per_prompt_2(
            parametri_dettagliati=parametri_dettagliati,
            risposta_prompt_1=risposta_prompt_1,
            rag_source=rag_source,
            kb_result=kb_result
        )

        try:
            logging.info("📝 Prompt 2: generazione query FHIR")
            logging.info("🔎 Parametri passati a Prompt 2 (A): %s", parametri_testo.replace("\n", " | ")[:500])
            prompt_2_text = build_prompt_2(parametri_testo)
            query_fhir = client_ai.chat_completion(
                model=ai_foundry_model,
                messages=[
                    {"role": "system", "content": "Sei un esperto FHIR."},
                    {"role": "user",   "content": prompt_2_text}
                ],
                temperature=0.0
            )
            query_fhir = normalizza_query_fhir_lastn_unica(query_fhir, parametri_dettagliati)
            logging.info(f"✅ Prompt 2 completato.")
            logging.info("📤 Query FHIR normalizzata (A): %s", query_fhir.replace("\n", " | ")[:500])

        except Exception as e:
            logging.error(f"❌ Errore Prompt 2: {e}")
            return func.HttpResponse(f"Errore AI Prompt 2: {e}", status_code=500)

        try:
            logging.info("📝 Prompt 3: LLM-as-a-Judge")
            prompt_3_text = build_prompt_judge(parametri_testo, paziente, professionista)
            verifica_fhir = client_ai.chat_completion(
                model=model_judge,
                messages=[
                    {"role": "system", "content": "Sei un medico esperto valutatore."},
                    {"role": "user",   "content": prompt_3_text}
                ],
                temperature=0.0
            )
            logging.info(f"✅ Prompt 3 completato.")

        except Exception as e:
            logging.error(f"❌ Errore Prompt 3: {e}")
            return func.HttpResponse(f"Errore AI Prompt 3: {e}", status_code=500)

        try:
            with get_mongo_client() as mongo_client:
                db   = mongo_client[mongo_db_name]
                coll = db[mongo_coll_params]
                doc_parametri = {
                    "transaction_id":          transaction_id,
                    "chiave_input":            chiave_input,
                    "timestamp":               timestamp_now,
                    "output_parametri":        {param: 0 for param in lista_paramerti},
                    "parametri_con_dettaglio": parametri_dettagliati,
                    "proposta_visita_prestazioni_icd9cm": proposta_visita_prestazioni_icd9cm,
                    "rag_source":              rag_source,
                    "kb_patologia":            kb_result.get("patologia") if kb_result else None,
                    "kb_fonti":                kb_result.get("fonti") if kb_result else [],
                    "livello_complessita":     livello_complessita,
                    "parametri_aggiunti":      {},
                    "Flag_qualità":            0,
                    "GET_FHIR":                query_fhir,
                    "Feedback_number":         0
                }
                coll.insert_one(doc_parametri)
                logging.info(f"✅ Documento salvato in MongoDB")

        except Exception as e:
            logging.error(f"❌ Errore salvataggio MongoDB: {e}")

        output_completo = {
            "transaction_id":                   transaction_id,
            "chiave_input":                     chiave_input,
            "rag_source":                       rag_source,
            "kb_patologia":                     kb_result.get("patologia") if kb_result else None,
            "kb_fonti":                         kb_result.get("fonti") if kb_result else [],
            "livello_complessita":              livello_complessita,
            "risultato_prompt_1_parametri":     lista_paramerti,
            "parametri_con_dettaglio":          parametri_dettagliati,
            "proposta_visita_prestazioni_icd9cm": proposta_visita_prestazioni_icd9cm,
            "risultato_prompt_2_fhir":          query_fhir,
            "risultato_prompt_3_verifica_fhir": verifica_fhir
        }

        logging.info(f"✅ Risposta PATH A pronta. rag_source={rag_source}")
        return func.HttpResponse(
            json.dumps(output_completo, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )

    # ==================================================================
    # PATH B — CASO GIÀ VISTO
    # ==================================================================
    else:
        Flag = leggi_flag(mongo_db_name, mongo_coll_params, chiave_input)
        logging.info(f"🔄 PATH B: Caso esistente | Flag_qualità={Flag}")

        # PATH B1 — Flag 0: restituzione parametri da DB
        if Flag == 0:
            logging.info("📦 PATH B1: Restituzione parametri da DB (no ricalcolo)")
            parametri_modello, query_fhir, parametri_con_dettaglio, proposta_visita_prestazioni_icd9cm = leggi_parametrimodello_e_query(
                mongo_db_name, mongo_coll_params, chiave_input
            )
            parametri_modello_lista = list(parametri_modello.keys()) if parametri_modello else []
            parametri_con_dettaglio, proposta_visita_prestazioni_icd9cm, esami_spostati = separa_parametri_da_esami(
                parametri_con_dettaglio or [],
                proposta_visita_prestazioni_icd9cm
            )
            if esami_spostati:
                logging.info(
                    "🧹 [B1] Pulizia cache: spostati %d esami in proposta visita: %s",
                    len(esami_spostati),
                    ", ".join(esami_spostati)
                )
                parametri_modello_lista = [p.get("nome", "") for p in parametri_con_dettaglio if p.get("nome")]

            output_completo = {
                "transaction_id":                   transaction_id,
                "chiave_input":                     chiave_input,
                "rag_source":                       None,
                "kb_patologia":                     None,
                "kb_fonti":                         [],
                "risultato_prompt_1_parametri":     parametri_modello_lista,
                "parametri_con_dettaglio":          parametri_con_dettaglio or [],
                "proposta_visita_prestazioni_icd9cm": proposta_visita_prestazioni_icd9cm,
                "risultato_prompt_2_fhir":          query_fhir,
                "risultato_prompt_3_verifica_fhir": "Parametri recuperati dal Database"
            }

            logging.info(f"✅ PATH B1 completato: {len(parametri_modello_lista)} parametri")
            return func.HttpResponse(
                json.dumps(output_completo, ensure_ascii=False),
                mimetype="application/json",
                status_code=200
            )

        # PATH B2 — Flag 1: ricalcola con RAG + feedback
        else:
            logging.info("🔄 PATH B2: Ricalcolo con KB feedback + RAG")

            diagnosi            = paziente.get("diagnosi", "")
            livello_complessita = determina_livello_complessita(comorbidita=paziente.get("comorbidita"))

            rag_source_db = leggi_rag_source(mongo_db_name, mongo_coll_params, chiave_input)
            kb_result  = None
            rag_source = False

            kb_feedback = leggi_kb_completa(mongo_db_name, mongo_coll_params, chiave_input)

            try:
                if rag_source_db:
                    logging.info("🔍 [B2] rag_source DB=true → rieseguo retrieval su AI Search")
                    kb_result = cerca_patologia_in_kb(diagnosi, livello=livello_complessita)
                    if kb_result:
                        rag_source = True
                        logging.info(f"✅ [RAG] Trovata in KB: '{kb_result['patologia']}'")
                    else:
                        logging.info("⚠️ [RAG] Retrieval senza risultati utili, fallback a KB feedback MongoDB")
                else:
                    logging.info("ℹ️ [B2] rag_source DB=false → salto retrieval RAG e uso flusso KB feedback")
            except RAGError as e:
                logging.warning(f"⚠️ [RAG] Errore retrieval B2, continuo senza RAG: {e}")

            try:
                if rag_source_db and rag_source and kb_result and kb_feedback:
                    logging.info("📝 Prompt 1 (B2) — PATH: KB feedback + candidati RAG")
                    prompt_1_text = build_prompt1_CON_kb_con_rag_suggeriti(
                        paziente,
                        professionista,
                        kb_feedback,
                        kb_result
                    )
                elif kb_feedback:
                    logging.info("📝 Prompt 1 (B2) — PATH: KB feedback MongoDB")
                    prompt_1_text = build_prompt1_CON_kb(paziente, professionista, kb_feedback)
                elif rag_source and kb_result:
                    logging.info("📝 Prompt 1 (B2) — PATH: RAG")
                    prompt_1_text = build_prompt_1_con_rag(paziente, professionista, kb_result)
                else:
                    logging.info("📝 Prompt 1 (B2) — PATH: LLM puro")
                    prompt_1_text = build_prompt_1_no_KB(paziente, professionista)

                risposta_prompt_1 = client_ai.chat_completion(
                    model=ai_foundry_model,
                    messages=[
                        {"role": "system", "content": "Sei un medico esperto. Rispondi SEMPRE e SOLO con un JSON valido, senza testo aggiuntivo."},
                        {"role": "user",   "content": prompt_1_text}
                    ],
                    temperature=0.0
                )
                logging.info("✅ Prompt 1 (B2) completato.")

            except Exception as e:
                logging.error(f"❌ Errore Prompt 1 (B2): {e}")
                return func.HttpResponse(f"Errore AI Prompt 1: {e}", status_code=500)

            try:
                dati_prompt_1         = _parse_llm_json(risposta_prompt_1)
                proposta_visita_prestazioni_icd9cm = dati_prompt_1.get("proposta_visita_prestazioni_icd9cm")
                parametri_dettagliati = dati_prompt_1.get("parametri", [])
                if rag_source and kb_result:
                    parametri_dettagliati = normalizza_fonti_parametri_rag(parametri_dettagliati, kb_result)
                else:
                    parametri_dettagliati = normalizza_parametri_llm_no_rag(parametri_dettagliati)
                parametri_dettagliati = normalizza_parametri_dettaglio_output(parametri_dettagliati)
                parametri_dettagliati, proposta_visita_prestazioni_icd9cm, esami_spostati = separa_parametri_da_esami(
                    parametri_dettagliati,
                    proposta_visita_prestazioni_icd9cm
                )
                parametri_dettagliati, parametri_scartati_no_loinc = filtra_parametri_clinici_con_loinc(parametri_dettagliati)
                if parametri_scartati_no_loinc:
                    logging.info(
                        "🧹 [B2] Parametri scartati senza codice LOINC: %s",
                        ", ".join(parametri_scartati_no_loinc)
                    )
                parametri_dettagliati, parametri_aggiunti = completa_parametri_a_sei(
                    parametri_dettagliati,
                    kb_result=kb_result if rag_source else None
                )
                if parametri_aggiunti:
                    logging.info("➕ [B2] Parametri aggiunti automaticamente per arrivare a 6: %d", parametri_aggiunti)
                if esami_spostati:
                    logging.info(
                        "🧹 [B2] Parametri filtrati: spostati %d esami in proposta visita: %s",
                        len(esami_spostati),
                        ", ".join(esami_spostati)
                    )
                lista_paramerti = [p.get("nome", "") for p in parametri_dettagliati if p.get("nome")]
                logging.info(
                    "📊 [B2] Prompt 1 normalizzato | parametri_finali=%d | proposta_visita_presente=%s",
                    len(lista_paramerti),
                    bool(proposta_visita_prestazioni_icd9cm)
                )
            except Exception:
                proposta_visita_prestazioni_icd9cm = None
                parametri_dettagliati = []
                lista_paramerti       = estrai_parametri(risposta_prompt_1)

            parametri_testo = prepara_parametri_per_prompt_2(
                parametri_dettagliati=parametri_dettagliati,
                risposta_prompt_1=risposta_prompt_1,
                rag_source=rag_source,
                kb_result=kb_result
            )

            try:
                logging.info("🔎 Parametri passati a Prompt 2 (B2): %s", parametri_testo.replace("\n", " | ")[:500])
                prompt_2_text = build_prompt_2(parametri_testo)
                query_fhir = client_ai.chat_completion(
                    model=ai_foundry_model,
                    messages=[
                        {"role": "system", "content": "Sei un esperto FHIR."},
                        {"role": "user",   "content": prompt_2_text}
                    ],
                    temperature=0.0
                )
                query_fhir = normalizza_query_fhir_lastn_unica(query_fhir, parametri_dettagliati)
                logging.info("✅ Prompt 2 (B2) completato.")
                logging.info("📤 Query FHIR normalizzata (B2): %s", query_fhir.replace("\n", " | ")[:500])

            except Exception as e:
                logging.error(f"❌ Errore Prompt 2 (B2): {e}")
                return func.HttpResponse(f"Errore AI Prompt 2: {e}", status_code=500)

            try:
                prompt_3_text = build_prompt_judge(parametri_testo, paziente, professionista)
                verifica_fhir = client_ai.chat_completion(
                    model=model_judge,
                    messages=[
                        {"role": "system", "content": "Sei un medico esperto valutatore."},
                        {"role": "user",   "content": prompt_3_text}
                    ],
                    temperature=0.0
                )
                logging.info("✅ Prompt 3 (B2) completato.")

            except Exception as e:
                logging.error(f"❌ Errore Prompt 3 (B2): {e}")
                return func.HttpResponse(f"Errore AI Prompt 3: {e}", status_code=500)

            try:
                with get_mongo_client() as mongo_client:
                    db   = mongo_client[mongo_db_name]
                    coll = db[mongo_coll_params]
                    coll.update_one(
                        {"chiave_input": chiave_input},
                        {
                            "$set": {
                                "transaction_id":          transaction_id,
                                "output_parametri":        {param: 0 for param in lista_paramerti},
                                "parametri_con_dettaglio": parametri_dettagliati,
                                "proposta_visita_prestazioni_icd9cm": proposta_visita_prestazioni_icd9cm,
                                "rag_source":              rag_source,
                                "kb_patologia":            kb_result.get("patologia") if kb_result else None,
                                "kb_fonti":                kb_result.get("fonti") if kb_result else [],
                                "livello_complessita":     livello_complessita,
                                "parametri_aggiunti":      {},
                                "Feedback_number":         0,
                                "Flag_qualità":            0,
                                "GET_FHIR":                query_fhir,
                                "timestamp":               timestamp_now,
                            }
                        }
                    )
                    logging.info("✅ Documento aggiornato in MongoDB (B2)")

            except Exception as e:
                logging.error(f"❌ Errore aggiornamento MongoDB (B2): {e}")

            output_completo = {
                "transaction_id":                   transaction_id,
                "chiave_input":                     chiave_input,
                "rag_source":                       rag_source,
                "kb_patologia":                     kb_result.get("patologia") if kb_result else None,
                "kb_fonti":                         kb_result.get("fonti") if kb_result else [],
                "livello_complessita":              livello_complessita,
                "risultato_prompt_1_parametri":     lista_paramerti,
                "parametri_con_dettaglio":          parametri_dettagliati,
                "proposta_visita_prestazioni_icd9cm": proposta_visita_prestazioni_icd9cm,
                "risultato_prompt_2_fhir":          query_fhir,
                "risultato_prompt_3_verifica_fhir": verifica_fhir
            }

            logging.info(f"✅ Risposta PATH B2 pronta. rag_source={rag_source}")
            return func.HttpResponse(
                json.dumps(output_completo, ensure_ascii=False),
                mimetype="application/json",
                status_code=200
            )


@app.route(route="ui", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def serve_ui(req: func.HttpRequest) -> func.HttpResponse:
    """Serve il frontend HTML. L'URL dell'API non è mai esposto al client."""
    html_path = Path(__file__).parent / "index.html"
    try:
        html = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return func.HttpResponse("Frontend non trovato.", status_code=404)
    return func.HttpResponse(html, mimetype="text/html", status_code=200)


@app.route(route="Elaborafeedback", methods=["POST"])
def ProxyElaboraFeedback(req: func.HttpRequest) -> func.HttpResponse:
    feedback_url = os.environ.get("FEEDBACK_FUNCTION_URL")
    if not feedback_url:
        return func.HttpResponse(
            "Variabile d'ambiente FEEDBACK_FUNCTION_URL mancante.",
            status_code=500,
        )

    try:
        payload = req.get_body()
        proxy_req = Request(
            feedback_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(proxy_req, timeout=30) as response:
            body = response.read().decode("utf-8")
            status = response.getcode()
            content_type = response.headers.get("Content-Type", "application/json")

        return func.HttpResponse(body=body, status_code=status, mimetype=content_type)

    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logging.error(f"❌ Proxy feedback HTTPError {e.code}: {body[:400]}")
        return func.HttpResponse(body=body, status_code=e.code)
    except URLError as e:
        logging.error(f"❌ Proxy feedback URLError: {e}")
        return func.HttpResponse("Errore connessione servizio feedback.", status_code=502)
    except Exception as e:
        logging.error(f"❌ Proxy feedback generic error: {e}")
        return func.HttpResponse("Errore interno proxy feedback.", status_code=500)