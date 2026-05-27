import requests
import json
import re , os
from rag_store import load_rules_file ,extract_evidence ,pre_validate



def extract_json(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group() if match else None

def extract_policy_patient_from_text(text: str):
    # Flexible pattern: match letters, numbers, dashes, underscores, slashes
    policy_match = re.search(r"(?:Policy\s*(?:No|Number|#|ID)[:\s]*)([A-Za-z0-9\-/]+)", text, re.IGNORECASE)
    policy_number = policy_match.group(1).strip() if policy_match else ""

    patient_match = re.search(r"(?:Patient\s*(?:ID|No|#|MRN)[:\s]*)([A-Za-z0-9\-/]+)", text, re.IGNORECASE)
    patient_id = patient_match.group(1).strip() if patient_match else ""

    return policy_number, patient_id


def build_rules_text(rules):
    text = ""
    for rule in rules:
        text += f"""
        Rule ID: {rule['rule_id']}
        Condition: {rule['condition']}
        Required Evidence: {', '.join(rule['required_evidence'])}
        """
    return text

def normalize_value(value):
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if value is None:
        return ""
    return str(value)

def run_mistral_agent(ocr_text: str):
    # 1. Load rules (RAG source)
    relevant_rules = load_rules_file()

    # 2. Extract evidence deterministically
    extracted_evidence = sorted(set(extract_evidence(ocr_text)))

    # 3. Build extraction-only prompt (NO approval logic here)
    prompt = f"""
    You are a medical information extraction AI.

    RULES:
    - Extract structured data ONLY
    - Return VALID JSON ONLY
    - DO NOT decide approval or rejection
    - DO NOT apply compliance rules
    - DO NOT invent information
    - Use "" if a field is missing
    - Normalize dates to DD/MM/YYYY

    OCR TEXT:
    {ocr_text}

    OUTPUT FORMAT:
    {{
    "policy_number": "",
    "patient_id": "",
    "diagnosis": "",
    "hospital_name": "",
    "doctor_id": "",
    "doctor_name": "",
    "doctor_notes": "",
    "date": ""
    }}
    """

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1
        }
    }

    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
    response.raise_for_status()

    raw_text = response.json().get("response", "")
    json_text = extract_json(raw_text)

    try:
        data = json.loads(json_text) if json_text else {}
    except json.JSONDecodeError:
        data = {}

    # 4. Regex fallback
    policy_number, patient_id = extract_policy_patient_from_text(ocr_text)

    # 5. Normalize extracted values
    diagnosis = normalize_value(data.get("diagnosis"))
    visit_date = (data.get("date"))

    # 6. Apply RULE VALIDATION (outside LLM)
    validation = pre_validate(
        relevant_rules=relevant_rules,
        extracted_evidence=extracted_evidence,
        diagnosis=diagnosis
    )

    failed = [r for r in validation if not r["is_valid"]]

    if failed:
        status = "REJECTED"
        reason = f"Missing required evidence: {', '.join(failed[0]['missing_evidence'])}"
    else:
        status = "APPROVED"
        reason = "All required medical evidence is present"

    # 7. Final result (DB-safe)
    result= {
        "status": status,
        "reason": reason,
        "policy_number": normalize_value(data.get("policy_number")) or policy_number,
        "patient_id": normalize_value(data.get("patient_id")) or patient_id,
        "diagnosis": diagnosis,
        "hospital_name": normalize_value(data.get("hospital_name")),
        "doctor_id": normalize_value(data.get("doctor_id")),
        "doctor_name": normalize_value(data.get("doctor_name")),
        "doctor_notes": normalize_value(data.get("doctor_notes")),
        "visit_date": visit_date,
        "ocr_text": ocr_text,
        "raw_text": raw_text
    }

    return result