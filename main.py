from fastapi import FastAPI, UploadFile, File
import os
from database import Base, engine, SessionLocal
from models import ImageRecord, Claim
from ocr_service import extract_text ,extract_text_local ,extract_text_trocr  ,extract_text_paddle
from llm_service import run_mistral_agent
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

# Create DB tables
Base.metadata.create_all(bind=engine)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

origins = [
    "http://localhost:3000",   # React
    "http://127.0.0.1:3000",
    "http://localhost:5173",   # Vite
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 👈 specific origins
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE
    allow_headers=["*"],        # Authorization, Content-Type
)


def parse_date(date_str):
    if not date_str:
        return None

    date_str = date_str.strip().lower()

    # OCR garbage check
    if date_str in ["date", "dt", "dated", ""]:
        return None

    formats = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d.%m.%Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None   # don't crash production


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    db = SessionLocal()

    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Step 1: OCR
    extracted_text = extract_text_trocr(file_path)

    # Save image record
    image_record = (
    db.query(ImageRecord)
    .filter(ImageRecord.filename == file.filename)
    .first()
)

    if image_record:
        # update existing record
        image_record.ocr_text = extracted_text
    else:
        # create new record
        image_record = ImageRecord(
            filename=file.filename,
            ocr_text=extracted_text
        )
        db.add(image_record)

    db.commit()
    db.refresh(image_record)

    # Step 2: Run Local LLM Auditor
    claim_result = run_mistral_agent(extracted_text)
    claim_data = claim_result

    claim = Claim(
        image_id=image_record.id,

        # top-level fields
        status=claim_result.get("status"),
        reason=claim_result.get("reason"),

        policy_number=claim_result.get("policy_number"),
        patient_id=claim_result.get("patient_id"),

        # claim object fields
        diagnosis=claim_data.get("diagnosis"),
        hospital_name=claim_data.get("hospital_name"),

        doctor_id=claim_data.get("doctor_id"),
        doctor_name=claim_data.get("doctor_name"),
        doctor_notes=claim_data.get("doctor_notes"),

        visit_date = parse_date(claim_data.get("date"))
        )

    db.add(claim)
    db.commit()
    db.refresh(claim)

    return {
        "image_id": image_record.id,
        "ocr_text": extracted_text,
        "claim": claim_result
    }


@app.get("/claims")
async def list_claims():
    db = SessionLocal()
    claims = db.query(Claim).all()

    return claims
