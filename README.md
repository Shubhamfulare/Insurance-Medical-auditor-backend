# Insurance-Medical-auditor
AI based medical insurance claim auditor



What is does 

- Accepts insurance clain documents (Pdf/image)
- Extracts text via ocr
- Cleans and normalises  messy OCR input using LLM
- Runs claims through rag pipeline against policy rules
- returns : valid / Invalid / Need Review +  Reasoning

Why Local LLM ?
Medical data is sensitive , this system runs 100% offline using ollama - no api keys no cloud , no data exposure

Tech stack
React js 
Tailwind
Fast api
ollma mistral 
Tr OCR




To run 
backend code with ollma model on local 



Architechure Daigram


┌─────────────────────────────────────────────────────────────────────┐
│                         USER / AUDITOR PORTAL                      │
│                         React JS Frontend                          │
│                                                                     │
│  Features:                                                          │
│  - Claim Upload                                                     │
│  - Document Viewer                                                  │
│  - Audit Dashboard                                                  │
│  - Claim Status Tracking                                            │
│  - AI Justification Review                                          │
│  - Reports & Analytics                                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTPS / REST API
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND                           │
│                                                                     │
│  Modules:                                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Authentication & Authorization                               │  │
│  │ JWT / Role Based Access                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Claim Management Service                                      │  │
│  │ Upload / Validation / Status Workflow                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ OCR Processing Service                                        │  │
│  │ TR OCR extracts:                                               │  │
│  │ - Patient Details                                              │  │
│  │ - Hospital Info                                                │  │
│  │ - Billing Data                                                 │  │
│  │ - Diagnosis & Treatment                                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ AI Processing Pipeline                                         │  │
│  │ Ollama + Mistral LLM                                           │  │
│  │ - Medical document understanding                               │  │
│  │ - Fraud detection hints                                        │  │
│  │ - Missing field analysis                                       │  │
│  │ - Claim validation                                              │  │
│  │ - AI justification generation                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ RAG Engine                                                     │  │
│  │ Retrieval Augmented Generation                                 │  │
│  │ - Fetch policy rules                                           │  │
│  │ - Fetch historical claims                                      │  │
│  │ - Fetch compliance guidelines                                  │  │
│  │ - Context injection to LLM                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                 │                         │
                 │                         │
                 ▼                         ▼

┌─────────────────────────┐      ┌──────────────────────────────┐
│        SQL DATABASE     │      │      VECTOR DATABASE         │
│                         │      │                              │
│ PostgreSQL / MySQL      │      │ ChromaDB / FAISS            │
│                         │      │                              │
│ Stores:                 │      │ Stores:                     │
│ - Users                 │      │ - Embeddings                │
│ - Claims                │      │ - Medical policies          │
│ - Audit logs            │      │ - Claim knowledge base      │
│ - OCR extracted data    │      │ - Historical audit context  │
│ - AI decisions          │      │                              │
└─────────────────────────┘      └──────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES                           │
│                                                                     │
│  - TR OCR Engine                                                    │
│  - Hospital APIs                                                    │
│  - Insurance Provider APIs                                          │
│  - Email / Notification Services                                    │
│  - File Storage (S3 / MinIO)                                        │
└─────────────────────────────────────────────────────────────────────┘


===========================
CLAIM PROCESSING FLOW
===========================

1. User uploads medical claim documents from React UI
2. FastAPI receives and stores files
3. OCR extracts structured medical data
4. Extracted text sent to Ollama Mistral
5. RAG fetches policy and historical claim context
6. LLM validates and justifies claim
7. AI response stored in SQL DB
8. Auditor reviews AI recommendation in dashboard
9. Final approve/reject decision generated

 
