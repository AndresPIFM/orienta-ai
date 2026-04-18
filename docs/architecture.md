# Orienta.ai — System Architecture

## Overview

Orienta.ai is a multi-agent vocational guidance system deployed on Oracle Cloud Infrastructure (OCI). The user interacts through a web interface; all messages flow through an orchestrator in Oracle Digital Assistant (ODA) that routes requests to specialized agents.

---

## Component Map

```
User (Browser)
    │
    ▼
Oracle APEX + JavaScript (Web Frontend)
    │  REST / WebSocket
    ▼
Oracle Digital Assistant — ODA Orchestrator
    │
    ├──► Conversation & Profile Agent
    │        │ Collects interests, aptitudes, values, environment preferences
    │        │ Builds structured perfil_usuario_final
    │        ▼
    │    Recommendation Engine (OCI Functions — Python)
    │        │ Hybrid scorer: content-based + semantic RAG
    │        │ Returns Top 3 careers with global score + explanations
    │        ▼
    │    TimescaleDB Vector Store (pgvector)
    │        176 careers × 3072-dim embeddings (text-embedding-3-large)
    │
    ├──► RAG Semantic Retrieval Module
    │        │ API: semantic search over careers + universities
    │        │ Returns relevant text passages for LLM context
    │        ▼
    │    Oracle Autonomous Database (PostgreSQL)
    │        base_carreras (176 rows) · base_universidades (185 rows)
    │
    ├──► University Agent
    │        REST microservice — filters universities by:
    │        state, institution type (public/private), cost, scholarship rules
    │        Crosses student GPA against scholarship thresholds
    │
    └──► OCR / Document Understanding Agent
             OCI Document Understanding (pre-trained models)
             Input:  PDF or image of academic transcript (Kardex)
             Output: JSON { materia, calificacion } per subject
                     → GPA calculation
                     → Scholarship eligibility
                     → Subject strength weights for recommendation engine
```

---

## Data Layer

| Store | Technology | Contents |
|---|---|---|
| Relational DB | Oracle Autonomous Database (PostgreSQL) | Careers, universities, scholarship rules |
| Vector Store | TimescaleDB + pgvector | 3072-dim embeddings per career field |
| Document Store | OCI Object Storage | Uploaded Kardex PDFs |

---

## Recommendation Engine Detail

### Input
Structured `perfil_usuario_final` JSON with:
- `areas_preferidas` — ordered list from 7-category catalog
- `industria_preferida` — single value from 10-category catalog
- `materias_like` / `materias_dislike` — mapped to 16 standardized subject tags
- `codigo_riasec_usuario` — 2–3 letter RIASEC code
- `tipo_trabajo_dominante_usuario` — 7-category work style
- `entornos_preferidos` — 1–2 values from 10 environment types
- `nivel_contacto_personas_usuario` — ordinal 1–3
- `nivel_trabajo_equipo_usuario` — ordinal 1–3
- `quiere_ayuda_directa` — ordinal 1–3
- Free-text fields for semantic comparison via embeddings

### Scoring Pipeline

```
For each career j in catalog (176 careers):
    1. Compute sub-scores s_jk ∈ [0, 1] per dimension k
       - Embedding fields: s = 1 / (1 + 0.6 * cosine_distance(u, v))
       - Chunk boost:      s = 0.2 * s_embed + 0.8 * s_chunk
       - RIASEC:           s = 0.6 * letter_overlap + 0.4 * s_embed
       - Ordinal fields:   s = {1.0, 0.8, 0.5} for |u - c| = {0, 1, ≥2}
       - Categorical:      s = 1.0 if match, else 0.5
       - Subjects:         s = f(like_matches, dislike_matches)

    2. Weighted sum:
       Score_Global(j) = Σ_k  w_k · s_jk

    3. Sort descending → select Top 3
```

### Dimension Weights

```
BLOCK 1 — Interests & Vocation (40%)
  que_se_hace_mundo_profesional  0.15
  riasec                         0.12
  perfil_laboral_y_valores       0.12 (+ chunk boost)

BLOCK 2 — Aptitudes & Skills (25%)
  aptitudes_clave                0.12
  actitudes_clave                0.08
  como_ayuda                     0.05

BLOCK 3 — Academic (20%)
  materias_fuertes_prepa         0.10
  que_se_hace_carrera            0.06 (+ chunk boost)
  materias_carrera               0.04

BLOCK 4 — Work Environment (10%)
  tipo_trabajo_dominante         0.05
  entorno_trabajo_tipico         0.03
  salida_laboral                 0.02 (+ chunk boost)

BLOCK 5 — Context (5%)
  area_general                   0.03
  industria                      0.01
  contacto_personas              0.01
  nivel_trabajo_equipo           0.005
  ayuda_directamente_personas    0.005
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Embedding model | `text-embedding-3-large` (OpenAI) |
| Embedding dimensions | 3072 |
| Similarity metric | Cosine distance |
| LLM max_tokens | 600 |
| LLM temperature | 1.0 |
| LLM top_p | 0.75 |

---

## Deployment

All services run within OCI:

| Component | OCI Service |
|---|---|
| Conversational agent | Oracle Digital Assistant |
| Backend functions | OCI Functions (serverless) |
| Relational + vector DB | Autonomous Database + TimescaleDB |
| Semantic search | OCI Search with OpenSearch |
| Document processing | OCI Document Understanding |
| Frontend | Oracle APEX |
| Secrets / networking | OCI Vault + VCN |

Auto-scaling rules are configured on both OCI Functions and Autonomous Database to handle concurrent users without performance degradation.

---

## Security

- Data in transit: TLS 1.2+
- Data at rest: AES-256 encryption (OCI managed keys)
- Access control: Role-based policies per module via OCI IAM
- Student data: Anonymized in analytics pipelines
