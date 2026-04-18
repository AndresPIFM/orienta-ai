# Orienta.ai — Vocational Guidance Agent for Mexican Students

**Advanced AI for Data Science II · Tecnológico de Monterrey**  
Multi-agent system that centralizes academic and labor market data to deliver personalized career recommendations in natural language.

---

## The Problem

In Mexico, the information students need to choose a university career is scattered across institutional portals, PDFs, internal databases, and unstructured documents. This makes comparing options genuinely difficult and increases uncertainty at one of the most important decision points in a student's life.

Orienta.ai centralizes all of that into a single conversational agent.

---

## What It Does

| Feature | Description |
|---|---|
| Career Recommendation | Conversational questionnaire → top 3 career matches based on interests, aptitudes, and academic performance |
| University Comparison | Public and private universities across all 32 Mexican states, with costs and scholarship availability |
| Labor Statistics | Employment rates, average salaries, informality index, and growth trends per career |
| Kardex OCR | Extracts grades from academic transcripts (PDF/image) using OCI Document Understanding |
| Web Chatbot | Conversational interface built with Oracle APEX + JavaScript |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Oracle APEX)                │
│                     Web Chatbot Interface                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Orchestrator — Oracle Digital Assistant (ODA)   │
│                     Oracle Gen AI Service                    │
└──┬────────────────┬──────────────┬─────────────────┬────────┘
   │                │              │                  │
   ▼                ▼              ▼                  ▼
Conversation   RAG Semantic   Universities       OCR / Document
& Profile      Retrieval      Module (REST)      Understanding
Agent          Module         185 universities   (Kardex grades)
   │                │
   ▼                ▼
Recommendation  TimescaleDB
Engine (Top 3)  Vector Store
   │            (embeddings)
   ▼
PostgreSQL (Autonomous DB)
176 careers · 185 universities
```

**Core components:**

- **Oracle Digital Assistant (ODA)** — Orchestrates all agents and conversational flows
- **Recommendation Engine** — Hybrid content-based + semantic scorer using cosine distance on 3072-dim embeddings (`text-embedding-3-large`)
- **RAG Module** — Semantic search over career and university databases via TimescaleDB vector extensions
- **OCR Module** — OCI Document Understanding extracts grades from scanned transcripts
- **University Module** — REST microservice filtering 185 institutions by state, type, cost, and scholarship rules

---

## Recommendation Model

Each career in the catalog gets scored against the student profile across weighted dimensions:

| Block | Weight | Dimensions |
|---|---|---|
| Interests & Vocation | 40% | RIASEC code, professional world description, labor values |
| Aptitudes & Skills | 25% | Key aptitudes, attitudes, how they want to help |
| Academic | 20% | Strong subjects, career-required subjects |
| Work Environment | 10% | Dominant work type, typical environment, labor output |
| Context | 5% | General area, people contact level, industry |

Scoring approaches: cosine similarity on text embeddings, discrete intersection for aptitude tags, ordinal comparison for contact/teamwork levels, and catalog matching for RIASEC codes and work environments.

$$\text{Score}(j) = \sum_k w_k \cdot s_j^k$$

---

## Results

| Test Group | Accuracy | Avg. Score |
|---|---|---|
| Internal tests (7 cases) | 100% | 0.87 |
| Real user tests (7 cases) | 85.7% | 0.79 |

The one misclassification (Medicine → Nutrition) happened because the student described wanting to help people and promote healthy habits — without mentioning clinical environments or hard sciences. A known problem in vocational guidance: how someone describes themselves doesn't always match what their chosen career actually involves.

---

## Datasets

Built manually from official educational portals, university websites, and labor statistics sources.

- `base_carreras` — 176 university careers with RIASEC codes, aptitudes, work environments, and labor trends
- `base_universidades` — 185 universities across 32 Mexican states with academic offerings, costs, and scholarship rules

Scholarship thresholds for private universities: 20% for GPA 8.0–8.5, 50% for 8.5–9.5, and 100% above 9.5.

---

## My Contribution

My role was data engineering and statistical analysis:

- **Web scraping pipeline** (`data_pipeline/web_scraping.py`) — automated extraction of employer reputation scores from QS World University Rankings for UNAM and Tec de Monterrey across 17 career fields, merged with labor statistics from IMCO's Compara Carreras portal (employment rate, informality, average salary, gender wage gap, postgrad premium)
- **Time series modeling** (`modeling/notebook7.py`) — built and evaluated CNN and LSTM architectures for forecasting as part of the predictive analytics work
- **Data preparation** — cleaning, normalization, duplicate removal, RIASEC labeling, and merging of multi-source datasets
- **Database schema** — relational schema for PostgreSQL/TimescaleDB supporting semantic queries and similarity calculations

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud Platform | Oracle Cloud Infrastructure (OCI) |
| AI Orchestration | Oracle Digital Assistant + Oracle Gen AI |
| Embeddings | OpenAI `text-embedding-3-large` (3072 dims) |
| Generative Model | OCI Generative AI (LLM Blocks) |
| Vector Store | TimescaleDB with pgvector |
| Relational DB | Oracle Autonomous Database (PostgreSQL) |
| OCR | OCI Document Understanding |
| Backend | OCI Functions (serverless) |
| Frontend | Oracle APEX + JavaScript |
| Data Collection | Selenium, Requests |
| ML/DL | TensorFlow/Keras, scikit-learn, pandas, numpy |

---

## Repository Structure

```
orienta-ai/
├── README.md
├── data_pipeline/
│   └── web_scraping.py       # QS Rankings + IMCO labor stats scraper
├── modeling/
│   └── notebook7.py          # Time series forecasting (CNN/LSTM)
└── docs/
    ├── architecture.md       # Detailed system architecture
    └── Orienta_ai_Report.pdf # Full project report
```

---

## Demo

[Watch the agent in action](https://youtu.be/2CnAWxUn4kQ)

---

## Team

Tecnológico de Monterrey — Advanced AI for Data Science II (Nov–Dec 2025)

Andrew Williams · Alejandro Miloslavich · Marco Miloslavich · Luis Navarro · Bruno Zamora · Andrés Pi · Samuel López

---

## References

- Soumya & Ahmed (2025). *AI powered career path recommender*. Foundry Journal.
- National High School Journal of Science (2025). *RAG for AI-powered career guidance in high schools*.
- Observatorio Laboral México (2017). *Carreras con mayor ingreso*.
- Fishbowl Solutions (2024). *Using generative AI inside Oracle Gen AI: Knowledge Dialogs and LLM Blocks*.
