---
title: Phi2 JSON Extractor
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Phi-2 Nested JSON Extractor

Fine-tuned Microsoft Phi-2 (2.78B) using QLoRA to extract 
structured nested JSON from natural language sentences.

**Live API:** https://Vimal1415-phi2-json-extractor.hf.space

## Results

| Metric | Score |
|--------|-------|
| Base model accuracy | 60% |
| Fine-tuned accuracy | 90% |
| Field accuracy | 79% |
| CI/CD eval gate | 85% threshold |
| Trainable parameters | 5.2M / 2.78B (0.19%) |

## Architecture
Natural language input
↓
FastAPI server (POST /extract)
↓
Phi-2 + LoRA adapter (HuggingFace Hub)
↓
3-stage JSON parser
↓
Structured nested JSON response

## Example

**Input:**
```json
{
  "text": "Sneha is a senior QA engineer in Chennai 
           with 6 years experience. She is 29 and 
           holds AWS certifications."
}
```

**Output:**
```json
{
  "person": {"name": "Sneha", "age": 29},
  "work": {
    "job": "QA engineer",
    "city": "Chennai", 
    "years_experience": 6
  },
  "details": {
    "is_senior": true,
    "has_certification": true
  }
}
```

## Tech Stack

- **Model:** Microsoft Phi-2 (2.78B parameters)
- **Fine-tuning:** QLoRA (rank=16, alpha=32) via HuggingFace PEFT
- **Training:** TRL SFTTrainer on Kaggle T4 GPU
- **API:** FastAPI + Pydantic
- **Deployment:** Docker + HuggingFace Spaces
- **CI/CD:** GitHub Actions with regression gate

## Training Details

- Dataset: 178 synthetic examples (170 train, 30 eval)
- Epochs: 5
- Learning rate: 2e-4
- Final eval loss: 0.09
- Training time: ~4.5 minutes on Kaggle T4

## CI/CD Pipeline

Every push to main triggers:
1. Fresh Ubuntu VM spins up
2. Dependencies installed
3. FastAPI server started
4. 10-test eval suite runs
5. If accuracy >= 85% → deploy to HF Spaces
6. If accuracy < 85% → deployment blocked

## Project Structure
phi2-json-extractor/
├── app/
│   ├── main.py       # FastAPI server
│   ├── model.py      # Model loading + inference
│   └── schemas.py    # Request/response schemas
├── eval/
│   └── test_api.py   # Regression eval suite
├── Dockerfile
├── requirements.txt
└── .github/
└── workflows/
└── ci.yml    # CI/CD pipeline
