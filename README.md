---
title: Phi2 JSON Extractor
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Phi-2 Nested JSON Extractor

Fine-tuned Microsoft Phi-2 (2.78B) using QLoRA to extract structured nested JSON from natural language sentences.

## API Endpoints

- `GET /health` — health check
- `POST /extract` — extract nested JSON from text

## Example

**Input:**
```json
{
  "text": "Aarav is 28 years old and works as a senior software engineer in Bangalore with 6 years experience."
}
```

**Output:**
```json
{
  "person": {"name": "Aarav", "age": 28},
  "work": {"job": "software engineer", "city": "Bangalore", "years_experience": 6},
  "details": {"is_senior": true, "has_certification": false}
}
```

## Model

- Base: microsoft/phi-2 (2.78B parameters)
- Fine-tuned with QLoRA (rank=16, alpha=32)
- Trained on 178 nested JSON extraction examples
- Accuracy: 60% → 90% improvement over base model