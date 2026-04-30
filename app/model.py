# app/model.py
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from huggingface_hub import hf_hub_download
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_ID  = "microsoft/phi-2"
ADAPTER_ID     = "Vimal1415/phi2-nested-json-lora"
MAX_NEW_TOKENS = 400
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION = """Extract the person's details from the sentence and return a nested JSON object with this exact structure:
{"person": {"name": "...", "age": ...}, "work": {"job": "...", "city": "...", "years_experience": ...}, "details": {"is_senior": true/false, "has_certification": true/false}}

Rules:
- is_senior is true if years_experience >= 5 OR the sentence mentions "senior" or "promoted"
- has_certification is true if the sentence mentions "certified" or "certification"
- years_experience is 0 if they are a fresher or it is their first job
- Return compact JSON on a single line with no spaces or newlines"""

_model     = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    logger.info("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True
    )
    _tokenizer.pad_token    = _tokenizer.eos_token
    _tokenizer.padding_side = "right"

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    logger.info("Loading LoRA adapter...")
    _model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
        is_trainable=False,
    )

    _model.eval()
    logger.info("✓ Model ready on cpu")


def get_model():
    if _model is None or _tokenizer is None:
        load_model()
    return _model, _tokenizer


def build_prompt(sentence: str) -> str:
    return f"""### Instruction:
{INSTRUCTION}

### Input:
{sentence}

### Response:"""


def parse_json_from_output(raw_text: str) -> dict:
    start = raw_text.find('{')
    if start == -1:
        raise ValueError(f"No JSON found in output: {raw_text[:100]}")

    json_str = raw_text[start:]

    # Try 1 — parse complete JSON
    end = json_str.rfind('}') + 1
    if end > 0:
        try:
            parsed = json.loads(json_str[:end])
            return _validate_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 2 — repair truncated JSON
    repaired = _repair_json(json_str)
    if repaired:
        try:
            parsed = json.loads(repaired)
            return _validate_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 3 — regex extraction as last resort
    parsed = _extract_fields_regex(raw_text)
    if parsed:
        return _validate_parsed(parsed)

    raise ValueError(f"Could not parse JSON from output: {raw_text[:150]}")


def _repair_json(json_str: str) -> str:
    clean = json_str.rstrip()
    if clean.endswith(','):
        clean = clean[:-1]
    open_count   = clean.count('{')
    closed_count = clean.count('}')
    missing      = open_count - closed_count
    if missing > 0:
        clean += '}' * missing
    return clean if missing >= 0 else None


def _extract_fields_regex(text: str) -> dict | None:
    import re
    def find(pattern, default=None):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else default

    name   = find(r'"name"\s*:\s*"([^"]+)"')
    age    = find(r'"age"\s*:\s*(\d+)')
    job    = find(r'"job"\s*:\s*"([^"]+)"')
    city   = find(r'"city"\s*:\s*"([^"]+)"')
    exp    = find(r'"years_experience"\s*:\s*(\d+)', '0')
    senior = find(r'"is_senior"\s*:\s*(true|false)', 'false')
    cert   = find(r'"has_certification"\s*:\s*(true|false)', 'false')

    if not all([name, age, job, city]):
        return None

    return {
        "person": {"name": name, "age": int(age)},
        "work":   {"job": job, "city": city, "years_experience": int(exp)},
        "details": {
            "is_senior":         senior == 'true',
            "has_certification": cert   == 'true'
        }
    }


def _validate_parsed(parsed: dict) -> dict:
    required = {
        "person":  ["name", "age"],
        "work":    ["job", "city", "years_experience"],
        "details": ["is_senior", "has_certification"]
    }
    for section, keys in required.items():
        if section not in parsed:
            raise ValueError(f"Missing section '{section}'")
        for key in keys:
            if key not in parsed[section]:
                raise ValueError(f"Missing field '{section}.{key}'")
    return parsed


def extract(sentence: str) -> tuple[dict, str]:
    model, tokenizer = get_model()
    prompt  = build_prompt(sentence)
    inputs  = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    generated  = outputs[0][inputs['input_ids'].shape[1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    logger.info(f"Raw output: {raw_output[:100]}")

    # Clean raw output — remove hallucinated code after JSON
    json_start   = raw_output.find('{')
    json_end     = raw_output.rfind('}') + 1
    clean_output = raw_output[json_start:json_end] if json_start != -1 else raw_output

    parsed = parse_json_from_output(raw_output)
    return parsed, clean_output