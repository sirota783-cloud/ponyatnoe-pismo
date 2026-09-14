import base64
import json
import logging
import mimetypes
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ponyatnoe-pismo")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

MAX_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | {"application/pdf"}

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Понятное письмо API", version="4.0")


LETTER_SCHEMA = {
    "type": "object",
    "properties": {
        "sender": {"type": "string"},
        "document_type": {"type": "string"},
        "source_language": {"type": "string"},
        "summary": {"type": "string"},
        "action_required": {
            "type": "string",
            "enum": ["yes", "no", "unclear"]
        },
        "action_text": {"type": "string"},
        "deadline": {"type": "string"},
        "payment_now": {"type": "string"},
        "important": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "full_translation": {"type": "string"},
        "uncertainties": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        }
    },
    "required": [
        "sender",
        "document_type",
        "source_language",
        "summary",
        "action_required",
        "action_text",
        "deadline",
        "payment_now",
        "important",
        "steps",
        "full_translation",
        "uncertainties",
        "confidence"
    ],
    "additionalProperties": False
}


def instructions_for(output_language: str) -> str:
    if output_language == "he":
        answer_language = "Hebrew"
    else:
        answer_language = "Russian"

    return f"""
You explain official letters to older adults in clear, simple language.

Analyze ONLY the uploaded document. Do not invent missing details.
Return every human-readable field in {answer_language}.

Important rules:
1. Distinguish a DATE WHEN CHANGES TAKE EFFECT from a PERSONAL DEADLINE for the recipient.
2. Do not call an effective date a response deadline unless the document explicitly requires action by that date.
3. Distinguish a future price/fee change from a demand to pay money now.
4. If the letter is informational only, clearly say that no action is required now.
5. If the document asks the recipient to submit, sign, pay, call, appeal, book, renew, or reply, identify that action clearly.
6. Preserve important amounts, dates, medicine names, organization names, reference numbers, and percentages.
7. For the full translation, translate the whole meaningful body of the letter. Tables may be summarized faithfully if reproducing every cell would be confusing, but retain values relevant to the recipient.
8. If text is unreadable or ambiguous, say so in "uncertainties". Never guess.
9. Do not give legal, medical, or financial advice beyond explaining what the letter says.
10. Use short sentences and simple words suitable for an older reader.

For "action_required":
- "yes" = the recipient is explicitly required to do something.
- "no" = informational notice; no action is requested now.
- "unclear" = the document is too unclear to decide safely.

For "deadline":
- If no personal deadline exists, explicitly say so.
- If a date is only an effective date, explain that distinction.

For "payment_now":
- State whether the document demands payment now.
- If it only announces changed future fees/copayments, say that instead.
""".strip()


@app.get("/")
def home():
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "api_key_configured": bool(API_KEY),
    }


@app.post("/api/analyze")
async def analyze_letter(
    file: UploadFile = File(...),
    language: str = Form("ru"),
):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="На сервере не настроен OPENAI_API_KEY."
        )

    language = "he" if language == "he" else "ru"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Файл слишком большой. Максимум 20 МБ."
        )

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""
    if mime not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Поддерживаются PDF, JPG, PNG, WEBP и GIF."
        )

    client = OpenAI(api_key=API_KEY)
    uploaded_file_id = None
    temp_path = None

    try:
        user_content = [
            {
                "type": "input_text",
                "text": (
                    "Read this official letter carefully and explain it according "
                    "to the instructions. Pay special attention to whether the "
                    "recipient actually has to do anything, whether there is a "
                    "personal deadline, and whether payment is demanded now."
                ),
            }
        ]

        if mime == "application/pdf":
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            with open(temp_path, "rb") as f:
                uploaded = client.files.create(
                    file=f,
                    purpose="user_data",
                )
            uploaded_file_id = uploaded.id

            user_content.append({
                "type": "input_file",
                "file_id": uploaded_file_id,
                "detail": "high",
            })

        else:
            b64 = base64.b64encode(content).decode("ascii")
            user_content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{b64}",
                "detail": "high",
            })

        response = client.responses.create(
            model=MODEL,
            instructions=instructions_for(language),
            input=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "letter_explanation",
                    "strict": True,
                    "schema": LETTER_SCHEMA,
                }
            },
        )

        data = json.loads(response.output_text)
        data["model"] = MODEL
        return data

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Letter analysis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Не удалось проанализировать письмо. Попробуйте ещё раз."
        )
    finally:
        if uploaded_file_id:
            try:
                client.files.delete(uploaded_file_id)
            except Exception:
                log.warning("Could not delete temporary OpenAI file %s", uploaded_file_id)

        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
