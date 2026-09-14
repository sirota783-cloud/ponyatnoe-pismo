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

app = FastAPI(title="Понятное письмо API", version="6.0")

LETTER_SCHEMA = {
    "type": "object",
    "properties": {
        "sender": {"type": "string"},
        "document_type": {"type": "string"},
        "source_language": {"type": "string"},
        "summary": {"type": "string"},
        "action_required": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "action_text": {"type": "string"},
        "deadline": {"type": "string"},
        "payment_now": {"type": "string"},
        "important": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "full_translation": {"type": "string"},
        "uncertainties": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": [
        "sender","document_type","source_language","summary","action_required","action_text",
        "deadline","payment_now","important","steps","full_translation","uncertainties","confidence"
    ],
    "additionalProperties": False
}

def instructions_for(output_language: str) -> str:
    answer_language = "Hebrew" if output_language == "he" else "Russian"
    return f"""
You explain official letters to older adults in clear, simple language.
Analyze ONLY the uploaded document. Do not invent missing details.
Return every human-readable field in {answer_language}.

Rules:
1. Distinguish a date when changes take effect from a personal deadline.
2. Do not call an effective date a response deadline unless the letter explicitly requires action by that date.
3. Distinguish a future price or fee change from a demand to pay now.
4. If the letter is informational only, clearly say no action is required now.
5. If the recipient must submit, sign, pay, call, appeal, book, renew or reply, say that clearly.
6. Preserve important amounts, dates, medicine names, reference numbers and percentages.
7. Translate the whole meaningful body. Tables may be summarized faithfully, but keep values relevant to the recipient.
8. If anything is unreadable or ambiguous, say so in uncertainties. Never guess.
9. Do not give legal, medical or financial advice beyond explaining the letter.
10. Use short sentences and simple words.
11. For sender, prefer the familiar short organization name when unambiguous. Example: write "Маккаби" instead of a long bilingual corporate title.

For action_required:
- yes = recipient explicitly must do something
- no = informational notice; no action requested now
- unclear = document is too unclear to decide safely

For deadline:
- if no personal deadline exists, say so
- if a date is only an effective date, explain that

For payment_now:
- say whether the document demands payment now
- if it only announces future fees/copayments, say that instead
""".strip()

@app.get("/")
def home():
    return FileResponse(INDEX_FILE)

@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "api_key_configured": bool(API_KEY)}

@app.post("/api/analyze")
async def analyze_letter(
    files: list[UploadFile] = File(...),
    language: str = Form("ru")
):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="На сервере не настроен OPENAI_API_KEY.")

    language = "he" if language == "he" else "ru"

    if not files:
        raise HTTPException(status_code=400, detail="Не выбран ни один файл.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Можно загрузить не более 10 страниц за один раз.")

    prepared = []
    total_bytes = 0

    for upload in files:
        content = await upload.read()
        if not content:
            continue

        total_bytes += len(content)
        if total_bytes > 30 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Общий размер файлов слишком большой. Максимум 30 МБ.")

        mime = upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or ""
        if mime not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Поддерживаются PDF, JPG, PNG, WEBP и GIF."
            )

        prepared.append({
            "filename": upload.filename or "document",
            "mime": mime,
            "content": content,
        })

    if not prepared:
        raise HTTPException(status_code=400, detail="Файлы пустые.")

    # For the simple elderly-user flow:
    # - one or more photos are treated as consecutive pages of the same letter;
    # - PDFs are also supported;
    # - mixed image/PDF batches are accepted, but the model is told to preserve order.
    client = OpenAI(api_key=API_KEY)
    uploaded_file_ids = []
    temp_paths = []

    try:
        user_content = [{
            "type": "input_text",
            "text": (
                "These files are pages of one official letter, in the order supplied. "
                "Read all pages together. Explain whether the recipient actually has to do "
                "anything, whether there is a personal deadline, and whether payment is demanded now. "
                "If a page is blurry or unreadable, say exactly what cannot be read and lower confidence."
            )
        }]

        for item in prepared:
            mime = item["mime"]
            content = item["content"]

            if mime == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    temp_path = tmp.name
                temp_paths.append(temp_path)

                with open(temp_path, "rb") as f:
                    uploaded = client.files.create(file=f, purpose="user_data")
                uploaded_file_ids.append(uploaded.id)

                user_content.append({
                    "type": "input_file",
                    "file_id": uploaded.id,
                    "detail": "high"
                })
            else:
                b64 = base64.b64encode(content).decode("ascii")
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{b64}",
                    "detail": "high"
                })

        response = client.responses.create(
            model=MODEL,
            instructions=instructions_for(language),
            input=[{"role": "user", "content": user_content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "letter_explanation",
                    "strict": True,
                    "schema": LETTER_SCHEMA
                }
            }
        )

        data = json.loads(response.output_text)
        data["model"] = MODEL
        data["pages_received"] = len(prepared)
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
        for file_id in uploaded_file_ids:
            try:
                client.files.delete(file_id)
            except Exception:
                pass

        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except OSError:
                pass

