"""
استخراج بيانات التدريب من الكتب لـ Fine-tuning.

يأخذ نصوص الكتب ويحوّلها لـ Q&A pairs بصيغة Alpaca:
  {"instruction": "سؤال", "input": "", "output": "جواب من الكتب"}

الاستخدام:
  python scripts/prepare_finetune_data.py --books-dir books/ --output data/finetune.jsonl

المتطلبات:
  pip install PyPDF2 tqdm  (+ إما Ollama أو GROQ_API_KEY لتوليد الأسئلة)
"""

import os
import json
import re
import argparse
import random
from pathlib import Path
from tqdm import tqdm

try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ── Config ────────────────────────────────────────────────────
CHUNK_WORDS   = 300
OVERLAP_WORDS = 50
MIN_WORDS     = 80
QA_PER_CHUNK  = 2   # عدد أسئلة من كل قطعة نص


def extract_pdf_text(path: str) -> str:
    if not PDF_OK:
        raise ImportError("pip install PyPDF2")
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def extract_txt_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, title: str) -> list[dict]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + CHUNK_WORDS, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= MIN_WORDS:
            chunks.append({"title": title, "text": chunk})
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def generate_qa_ollama(chunk: dict) -> list[dict]:
    """توليد أسئلة وأجوبة باستخدام Ollama (محلي)."""
    import requests
    prompt = f"""أنت معلم متخصص. بناءً على النص التالي من كتاب "{chunk['title']}"، 
اكتب {QA_PER_CHUNK} سؤالاً وجوابه بالعربية.
الصيغة: سطر السؤال يبدأ بـ "س:" وسطر الجواب يبدأ بـ "ج:"

النص:
{chunk['text']}

الأسئلة والأجوبة:"""

    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 400}
        }, timeout=60)
        response = r.json().get("response", "")
        return _parse_qa(response, chunk["text"])
    except Exception:
        return _generate_qa_simple(chunk)


def generate_qa_groq(chunk: dict) -> list[dict]:
    """توليد أسئلة وأجوبة باستخدام Groq (إنترنت)."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""من كتاب "{chunk['title']}"، اكتب {QA_PER_CHUNK} سؤالاً وجوابه.
صيغة: "س: [السؤال]\nج: [الجواب]"

النص: {chunk['text']}"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.7
        )
        return _parse_qa(resp.choices[0].message.content, chunk["text"])
    except Exception:
        return _generate_qa_simple(chunk)


def _parse_qa(text: str, context: str) -> list[dict]:
    """تحليل نص الأسئلة والأجوبة المولّدة."""
    pairs = []
    lines = text.strip().split("\n")
    q = a = ""
    for line in lines:
        line = line.strip()
        if line.startswith("س:"):
            q = line[2:].strip()
        elif line.startswith("ج:") and q:
            a = line[2:].strip()
            if q and a:
                pairs.append({
                    "instruction": q,
                    "input":       "",
                    "output":      a,
                    "context":     context[:200]
                })
            q = a = ""
    return pairs if pairs else _generate_qa_simple_from_text(context)


def _generate_qa_simple(chunk: dict) -> list[dict]:
    return _generate_qa_simple_from_text(chunk["text"])


def _generate_qa_simple_from_text(text: str) -> list[dict]:
    """أسئلة بسيطة بدون LLM — احتياطي."""
    sentences = [s.strip() for s in re.split(r'[.!؟\n]', text) if len(s.strip()) > 30]
    if not sentences:
        return []
    s = random.choice(sentences[:5])
    return [{
        "instruction": f"ماذا يقول النص عن: {s[:60]}...؟",
        "input":       "",
        "output":      s,
        "context":     text[:200]
    }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-dir",  default="books",         help="مجلد الكتب")
    parser.add_argument("--output",     default="data/finetune.jsonl", help="ملف الإخراج")
    parser.add_argument("--backend",    default="auto",          help="auto|ollama|groq|simple")
    parser.add_argument("--max-chunks", type=int, default=2000,  help="حد عدد القطع")
    args = parser.parse_args()

    # Determine backend
    if args.backend == "auto":
        try:
            import requests
            requests.get("http://localhost:11434/api/tags", timeout=2)
            backend = "ollama"
            print("[INFO] سيُستخدم Ollama لتوليد الأسئلة")
        except Exception:
            if os.getenv("GROQ_API_KEY"):
                backend = "groq"
                print("[INFO] سيُستخدم Groq لتوليد الأسئلة")
            else:
                backend = "simple"
                print("[WARN] لا Ollama ولا Groq — أسئلة بسيطة فقط")
    else:
        backend = args.backend

    gen_fn = {
        "ollama": generate_qa_ollama,
        "groq":   generate_qa_groq,
        "simple": _generate_qa_simple,
    }[backend]

    # Collect all text files
    books_dir = Path(args.books_dir)
    files = list(books_dir.glob("**/*.pdf")) + list(books_dir.glob("**/*.txt"))
    if not files:
        print(f"[ERROR] لم تجد ملفات في {books_dir}")
        return

    print(f"[INFO] {len(files)} كتاب/ملف")

    all_chunks = []
    for fpath in files:
        title = fpath.stem
        try:
            text = extract_pdf_text(str(fpath)) if fpath.suffix == ".pdf" \
                   else extract_txt_text(str(fpath))
            chunks = chunk_text(text, title)
            all_chunks.extend(chunks)
            print(f"  ✓ {title}: {len(chunks)} قطعة")
        except Exception as e:
            print(f"  ✗ {title}: {e}")

    # Shuffle and limit
    random.shuffle(all_chunks)
    all_chunks = all_chunks[:args.max_chunks]
    print(f"\n[INFO] توليد أسئلة لـ {len(all_chunks)} قطعة...")

    # Generate Q&A
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for chunk in tqdm(all_chunks):
            pairs = gen_fn(chunk)
            for pair in pairs:
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1

    print(f"\n[OK] {total} زوج سؤال-جواب محفوظ في: {args.output}")
    print("     استخدمه في: notebooks/finetune_qwen25.ipynb")


if __name__ == "__main__":
    main()
