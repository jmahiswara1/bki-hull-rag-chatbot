# BKI Hull Rules Chatbot CLI v0

CLI chatbot berbasis RAG untuk menjawab pertanyaan teknis dari dokumen BKI Rules for Hull 2026 menggunakan LLM lokal via Ollama.

## Prasyarat

- Python 3.10+
- Ollama
- Microsoft C++ Build Tools jika instalasi ChromaDB di Windows gagal

Pull model Ollama yang dibutuhkan:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Instalasi

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Jika memakai PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Siapkan PDF

Letakkan dokumen sumber di:

```text
data/bki-rules-hull-2026.pdf
```

## Ingestion

```bash
python src/ingest.py
```

Script akan mengekstrak teks PDF, membuat chunk, menghasilkan embedding dengan `nomic-embed-text`, lalu menyimpan vector store versi terbaru di `chroma_db_v2/`.

## Menjalankan Chatbot

```bash
python src/chat.py
```

Command CLI:

```text
/help                    Show help
/clear                   Clear conversation history
/fast                    Use qwen2.5:3b with grounded concise answers
/llama                   Use llama3.2:3b with grounded concise answers
/normal                  Use qwen2.5:7b fallback with deeper retrieval
/debug-retrieve <q>      Show retrieved chunks and scores for a question
/import-json [path]      Ask independent questions from JSON, default: questions.json
/import-pdf [path]       Ask independent questions from PDF, default: data/AI testing.pdf
/quit                    Exit
/exit                    Exit
```

## Testing Batch

Untuk menjalankan daftar pertanyaan di `questions.json`:

```text
/import-json
```

Untuk menjalankan pertanyaan dari PDF testing:

```text
/import-pdf "data/AI testing.pdf"
```

Setiap pertanyaan JSON/PDF diproses secara independen agar hasil pertanyaan sebelumnya tidak memengaruhi retrieval pertanyaan berikutnya. PDF testing hanya dipakai sebagai sumber pertanyaan; jawaban tetap memakai vector store BKI Hull Rules.

Jika tabel atau pertanyaan di PDF/DOCX berbentuk gambar, teks di dalam gambar tidak terbaca tanpa OCR. Untuk data teknis berbentuk tabel, format yang lebih aman adalah JSON, CSV, XLSX, atau DOCX dengan tabel asli yang editable/selectable.

## Akurasi dan Mode

Semua mode memakai guardrail yang sama: jawaban harus didukung retrieved context. Jika angka, formula, atau aturan tidak muncul jelas di context, chatbot diarahkan untuk mengatakan bahwa informasi tidak tersedia dalam context yang ditemukan.

- `/normal` cocok untuk kualitas terbaik karena memakai model lebih besar dan retrieval lebih dalam.
- `/fast` cocok untuk respons cepat dengan jawaban ringkas.
- `/llama` cocok untuk pembanding cepat memakai `llama3.2:3b`.
- `/debug-retrieve <q>` membantu melihat apakah halaman/chunk yang benar sudah terambil.

## Catatan v0

- Hanya mendukung satu dokumen PDF BKI Hull 2026.
- Conversation history hanya tersimpan selama satu sesi.
- Gambar, diagram, tabel kompleks, dan rumus belum diproses secara khusus.
- Sistem dapat berjalan offline setelah dependency, model Ollama, PDF, dan vector store tersedia lokal.
