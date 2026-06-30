# Chay Lab 24 voi NVIDIA NIM

Repo nay dung NVIDIA qua OpenAI-compatible API. Code van dung SDK `openai`, nhung `OPENAI_BASE_URL` tro sang NVIDIA.

## 1. Tao file `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Dien cau hinh:

```env
NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=nvapi-...
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=meta/llama-3.1-70b-instruct
JUDGE_MODEL=meta/llama-3.1-70b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
EMBEDDING_DIM=0
NVIDIA_CHAT_RPM=8
NVIDIA_MAX_RETRIES=5
```

## 2. Chay pipeline

```powershell
docker compose up -d
python setup_answers.py
python src/phase_a_ragas.py
python src/phase_b_judge.py
python src/phase_c_guard.py
python check_lab.py
```

## 3. Ghi chu loi thuong gap

- `429 Too Many Requests`: ha `NVIDIA_CHAT_RPM` xuong `4`, roi chay lai phase bi loi.
- `pytest` khong nhan lenh tren PowerShell: dung `.\pytest.cmd tests -v` hoac `python check_lab.py`.
- `sentence-transformers/torch` crash tren Windows: repo dang dung `EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5`, nen khong can local embedding model.
- `Enrichment API failed: Expecting value`: M5 fallback sang heuristic, khong chan pipeline.

## 4. Test nhanh NVIDIA

```powershell
python -c "from config import openai_client, OPENAI_MODEL; c=openai_client(); print(c.chat.completions.create(model=OPENAI_MODEL, messages=[{'role':'user','content':'Tra loi ngan: xin chao'}]).choices[0].message.content)"
```

## 5. Kiem tra cuoi

```powershell
python check_lab.py
```

Pass mong doi:

```text
Score: 22/22 checks passed
San sang nop bai.
```
