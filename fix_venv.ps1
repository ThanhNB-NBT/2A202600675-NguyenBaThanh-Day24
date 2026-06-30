$ErrorActionPreference = "Stop"

if (Test-Path ".venv") {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backup = ".venv.broken-$stamp"
    Write-Host "Renaming existing .venv -> $backup"
    Rename-Item -LiteralPath ".venv" -NewName $backup
}

Write-Host "Creating fresh .venv..."
uv venv .venv --python 3.11

Write-Host "Installing requirements..."
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Testing embedding model..."
.\.venv\Scripts\python.exe -c "from config import EMBEDDING_MODEL; from sentence_transformers import SentenceTransformer; print('loading', EMBEDDING_MODEL); m=SentenceTransformer(EMBEDDING_MODEL); print('loaded'); print(m.encode(['xin chao']).shape)"

Write-Host "Done. Activate with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
