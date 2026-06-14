# reset_to_per_user.ps1
# Moves existing shared library data to _archive/ before switching to per-user storage.
# Run ONCE from the project root: .\scripts\reset_to_per_user.ps1
# The server must be stopped before running this script.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "=== مكتبة ناطقة — إعادة ضبط التخزين للمكتبة الشخصية ===" -ForegroundColor Cyan
Write-Host "مجلد المشروع: $ProjectRoot"
Write-Host ""

# ── Stop any running Python processes on port 8000 ────────────────────────────
Write-Host "1/5  إيقاف عمليات Python المشغّلة..." -ForegroundColor Yellow
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# ── Create _archive directory ─────────────────────────────────────────────────
$Archive = Join-Path $ProjectRoot "_archive"
New-Item -ItemType Directory -Force -Path $Archive | Out-Null
Write-Host "2/5  مجلد الأرشيف: $Archive" -ForegroundColor Yellow

# ── Archive books/ ────────────────────────────────────────────────────────────
$BooksDir = Join-Path $ProjectRoot "books"
if (Test-Path $BooksDir) {
    $ArchiveBooks = Join-Path $Archive "books"
    New-Item -ItemType Directory -Force -Path $ArchiveBooks | Out-Null
    $pdfs = Get-ChildItem -Path $BooksDir -Filter "*.pdf" -File -ErrorAction SilentlyContinue
    if ($pdfs) {
        foreach ($pdf in $pdfs) {
            $dest = Join-Path $ArchiveBooks $pdf.Name
            Move-Item -Path $pdf.FullName -Destination $dest -Force
            Write-Host "   نقل: $($pdf.Name)" -ForegroundColor Gray
        }
        Write-Host "   تم نقل $($pdfs.Count) ملف PDF إلى $ArchiveBooks" -ForegroundColor Green
    } else {
        Write-Host "   لا توجد ملفات PDF في books/" -ForegroundColor Gray
    }
} else {
    Write-Host "3/5  مجلد books/ غير موجود — تم تخطيه" -ForegroundColor Gray
}

# ── Archive qdrant_data/ ──────────────────────────────────────────────────────
Write-Host "3/5  أرشفة Qdrant..." -ForegroundColor Yellow
$QdrantDir = Join-Path $ProjectRoot "qdrant_data"
if (Test-Path $QdrantDir) {
    $ArchiveQdrant = Join-Path $Archive "qdrant_data"
    if (Test-Path $ArchiveQdrant) { Remove-Item -Recurse -Force $ArchiveQdrant }
    Move-Item -Path $QdrantDir -Destination $ArchiveQdrant -Force
    Write-Host "   qdrant_data/ → _archive/qdrant_data/" -ForegroundColor Green
} else {
    Write-Host "   qdrant_data/ غير موجود — تم تخطيه" -ForegroundColor Gray
}

# ── Archive bm25_cache/ ───────────────────────────────────────────────────────
Write-Host "4/5  أرشفة BM25 cache..." -ForegroundColor Yellow
$Bm25Dir = Join-Path $ProjectRoot "bm25_cache"
if (Test-Path $Bm25Dir) {
    $ArchiveBm25 = Join-Path $Archive "bm25_cache"
    if (Test-Path $ArchiveBm25) { Remove-Item -Recurse -Force $ArchiveBm25 }
    Move-Item -Path $Bm25Dir -Destination $ArchiveBm25 -Force
    Write-Host "   bm25_cache/ → _archive/bm25_cache/" -ForegroundColor Green
} else {
    Write-Host "   bm25_cache/ غير موجود — تم تخطيه" -ForegroundColor Gray
}

# ── Archive ingestion_ledger.json ─────────────────────────────────────────────
Write-Host "5/5  أرشفة سجل الاستيعاب..." -ForegroundColor Yellow
$LedgerPath = Join-Path $ProjectRoot "ingestion_ledger.json"
if (Test-Path $LedgerPath) {
    $ArchiveLedger = Join-Path $Archive "ingestion_ledger.json"
    Move-Item -Path $LedgerPath -Destination $ArchiveLedger -Force
    Write-Host "   ingestion_ledger.json → _archive/" -ForegroundColor Green
} else {
    Write-Host "   ingestion_ledger.json غير موجود — تم تخطيه" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== اكتمل الإعداد ===" -ForegroundColor Cyan
Write-Host "يمكنك الآن تشغيل الموقع بشكل طبيعي." -ForegroundColor Green
Write-Host "كل مستخدم سيبني مكتبته الشخصية عبر تبويب 'المكتبة'." -ForegroundColor Green
Write-Host ""
