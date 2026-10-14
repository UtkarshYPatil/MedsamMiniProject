<#
    One-shot environment setup for Windows.

        powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

    Installs torch (CUDA 12.1), the project requirements, the MedSAM2 package,
    and downloads the checkpoints. Safe to re-run -- every step is skipped if
    already satisfied.

    The MedSAM2 repository is already vendored at external\MedSAM2.
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "project: $root`n" -ForegroundColor Cyan

function Step($n, $msg) { Write-Host "[$n] $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 1. torch
Step 1 "PyTorch with CUDA"
$hasCuda = $false
try {
    $out = python -c "import torch;print(torch.__version__,torch.cuda.is_available())" 2>$null
    if ($out -match "True") { $hasCuda = $true; Write-Host "    already present: $out" -ForegroundColor Green }
} catch {}
if (-not $hasCuda) {
    Write-Host "    installing torch cu121 (large download)..."
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

# ---------------------------------------------------------------- 2. deps
Step 2 "Project requirements"
python -m pip install -q -r requirements.txt

# ---------------------------------------------------------------- 3. sam2
Step 3 "MedSAM2 / sam2 package"
$sam2 = $null
try { $sam2 = python -c "import sam2;print(sam2.__file__)" 2>$null } catch {}
if ($sam2) {
    Write-Host "    already importable: $sam2" -ForegroundColor Green
} else {
    if (-not (Test-Path "external\MedSAM2\setup.py")) {
        Write-Host "    external\MedSAM2 missing - cloning" -ForegroundColor Yellow
        git clone --depth 1 https://github.com/bowang-lab/MedSAM2.git external\MedSAM2
    }
    python -m pip install -e external\MedSAM2
}

# ---------------------------------------------------------------- 4. weights
Step 4 "MedSAM2 checkpoints"
python scripts\download_checkpoint.py

# ---------------------------------------------------------------- 5. verify
Step 5 "Verification"
python -c @"
import sys
ok = True
try:
    import torch
    print(f'    torch {torch.__version__}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'    gpu  {torch.cuda.get_device_name(0)}')
    else:
        print('    WARNING: no CUDA - inference will be very slow'); ok = False
except Exception as e:
    print(f'    torch FAILED: {e}'); ok = False
try:
    from sam2.build_sam import build_sam2_video_predictor_npz
    print('    sam2 import OK')
except Exception as e:
    print(f'    sam2 FAILED: {e}'); ok = False
from pathlib import Path
for f in ('MedSAM2_CTLesion.pt','MedSAM2_latest.pt'):
    p = Path('checkpoints')/f
    print(f'    {f}: ' + (f'{p.stat().st_size/1e6:.0f} MB' if p.exists() else 'MISSING'))
    if not p.exists(): ok = False
sys.exit(0 if ok else 1)
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nREADY. Next:" -ForegroundColor Green
    Write-Host "    python scripts\03_run_baseline.py --limit 50"
} else {
    Write-Host "`nSomething is missing - see above." -ForegroundColor Red
}
