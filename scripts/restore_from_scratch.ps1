<#
    Full restore after a clean Windows install.

    Run it from anywhere using the full path:

        powershell -ExecutionPolicy Bypass -File E:\MiniProject\MedsamMiniProject\scripts\restore_from_scratch.ps1

    Options:
        -SkipDataset     environment only, no 125 GB download
        -DatasetOnly     resume an interrupted dataset download
        -PythonExe <p>   use a specific interpreter instead of auto-detecting

    PYTHON VERSION MATTERS.
    PyTorch on Windows ships wheels for Python 3.10-3.14 only. On 3.15 pip
    finds no wheel for torch, duckdb or scikit-image, tries to compile them
    from source, and dies for want of a C compiler. This script ignores
    whatever `python` is on PATH and installs into a supported interpreter
    found via the `py` launcher, preferring 3.12.

    No virtual environment: packages go straight into that interpreter.
    So run every project command with python3.12, NOT python:

        python3.12 scripts\03_run_baseline.py

    Everything is idempotent and the dataset download resumes, so re-running
    after a failure or Ctrl+C costs nothing.
#>
param(
    [switch]$SkipDataset,
    [switch]$DatasetOnly,
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$DATASET   = "E:/MiniProject/DATASET_ct"
$CUDA_URLS = @("https://download.pytorch.org/whl/cu128",
               "https://download.pytorch.org/whl/cu126")
$WANTED    = @("3.12", "3.11", "3.13", "3.10", "3.14")   # preference order

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Yellow }
function Ok($msg)       { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "    $msg" -ForegroundColor Red }

# Native commands do not trip $ErrorActionPreference, so every external call
# goes through here and checks the exit code.
function Run($exe, [string[]]$cmdArgs, $what) {
    & $exe @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

function Find-Python {
    if ($PythonExe) {
        if (-not (Test-Path $PythonExe)) { throw "-PythonExe '$PythonExe' does not exist" }
        return $PythonExe
    }
    # The new Python Install Manager wants -V:3.12; the legacy launcher wants
    # -3.12. Try both so this works either way.
    foreach ($v in $WANTED) {
        foreach ($flag in @("-V:$v", "-$v")) {
            try {
                $probe = (& py $flag -c "import sys;print(sys.executable)" 2>$null)
                if ($LASTEXITCODE -eq 0 -and $probe) { return $probe.Trim() }
            } catch {}
        }
    }
    return $null
}

Write-Host "Restoring: $root" -ForegroundColor Cyan

$PY = Find-Python
if (-not $PY) {
    $sys = "?"; try { $sys = (python --version 2>&1 | Out-String).Trim() } catch {}
    Warn "No Python 3.10-3.14 found. System default is $sys."
    Write-Host ""
    Write-Host "      py install 3.12" -ForegroundColor White
    Write-Host ""
    Write-Host "  then re-run this script."
    exit 1
}

# =================================================== 0. preflight
if (-not $DatasetOnly) {
    Step 0 "Environment"
    Ok "python  $((& $PY --version 2>&1 | Out-String).Trim())"
    Ok "        $PY"

    $git = $null
    try { $git = (git --version 2>&1 | Out-String).Trim() } catch {}
    if ($git -match "git version") { Ok "git     $git" }
    else { throw "git not found - install https://git-scm.com/download/win" }

    $gpu = $null
    try { $gpu = (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | Out-String).Trim() } catch {}
    if ($gpu -and $gpu -notmatch "not recognized") { Ok "gpu     $gpu" }
    else { Warn "gpu     nvidia-smi missing - CPU only, inference will be very slow" }

    Run $PY @("-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel") "pip upgrade"

    # =============================================== 1. torch
    Step 1 "PyTorch with CUDA"
    $hasCuda = $false
    try {
        $out = & $PY -c "import torch;print(torch.__version__,torch.cuda.is_available())" 2>$null
        if ($out -match "True") { $hasCuda = $true; Ok "already present: $out" }
    } catch {}
    if (-not $hasCuda) {
        $done = $false
        foreach ($url in $CUDA_URLS) {
            Write-Host "    trying $url  (~2.5 GB)"
            & $PY -m pip install torch torchvision --index-url $url
            if ($LASTEXITCODE -eq 0) { $done = $true; break }
            Warn "that index did not work - trying the next"
        }
        if (-not $done) { throw "could not install torch from any CUDA index" }
        $out = & $PY -c "import torch;print(torch.__version__,torch.cuda.is_available())" 2>$null
        Ok "installed: $out"
        if ($out -notmatch "True") { Warn "torch is in but CUDA is not visible - check your driver" }
    }

    # =============================================== 2. requirements
    Step 2 "Project requirements"
    Run $PY @("-m", "pip", "install", "-r", "requirements.txt") "requirements.txt"
    Ok "installed"

    # =============================================== 3. idc-index
    Step 3 "idc-index (dataset downloader)"
    $hasIdc = $false
    try { & $PY -c "import idc_index" 2>$null; if ($LASTEXITCODE -eq 0) { $hasIdc = $true } } catch {}
    if ($hasIdc) { Ok "already present" }
    else { Run $PY @("-m", "pip", "install", "idc-index") "idc-index"; Ok "installed" }

    # =============================================== 4. sam2
    Step 4 "MedSAM2 / sam2 package"
    $sam2ok = $false
    try { & $PY -c "import sam2" 2>$null; $sam2ok = ($LASTEXITCODE -eq 0) } catch {}
    if ($sam2ok) {
        Ok "already importable"
    } else {
        if (-not (Test-Path "external\MedSAM2\setup.py")) {
            if (Test-Path "external\MedSAM2") { Remove-Item -Recurse -Force "external\MedSAM2" }
            New-Item -ItemType Directory -Force -Path external | Out-Null
            Run "git" @("clone", "--depth", "1",
                        "https://github.com/bowang-lab/MedSAM2.git",
                        "external\MedSAM2") "MedSAM2 clone"
        }
        # torch is already installed; --no-build-isolation stops pip from
        # fetching its own copy of torch just to satisfy the build deps.
        Run $PY @("-m", "pip", "install", "-e", "external\MedSAM2", "--no-build-isolation") "sam2 install"
        Ok "installed"
    }

    # =============================================== 5. checkpoints
    Step 5 "MedSAM2 checkpoints"
    Run $PY @("scripts\download_checkpoint.py") "checkpoint download"

    # =============================================== 6. tests
    Step 6 "Test suite (no dataset needed)"
    & $PY -m pytest -q tests/
    if ($LASTEXITCODE -eq 0) { Ok "tests pass - the code side is fully restored" }
    else { Warn "tests failed - see above" }
}

# =================================================== 7. dataset
if (-not $SkipDataset) {
    Step 7 "LIDC-IDRI CT series  ->  $DATASET"

    $have = 0
    if (Test-Path "$DATASET/lidc_idri") {
        $have = (Get-ChildItem "$DATASET/lidc_idri" -Directory -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -like "LIDC-IDRI-*" }).Count
    }
    Write-Host "    patient folders present: $have / ~1010"

    if ($have -ge 1000) {
        Ok "dataset looks complete - skipping"
    } else {
        Write-Host "    ~125 GB. Resumable - re-run this script to continue." -ForegroundColor Cyan
        Write-Host "    Leave the window open; expect several hours.`n" -ForegroundColor Cyan
        & $PY scripts\download_ct.py --out $DATASET --apply
        if ($LASTEXITCODE -ne 0) { Warn "download stopped early - re-run to resume" }
    }

    Step 8 "Point pylidc at the dataset"
    if (Test-Path "$DATASET/lidc_idri") {
        & $PY scripts\01_setup_pylidc.py --dicom "$DATASET/lidc_idri"
    } else {
        Warn "dataset not there yet - re-run once it is"
    }
}

# =================================================== summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "WHERE THINGS STAND" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

function Report($label, $cond, $detail) {
    $mark = if ($cond) { "  OK  " } else { " MISS " }
    $col  = if ($cond) { "Green" } else { "Red" }
    Write-Host ("[{0}] {1,-18} {2}" -f $mark, $label, $detail) -ForegroundColor $col
}

Report "python" $true "$((& $PY --version 2>&1 | Out-String).Trim())"

$t = $null; try { $t = & $PY -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>$null } catch {}
Report "torch + cuda" ($t -match "True") "$t"

$s = $false; try { & $PY -c "import sam2" 2>$null; $s = ($LASTEXITCODE -eq 0) } catch {}
Report "sam2" $s "external\MedSAM2"

$ck = Test-Path "checkpoints\MedSAM2_CTLesion.pt"
Report "checkpoint" $ck $(if ($ck) { "{0:N0} MB" -f ((Get-Item "checkpoints\MedSAM2_CTLesion.pt").Length / 1MB) } else { "missing" })

$n = 0
if (Test-Path "$DATASET/lidc_idri") {
    $n = (Get-ChildItem "$DATASET/lidc_idri" -Directory -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -like "LIDC-IDRI-*" }).Count
}
Report "dataset" ($n -ge 1000) "$n / ~1010 patients"
Report "pylidc config" (Test-Path (Join-Path $env:USERPROFILE "pylidc.conf")) (Join-Path $env:USERPROFILE "pylidc.conf")
Report "processed ROIs" (Test-Path "E:/MiniProject/processed/manifest.json") "built by 02_prepare_data.py"

Write-Host "`nIMPORTANT - use python3.12, not python:" -ForegroundColor Cyan
Write-Host "  'python' on this machine is 3.15 and has none of these packages."
Write-Host ""
Write-Host "    cd E:\MiniProject\MedsamMiniProject"
Write-Host "    python3.12 scripts\02_prepare_data.py --limit 50"
Write-Host "    python3.12 scripts\03_run_baseline.py"
