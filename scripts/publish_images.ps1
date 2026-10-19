<#
Build the three images and push them to a registry, so someone else can run
the project without the repository.

    .\scripts\publish_images.ps1                      # GHCR, tag "latest"
    .\scripts\publish_images.ps1 -Tag v1.0            # a fixed version
    .\scripts\publish_images.ps1 -Gpu                 # also the CUDA worker
    .\scripts\publish_images.ps1 -Registry docker.io/utkarshypatil

The recipient then needs only docker-compose.pull.yml and the checkpoint.

THE WEIGHTS ARE NOT PUBLISHED. MedSAM2_CTLesion.pt is mounted from the host at
runtime and is deliberately absent from every image: the weights are
research-and-education-use only, so redistributing them is not ours to do.
This script checks that before pushing anything.
#>
[CmdletBinding()]
param(
    [string]$Registry = "ghcr.io/utkarshypatil",
    [string]$Tag = "latest",
    [switch]$Gpu,
    [switch]$SkipPush
)

$ErrorActionPreference = "Continue"

function Run($what, [scriptblock]$cmd) {
    Write-Host ">> $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        # $ErrorActionPreference does not catch native exit codes, so this is
        # checked by hand. Without it the script cheerfully reports success
        # after a failed build.
        Write-Host "FAILED: $what (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $env:REGISTRY = $Registry
    $env:TAG = $Tag

    Write-Host "registry : $Registry"
    Write-Host "tag      : $Tag`n"

    Run "build" { docker compose build }
    if ($Gpu) {
        Run "build (gpu worker)" {
            docker compose -f docker-compose.yml -f docker-compose.gpu.yml build worker
        }
    }

    # Refuse to publish an image carrying the weights.
    Write-Host ">> checking the images do not contain the checkpoint" -ForegroundColor Cyan
    $leaked = $false
    $images = @("$Registry/nodule-api:$Tag",
                "$Registry/nodule-worker:$Tag",
                "$Registry/nodule-ui:$Tag")
    if ($Gpu) { $images += "$Registry/nodule-worker:$Tag-gpu" }
    foreach ($img in $images) {
        $svc = $img.Split("/")[-1]
        $found = docker run --rm --entrypoint sh $img -c `
            "find / -name 'MedSAM2*.pt' -not -path '/proc/*' 2>/dev/null | head -5"
        if ($LASTEXITCODE -eq 0 -and $found) {
            Write-Host "  $svc CONTAINS WEIGHTS:" -ForegroundColor Red
            $found | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            $leaked = $true
        } else {
            Write-Host "  $svc clean" -ForegroundColor Green
        }
    }
    if ($leaked) {
        Write-Host "`nNot pushing. Remove the weights from the image first." -ForegroundColor Red
        exit 1
    }

    if ($SkipPush) {
        Write-Host "`n-SkipPush set: built and verified, nothing pushed." -ForegroundColor Yellow
        exit 0
    }

    if ($Registry -like "ghcr.io/*") {
        Write-Host "`nIf the push 401s, log in first:" -ForegroundColor Yellow
        Write-Host "  echo <your-PAT> | docker login ghcr.io -u UtkarshYPatil --password-stdin"
        Write-Host "  (a classic token with write:packages)`n"
    }

    Run "push" { docker compose push }
    if ($Gpu) {
        Run "push (gpu worker)" {
            docker compose -f docker-compose.yml -f docker-compose.gpu.yml push worker
        }
    }

    Write-Host "`nPublished. Send them docker-compose.pull.yml and this:" -ForegroundColor Green
    Write-Host "  1. put MedSAM2_CTLesion.pt in ./checkpoints"
    if ($Registry -ne "ghcr.io/utkarshypatil" -or $Tag -ne "latest") {
        Write-Host "  2. REGISTRY=$Registry TAG=$Tag docker compose -f docker-compose.pull.yml up"
    } else {
        Write-Host "  2. docker compose -f docker-compose.pull.yml up"
    }
    Write-Host "  3. http://localhost:8501"
    Write-Host "`nPackages are PRIVATE by default on GHCR. To let anyone pull without"
    Write-Host "logging in, set each package to public at"
    Write-Host "  https://github.com/UtkarshYPatil?tab=packages"
}
finally {
    Pop-Location
}
