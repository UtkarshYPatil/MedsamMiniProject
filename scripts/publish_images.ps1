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
        Write-Host "`nPushing to GHCR. The two ways this fails look different:" -ForegroundColor Yellow
        Write-Host "  401 / unauthorized -> not logged in:"
        Write-Host "      docker login ghcr.io -u UtkarshYPatil"
        Write-Host "  denied -> logged in, but the token cannot WRITE packages."
        Write-Host "      Login only needs a valid token; pushing needs the"
        Write-Host "      write:packages scope, so a read-only token logs in fine"
        Write-Host "      and is refused here, sometimes after uploading most of"
        Write-Host "      the image. Fix the scope at github.com/settings/tokens"
        Write-Host "      (Tokens CLASSIC -- fine-grained tokens are unreliable"
        Write-Host "      with GHCR), then: docker logout ghcr.io && docker login ghcr.io`n"
    }

    # `docker compose push` was denied here while a plain `docker push` of the
    # same image succeeded -- the blobs uploaded either way, only compose's
    # manifest PUT was refused. Pushing each image directly avoids that, and
    # gives one error per image instead of three fired at once.
    $pushFailed = $false
    foreach ($img in $images) {
        Write-Host ">> push $img" -ForegroundColor Cyan
        docker push $img
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   FAILED: $img" -ForegroundColor Red
            $pushFailed = $true
        }
    }

    if ($pushFailed) {
        Write-Host "`nOne or more pushes failed." -ForegroundColor Red
        if ($Registry -like "ghcr.io/*") {
            Write-Host "Check, in this order:" -ForegroundColor Yellow
            Write-Host "  1. Is the token valid?   docker login ghcr.io -u UtkarshYPatil"
            Write-Host "  2. Can it write at all?  push a throwaway image:"
            Write-Host "       docker pull hello-world"
            Write-Host "       docker tag hello-world $Registry/scope-test:latest"
            Write-Host "       docker push $Registry/scope-test:latest"
            Write-Host "     'denied' there means the token lacks write:packages."
            Write-Host "     It working means the token is fine and the problem is"
            Write-Host "     this image. Delete the test package afterwards."
            Write-Host "  3. Namespace must be your username in LOWERCASE."
            Write-Host "     Pushing to: $Registry"
        }
        exit 1
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
