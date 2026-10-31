$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$statusPath = Join-Path $projectRoot 'results/baseline-setup.json'

function Set-SetupStatus([string]$stage, [string]$detail = '') {
    @{ stage = $stage; detail = $detail; updated_at = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
    Write-Output "Stage: $stage $detail"
}

try {
    Set-SetupStatus 'installing_gpu_support'
    & $pythonPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
    & $pythonPath -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --retries 10 --timeout 90
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed; existing downloads may be cached' }
    Set-SetupStatus 'verifying_gpu_support'
    & $pythonPath -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Package consistency check failed' }
    & $pythonPath -c "import json,torch; from pathlib import Path; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(32,32,device='cuda'); y=x@x; torch.cuda.synchronize(); r={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_matmul_passed':bool(torch.isfinite(y).all().item())}; assert r['gpu_matmul_passed']; Path('results/gpu-verification.json').write_text(json.dumps(r,indent=2)); print(r)"
    if ($LASTEXITCODE -ne 0) { throw 'GPU computation verification failed' }
    & $pythonPath -m pip freeze | Set-Content -LiteralPath 'requirements-gpu-lock.txt' -Encoding utf8
    if (-not (Test-Path -LiteralPath 'data/dataset_ppe/train_only.yaml')) { throw 'Prepared training configuration is missing' }
    Set-SetupStatus 'training'
    & $pythonPath -u src/train.py
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect ppe-training-status.json and the training log' }
    Set-SetupStatus 'complete' 'Training finished; test evaluation remains a separate step'
} catch {
    Set-SetupStatus 'failed' $_.Exception.Message
    Write-Error $_.Exception.Message
    exit 1
}
