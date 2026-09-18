# ==============================================================================
# BlasterOPT Botswana - Ollama Setup Script for Windows PowerShell
# ==============================================================================
# DOMAIN CONTEXT & OFFLINE-FIRST CAPABILITY:
# ------------------------------------------------------------------------------
# 1. WHAT OLLAMA IS & WHY IT MATTERS:
#    Ollama is an open-source local Large Language Model (LLM) execution engine.
#    In diamond open-pit operations across Botswana (e.g., Debswana Jwaneng, Orapa,
#    Damtshaa, Letlhakane), field engineers operate in remote pit floors where
#    cellular connectivity (4G/5G) and enterprise Wi-Fi are frequently unavailable
#    or intermittent due to high bench walls and heavy equipment shielding.
#    Ollama enables 100% offline LLM inference directly on field laptops and edge nodes.
#
# 2. "MULTIPLY, NOT REPLACE" POSITIONING:
#    BlasterOPT AI functions as a decision-support assistant to amplify human engineering
#    precision ("multiply"), never replacing certified blasters or autonomous firing.
#    Under Botswana's Mines, Quarries, Works and Machinery Act (Cap. 44:02), certified
#    blaster sign-off is legally mandatory before every detonation.
#
# 3. LLAMA 3.1 8B vs LLAMA 3.2 3B MODEL SELECTION:
#    - Llama 3.1 8B (Primary): High-capacity 8-billion parameter model for deep reasoning
#      on complex powder factor, multi-objective Pareto optimization, and vibration thresholds.
#    - Llama 3.2 3B (Edge Fallback): Ultra-fast 3-billion parameter model designed for rapid
#      low-latency responses on battery-constrained field tablets and mobile units.
#
# 4. 5-SECOND NON-BLOCKING HEALTH CHECK CONSTRAINT:
#    System health checks must complete within a strict 5-second timeout window to prevent
#    blocking Streamlit UI rendering or delaying critical pit floor blast design decisions.
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "=== [1/5] Checking Ollama Installation ===" -ForegroundColor Cyan
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $ver = & ollama --version
    Write-Host "✔ Ollama is already installed: $ver" -ForegroundColor Green
} else {
    Write-Host "ℹ Ollama not found. Downloading OllamaSetup.exe installer..." -ForegroundColor Yellow
    $installerPath = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installerPath
    Write-Host "Running OllamaSetup.exe installer..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/silent" -Wait
    Write-Host "✔ Ollama installation completed." -ForegroundColor Green
}

Write-Host "`n=== [2/5] Verifying Ollama Background Service ===" -ForegroundColor Cyan
try {
    $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3
    Write-Host "✔ Ollama background service is running on http://localhost:11434" -ForegroundColor Green
} catch {
    Write-Host "ℹ Starting Ollama app service in background..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "✔ Started Ollama background process." -ForegroundColor Green
}

Write-Host "`n=== [3/5] Pulling Primary Model (Llama 3.1 8B) ===" -ForegroundColor Cyan
Write-Host "Pulling llama3.1:8b for offline domain reasoning..." -ForegroundColor Yellow
& ollama pull llama3.1:8b

Write-Host "`n=== [4/5] Pulling Edge Fallback Model (Llama 3.2 3B) ===" -ForegroundColor Cyan
Write-Host "Pulling llama3.2:3b for low-power edge tablet fallback..." -ForegroundColor Yellow
& ollama pull llama3.2:3b

Write-Host "`n=== [5/5] Executing BlasterOPT Python Health Check Verification ===" -ForegroundColor Cyan
python -c "import json; from src.agent.ollama_health import full_health_check; res = full_health_check(); print(json.dumps(res, indent=2))"

Write-Host "`n=== Ollama Windows Setup Finished ===" -ForegroundColor Green
