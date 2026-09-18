#!/usr/bin/env bash
# ==============================================================================
# BlasterOPT Botswana - Ollama Setup Script for Linux/macOS
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

set -e

echo "=== [1/5] Checking Ollama Installation ==="
if command -v ollama >/dev/null 2>&1; then
    echo "✔ Ollama is already installed: $(ollama --version)"
else
    echo "ℹ Ollama not found. Downloading and installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install ollama
        else
            echo "Installing via Official Ollama Script for macOS..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    else
        echo "Installing via Official Ollama Script for Linux..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    echo "✔ Ollama installation completed."
fi

echo ""
echo "=== [2/5] Verifying Ollama Background Service ==="
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✔ Ollama background service is running on http://localhost:11434"
else
    echo "ℹ Ollama service is not responding. Starting Ollama serve in background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✔ Ollama service started successfully."
    else
        echo "⚠ Could not automatically verify background service. Continuing..."
    fi
fi

echo ""
echo "=== [3/5] Pulling Primary Model (Llama 3.1 8B) ==="
echo "Pulling llama3.1:8b for offline domain reasoning..."
ollama pull llama3.1:8b || echo "⚠ Warning: Failed to pull llama3.1:8b"

echo ""
echo "=== [4/5] Pulling Edge Fallback Model (Llama 3.2 3B) ==="
echo "Pulling llama3.2:3b for low-power edge tablet fallback..."
ollama pull llama3.2:3b || echo "⚠ Warning: Failed to pull llama3.2:3b"

echo ""
echo "=== [5/5] Executing BlasterOPT Python Health Check Verification ==="
python3 -c "
import json
from src.agent.ollama_health import full_health_check

res = full_health_check()
print(json.dumps(res, indent=2))
if res['can_use_offline_llm']:
    print('\n✔ SUCCESS: Ollama is fully configured and ready for offline pit floor operation!')
else:
    print('\n⚠ NOTICE: Ollama health check returned offline capability = False. Check recommendations above.')
"

echo "=== Ollama Setup Script Finished ==="
