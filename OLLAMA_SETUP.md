
## Ollama setup

The app uses a real Ollama HTTP endpoint when `OLLAMA_HOST` is reachable. The repository cannot start a persistent Ollama daemon inside Streamlit Cloud, so use one of the following deployment options.

### Local development

Install Ollama, then run:

```bash
ollama serve
ollama pull llama3.1:8b
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b
streamlit run app.py
```

You can also use the supplied helper:

```bash
bash scripts/start_ollama.sh
```

### Docker host

Run the included service:

```bash
docker compose -f docker-compose.ollama.yml up -d
curl http://localhost:11434/api/tags
```

Expose the host securely through a reverse proxy or private network. Do not expose an unauthenticated Ollama endpoint to the public internet.

### Streamlit Cloud

Copy `.streamlit/secrets.example.toml` into the Streamlit Cloud Secrets editor and replace the placeholder with the HTTPS URL of your secured Ollama host:

```toml
OLLAMA_HOST = "https://ollama.example.com"
OLLAMA_MODEL = "llama3.1:8b"
```

The endpoint must be reachable from Streamlit Cloud and must respond to `/api/tags` and `/api/generate`. After saving secrets, reboot the app and open **System Health → Test Ollama Status Now**. A working deployment must report **Online & Reachable**, the model as **Present**, and **Generation Working**.
