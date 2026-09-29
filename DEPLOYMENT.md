# Deployment Guide

## GitHub Setup

### 1. Initialize Git Repository

If you haven't already initialized git, run:

```bash
cd "C:\Users\User 2\Downloads\jules_session_2175162121400205044 (1)"
git init
```

### 2. Connect to Existing GitHub Repository

The repository already exists at: https://github.com/mndolo2x/BlasterOPT

```bash
git remote add origin https://github.com/mndolo2x/BlasterOPT.git
git branch -M main
```

### 4. Stage and Commit Files

```bash
git add .
git commit -m "Initial commit: BlastOpt Botswana platform with all modules"
```

### 4. Push to GitHub

```bash
git push -u origin main
```

**Quick Setup:** Run the provided script:
```bash
connect_to_repo.bat
```

## Streamlit Cloud Deployment

### Option 1: Deploy from GitHub (Recommended)

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select `mndolo2x/BlasterOPT` repository
5. Select `app.py` as the main file
6. Click "Deploy"

### Option 2: Deploy from Command Line

```bash
pip install streamlit
streamlit run app.py
```

## Deployment Requirements

The following files are required for Streamlit Cloud:

- ✅ `app.py` - Main application file
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System packages (optional)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `README.md` - Project documentation

## Environment Variables

Set these in Streamlit Cloud:

- `DEMO_MODE` - Set to `true` for demo mode, `false` for live data

## Troubleshooting

### Import Errors

If you encounter import errors on Streamlit Cloud:

1. Check that all dependencies are in `requirements.txt`
2. Verify file paths are correct (use relative paths)
3. Check the deployment logs for specific errors

### Database Connections

For database connections, use environment variables:

```python
import os
db_url = os.getenv('DATABASE_URL')
```

### Performance Issues

For better performance:
- Enable caching in Streamlit
- Optimize data loading
- Use efficient data structures

## Monitoring

Streamlit Cloud provides:
- Resource usage monitoring
- Error logs
- Deployment history
- Custom domain support (paid tier)

## Scaling

For production deployment:
- Consider paid Streamlit Cloud tier for more resources
- Implement proper error handling
- Add logging and monitoring
- Set up CI/CD pipeline
