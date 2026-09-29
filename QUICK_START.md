# Quick Start Guide - BlastOpt Botswana

## Current Status

✅ **Import errors fixed** - All 35+ modules created
✅ **Python syntax validated** - All files compile correctly
✅ **Deployment files created** - Ready for Streamlit Cloud
✅ **Encoding issues resolved** - Removed problematic characters

## Deployment Checklist

### Step 1: Install Git (Required)

1. Download Git from: https://git-scm.com/download/win
2. Run the installer with default settings
3. Restart your terminal/command prompt
4. Verify installation: `git --version`

### Step 2: GitHub Repository

Repository already exists at: https://github.com/mndolo2x/BlasterOPT

### Step 3: Setup Local Git Repository

Open Command Prompt in the project directory and run:

```bash
cd "C:\Users\User 2\Downloads\jules_session_2175162121400205044 (1)"
git init
git add .
git commit -m "Initial commit: BlastOpt Botswana platform with all modules"
git branch -M main
git remote add origin https://github.com/mndolo2x/BlasterOPT.git
git push -u origin main
```

**Quick Setup:** Run the provided script:
```bash
connect_to_repo.bat
```

### Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Click "Connect with GitHub" (if not already connected)
4. Select `mndolo2x/BlasterOPT` repository
5. Main file path: `app.py`
6. Click "Deploy"

### Step 5: Configure Environment Variables (Optional)

In Streamlit Cloud settings, add:
- Key: `DEMO_MODE`
- Value: `true`

## Local Testing

Before deploying, test locally:

```bash
cd "C:\Users\User 2\Downloads\jules_session_2175162121400205044 (1)"
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

### Git Installation Issues
- Make sure to restart your terminal after installing Git
- Check that Git is in your system PATH

### GitHub Authentication
- If asked for credentials, use your GitHub username and personal access token
- Create a token at: https://github.com/settings/tokens

### Streamlit Deployment Issues
- Ensure repository is PUBLIC
- Check that `app.py` is in the root directory
- Verify `requirements.txt` includes all dependencies
- Check deployment logs in Streamlit Cloud

### Import Errors on Streamlit Cloud
- Some modules require external dependencies (numpy, pandas, etc.)
- These are listed in `requirements.txt` and will be installed automatically
- The app handles missing dependencies gracefully with error messages

## Project Structure Verification

Your repository should contain:
```
blastopt-botswana/
├── app.py                    # Main application ✅
├── requirements.txt          # Python dependencies ✅
├── packages.txt             # System packages ✅
├── README.md                # Documentation ✅
├── .gitignore               # Git ignore rules ✅
├── .streamlit/
│   └── config.toml         # Streamlit config ✅
├── src/                     # Source modules ✅
├── models/                  # Model registry ✅
├── data/                    # Data directory ✅
└── tests/                   # Unit tests ✅
```

## Next Steps After Deployment

1. **Test the deployed app** - Verify all modules load correctly
2. **Configure production settings** - Set DEMO_MODE=false for live data
3. **Set up monitoring** - Enable error tracking and analytics
4. **Customize branding** - Update colors, logos, and branding
5. **Add real data** - Connect to actual mining data sources

## Support

For issues or questions:
- Check the deployment logs in Streamlit Cloud
- Review the `DEPLOYMENT.md` file for detailed instructions
- Run the startup diagnostics in the app to check module status
