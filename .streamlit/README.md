# Streamlit Cloud Deployment Setup

## Environment Variables (Secrets)

For **local development**, create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
HF_TOKEN = "your-huggingface-token-here"  # optional
```

> ⚠️ **IMPORTANT**: Never commit `secrets.toml` to git. Add to `.gitignore`.

For **Streamlit Cloud deployment**, set secrets in the Streamlit Cloud dashboard:
1. Go to your app settings: https://share.streamlit.io/your-username/repo-name/settings
2. Click "Secrets" in the menu
3. Paste your keys in TOML format:
   ```toml
   GROQ_API_KEY = "your-production-key"
   HF_TOKEN = "your-hf-token"
   ```

## Getting API Keys

- **Groq**: https://console.groq.com/keys (free tier: 100k tokens/day)
- **HuggingFace**: https://huggingface.co/settings/tokens (optional for rate limits)

## Deployment Commands

```bash
# Local development
streamlit run streamlit_app.py

# Deploy to Streamlit Cloud
git push  # Changes automatically deploy to Streamlit Cloud
```
