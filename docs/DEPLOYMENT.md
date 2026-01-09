***

```markdown
# 🚀 Deployment Guide

## Heroku
```bash
heroku create kplc-assistant
heroku config:set GOOGLE_API_KEY=$GOOGLE_API_KEY
heroku config:set TAVILY_API_KEY=$TAVILY_API_KEY
git push heroku main
```

## Railway
```
1. Connect GitHub repo
2. Add env vars: GOOGLE_API_KEY, TAVILY_API_KEY
3. Deploy → https://railway.app
```

## Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_kplc_app.py", "--server.port=8501"]
```
```

***
