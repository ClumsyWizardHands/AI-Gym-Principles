# Railway Deployment Guide for AI Principles Gym

This guide will walk you through deploying your AI Principles Gym to Railway with a custom subdomain.

## ⚠️ SECURITY ALERT
**IMMEDIATELY go to https://console.anthropic.com and:**
1. Revoke the API key you shared
2. Create a new one
3. Keep it secret - we'll add it securely in Railway

## Quick Start (10-15 minutes)

### Step 1: Sign Up for Railway
1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign up with GitHub (use the account with your code)

### Step 2: Deploy the Backend

1. Click **"Deploy from GitHub repo"**
2. Select **"AI-Gym-Principles"** repository
3. Railway will auto-detect it's a Python app

**Environment Variables** - Click on the service, then "Variables":
```
# Core Settings
PORT=8000
ENVIRONMENT=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}

# Your API Keys (Add these SECURELY)
ANTHROPIC_API_KEY=your-new-key-here
OPENAI_API_KEY=your-openai-key-if-you-have-one

# Security
JWT_SECRET_KEY=generate-a-random-string-here
ENABLE_AUTH=true
CORS_ORIGINS=["https://app.yourdomain.com","https://yourdomain.com"]
```

4. Click **"Add Database"** → PostgreSQL
5. Click **"Add Database"** → Redis
6. Wait for deployment (2-3 minutes)

### Step 3: Deploy the Frontend

1. Click **"New"** → **"Empty Service"** in your Railway project
2. Name it "frontend"
3. Connect the same GitHub repo
4. In Settings, set:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npm run preview -- --port $PORT --host 0.0.0.0`

5. Add environment variable:
```
VITE_API_URL=https://your-backend-url.railway.app
```
(Replace with your actual backend URL from Railway)

### Step 4: Set Up Custom Domain

#### In Railway:
1. Click on your frontend service
2. Go to **Settings** → **Domains**
3. Click **"Generate Domain"** to get a temporary domain
4. Click **"+ Custom Domain"**
5. Enter: `app.yourdomain.com`
6. Railway will show you DNS records

#### In Squarespace:
1. Go to **Settings** → **Domains** → **Advanced Domain Settings**
2. Click **"External Domain" or "DNS Settings"**
3. Add a new CNAME record:
   - **Host**: `app`
   - **Points to**: `your-frontend.railway.app` (from Railway)
   - **TTL**: Automatic

4. Save and wait 5-10 minutes for DNS to propagate

### Step 5: Update Your Squarespace Site

1. In Squarespace, go to **Pages**
2. Add a **Link** to your navigation:
   - **Title**: "Launch App"
   - **URL**: `https://app.yourdomain.com`

3. Or add a button on any page:
   - Add a **Button Block**
   - Link to: `https://app.yourdomain.com`
   - Text: "Access AI Principles Gym"

## Environment Variables Reference

### Required Variables:
```bash
# Backend Service
DATABASE_URL          # Auto-provided by Railway PostgreSQL
REDIS_URL            # Auto-provided by Railway Redis
ANTHROPIC_API_KEY    # Your Anthropic API key
JWT_SECRET_KEY       # Generate with: openssl rand -hex 32

# Frontend Service  
VITE_API_URL         # Your backend Railway URL
```

### Optional Variables:
```bash
OPENAI_API_KEY       # If using OpenAI
MAX_SCENARIOS_PER_SESSION=100
MAX_CONCURRENT_SESSIONS=50
LOG_LEVEL=INFO
```

## Troubleshooting

### "Build Failed" Error
- Check the build logs in Railway
- Make sure all dependencies are in requirements.txt
- Ensure Python version is 3.11+

### "Cannot Connect to Backend"
- Verify VITE_API_URL is set correctly in frontend
- Check CORS_ORIGINS includes your frontend URL
- Ensure backend is deployed and running

### "Database Connection Failed"
- Railway auto-injects DATABASE_URL
- Don't manually set database credentials
- Check if PostgreSQL service is running

### Custom Domain Not Working
- DNS can take up to 48 hours (usually 5-10 minutes)
- Verify CNAME record is correct
- Try: `nslookup app.yourdomain.com`

## Next Steps

1. **Monitor your app**: Railway dashboard shows logs and metrics
2. **Set up backups**: Enable daily backups in PostgreSQL settings
3. **Configure alerts**: Set up uptime monitoring
4. **Scale if needed**: Upgrade to Team plan for more resources

## Costs

With Railway Team plan (~$20/month):
- Backend: ~$5-10/month
- Frontend: ~$5/month  
- PostgreSQL: ~$5-10/month
- Redis: ~$5/month
- **Total**: ~$20-30/month

Free tier available but limited to 500 hours/month.

## Support

- Railway Discord: https://discord.gg/railway
- Railway Docs: https://docs.railway.app
- Your app logs: Check Railway dashboard

---

**Remember**: 
1. Rotate that Anthropic API key NOW
2. Never commit API keys to GitHub
3. Always use environment variables for secrets

Ready? Go to https://railway.app and let's deploy! 🚀
