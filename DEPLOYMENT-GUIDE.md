# Deployment Guide for Healthcare RAG

This guide explains how to properly connect your frontend and backend using environment variables, and how to securely deploy them to platforms like Render and Railway.

## 1. Environment Variable Setup (Connecting Frontend and Backend)

We've updated the codebase so both the Frontend and Backend can be securely linked using environment variables without hardcoding `localhost`.

### Backend Environment Variables (`.env`)
You can now restrict which websites are allowed to make requests to your API using CORS (Cross-Origin Resource Sharing). 

Add this to your backend `.env` (or cloud dashboard):
```env
# Comma-separated list of allowed frontend URLs
# Start with localhost for testing, then add your deployed URL later
CORS_ORIGINS="http://localhost:3000,https://your-frontend.vercel.app"
```
*(If you leave this out, it defaults to `*` which allows all connections).*

### Frontend Environment Variables (`.env.local`)
The frontend needs to know where the backend API lives.

Add this to your frontend `.env.local` (or cloud dashboard):
```env
# The URL of your backend server
NEXT_PUBLIC_API_BASE="https://your-backend.onrender.com"
```

---

## 2. Deploying the Backend (FastAPI) on Render

Render is excellent for hosting Python web services.

1. **Create a New Web Service** on [Render.com](https://render.com).
2. **Connect your GitHub repository** (`healtcare-Rag-1`).
3. **Configure the Service**:
   - **Root Directory**: `backend` (This is crucial, it tells Render your app is inside the backend folder).
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables**:
   - Scroll down to "Environment Variables" and add all the keys from your backend `.env` file (OpenAI, Supabase, Qdrant, LlamaCloud).
   - Add `CORS_ORIGINS` with the URL of where your frontend will be hosted (you can add this later once the frontend is deployed).
5. **Click Deploy**. 
   *Note: Render free tiers sleep after 15 minutes of inactivity, so the first request might take 50 seconds to wake it up.*

---

## 3. Deploying the Frontend (Next.js) on Railway

Railway is fantastic for hosting Next.js applications seamlessly.

1. **Create a New Project** on [Railway.app](https://railway.app).
2. Select **Deploy from GitHub repo** and choose `healtcare-Rag-1`.
3. **Configure the Service**:
   - Go to **Settings** > **Build**.
   - Under **Root Directory**, type `/Frontend` (This tells Railway the Next.js app is in this folder).
   - Railway will automatically detect it is a Next.js app and run `npm install` and `npm run build`.
4. **Environment Variables**:
   - Go to the **Variables** tab.
   - Add: `NEXT_PUBLIC_API_BASE` and set its value to the URL of your deployed Render backend (e.g., `https://my-backend.onrender.com`).
5. **Generate a Domain**:
   - Go to **Settings** > **Networking** and click **Generate Domain** (or attach a custom domain).
6. **Deploy**.

---

## 4. Finalizing the Connection

Once both are deployed:
1. Copy the public URL of your Railway frontend (e.g., `https://my-frontend.up.railway.app`).
2. Go back to your **Render** backend dashboard.
3. Update the `CORS_ORIGINS` environment variable to include your Railway URL:
   `CORS_ORIGINS="https://my-frontend.up.railway.app,http://localhost:3000"`
4. Save and let Render restart the service.

Your application is now fully deployed, securely connected via CORS, and communicating via environment variables!
