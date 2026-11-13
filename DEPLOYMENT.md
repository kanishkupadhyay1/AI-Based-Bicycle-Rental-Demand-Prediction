# Deployment Guide: Render (Backend) & Vercel (Frontend)

This guide documents the exact steps to deploy the AI-Based Bicycle Rental Demand Prediction project using Render for the FastAPI backend and Vercel for the React frontend.

---

## 1. Architecture Overview
- **Backend**: FastAPI serving trained models (PoissonGAM, Gradient Boosting, Random Forest, Linear Regression) and real dataset analytics.
  - Hosted on: **Render (Web Service)**
  - Runtime: **Python 3.11.11** (enforced by `.python-version`)
- **Frontend**: React + Vite + Tailwind CSS + Recharts
  - Hosted on: **Vercel**
  - Communicates with backend using `VITE_API_BASE_URL`

---

## 2. Render Deployment (FastAPI Backend)

### Why `.python-version` is required
By default, Render defaults to Python 3.14+ on newer builds. Dependencies such as `scipy`, `numpy`, and `pygam` require pre-built binary wheels for Python 3.11 to avoid compilation from source (which fails due to missing Fortran compilers). The repository root contains:
```text
.python-version -> 3.11.11
```
Render automatically reads this file and installs Python 3.11.11 with pre-built binary wheels for all requirements.

### Render Service Settings:
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect your GitHub repository: `kanishkupadhyay1/AI-Based-Bicycle-Rental-Demand-Prediction`.
3. Configure settings:
   - **Name**: `bike-demand-api` (or your preferred name)
   - **Language / Environment**: `Python 3`
   - **Region**: Closest to your users (e.g. Frankfurt, Oregon, Ohio, Singapore)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (uses repo root)
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
     ```
4. **Environment Variables** (in Render Web Service -> Environment):
   - `PYTHON_VERSION`: `3.11.11` (optional redundancy, Render also respects `.python-version`)
   - `FRONTEND_URL`: `https://your-frontend-app.vercel.app` (set this after creating the Vercel project or set `*` initially)
5. Click **Create Web Service**.
6. Note your Render URL once deployed (e.g., `https://bike-demand-api.onrender.com`).
   - Health check: `https://bike-demand-api.onrender.com/`
   - Swagger docs: `https://bike-demand-api.onrender.com/docs`

---

## 3. Vercel Deployment (React Frontend)

### Vercel Project Settings:
1. Go to [Vercel Dashboard](https://vercel.com/) and click **Add New...** -> **Project**.
2. Import the repository: `kanishkupadhyay1/AI-Based-Bicycle-Rental-Demand-Prediction`.
3. Configure settings:
   - **Project Name**: `ai-bicycle-rental-demand`
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend` *(Click Edit and select the `frontend` directory)*
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
4. **Environment Variables**:
   - `VITE_API_BASE_URL`: `https://bike-demand-api.onrender.com` (replace with your actual Render backend URL)
5. Click **Deploy**.

---

## 4. Connecting Frontend & Backend (CORS Alignment)

Once both are deployed:
1. Copy your Vercel deployment URL (e.g., `https://ai-bicycle-rental-demand.vercel.app`).
2. Go to **Render Dashboard** -> Your Web Service -> **Environment**.
3. Set or update:
   - `FRONTEND_URL`: `https://ai-bicycle-rental-demand.vercel.app`
4. Save changes. Render will automatically redeploy or restart with the new CORS origin whitelist.

---

## 5. Verification Checklist Post-Deployment

- [ ] `GET https://<render-backend-url>/` returns `{"status":"healthy", "models_loaded":true}`.
- [ ] `GET https://<render-backend-url>/api/metrics` returns the 4 model evaluations.
- [ ] `GET https://<render-backend-url>/api/gam-effects` returns the 6 smooth partial curves.
- [ ] Open the Vercel frontend in your browser:
  - Header pill displays `API Connected` in green.
  - Dashboard charts render real demand curves.
  - "Predict Demand" form returns real predicted counts from the deployed backend models.
  - "GAM Smooth Effects" tab renders all 6 spline curves with 95% confidence bands.
