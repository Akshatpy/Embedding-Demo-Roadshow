# Deploying "Words → Vectors"

This folder is everything the app needs and nothing it doesn't — 4 files, ~560KB total
(the big 73MB NumPy index it's based on already got reduced down to a 540KB stats file).
Send someone this **folder**, or push it as its own repo; either works.

| File | Why it's here |
|---|---|
| `vector_demo.py` | The Streamlit app |
| `vector_core.py` | Model + vector math it imports |
| `numpy_bm25_stats.json` | Real NumPy-corpus word-frequency stats (for the "classic sparse" panel) |
| `requirements.txt` | Pinned deps, CPU-only torch |

The two ML models (MiniLM, SpladeX) are **not** bundled — they download from Hugging
Face automatically on first run and get cached after that. That means the very first
query after a cold start takes a few extra seconds; after that it's fast.

## Free hosting: two good options

### Option A — Streamlit Community Cloud (simplest, made for exactly this)

1. Push this folder to a **public GitHub repo** (see "Getting it onto GitHub" below).
2. Go to **share.streamlit.io**, sign in with GitHub.
3. **Create app** → pick the repo, branch `main`, main file path `vector_demo.py` → **Deploy**.
4. Wait ~2-5 min for the first build (installing torch/transformers takes a bit). You get
   a URL like `https://<something>.streamlit.app` — that's what you send people.

Free tier: 1 CPU / ~1GB RAM, the app sleeps after inactivity and wakes on the next visit
(~20-30s wake-up delay). Totally fine for this app's size.

### Option B — Hugging Face Spaces (also free, arguably a better fit)

The SpladeX model already lives on Hugging Face, so keeping the demo there too is a
nice fit, and the free CPU tier is roomier than Streamlit Cloud's.

1. Go to **huggingface.co/new-space**.
2. Name it, pick **SDK: Streamlit**, **Hardware: CPU basic (free)** → **Create Space**.
3. Either:
   - Use the web **"Add file" → "Upload files"** button and drag in all 4 files, or
   - `git clone` the Space's own git URL (shown on its page) and copy these 4 files in,
     then `git add . && git commit -m "deploy" && git push`.
4. It builds automatically from `requirements.txt` and starts `vector_demo.py`. URL is
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

## Getting it onto GitHub (for Option A)

From this `deploy` folder:

```
git init
git add .
git commit -m "SpladeX words-to-vectors demo"
git branch -M main
git remote add origin https://github.com/<you>/<repo-name>.git
git push -u origin main
```

(Create the empty repo on github.com first, don't initialize it with a README there —
keeps `git push` simple.)

## Before you send the link to anyone

- **It's public.** Both platforms' free tiers put the app at a guessable-but-unlisted
  URL, not password-protected. Fine for a roadshow demo link; don't put anything
  sensitive in front of it.
- **Cold start ≈ 30-60s** the first time anyone hits it after it's been idle — the
  container wakes up and downloads/loads the models. Worth a heads-up if you're sharing
  it live rather than pre-warming it yourself first.
- Test the deployed URL yourself once before sharing — type a query, confirm all three
  panels render — same as you'd do for any new environment.
