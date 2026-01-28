# On device web summarizer

A test app for downloading models locally and summarizing text with various parameters, and in various browsers.


# Quick Setup (Tailwind v4)

## Install dependencies

npm install lucide-react
npm install -D tailwindcss @tailwindcss/vite

## Run it

npm run dev

Open http://localhost:5173 in Chrome/Edge (WebGPU) or Firefox (CPU fallback).


## Run the API service (optional)

npm install
npm run server

Then POST to http://localhost:3001/api/summarize :


curl -X POST http://localhost:3001/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "detailLevel": 3}'
Response:


{"summary": "...", "duration": 45.2}

About timeouts:

* First request is slow (model download + load ~30s)
* Subsequent requests: 30-90s for long text on CPU
* If deploying to serverless, you'd need GPU inference or a persistent server
* For production, consider running on a VPS or using a service like Modal/Replicate that supports longer-running inference
