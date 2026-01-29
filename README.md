# On device web summarizer

A test app for downloading models locally and summarizing text with various parameters, and in various browsers.


# Quick Setup (Tailwind v4)

## Install dependencies

npm install lucide-react
npm install -D tailwindcss @tailwindcss/vite

## Run it

bun run dev

Open http://localhost:5173 in Chrome/Edge (WebGPU) or Firefox (CPU fallback).


## Run the API service (optional)

npm install
bun run server

### API Endpoints

**POST /api/summarize**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | The text to summarize |
| `detailLevel` | number (0-4) | 3 | Level of detail in the summary |
| `provider` | string | `"local"` | Provider: `local`, `groq`, `openai`, or `google` |
| `model` | string | varies | Optional model override |

**GET /api/health**

Returns server status and available providers.

### Cloud Provider Setup

Set environment variables for the providers you want to use:

```bash
export GROQ_API_KEY=your-groq-key
export OPENAI_API_KEY=your-openai-key
export GOOGLE_API_KEY=your-google-key
```

NOTE: I have only tested groq. My understanding is gemini free tier still requires a credit card to setup.

Default models per provider:
- **Groq**: `llama-3.3-70b-versatile`
- **OpenAI**: `gpt-4o-mini`
- **Google**: `gemini-2.0-flash`

### Examples

**Using the local model:**

```bash
curl -X POST http://localhost:3001/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "detailLevel": 3}'
```

**Using Groq:**

```bash
curl -X POST http://localhost:3001/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "provider": "groq"}'
```

**Using OpenAI with a specific model:**

```bash
curl -X POST http://localhost:3001/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "provider": "openai", "model": "gpt-4o"}'
```

**Summarize a file (e.g., sample-text.txt):**

```bash
curl -X POST http://localhost:3001/api/summarize \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg text "$(cat src/sample-text.txt)" '{text: $text, provider: "groq"}')"
```

Or with a heredoc:

```bash
curl -X POST http://localhost:3001/api/summarize \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "text": $(cat src/sample-text.txt | jq -Rs .),
  "provider": "groq",
  "detailLevel": 2
}
EOF
```

**Response:**

```json
{"summary": "...", "duration": 1.23, "provider": "groq"}
```

### About timeouts

* **Local model**: First request is slow (model download + load ~30s), subsequent requests 30-90s for long text on CPU
* **Cloud providers**: Typically respond in 1-5 seconds
* If deploying to serverless, you'd need GPU inference or a persistent server
* For production, consider running on a VPS or using a service like Modal/Replicate that supports longer-running inference
