import express from 'express';
import cors from 'cors';
import { AutoTokenizer, AutoModelForCausalLM } from '@huggingface/transformers';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

let tokenizer = null;
let model = null;
let isLoading = false;

const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct';

// Cloud provider configurations
const cloudProviders = {
  groq: {
    endpoint: 'https://api.groq.com/openai/v1/chat/completions',
    defaultModel: 'llama-3.3-70b-versatile',
    apiKeyEnv: 'GROQ_API_KEY',
  },
  openai: {
    endpoint: 'https://api.openai.com/v1/chat/completions',
    defaultModel: 'gpt-4o-mini',
    apiKeyEnv: 'OPENAI_API_KEY',
  },
  google: {
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models',
    defaultModel: 'gemini-2.0-flash',
    apiKeyEnv: 'GOOGLE_API_KEY',
  },
};

async function callCloudApi(provider, prompt, modelOverride) {
  const config = cloudProviders[provider];
  if (!config) {
    throw new Error(`Unknown provider: ${provider}`);
  }

  const apiKey = process.env[config.apiKeyEnv];
  if (!apiKey) {
    throw new Error(`API key not configured. Set ${config.apiKeyEnv} environment variable.`);
  }

  const modelId = modelOverride || config.defaultModel;

  if (provider === 'google') {
    const url = `${config.endpoint}/${modelId}:generateContent?key=${apiKey}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 1000 },
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error?.message || `Google API error: ${response.status}`);
    }

    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
  }

  // OpenAI-compatible format (OpenAI, Groq)
  const response = await fetch(config.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 1000,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `${provider} API error: ${response.status}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

async function loadModel() {
  if (model && tokenizer) return;
  if (isLoading) {
    while (isLoading) await new Promise(r => setTimeout(r, 100));
    return;
  }

  isLoading = true;
  console.log('Loading model...');

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { dtype: 'q4' });

  console.log('Model loaded');
  isLoading = false;
}

app.post('/api/summarize', async (req, res) => {
  const { text, detailLevel = 3, provider = 'local', model: modelOverride } = req.body;

  if (!text?.trim()) {
    return res.status(400).json({ error: 'Text is required' });
  }

  const detailInstructions = {
    0: 'Provide a very brief 1-2 sentence summary capturing only the core message.',
    1: 'Provide a concise summary in 3-4 sentences covering the main points.',
    2: 'Provide a moderate summary covering key points and some supporting details.',
    3: 'Provide a detailed summary with main points, supporting details, and examples.',
    4: 'Provide a comprehensive summary capturing all significant points, details, nuances, and examples.'
  };

  const prompt = `Summarize the following text. ${detailInstructions[detailLevel]} Use bullet points to organize the information.\n\nText:\n${text}\n\nSummary:`;

  try {
    const startTime = Date.now();
    let summary;

    if (provider === 'local') {
      // Use local HuggingFace model
      await loadModel();

      const messages = [{ role: 'user', content: prompt }];
      const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });

      const output = await model.generate({ ...inputs, max_new_tokens: 800, temperature: 0.5, do_sample: true });

      const promptLength = inputs.input_ids.dims[1];
      const newTokens = output.slice(null, [promptLength, null]);
      summary = tokenizer.batch_decode(newTokens, { skip_special_tokens: true })[0].trim();
    } else {
      // Use cloud provider (groq, openai, google)
      summary = await callCloudApi(provider, prompt, modelOverride);
    }

    const duration = (Date.now() - startTime) / 1000;
    res.json({ summary, duration, provider });
  } catch (err) {
    console.error('Summarization error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (_req, res) => {
  const availableProviders = ['local'];
  for (const [name, config] of Object.entries(cloudProviders)) {
    if (process.env[config.apiKeyEnv]) {
      availableProviders.push(name);
    }
  }
  res.json({ status: 'ok', modelLoaded: !!model, availableProviders });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('Model will load on first request');
});
