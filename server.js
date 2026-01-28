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
  const { text, detailLevel = 3 } = req.body;

  if (!text?.trim()) {
    return res.status(400).json({ error: 'Text is required' });
  }

  try {
    await loadModel();

    const detailInstructions = {
      0: 'Provide a very brief 1-2 sentence summary capturing only the core message.',
      1: 'Provide a concise summary in 3-4 sentences covering the main points.',
      2: 'Provide a moderate summary covering key points and some supporting details.',
      3: 'Provide a detailed summary with main points, supporting details, and examples.',
      4: 'Provide a comprehensive summary capturing all significant points, details, nuances, and examples.'
    };

    const prompt = `Summarize the following text. ${detailInstructions[detailLevel]} Use bullet points to organize the information.\n\nText:\n${text}\n\nSummary:`;

    const messages = [{ role: 'user', content: prompt }];
    const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });

    const startTime = Date.now();
    const output = await model.generate({ ...inputs, max_new_tokens: 800, temperature: 0.5, do_sample: true });
    const duration = (Date.now() - startTime) / 1000;

    const promptLength = inputs.input_ids.dims[1];
    const newTokens = output.slice(null, [promptLength, null]);
    const summary = tokenizer.batch_decode(newTokens, { skip_special_tokens: true })[0].trim();

    res.json({ summary, duration });
  } catch (err) {
    console.error('Summarization error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modelLoaded: !!model });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('Model will load on first request');
});
