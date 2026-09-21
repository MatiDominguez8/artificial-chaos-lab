# Next milestone — Local agents

Target: replace `RandomPolicy` with a local model while keeping the simulation deterministic everywhere except model inference.

Planned interface:

```json
{
  "action": "steal",
  "target": "Eva",
  "amount": 10,
  "message": null,
  "reason": "I need food and have little money"
}
```

The reason is logged for analysis but never grants new capabilities.

## Local stack candidate

- Ollama as the simplest local inference server
- Qwen-family instruct model sized for a 16 GB GPU
- JSON-schema / structured-output validation
- retries only for malformed actions

## After that

Add relationships, memory and a small 2D replay viewer before increasing agent count.
