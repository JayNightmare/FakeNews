# Unified Prompt Template

Use this template as a mental model for how `src/prompts.py` converts a `UnifiedRecord` into a prompt context for the LLM.

---

You are a misinformation detection system. Analyze the provided content and determine whether it is real or fake. Return a structured JSON response.

## Task

Classify the following content as real or fake.

## Content

Claim: {{text}}

## Additional Context

{{context_text}}

## Metadata

{{metadata_key_1}}: {{metadata_value_1}}
{{metadata_key_2}}: {{metadata_value_2}}
...

## Output Requirements

Return JSON with this shape:

```json
{
  "classification": "real" or "fake",
  "confidence": 0.0 to 1.0,
  "explanation": "1-3 sentence justification for the classification",
  "reasoning_signals": ["list of key signals used"],
  "requires_external_evidence": true/false
}
```

Rules:
- Use only the information provided.
- If context is missing or insufficient, lower your confidence.
- Do not invent evidence.
- Keep the explanation to 1-3 sentences.
