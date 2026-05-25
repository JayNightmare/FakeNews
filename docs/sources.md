# Sources consulted

## Official / primary

- Google Fact Check Tools API docs: https://developers.google.com/fact-check/tools/api/
- Claims resource reference: https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims
- Data Commons fact-check download page: https://datacommons.org/factcheck/download
- Data Commons fact-check FAQ: https://datacommons.org/factcheck/faq
- Daily ClaimReview feed: https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json
- Google Fact Check Explorer: https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=en
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/index
- Transformers chat templating docs: https://huggingface.co/docs/transformers/chat_templating
- PEFT docs: https://huggingface.co/docs/peft/index
- LoRA reference paper page: https://huggingface.co/papers/2106.09685
- Qwen model collection: https://huggingface.co/Qwen

## Project-context reference

- TechRxiv survey noted in the meeting notes: From Fact Verification to Understanding Misleadingness: A Survey and Roadmap on Reader-Centric Multimodal Misinformation Detection
- Recent repository history captured in `CHANGELOG.md`

## Practical takeaway from the sources

- Google/Data Commons provides structured ClaimReview metadata and a daily feed.
- The structured release does not include the full fact-check article body.
- The Explorer is useful for quick topic inspection and manual sampling.
- Google Fact Check Tools search returns structured claim-review matches and requires an API key for live runtime lookup.
- Transformers chat templates and PEFT/LoRA are the main references for the Hugging Face-first inference and adapter-training workflow.
