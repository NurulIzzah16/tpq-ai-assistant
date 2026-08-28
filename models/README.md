# Models Directory

This directory stores the fine-tuned model output.

## After Training

After running `python training/train.py`, the fine-tuned LoRA adapter will be saved here:

```
models/
└── qwen-tpq-sft/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## Important Notes

- Model weights are **not** committed to Git (see `.gitignore`).
- The fine-tuned model is a **LoRA adapter**, not a full model copy.
- To use the model, you need both the base model (`unsloth/Qwen2.5-1.5B-Instruct`) and this adapter.
