# Fine Tuning LLMs — Notebooks in this folder

This folder contains two Jupyter notebooks that demonstrate two different parameter-efficient fine-tuning workflows. The README below describes precisely what the code in each notebook does and any related artifacts the code produces.

## Repository contents
- `Fine_tune_Llama_2 (1).ipynb`  
  Fine-tuning a Llama 2 chat model (QLoRA + LoRA + SFTTrainer, HF/transformers + bitsandbytes).
- `lora_tuning.ipynb`  
  LoRA fine-tuning of a Gemma model using Keras/KerasNLP on the Databricks Dolly dataset.

---

## Notebook: Fine_tune_Llama_2 (1).ipynb
What the code does:
- Installs required Python packages: accelerate, peft, bitsandbytes, transformers, trl (specific versions are used in the notebook).
- Imports libraries (torch, datasets, transformers, peft, trl, etc.).
- Describes and applies a Llama 2 chat-style prompt template (system + user + model answer) for instruction-style data.
- Loads the instruction dataset `mlabonne/guanaco-llama2-1k` from Hugging Face (the notebook uses this dataset name).
  - Links mentioned in the notebook: original dataset [timdettmers/openassistant-guanaco], reformat versions [mlabonne/guanaco-llama2-1k] and [mlabonne/guanaco-llama2].
- Configures QLoRA / bitsandbytes parameters:
  - Loads base model (`NousResearch/Llama-2-7b-chat-hf`) in 4-bit precision via bitsandbytes config (nf4 quantization, compute dtype float16).
  - Sets LoRA hyperparameters (r=64, alpha=16, dropout=0.1).
- Sets training arguments (learning rate, optimizer, batch sizes, gradient checkpointing, epochs, logging, warmup, scheduler, etc.) and SFTTrainer parameters (dataset_text_field, packing, max_seq_length).
- Creates an SFTTrainer instance with the quantized model, tokenizer, peft (LoRA) config and dataset, then runs `trainer.train()`.
- Saves the trained model (calls `trainer.model.save_pretrained(new_model)`).
- Shows how to inspect training metrics with TensorBoard (`%tensorboard --logdir results/runs`).
- Demonstrates running a text-generation pipeline for quick inference using the fine-tuned model/tokenizer (example prompt).
- Shows cleanup steps to free VRAM (delete objects, gc.collect).
- Reloads the full base model in FP16, loads LoRA weights with `PeftModel.from_pretrained(...)`, merges and unloads (`merge_and_unload()`), then pushes merged model + tokenizer to the Hugging Face Hub (example push uses `model.push_to_hub` and `tokenizer.push_to_hub`).

Notes included in the code:
- The notebook explains resource constraints on Colab (GPU VRAM) and motivates QLoRA/LoRA to reduce memory usage.
- It recommends merging LoRA weights into the base model before pushing to HF Hub and includes example commands to log in and push.

Primary external resources referenced in the notebook:
- Model: `NousResearch/Llama-2-7b-chat-hf`
- Dataset: `mlabonne/guanaco-llama2-1k` (and links to related Guanaco datasets on HF)

---

## Notebook: lora_tuning.ipynb
What the code does:
- Provides a Keras/KerasNLP tutorial-style notebook to fine-tune Gemma models with LoRA.
- Shows setup steps for Colab/Kaggle access to Gemma (set `KAGGLE_USERNAME` / `KAGGLE_KEY`), selects runtime (T4 GPU recommendation).
- Installs dependencies used in the notebook (`keras-nlp` and `keras>=3`).
- Configures Keras backend environment (example sets `KERAS_BACKEND="jax"` and an XLA memory fraction).
- Loads the Databricks Dolly 15k dataset file (`databricks-dolly-15k.jsonl`) via wget and preprocesses it:
  - Filters examples with context (the notebook filters out examples that include `context`) and formats each example using a simple "Instruction / Response" template.
  - Uses a subset of 1000 examples in the notebook for faster execution.
- Loads a Gemma causal LM from a preset (`keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")`) and shows model summary.
- Enables LoRA on the Gemma backbone with a specified rank (`rank=4`) reducing trainable params.
- Sets training configuration in Keras:
  - Limits sequence length to 512, configures optimizer (AdamW) and loss (SparseCategoricalCrossentropy), compiles, and runs `fit()` for 1 epoch on the prepared data (batch_size=1 in the notebook example).
- Demonstrates inference before and after fine-tuning by generating text for example prompts (Europe trip, ELI5 photosynthesis).
- Notes on precision:
  - Mentions mixed precision is optional and comments how to enable it (commented example line in the notebook).
- The notebook contains instructional content (setup instructions, explanation of LoRA) and code to perform end-to-end fine-tuning with KerasNLP.

Primary external resources referenced in the notebook:
- Dataset download: `https://huggingface.co/datasets/databricks/databricks-dolly-15k`
- Gemma presets via KerasNLP (`gemma_2b_en`)

---

## Requirements (as used by the notebooks)
- For Llama/QLoRA notebook:
  - accelerate==0.21.0, peft==0.4.0, bitsandbytes==0.40.2, transformers==4.31.0, trl==0.4.7 (these are installed in a cell).
  - PyTorch + CUDA GPU (the notebook uses torch and checks device capability).
  - Hugging Face Hub credentials if pushing models.
- For Gemma LoRA (Keras) notebook:
  - keras-nlp (installed in the notebook), keras>=3
  - Colab runtime with a suitable GPU (T4 recommended in the notebook) or other environment that can load Gemma presets.
  - Kaggle credentials for Gemma access (the notebook instructs storing KAGGLE_USERNAME and KAGGLE_KEY).

---

## How to run these notebooks
- Open the notebook in Colab or a local Jupyter environment that has the required GPU and Python packages.
- Run cells top-to-bottom; notebooks install the necessary packages and set environment variables where needed.
- For the Llama notebook:
  - Ensure a GPU with sufficient memory or adjust quantization / device_map settings.
  - Log in to Hugging Face before push steps (`!huggingface-cli login`) if you want to push merged weights to the Hub.
- For the Gemma notebook:
  - Provide Kaggle credentials in Colab secrets (if using Gemma via Kaggle) and choose a runtime with a GPU that meets the notebook recommendations.

---

## Outputs produced by the code
- Fine-tuned model checkpoints / saved models (the Llama notebook saves to `./results` and `new_model` when calling `save_pretrained`).
- TensorBoard logs under `results/runs` (QLoRA notebook).
- Merged/merged-and-pushed model on Hugging Face Hub if push steps are executed.
- In the Keras notebook, `History` objects from `.fit()` and generated text outputs for inference cells.

---

## References (from the notebooks)
- Guanaco reformat datasets on Hugging Face:
  - mlabonne/guanaco-llama2-1k — used as training data in the Llama notebook.
  - mlabonne/guanaco-llama2 — full reformat suggested by the notebook.
- Databricks Dolly 15k dataset:
  - https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Models:
  - NousResearch/Llama-2-7b-chat-hf (used as base in the Llama-2 fine-tuning notebook)
  - Gemma presets in KerasNLP (gemma_2b_en) used in the Gemma LoRA notebook

---

