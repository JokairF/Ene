import os
import torch
import json
# from app import config
from datasets import Dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------------
# 1. PARAMÈTRES DE BASE
# -------------------------
HERE = Path(__file__).parent.resolve()
MODEL_NAME = r"D:\models\Mistral-7B-Instruct-v0.1" # ou ton modèle local
OUTPUT_DIR = HERE / "ene_lora_model"
DATASET_PATH = HERE / "ene_dataset.json"  # chemin vers le mini dataset JSON fourni

# QLoRA 4-bit (stable sous Windows)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # bf16 si possible (RTX 40xx) sinon fp16
    bnb_4bit_compute_dtype=torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8
    else torch.float16
)

MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
LR = 2e-4
MAX_LEN = 1024  # adapte si VRAM limitée (768/896)

# -------------------------
# 1. CHARGEMENT DATASET
# -------------------------
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset introuvable: {DATASET_PATH}")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

# Assure-toi que completion est bien un JSON string
def normalize_example(ex):
    prompt = ex["prompt"]
    comp = ex["completion"]
    if not isinstance(comp, str):
        comp = json.dumps(comp, ensure_ascii=False)
    return {"prompt": prompt, "completion": comp}

data = [normalize_example(x) for x in raw]
dataset = Dataset.from_list(data)

# -------------------------
# 2. TOKENIZER & MODÈLE
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
)

# Préparation QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

# -------------------------
# 3. TOKENISATION + MASQUAGE LABELS
#    (loss calculée uniquement sur la "completion")
# -------------------------
def tokenize_and_mask(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]

    input_ids_list = []
    attention_masks = []
    labels_list = []

    for p, c in zip(prompts, completions):
        # Texte complet = prompt + completion(JSON)
        full = p + "\n" + c

        # Tokenize séparément pour obtenir la longueur du prompt
        enc_prompt = tokenizer(p, add_special_tokens=False)
        enc_full = tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
        )

        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]

        # Longueur de la partie "prompt"
        n_prompt = len(enc_prompt["input_ids"])

        # Labels = -100 sur le prompt, labels identiques à input_ids sur la completion
        labels = [-100] * len(input_ids)
        for i in range(n_prompt, len(input_ids)):
            labels[i] = input_ids[i]

        input_ids_list.append(input_ids)
        attention_masks.append(attn)
        labels_list.append(labels)

    # Padding dynamique (left-pad par défaut chez HF ; on force pad à droite)
    batch_max = max(len(x) for x in input_ids_list)
    def pad_to(x, pad_id):
        return x + [pad_id] * (batch_max - len(x))
    def pad_labels(x):
        return x + [-100] * (batch_max - len(x))

    input_ids_list = [pad_to(x, tokenizer.pad_token_id) for x in input_ids_list]
    attention_masks = [pad_to(x, 0) for x in attention_masks]
    labels_list = [pad_labels(x) for x in labels_list]

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels_list,
    }

tokenized = dataset.map(tokenize_and_mask, batched=True, remove_columns=["prompt", "completion"])

# -------------------------
# 4. ARGUMENTS D'ENTRAÎNEMENT
# -------------------------
use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8
)

training_args = TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    output_dir=str(OUTPUT_DIR),
    report_to="none",
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="adamw_torch",
)

# -------------------------
# 5. TRAINER
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

model.config.use_cache = False  # nécessaire avec gradient_checkpointing
trainer.train()

# -------------------------
# 6. SAUVEGARDE ADAPTER LoRA
# -------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"✅ Entraînement terminé. Adapter LoRA sauvegardé dans: {OUTPUT_DIR}") 
