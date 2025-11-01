from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Trainer,
    TrainingArguments
)
import torch
from datasets import load_dataset

MODEL_NAME = "facebook/blenderbot-400M-distill"

# 🔹 Load tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

# 🔹 Load your dataset
dataset = load_dataset("empathetic_dialogues")

def preprocess_function(examples):
    inputs = [ex for ex in examples["input_text"]]
    targets = [ex for ex in examples["target_text"]]
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(preprocess_function, batched=True)

# 🔹 Training settings
training_args = TrainingArguments(
    output_dir="./models/inner_self_blenderbot",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_total_limit=1,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
)

# 🔹 Start fine-tuning
trainer.train()

# 🔹 Save final model
trainer.save_model("./models/inner_self_blenderbot")
tokenizer.save_pretrained("./models/inner_self_blenderbot")
