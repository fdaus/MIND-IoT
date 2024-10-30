import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Check if GPU is available
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("GPU not available, training on CPU")

# Load and preprocess the dataset
data = load_dataset("json", data_files="data/json/dataset_balanced.json", split='train')

# Initialize the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("./models/tokenizer_iot", max_len=576)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the entire dataset
tokenized_datasets = data.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])

# Configuration for the RoBERTa model
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=578,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Initialize the model
model = RobertaForMaskedLM(config=config)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/pretrained_iot-continue_2",
    do_train=True,
    overwrite_output_dir=True,
    num_train_epochs=500,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,  # Effective batch size = 16 * 8 = 128
    save_steps=500,
    save_total_limit=5,
    seed=42
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets
)

# Start the training process
trainer.train()