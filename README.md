import os
import torch
import pandas as pd
import logging
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Login to Hugging Face using your token
try:
    login(token="....")
    logging.info("Successfully logged into Hugging Face.")
except Exception as e:
    logging.error("Error during Hugging Face login: %s", e)
    raise

# Initialize accelerator (for potential custom multi-GPU training)
accelerator = Accelerator()

# Load dataset (Excel file)
def load_data(file_path):
    try:
        # If your file is a CSV, you can replace the following line with:
        # df = pd.read_csv(file_path)
        df = pd.read_excel(file_path)
        logging.info("Dataset loaded successfully from %s", file_path)
        return df
    except Exception as e:
        logging.error("Error loading dataset from %s: %s", file_path, e)
        raise

# Extract important features from dataset
def extract_features(df):
    try:
        # Example: Selecting key network traffic features
        important_columns = ["packet_size", "src_ip", "dst_ip", "protocol", "attack_type"]
        df = df[important_columns]
        df = df.dropna()
        logging.info("Important features extracted successfully.")
        return df
    except Exception as e:
        logging.error("Error extracting features: %s", e)
        raise

# Preprocess dataset by extracting features and converting to Hugging Face Dataset
def preprocess_data(df):
    try:
        df = extract_features(df)
        dataset = Dataset.from_pandas(df)
        logging.info("Dataset preprocessed successfully.")
        return dataset
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise

# Load tokenizer and model using the new model identifier and ensure GPU (cuda) usage
try:
    tokenizer = AutoTokenizer.from_pretrained(".....", use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision",
        torch_dtype=torch.float16,
        device_map="cuda",
        use_auth_token=True
    )
    logging.info("Tokenizer and model loaded successfully.")
except Exception as e:
    logging.error("Error loading tokenizer/model: %s", e)
    raise

# Apply LoRA for parameter-efficient fine-tuning
try:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    logging.info("LoRA applied successfully.")
except Exception as e:
    logging.error("Error applying LoRA: %s", e)
    raise

# Tokenization function; adjust key if needed based on your dataset's column names
def tokenize_function(examples):
    try:
        # Here, we assume that the feature for text input is in a column named "text".
        # Modify the field if your dataset uses a different column name.
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    except Exception as e:
        logging.error("Error during tokenization: %s", e)
        raise

# Main function to train the model
def train(file_path):
    try:
        # Load and preprocess data
        df = load_data(file_path)
        dataset = preprocess_data(df)
        
        # Split the dataset into train (80%), validation (10%), and test (10%)
        split1 = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split1["train"]
        temp_dataset = split1["test"]
        split2 = temp_dataset.train_test_split(test_size=0.5, seed=42)
        eval_dataset = split2["train"]
        test_dataset = split2["test"]  # Reserved for final evaluation
        logging.info("Dataset split into train, eval, and test sets.")
        
        # Tokenize the train and validation datasets
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        logging.info("Tokenization of train and eval datasets completed.")
        
        # Define training arguments with Hub integration (push the final model to Hugging Face Hub)
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            fp16=True,
            push_to_hub=True,
            hub_model_id=".....",  # Replace with your repo name if needed
            hub_token="....."
        )
        
        # Create the Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer
        )
        
        # Train the model
        logging.info("Starting model training.")
        trainer.train()
        logging.info("Model training completed.")
        
        # Push the model to the Hugging Face Hub
        trainer.push_to_hub()
        logging.info("Model pushed to Hugging Face Hub successfully.")
        
        # Optionally, also save the model locally
        model.save_pretrained("./fine_tuned_llama3")
        tokenizer.save_pretrained("./fine_tuned_llama3")
        logging.info("Model saved locally at ./fine_tuned_llama3")
        print("Fine-tuning complete. Model saved locally and pushed to Hugging Face Hub.")
    except Exception as e:
        logging.error("Error during training: %s", e)
        raise

# Run training using the specified file path
train(r".....")

