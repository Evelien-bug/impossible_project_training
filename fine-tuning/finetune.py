import random
import json
from pathlib import Path
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import yaml
from datasets import Dataset

def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


DEVICE = get_device()


def create_reversal_example(text, marker='🅁'):
    tokens = text.split()
    if len(tokens) < 3:
        return None

    # Choose random split point (not at the very end)
    split_idx = random.randint(1, len(tokens) - 2)

    before = tokens[:split_idx]
    after = tokens[split_idx:]

    # Create corrupted version with marker and reversed tokens
    corrupted = ' '.join(before) + marker + ' ' + ' '.join(reversed(after))
    original = text

    return corrupted, original


def load_sentences_from_file(input_file):
    sentences = []

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) >= 3:  # Must have at least 3 tokens
                sentences.append(line)

    if not sentences:
        raise ValueError(f"No valid sentences found in {input_file}")

    print(f"Loaded {len(sentences)} sentences from {input_file}")
    return sentences


def generate_training_data(input_file, num_samples=10000, marker='🅁', augment=True):
    training_data = []

    # Load sentences from file
    sentences = load_sentences_from_file(input_file)

    # Generate examples
    if augment:
        # Allow reusing sentences to reach num_samples
        while len(training_data) < num_samples:
            sentence = random.choice(sentences)
            example = create_reversal_example(sentence, marker)
            if example:
                training_data.append(example)
    else:
        for sentence in sentences:
            example = create_reversal_example(sentence, marker)
            if example:
                training_data.append(example)
            if len(training_data) >= num_samples:
                break

    return training_data


def format_for_training(corrupted, correct):
    return f"Fix this text: {corrupted}\nCorrected: {correct}<|endoftext|>"


def save_dataset(data, output_file='training_data.json'):
    """Save dataset to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_file}")


def prepare_dataset(training_data, tokenizer, train_split=0.9):
    # Format all examples
    formatted_data = [format_for_training(c, o) for c, o in training_data]

    # Split into train and eval
    split_idx = int(len(formatted_data) * train_split)
    train_texts = formatted_data[:split_idx]
    eval_texts = formatted_data[split_idx:]

    # Create datasets
    train_dataset = Dataset.from_dict({'text': train_texts})
    eval_dataset = Dataset.from_dict({'text': eval_texts})

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return train_dataset, eval_dataset


def train_model(
        train_dataset,
        eval_dataset,
        config,
        model_name,
        output_dir='./gpt2-reversal',
):
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Move model to device (MPS/CUDA/CPU)
    model = model.to(DEVICE)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked LM
    )

    training_config = config.get('training_arguments', {})
    training_args = TrainingArguments(**training_config)
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=num_epochs,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     warmup_steps=100,
    #     weight_decay=0.01,
    #     logging_dir=f'{output_dir}/logs',
    #     logging_steps=100,
    #     save_steps=save_steps,
    #     save_total_limit=-1,
    #     eval_strategy="steps",
    #     eval_steps=500,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     fp16=use_fp16,  # Only use fp16 on CUDA
    #     use_cpu=False,
    #     no_cuda=not torch.cuda.is_available(),  # Disable CUDA if not available
    # )
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer


# ============================================================================
# 3. INFERENCE
# ============================================================================

def test_model(model_path, test_examples):
    """
    Test the trained model on examples.

    Args:
        model_path: Path to saved model
        test_examples: List of corrupted texts to fix
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Move model to the appropriate device
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode

    print("\n" + "=" * 80)
    print("TESTING MODEL")
    print(f"Using device: {DEVICE}")
    print("=" * 80 + "\n")

    for test_input in test_examples:
        prompt = f"Fix this text: {test_input}\nCorrected:"

        # Tokenize input
        input_tokens = tokenizer.encode(test_input)
        prompt_encoding = tokenizer(prompt, return_tensors="pt")

        # Move input to device
        input_ids = prompt_encoding['input_ids'].to(DEVICE)
        attention_mask = prompt_encoding['attention_mask'].to(DEVICE)

        # Calculate max_new_tokens strictly
        max_new_tokens = len(input_tokens) + 5
        min_new_tokens = max(1, len(input_tokens) - 5)

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode output
        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the corrected part
        if "Corrected:" in generated:
            corrected = generated.split("Corrected:")[1].strip()
        else:
            corrected = generated

        # Count tokens for comparison
        input_token_count = len(input_tokens)
        output_token_count = len(tokenizer.encode(corrected))

        print(f"Input:  {test_input}")
        print(f"        (Tokens: {input_token_count})")
        print(f"Output: {corrected}")
        print(f"        (Tokens: {output_token_count})")
        print("-" * 80)


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main(config, input_file='input_sentences.txt', model_name='mission-impossible-lms/partial-reverse-gpt2'):
    """
    Main training pipeline.

    Args:
        input_file: Path to input text file (one sentence per line)
    """

    # Configuration
    NUM_SAMPLES = 10000
    MARKER = '🅁'
    OUTPUT_DIR = './gpt2-reversal'
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5

    # Step 1: Generate training data from input file
    print(f"Reading sentences from {input_file}...")
    training_data = generate_training_data(
        input_file=input_file,
        num_samples=NUM_SAMPLES,
        marker=MARKER,
        augment=True  # Set to False to use each sentence only once
    )

    # Optionally save the dataset
    save_dataset(training_data, 'training_data.json')

    # Step 2: Prepare tokenizer and datasets
    print("\nPreparing datasets...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_dataset(training_data, tokenizer)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Step 3: Train model
    train_model(
        train_dataset,
        eval_dataset,
        model_name=model_name,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    # # Step 4: Test model
    # test_examples = [
    #     "But it was her, and🅁 *. now retired have to sure's she after thinking was I",
    #     "First round with🅁. League Second the from teams",
    #     "The weather is🅁 today nice very",
    # ]
    #
    # test_model(OUTPUT_DIR, test_examples)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,)
    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()
    config = load_configs(args.config)

    config = load_configs(args.config)
    main(config, args.path, args.model)
