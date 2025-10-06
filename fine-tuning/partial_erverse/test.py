from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import argparse

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

def test_model(model_path, test_examples):

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
        print(f"Input:  {test_input}")
        print(f"Output: {corrected}")
        print("-" * 80)

def load_test_data(dataset_path):
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(line.strip())
    return dataset

def main(model_path, dataset_path):
    test_samples = load_test_data(dataset_path)

    test_model(model_path, test_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,)
    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    args = parser.parse_args()

    main(args.model, args.path)