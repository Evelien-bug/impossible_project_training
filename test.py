from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AddedToken
import torch
import json
import matplotlib.pyplot as plt
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os


def calculate_perplexity(model, tokenizer, dataset):
    model.eval()
    perplexities = []

    with torch.no_grad():
        for item in dataset:
            text = item['original_text'].strip()
            if not text:
                continue

            encoding = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            encoding = {k: v.to(model.device) for k, v in encoding.items()}

            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                labels=encoding['input_ids']
            )
            loss = outputs.loss.item()
            ppl = torch.exp(torch.tensor(loss)).item()
            perplexities.append(ppl)

    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('inf')
    return avg_perplexity


def plot_perplexity(perplexity, save_path='perplexity_evaluation.png'):
    plt.figure(figsize=(8, 6))
    plt.bar(['Model Perplexity'], [perplexity], color='blue', alpha=0.6)
    plt.title('Model Perplexity Evaluation')
    plt.ylabel('Perplexity Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.text(0, perplexity, f'{perplexity:.2f}',
             horizontalalignment='center',
             verticalalignment='bottom')

    plt.savefig(save_path)
    plt.close()


def calculate_bleu(origin_text: str, translated_text: str):
    reference_token = origin_text.split()
    candidate_token = translated_text.split()

    reference = [reference_token]

    bleu_score = sentence_bleu(reference, candidate_token)

    return bleu_score


def get_gpt2_tokenizer_with_markers():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    MARKER_HOP_SING = "🅂"
    MARKER_HOP_PLUR = "🄿"
    new_tokens = [
        AddedToken(MARKER_HOP_SING, lstrip=True, rstrip=False),
        AddedToken(MARKER_HOP_PLUR, lstrip=True, rstrip=False)
    ]
    tokenizer.add_tokens(new_tokens)
    return tokenizer


def translate(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True).to('cuda')

    output_ids = model.generate(
        input_ids,
        max_length=256,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Path to Test dataset.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Path to Model")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation (PPL/BLEU)")
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data['train']

    output = open(f'result_{args.type}.txt', 'w', encoding='utf-8')

    subdirs = [f.path for f in os.scandir(args.model) if f.is_dir()]
    subdirs_sorted = sorted(subdirs, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
    for i, model_path in enumerate(subdirs_sorted):

        model = GPT2LMHeadModel.from_pretrained(model_path).to('cuda')
        tokenizer = get_gpt2_tokenizer_with_markers()
        model.config.pad_token_id = tokenizer.eos_token_id
        if args.type == 'PPL':
            avg_perplexity = calculate_perplexity(model, tokenizer, data)
            print(f"epoch{i} ,  Average Model Perplexity: {avg_perplexity:.2f}")
            output.write(f"epoch{i} ,  Average Model Perplexity: {avg_perplexity:.2f}")

        if args.type == 'BLEU':
            bleu_sum = 0
            errors = 0
            for d in tqdm(data):
                try:
                    translated_text = translate(d['perturbed_text'])
                    bleu_sum += calculate_bleu(d['original_text'], translated_text)
                except Exception:
                    errors += 1

            avg_bleu = bleu_sum / (len(data) - errors)
            print(f"epoch{i} ,  Average Model BLEU: {avg_bleu:.2f}")
            output.write(f"epoch{i} ,  Average Model BLEU: {avg_bleu:.2f}")
