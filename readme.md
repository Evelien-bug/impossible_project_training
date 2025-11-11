# impossible_translation

## Overview

**impossible_translation** is a Python project for fine-tuning custom GPT-2-based language models to perform "impossible" text correction tasks. The core aim is to train models that can reconstruct or "fix" sentences corrupted in specific ways (via word hopping, partial reversal, or shuffling). The repository includes scripts for generating augmented training data with different perturbations and for running full fine-tuning using Hugging Face’s Trainer.

## Features

- **Custom Data Augmentation:** Easily generate paired training data by corrupting sentences using various algorithms.
- **Plug-and-Play Perturbations:** Modular perturbation techniques (`wordHop`, `partialReverse`, `localShuffle`), extensible via `utils/`.
- **Masked Language Modeling:** Custom loss masking so the model only learns to predict the corrected part.
- **Device-Aware Training:** Auto-selects CPU, CUDA, or MPS.
- **Configuration-Driven:** All crucial hyperparameters and paths are set via YAML config files.

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/amirhoseinMhmd/impossible_translation.git
cd impossible_translation
pip install -r requirements.txt  # Make sure to have torch, transformers, datasets, tqdm, pyyaml, etc.
```

## Usage: Fine-Tuning GPT-2 with Corrupted Sentences

### Preparing Your Data

- **Input file:** Plain text, one sentence per line, at least 3 words per line.
- **YAML config:** Specifies Hugging Face Trainer arguments (see example below).

### Fine-Tuning Script

**Run:**
```bash
python fine-tuning/finetune.py \
    --path data/input.txt \
    --config configs/finetune.yaml \
    --type wordHop
```

**Arguments:**
- `--path` (`-p`): Path to input text file of sentences
- `--config` (`-c`): Path to YAML config file (for Trainer)
- `--type` (`-t`): Perturbation type — one of `wordHop`, `partialReverse`, `localShuffle`

#### Example

```bash
python fine-tuning/finetune.py -p data/my_sentences.txt -c configs/finetune.yaml -t localShuffle
```

### What Does `finetune.py` Do?

- Loads the config and input file.
- Generates a dataset of pairs: `[corrupted, correct]` sentences using your chosen perturbation (tooling in `utils/`).
- Processes the data for language modeling, with masking so the model is only penalized on the correction.
- Splits into train/eval datasets, tokenizes, and loads a pre-defined model checkpoint.
- Trains the GPT-2 model and saves the output in the location set in your YAML config.

**Supported perturbation types:**
- `wordHop`
- `partialReverse`
- `localShuffle`

*You can add more perturbation techniques in the `utils/` folder.*

---

## Example YAML Configuration

```yaml
training_arguments:
  output_dir: "./models/gpt2-localShuffle"
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  num_train_epochs: 3
  save_strategy: "epoch"
  logging_dir: "./logs"
  evaluation_strategy: "epoch"
```

---

## Project Structure

- `fine-tuning/finetune.py` – Main script for training and data generation
- `utils/reverse.py`, `utils/HOP.py`, `utils/shuffle.py` – Perturbation logic
- `data/` – Place your input files here (optional)
- `configs/` – Place your YAML training configs here

---

## Contributing

Contributions and new perturbations are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit & push
4. Open a pull request

---

## License

(Currently no license specified; please add one if intending to permit use/reuse.)

---

## Contact

Created by [amirhoseinMhmd](https://github.com/amirhoseinMhmd)
