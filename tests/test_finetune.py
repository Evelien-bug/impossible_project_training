import pytest
from unittest.mock import patch, MagicMock
import yaml
import json

from fine_tuning.finetune import (
    load_configs,
    prepare_dataset,
    train_model,
    main,
    generate_training_data,
)
from utils.utils import load_sentences_from_file


# Test data
TEST_SENTENCES = [
    "This is a test sentence.",
    "Here is another sentence for testing.",
    "The quick brown fox jumps over the lazy dog.",
]
TEST_CONFIG = {
    "training_arguments": {
        "output_dir": "./test-output",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
    }
}

@pytest.fixture
def temp_files(tmp_path):
    # Create a temporary input file
    input_file = tmp_path / "test.txt"
    with open(input_file, "w") as f:
        for sentence in TEST_SENTENCES:
            f.write(sentence + "\n")

    # Create a temporary config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(TEST_CONFIG, f)

    return input_file, config_file

def test_load_configs(temp_files):
    _, config_file = temp_files
    config = load_configs(config_file)
    assert config == TEST_CONFIG

def test_load_sentences_from_file(temp_files):
    input_file, _ = temp_files
    sentences = load_sentences_from_file(input_file)
    assert sentences == TEST_SENTENCES

def test_load_sentences_from_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_sentences_from_file("non_existent_file.txt")

@patch('fine_tuning.finetune.functions', {
    "partialReverse": lambda x: [("corrupted " + s, s) for s in x]
})
def test_generate_training_data(temp_files):
    input_file, _ = temp_files
    training_data = generate_training_data(str(input_file), "partialReverse")

    assert len(training_data) == len(TEST_SENTENCES)
    for item in training_data:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert item[0].startswith("corrupted")
        assert item[1] in TEST_SENTENCES


@patch('fine_tuning.finetune.GPT2Tokenizer.from_pretrained')
@patch('fine_tuning.finetune.tqdm', lambda x: x)
def test_prepare_dataset(mock_tokenizer_from_pretrained):
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<|endoftext|>"
    corrected_marker_tokens = [5025, 331]
    mock_tokenizer.encode.return_value = corrected_marker_tokens

    prompt_tokens = [10, 20, 30, 40]
    answer_tokens = [60, 70, 80]
    end_token = [50256]

    prompt_only_ids = prompt_tokens + corrected_marker_tokens
    full_text_ids = prompt_tokens + corrected_marker_tokens + answer_tokens + end_token

    def tokenize_side_effect(text, **kwargs):
        if "correct sentence" in text:
            return {'input_ids': full_text_ids, 'attention_mask': [1] * len(full_text_ids)}
        else:
            return {'input_ids': prompt_only_ids, 'attention_mask': [1] * len(prompt_only_ids)}

    mock_tokenizer.side_effect = tokenize_side_effect
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    training_data = [("corrupted", "correct sentence")]
    train_dataset, eval_dataset = prepare_dataset(
        training_data, mock_tokenizer, train_split=1.0, max_length=20
    )

    assert len(train_dataset) == 1
    assert len(eval_dataset) == 0

    sample = train_dataset[0]

    # The prompt part should be masked
    assert sample['labels'][5].item() == -100
    # The answer part should not be masked
    assert sample['labels'][6].item() == 60
    assert sample['labels'][6].item() != -100


@patch('fine_tuning.finetune.GPT2LMHeadModel.from_pretrained')
@patch('fine_tuning.finetune.GPT2Tokenizer.from_pretrained')
@patch('fine_tuning.finetune.Trainer')
def test_train_model(mock_trainer, mock_tokenizer, mock_model):
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_instance
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance

    train_dataset = MagicMock()
    eval_dataset = MagicMock()

    train_model(train_dataset, eval_dataset, TEST_CONFIG, "gpt2", "./test-output")

    mock_trainer.assert_called_once()
    trainer_instance = mock_trainer.return_value
    trainer_instance.train.assert_called_once()
    trainer_instance.save_model.assert_called_once_with("./test-output")

@patch('fine_tuning.finetune.train_model')
@patch('fine_tuning.finetune.prepare_dataset')
@patch('fine_tuning.finetune.generate_training_data')
@patch('fine_tuning.finetune.GPT2Tokenizer.from_pretrained')
def test_main(mock_tokenizer, mock_generate_training_data, mock_prepare_dataset, mock_train_model, temp_files):
    input_file, config_file = temp_files

    mock_generate_training_data.return_value = [("corrupted", "original")]
    mock_prepare_dataset.return_value = (MagicMock(), MagicMock())

    with patch('builtins.open', MagicMock()) as mock_open:
        with patch('json.load', MagicMock(return_value=[("corrupted", "original")])) as mock_json_load:
            main(TEST_CONFIG, str(input_file), "gpt2", "partialReverse")

    mock_prepare_dataset.assert_called_once()
    mock_train_model.assert_called_once()
