import json
import os.path
from glob import glob

from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

from utils.utils import perturb_hop_words4, perturb_hop_tokens4, perturb_hop_control, perturb_reverse, \
    perturb_shuffle_deterministic, perturb_shuffle_local, perturb_shuffle_even_odd


def purturbe(text, type):
    if type == 'hop_control':
        return perturb_hop_control(text)
    elif type == 'hop_token4':
        return perturb_hop_tokens4(text)
    elif type == 'hop_word4':
        return perturb_hop_words4(text)
    elif type == 'reverse_full':
        return perturb_reverse(text, rng=default_rng(21), reverse=False, full=True)
    elif type == 'reverse_control':
        return perturb_reverse(text, rng=default_rng(21), reverse=False, full=False)
    elif type == 'reverse_partial':
        return perturb_reverse(text, rng=default_rng(21), reverse=True, full=False)
    elif type == 'shuffle_control':
        return perturb_shuffle_deterministic(text, seed=None, shuffle=False)
    elif type == 'shuffle_deterministic21':
        return perturb_shuffle_deterministic(text, seed=21, shuffle=True)
    elif type == 'shuffle_deterministic57':
        return perturb_shuffle_deterministic(text, seed=57, shuffle=True)
    elif type == 'shuffle_deterministic84':
        return perturb_shuffle_deterministic(text, seed=84, shuffle=True)
    elif type == 'shuffle_local3':
        return perturb_shuffle_local(text, seed=33, window=3)
    elif type == 'shuffle_local5':
        return perturb_shuffle_local(text, seed=33, window=5)
    elif type == 'shuffle_local10':
        return perturb_shuffle_local(text, seed=33, window=10)
    elif type == 'shuffle_even_odd':
        return perturb_shuffle_even_odd(text)

    else:
        raise ValueError("Invalid hop type. Choose from 'control', 'token4', or 'word4'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation")
    args = parser.parse_args()

    pert_type = args.type
    json_ext = ".json"
    babylm_data = glob(f"{args.path}/*{json_ext}")
    if not os.path.exists(pert_type):
        os.mkdir(pert_type)

    for file in babylm_data:
        print(file)
        f = open(file)
        data = json.load(f)
        f.close()

        results = []
        for line in data:
            for sent in line['sent_annotations']:
                results.append(sent)

        output_data = []
        for sent in tqdm(results):
            perturbed_text, original_text = purturbe(sent, pert_type)
            res = {
                'original_text': original_text,
                'perturbed_text': perturbed_text
            }
            output_data.append(res)

        train_data, validate_data = train_test_split(output_data, test_size=0.2, random_state=42)

        dataset = {
            "train": train_data,
            "validate": validate_data
        }
        with open(f'{pert_type}/{os.path.basename(file).split('.')[0]}_{pert_type}.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
