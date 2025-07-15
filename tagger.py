import os
import argparse
import json
from typing import List, Dict, Any, Optional

import tqdm
import stanza


def get_constituency_parse(sent, nlp: stanza.Pipeline) -> Optional[str]:

    try:
        parse_doc = nlp(sent.text)
        parse_trees = [str(sentence.constituency) for sentence in parse_doc.sentences]
        constituency_parse = "(ROOT " + " ".join(parse_trees) + ")"
        return constituency_parse
    except Exception as e:
        print(f"Warning: Failed to parse sentence: {e}")
        return None


def process_word_annotations(tokens: List, words: List) -> List[Dict[str, Any]]:
    word_annotations = []
    for token, word in zip(tokens, words):
        annotation = {
            'id': word.id,
            'text': word.text,
            'lemma': word.lemma,
            'upos': word.upos,
            'xpos': word.xpos,
            'feats': word.feats,
            'start_char': token.start_char,
            'end_char': token.end_char
        }
        word_annotations.append(annotation)
    return word_annotations


def process_sentence_annotations(sentences: List, include_parse: bool = False,
                                 nlp_parser: Optional[stanza.Pipeline] = None) -> List[Dict[str, Any]]:

    sent_annotations = []
    for sent in sentences:
        word_annotations = process_word_annotations(sent.tokens, sent.words)

        annotation = {
            'sent_text': sent.text,
            'word_annotations': word_annotations,
        }

        if include_parse and nlp_parser:
            constituency_parse = get_constituency_parse(sent, nlp_parser)
            annotation['constituency_parse'] = constituency_parse

        sent_annotations.append(annotation)

    return sent_annotations


def process_text_batches(text_batches: List[str], nlp_main: stanza.Pipeline,
                         include_parse: bool = False, nlp_parser: Optional[stanza.Pipeline] = None) -> List[
    Dict[str, Any]]:


    line_annotations = []

    print("Segmenting and parsing text batches...")
    for text in tqdm.tqdm(text_batches):
        doc = nlp_main(text)

        sent_annotations = process_sentence_annotations(
            doc.sentences, include_parse, nlp_parser
        )

        line_annotation = {
            'sent_annotations': sent_annotations
        }
        line_annotations.append(line_annotation)

    return line_annotations


def create_text_batches(lines: List[str], batch_size: int = 5000) -> List[str]:

    print("Concatenating lines...")
    cleaned_lines = [line.strip() for line in lines]
    line_batches = [cleaned_lines[i:i + batch_size]
                    for i in range(0, len(cleaned_lines), batch_size)]
    text_batches = [" ".join(batch) for batch in line_batches]
    return text_batches


def initialize_pipelines(include_parse: bool = False, use_gpu: bool = True) -> tuple:

    # Main pipeline for tokenization, POS tagging, and lemmatization
    nlp_main = stanza.Pipeline(
        lang='en',
        processors='tokenize,pos,lemma',
        package="default_accurate",
        use_gpu=use_gpu
    )

    # Optional constituency parsing pipeline
    nlp_parser = None
    if include_parse:
        nlp_parser = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,constituency',
            package="default_accurate",
            use_gpu=use_gpu
        )

    return nlp_main, nlp_parser


def save_annotations(annotations: List[Dict[str, Any]], output_filename: str) -> None:

    print("Writing JSON output file...")
    with open(output_filename, "w", encoding='utf-8') as outfile:
        json.dump(annotations, outfile, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        prog='Dataset Tagger',
        description='Tag Texual dataset using Stanza NLP library'
    )
    parser.add_argument('path', type=argparse.FileType('r', encoding='utf-8'),
                        nargs='+', help="Path to input file(s)")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")
    parser.add_argument('--batch-size', type=int, default=5000,
                        help="Number of lines per batch (default: 5000)")
    parser.add_argument('--no-gpu', action='store_true',
                        help="Disable GPU acceleration")

    args = parser.parse_args()

    # Initialize Stanza pipelines
    use_gpu = not args.no_gpu
    nlp_main, nlp_parser = initialize_pipelines(args.parse, use_gpu)

    # Process each input file
    for file in args.path:
        print(f"Processing file: {file.name}")

        # Read and prepare text
        lines = file.readlines()
        text_batches = create_text_batches(lines, args.batch_size)

        # Process text through pipelines
        line_annotations = process_text_batches(
            text_batches, nlp_main, args.parse, nlp_parser
        )

        # Generate output filename
        base_name = os.path.splitext(file.name)[0]
        extension = '_parsed.json' if args.parse else '.json'
        output_filename = base_name + extension

        # Save results
        save_annotations(line_annotations, output_filename)
        print(f"Saved annotations to: {output_filename}")


if __name__ == "__main__":
    main()

# python tagger.py -p --batch-size 2000 --no-gpu input_file.txt