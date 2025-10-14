from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def bleu_score(prediction: list, actual: list):
    if len(prediction) != len(actual):
        raise ValueError("Prediction and actual lists must have the same length")

    if len(prediction) == 0:
        return 0.0

    smoothing = SmoothingFunction().method1
    scores = []

    for pred, act in zip(prediction, actual):
        # Tokenize the strings into words
        pred_tokens = pred.split()
        act_tokens = act.split()

        # Calculate BLEU score (reference must be in a list of lists)
        score = sentence_bleu([act_tokens], pred_tokens, smoothing_function=smoothing)
        scores.append(score)

    # Return average BLEU score
    return sum(scores) / len(scores)

if __name__ == '__main__':
    predictions = ["the cat sat on the mat", "hello world"]
    actuals = ["the cat is on the mat", "hello there world"]

    score = bleu_score(predictions, actuals)
    print(f"Average BLEU score: {score:.4f}")