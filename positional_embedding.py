import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model
import matplotlib.pyplot as plt

# ----------------------------
# Load GPT-2
# ----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()


# ----------------------------
# Helper Functions
# ----------------------------
def get_token_embeddings(text: str) -> torch.Tensor:
    """Return GPT-2 token embeddings (word + position)"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]


# def get_token_embeddings(text: str) -> torch.Tensor:
#     # Tokenize input text
#     inputs = tokenizer(text, return_tensors='pt')
#     input_ids = inputs['input_ids']
#
#     # Get the sequence length
#     seq_length = input_ids.shape[1]
#
#     # Extract positional embeddings
#     with torch.no_grad():
#         position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)
#         positional_embeddings = model.wpe(position_ids)  # wpe = word position embeddings
#
#     return positional_embeddings.squeeze(0)

def get_similarity_matrix(text1: str, text2: str) -> torch.Tensor:
    """Compute cosine similarity matrix between tokens of two texts"""
    emb1 = get_token_embeddings(text1)  # [len1, hidden_dim]
    emb2 = get_token_embeddings(text2)  # [len2, hidden_dim]

    # Normalize embeddings for cosine similarity
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    return torch.matmul(emb1, emb2.T)  # [len1, len2]


def sequence_similarity(text1: str, text2: str) -> float:
    """Compute order-sensitive similarity (diagonal alignment)"""
    sim_matrix = get_similarity_matrix(text1, text2)
    min_len = min(sim_matrix.size(0), sim_matrix.size(1))
    diagonal_sims = [sim_matrix[i, i].item() for i in range(min_len)]
    return sum(diagonal_sims) / min_len  # mean similarity along the diagonal


def order_penalty(sim_matrix: torch.Tensor) -> float:
    """Measure average shift of best token matches (higher = more reordering)"""
    best_matches = sim_matrix.argmax(dim=1)  # index of best match for each token
    penalty = torch.mean(
        torch.abs(torch.arange(len(best_matches)) - best_matches.float())
    )
    return penalty.item()


def plot_similarity_heatmap(text1: str, text2: str, sim_matrix: torch.Tensor):
    """Visualize token similarity matrix as a heatmap"""
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)

    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix.numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine similarity")

    plt.xticks(range(len(tokens2)), tokens2, rotation=45, ha="right")
    plt.yticks(range(len(tokens1)), tokens1)

    plt.xlabel("Text 2 tokens")
    plt.ylabel("Text 1 tokens")
    plt.title("Token Similarity Heatmap")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Example Usage
# ----------------------------
text_a = "future The of AI is now."
text_b = "The future is of now AI."

# Compute similarity matrix
sim_matrix = get_similarity_matrix(text_a, text_b)

# Order-sensitive similarity
order_sensitive_score = sequence_similarity(text_a, text_b)

# Order penalty
penalty = order_penalty(sim_matrix)

# Print results
print(f"Text A: {text_a}")
print(f"Text B: {text_b}")
print(f"Order-sensitive similarity: {order_sensitive_score:.4f}")
print(f"Order penalty (avg token shift): {penalty:.2f}")

# Plot heatmap
plot_similarity_heatmap(text_a, text_b, sim_matrix)


# Text A: The future of AI is now.
# Text B: The future of now is AI.
# Order-sensitive similarity: 0.9946
# Order penalty (avg token shift): 0.43