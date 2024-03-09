"""Here lies the exploratory data analysis for the dataset"""

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from wordcloud import WordCloud, STOPWORDS
from src.utils import load_model
from src.inference import evaluate_code


def plot_wordcloud(text, training=True):
    """Plot a wordcloud of the dataset"""
    if training:
        desc = "Train"
    else:
        desc = "Test"

    wordcloud = WordCloud(
        stopwords=STOPWORDS, background_color="white", random_state=1337
    ).generate(text)

    plt.title(f"Wordcloud of {desc} Dataset")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"figures/wordcloud_{desc.lower()}.png", dpi=300)
    plt.clf()


def plot_top_tokens(token_counts, training=True, n=20):
    """Plot the top tokens in the dataset"""
    if training:
        desc = "Train"
    else:
        desc = "Test"

    plt.figure(figsize=(5, 5))
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:n]
    plt.bar(range(n), [x[1] for x in top_tokens])
    plt.xlabel("Token")
    plt.xticks(range(n), [f"'{x[0]}'" for x in top_tokens], rotation=45)
    plt.ylabel("Count")
    plt.title(f"Top {n} Tokens in {desc} Dataset")
    plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin
    plt.savefig(f"figures/top_tokens_{desc.lower()}.png", dpi=300)
    plt.clf()


def plot_token_prompt_lengths(token_lengths, prompt_lengths, training=True):
    """Plot the token and prompt lengths for the dataset"""
    if training:
        desc = "Train"
    else:
        desc = "Test"

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(token_lengths, bins=50)
    axs[0].set_title(f"{desc} Token Lengths")
    axs[0].set_xlabel("Token Length")
    axs[0].set_ylabel("Count")

    axs[1].hist(prompt_lengths, bins=50)
    axs[1].set_title(f"{desc} Prompt Lengths")
    axs[1].set_xlabel("Prompt Length")
    axs[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"figures/token_prompt_lengths_{desc.lower()}.png", dpi=300)
    plt.clf()


def plot(dataset, token_counts, token_lengths, prompt_lengths, training=True):
    """Plot the exploratory data analysis for the dataset"""
    os.makedirs("figures", exist_ok=True)

    if training:
        text = " ".join((example["output"]) for example in dataset)
    else:
        text = " ".join(
            (example["prompt"] + example["canonical_solution"]) for example in dataset
        )

    plot_wordcloud(text, training=training)

    plot_top_tokens(token_counts, training=training)

    plot_token_prompt_lengths(token_lengths, prompt_lengths, training=training)


def get_token_counts(tokenizer, dataset, training=True):
    """Get the token counts for the dataset"""
    token_counts = {}
    token_lengths = []
    prompt_lengths = []
    for example in tqdm(dataset):
        if training:
            code_solution = example["output"]
        else:
            code_solution = example["prompt"] + example["canonical_solution"]
        prompt_lengths.append(len(code_solution))
        ids = tokenizer(code_solution, return_tensors="pt").input_ids
        token_lengths.append(len(ids[0]))
        for token_id in ids[0]:
            token = tokenizer.decode(token_id, skip_special_tokens=True)
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

    return token_counts, token_lengths, prompt_lengths


def eda(model_path, tokenizer_path, dataset_path, training=True):
    """Exploratory data analysis for the dataset"""
    tokenizer, _ = load_model(model_path, tokenizer_path, lora=False)

    if training:
        dataset = load_dataset(dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="test")

    token_counts, token_lengths, prompt_lengths = get_token_counts(
        tokenizer, dataset, training=training
    )

    plot(dataset, token_counts, token_lengths, prompt_lengths, training=training)

    # We cannot evaluate the code on the training dataset
    if training:
        return

    evaluate_code(dataset)
