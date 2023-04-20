from torch.utils.data import Dataset
import numpy as np
import re

PATTERN = re.compile(r"<extra_id_\d+>")


class PandasDataset(Dataset):
    def __init__(self, pd_data):
        self.df = pd_data

    def __getitem__(self, x):
        return self.df.iloc[x, :]

    def __len__(self):
        return self.df.shape[0]


def pandas_collate(x):
    return {i: [j[i] for j in x] for i in x[0].index}


def count_masks(texts):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts
    ]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, model, tokenizer, device):
    n_expected = count_masks(texts)
    stop_id = tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def tokenize_and_mask(text, span_length=2, pct=1, buffer_size=1, ceil_pct=False):
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    count = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
        count += 1
        if count >= 1000:
            break

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [PATTERN.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def process_spaces(story):
    return (
        story.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(" '", "'")
        .replace(" ’ ", "'")
        .replace(" :", ":")
        .replace("<newline>", "\n")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace("''", '"')
        .replace(".. ", "... ")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace(" n't", "n't")
        .replace(" i ", " I ")
        .replace(" i'", " I'")
        .replace("\\'", "'")
        .replace("\n ", "\n")
        .strip()
    )
