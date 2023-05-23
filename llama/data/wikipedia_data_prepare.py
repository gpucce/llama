import json
from pathlib import Path
from datasets import load_dataset
from llama import Tokenizer
from tqdm.auto import tqdm


def main():
    
    seq_len = 256

    output_path = Path("/home/users/giovannipuccetti/Data/wikipedia/")

    output_path.mkdir(exist_ok=True, parents=True)

    tok = Tokenizer("/home/users/giovannipuccetti/Models/13B/tokenizer.model")
    ds = load_dataset("wikipedia", "20220301.en")

    with open(output_path / "wiki_test.json", "w") as jf:
        for i in tqdm(ds["train"]["text"]):
            length_processed = 0
            wiki_page = tok.encode(i, bos=True, eos=False)
            while length_processed <= len(wiki_page):
                _old_length = length_processed
                length_processed += seq_len
                newline = wiki_page[_old_length:length_processed]
                if len(newline) < 256:
                    continue
                jf.write(json.dumps(newline))
                jf.write("\n")
                

if __name__ == "__main__":
    main()
