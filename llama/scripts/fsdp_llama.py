import os
import sys
import torch
from tqdm.auto import tqdm


def main():
    n_splits = 8
    multiplier = 8
    vertical = ["wq", "wk", "wv", "w1", "w3", "output"]
    for split in range(n_splits):
        new_params = [{} for _ in range(multiplier)]
        params = torch.load(
            f"/home/users/giovannipuccetti/Models/65B/consolidated.{split:02}.pth",
            map_location="cpu",
        )
        for i in tqdm(params):
            print(i)
            for split_id in range(multiplier):
                if any([j in i for j in vertical]):
                    size = params[i].shape[0]
                    assert size % multiplier == 0
                    new_size = size // multiplier
                    new_params[split_id][i] = params[i][
                        split_id * new_size : (split_id + 1) * new_size
                    ].clone()
                    assert (
                        new_params[split_id][i].numel()
                        == params[i].numel() // multiplier
                    )
                    print("reduced")
                elif "norm" in i or "rope" in i:
                    new_params[split_id][i] = params[i].clone()
                else:
                    size = params[i].shape[1]
                    assert size % multiplier == 0
                    new_size = size // multiplier
                    new_params[split_id][i] = params[i][
                        :, split_id * new_size : (split_id + 1) * new_size
                    ].clone()
                    assert (
                        new_params[split_id][i].numel()
                        == params[i].numel() // multiplier
                    )
                    print("reduced")

        for split_id in range(multiplier):
            n = multiplier * split + split_id
            torch.save(
                new_params[split_id],
                f"/home/users/giovannipuccetti/Models/65B_spread_64/consolidated.{n:02}.pth",
            )


if __name__ == "__main__":
    main()
