import pandas as pd
import random
from pathlib import Path
import regex as re
import sys

from ..data_utils import process_spaces

_SHUFFLING_ORDER = [
    17,
    13,
    7,
    14,
    8,
    12,
    18,
    0,
    2,
    6,
    5,
    4,
    11,
    1,
    19,
    16,
    15,
    10,
    3,
    9,
]


def template(txta, txtb):
    txta = " ".join(txta.split(" ")[:-1]) + "..."
    txtb = " ".join(txtb.split(" ")[:-1]) + "..."
    txta = process_spaces(txta)
    txtb = process_spaces(txtb)
    return f"""<p style="text-align: justify;"><span style="font-size: 14px;">Il testo B segue il testo A. Pensi che B sia stato generato automaticamente?<br><br><strong>A:</strong>&nbsp;{txta}</span></p><hr id="null"><span style="font-size: 16px;"><strong>B:&nbsp;</strong>{txtb}</span>"""


control_txta = """Nel giro di poco, Facebook potrebbe potenzialmente raggiungere 400 milioni di iscritti al miliardo e 800 milioni attuali. Si tratta della popolazione della Cina , mercato che Mark Zuckenberg punta da tempo."""
control_txtb = """ E' quello che ha dichiarato il fondatore di Facebook in un'intervista a Bloomberg. "Siamo pronti a fare tutto quello che è necessario per entrare in Cina" ha detto Zuckerberg. "Siamo pronti a fare tutto quello che è necessario per entrare in Cina". "Siamo pronti a fare tutto quello che è necessario per entrare in Cina". "Siamo pronti a fare tutto quello che è necessario per entrare in Cina". "Siamo pronti a fare..."""
control_q = template(control_txta, control_txtb)
control_id = 6

random.seed(42)
rephrased_df = pd.read_csv(
    "./llama/human_eval/input/change-it.ilgiornale.test_1000_rephrased_epoch_00006.csv",
    sep="\t",
    index_col=0,
    lineterminator="\n",
)

n_questionaries = 6
q_per_questionary = 20


evaldf = pd.read_csv("./llama/human_eval/questions_import_sample_IT.csv")
dfs1 = []
dfs2 = []
for n_split in range(n_questionaries):
    df = rephrased_df.iloc[q_per_questionary * n_split : q_per_questionary * (n_split + 1), :]
    split1 = range(q_per_questionary // 2)
    split2 = range(q_per_questionary // 2, q_per_questionary)
    future_df1 = evaldf.to_dict()
    for key in list(future_df1.keys()):
        future_df1[key] = [future_df1[key][0]]
    future_df2 = evaldf.to_dict()
    for key in list(future_df2.keys()):
        future_df2[key] = [future_df2[key][0]]

    future_df1["is_human"] = ["boh"]
    future_df2["is_human"] = ["boh"]

    for idx in split1:
        for key in list(future_df1):
            if key not in ["Testo domanda", "is_human"]:
                future_df1[key].append(future_df1[key][-1])
                future_df2[key].append(future_df2[key][-1])
                continue
            if key == "Testo domanda":
                future_df1[key].append(
                    template(df.iloc[idx, :].prompts, df.iloc[idx, :].true_continuations)
                )
                future_df2[key].append(
                    template(
                        df.iloc[idx, :].prompts, df.iloc[idx, :].generated_continuations
                    )
                )
            elif key == "is_human":
                future_df1[key].append(True)
                future_df2[key].append(False)

    for idx in split2:
        for key in list(future_df1):
            if key not in ["Testo domanda", "is_human"]:
                future_df1[key].append(future_df1[key][-1])
                future_df2[key].append(future_df2[key][-1])
                continue
            if key == "Testo domanda":
                future_df1[key].append(
                    template(
                        df.iloc[idx, :].prompts, df.iloc[idx, :].generated_continuations
                    )
                )
                future_df2[key].append(
                    template(df.iloc[idx, :].prompts, df.iloc[idx, :].true_continuations)
                )
            elif key == "is_human":
                future_df1[key].append(False)
                future_df2[key].append(True)

    # drop first
    for key in list(future_df1.keys()):
        future_df1[key] = future_df1[key][1:]
    for key in list(future_df2.keys()):
        future_df2[key] = future_df2[key][1:]
    
    future_df1["order"] = range(q_per_questionary)
    future_df2["order"] = range(q_per_questionary)
    
    shuffled_df1 = {}
    for key, val in future_df1.items():
        shuffled_df1[key] = [val[idx] for idx in _SHUFFLING_ORDER]
    shuffled_df2 = {}
    for key, val in future_df2.items():
        shuffled_df2[key] = [val[idx] for idx in _SHUFFLING_ORDER]
    
    if n_split == 0:
        assert shuffled_df1["Testo domanda"][6] == control_q, "missing control sequence"
        
    shuffled_df1["Testo domanda"][6] = control_q
    shuffled_df2["Testo domanda"][6] = control_q
    dfs1.append(shuffled_df1)
    dfs2.append(shuffled_df2)

    pd.DataFrame.from_dict(shuffled_df1).to_csv(f"./llama/human_eval/data/human_eval_df1_{n_split}.csv", encoding="utf-8")
    pd.DataFrame.from_dict(shuffled_df2).to_csv(f"./llama/human_eval/data/human_eval_df2_{n_split}.csv", encoding="utf-8")
