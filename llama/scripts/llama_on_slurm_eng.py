import sys
from llama_on_slurm import main
from fire import Fire

if __name__ == "__main__":
    
    ingredients = [
        "onions",
        "chicken",
        "turkey",
        "meat",
        "porridge",
        "sugar",
        "7 different ingredients",
        "with codfish",
    ]

    global prompts = [f"This is a recipe with {i}:" for i in ingredients]
    
    Fire(main)

