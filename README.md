# Info

This repo originates from a fork of the official [Llama](https://github.com/facebookresearch/llama) repository.

The data for the human evaluation is stored in `/llama/human_eval/`

And running `human_eval_study.ipynb` should replicate all the results from the human evaluation.

The `slurm_scripts` folder allows to replicate the other experiments, it might need adjustments to different systems.

The `slurm_scripts/train` folder contain the scripts useful for fine-tuning llama models, which one should download after formally requesting them from the authors.

Generally, calling `bash slurm_scripts/run_experiment.sh` should run the respective experiment.