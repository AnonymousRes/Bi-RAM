<h1 align="center">Bi-RAM: Bidirectional Recalibrated Aggregated Memory-Based Transformer for Efficient Sequential Modeling</h1>

This is the PyTorch implementation of the Bi-RAM paper.

<p align="center">
  <img src="Bi-RAM.png" alt="Bi-RAM" width="888">
</p>

> [**Bi-RAM: Bidirectional Recalibrated Aggregated Memory-Based Transformer for Efficient Sequential Modeling**]  
> (Under Review)


## Acknowledgment

This codebase is based on and modified from the highly successful **[MEGA repository](https://github.com/facebookresearch/mega/blob/main/examples/mega/README.lra.md)**. We sincerely thank the authors of MEGA for their excellent open-source framework.


## Repository Structure

To ensure reproducibility, we provide the following core assets:

* `LRA_checkpoint/`: Contains the pre-trained model weights for LRA.
* `out_log/`: Contains the complete training logs. These logs record the hyperparameter settings and configurations used for LRA.
* `run_lra/`: Contains executable shell scripts (`.sh`) for training and evaluation on the LRA benchmark.
* `Biram_TimeSeries/out_log/`: Contains the complete training logs. These logs record the hyperparameter settings and configurations used for UEA.
* `Biram_TimeSeries/results/`: Contains the complete results.
* `Biram_TimeSeries/scripts/biram.sh`: `biram.sh` for training and evaluation on the UEA multivariate time-series classification benchmark.


## LRA Data Preparation

Before running the LRA scripts, please download the processed LRA datasets.

Download the [processed data here](https://dl.fbaipublicfiles.com/mega/data/lra.zip), which is provided by the MEGA repository.

*Note: The original raw data is from the [Google LRA repository](https://github.com/google-research/long-range-arena).*

Extract the downloaded `lra.zip` to a directory on your machine.


## UEA Data Preparation

Before running the UEA scripts, please prepare the UEA multivariate time-series classification datasets according to the data format used by the corresponding experiment scripts.

Please place the processed UEA datasets in your local data directory and update the data path in the scripts under `Biram_TimeSeries/scripts/` before execution.


## How to Use

You can train or evaluate Bi-RAM by running the scripts provided in the corresponding script directories.

For LRA tasks:

```bash
cd run_lra
bash listops.sh
