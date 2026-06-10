<h1 align="center">Bi-RAM: Bidirectional Recalibrated Aggregated Memory-Based Transformer for Efficient Sequential Modeling</h1>

<p align="center">
  <img src="Bi-RAM.png" alt="Bi-RAM" width="888">
</p>

<p align="center">
  <b>PyTorch implementation of Bi-RAM</b>
</p>

> [**Bi-RAM: Bidirectional Recalibrated Aggregated Memory-Based Transformer for Efficient Sequential Modeling**]  
> Status: Under Review


## Overview

This repository provides the official PyTorch implementation of **Bi-RAM**, a bidirectional recalibrated aggregated memory-based Transformer designed for efficient sequential modeling. It includes the core implementation, training and evaluation scripts, pretrained checkpoints, training logs, and experimental results for the Long Range Arena (LRA) benchmark and the UEA multivariate time-series classification benchmark.


## Acknowledgment

This codebase is built upon and adapted from the excellent open-source implementations of **[MEGA](https://github.com/facebookresearch/mega/tree/main/examples/mega)** and **[Flowformer](https://github.com/thuml/Flowformer/tree/main/Flowformer_TimeSeries)**.

We sincerely thank the authors of MEGA and Flowformer for releasing their codebases and experimental frameworks. The LRA and UEA datasets used in this repository can also be prepared by following the data preparation instructions provided in the corresponding MEGA and Flowformer repositories.


## Repository Structure

The repository is organized as follows:

* `LRA_checkpoint/`: Pretrained model checkpoints for LRA experiments.
* `out_log/`: Complete training logs for LRA experiments, including hyperparameter settings and configurations.
* `run_lra/`: Shell scripts for training and evaluating Bi-RAM on the LRA benchmark.
* `fairseq/`: Core implementation of Bi-RAM for LRA experiments.
* `Biram_TimeSeries/`: Implementation and scripts for UEA multivariate time-series classification.
* `Biram_TimeSeries/out_log/`: Complete training logs for UEA experiments, including hyperparameter settings and configurations.
* `Biram_TimeSeries/results/`: Complete experimental results on the UEA benchmark.
* `Biram_TimeSeries/scripts/biram.sh`: Shell script for training and evaluating Bi-RAM on the UEA benchmark.


## LRA Data Preparation

Before running the LRA experiments, please download the processed LRA datasets.

The processed LRA data can be obtained from the **[MEGA repository](https://github.com/facebookresearch/mega/tree/main/examples/mega)**.

The original raw LRA data is provided by the **[Google Long Range Arena repository](https://github.com/google-research/long-range-arena)**.

After downloading the processed `lra.zip` file, extract it to a local directory and update the `DATA` path in the scripts under `run_lra/` accordingly.


## UEA Data Preparation

Before running the UEA experiments, please prepare the UEA multivariate time-series classification datasets.

The UEA data preparation follows the protocol used in the **[Flowformer TimeSeries repository](https://github.com/thuml/Flowformer/tree/main/Flowformer_TimeSeries)**. Please download and organize the datasets according to the instructions provided in the Flowformer project.

After the datasets are prepared, place them in your local data directory and update the corresponding data paths in the scripts under `Biram_TimeSeries/scripts/`.


## Training and Evaluation

Bi-RAM can be trained and evaluated by running the provided shell scripts.


### LRA Experiments

~~~bash
cd run_lra
bash lra_all.sh
~~~


### UEA Experiments

~~~bash
cd Biram_TimeSeries/scripts/
bash biram.sh
~~~


## Alternative Integration with the MEGA Framework

If you would like to run Bi-RAM directly within the original MEGA experimental framework, you can integrate the core Bi-RAM files into the MEGA repository.

Please follow the steps below:

1. Copy `fairseq/models/lra/biram_lra_encoder.py` from this repository to MEGA's `fairseq/models/lra/` directory.

2. Copy `fairseq/modules/biram_sentence_encoder_layer.py` from this repository to MEGA's `fairseq/modules/` directory.

3. Copy `fairseq/models/lra/model.py` from this repository and use it to replace the original `model.py` file in MEGA's `fairseq/models/lra/` directory.

After these files are replaced, Bi-RAM can be trained and evaluated using MEGA's original experimental pipeline.


## Citation

If you find this repository useful for your research, please consider citing our paper:

~~~bibtex
@article{biram,
  title={Bi-RAM: Bidirectional Recalibrated Aggregated Memory-Based Transformer for Efficient Sequential Modeling},
  author={},
  journal={Under Review},
  year={}
}
~~~
