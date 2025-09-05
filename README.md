

# Enhancing Large Language Model Alignment through Preference Data Selection

This repository accompanies the LLM course project *Enhancing Large Language Model Alignment through Preference Data Selection* (replication + novel fusion methods).&#x20;

---

## Overview

Large Language Models (LLMs) achieve strong generative performance but can be difficult to align with human preferences. This project (1) reproduces the Dual-Margin (DM) preference-selection results from Deng et al. and (2) introduces two new reward-fusion strategies — **Attention-Based Fusion** and **Orthogonal Fusion** — to improve preference data selection for Direct Preference Optimization (DPO). Key datasets used: Anthropic Helpful & Harmless (HH) and UltraFeedback (UF). Core reproduction and experimental details are available in the [paper](https://github.com/your-repo/your-path/yourfile.odf).&#x20;

---

## Highlights / Key results

* Successfully reproduced the general trends of Deng et al.&#x20;
* Implemented **Attention-Based Fusion** (adaptive confidence weighting) and **Orthogonal Fusion** (penalizes correlated reward signals).
* **Orthogonal Fusion** showed consistent gains on UF and competitive performance on HH (see paper Table 1).&#x20;



## Implemented methods

* **External Margin** — uses an external reward model margin.&#x20;
* **Implicit Margin** — DPO implicit reward margin.&#x20;
* **IFD Margin** — Conditional PPL.&#x20;
* **Dual-Margin (DM)** — ADD and MUL fusion variants of BeeS (replication of Deng et al.).&#x20;
* **Attention-Based Fusion** — adaptively weights reward signals by confidence (softmax over absolute margins).&#x20;
* **Orthogonal Fusion** — penalizes redundancy when Pearson correlation between signals is high; shown to improve UF performance in our experiments.&#x20;


## Models and Datasets

* **Trained Models** – \[link]
* **Model Generation Process** – \[link]
* **Reward Margins of Different Methods** – \[link]


## License

Suggested: MIT License. Add `LICENSE` file with MIT text.


## Contributing

Contributions welcome — please open issues for bug reports and feature requests. If you add new judge models or reproduce on larger backbones, please attach reproducible config files and a small README describing compute used.

