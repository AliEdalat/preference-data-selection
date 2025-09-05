

# Enhancing Large Language Model Alignment through Preference Data Selection

This repository accompanies the project *Enhancing Large Language Model Alignment through Preference Data Selection* (replication + novel fusion methods).&#x20;

---

## Overview

Large Language Models (LLMs) achieve strong generative performance but can be difficult to align with human preferences. This project (1) reproduces the Dual-Margin (DM) preference-selection results from Deng et al. and (2) introduces two new reward-fusion strategies — **Attention-Based Fusion** and **Orthogonal Fusion** — to improve preference data selection for Direct Preference Optimization (DPO). Key datasets used: Anthropic Helpful & Harmless (HH) and UltraFeedback (UF). Core reproduction and experimental details are available in the paper.&#x20;

---

## Highlights / Key results

* Successfully reproduced the general trends of Deng et al. (Dual-Margin advantage vs. random sampling).&#x20;
* Implemented **Attention-Based Fusion** (adaptive confidence weighting) and **Orthogonal Fusion** (penalizes correlated reward signals).
* **Orthogonal Fusion** showed consistent gains on UF and competitive performance on HH (see paper Table 1).&#x20;



## Implemented methods

* **External Margin** — uses an external reward model margin.&#x20;
* **Implicit Margin** — internal LM-based margin (e.g., conditional PPL).&#x20;
* **IFD Margin** — Inverse-Form Distance style margin (as used in the paper).&#x20;
* **Dual-Margin (DM)** — ADD and MUL fusion variants (replication of Deng et al.).&#x20;
* **Attention-Based Fusion** — adaptively weights reward signals by confidence (softmax over absolute margins).&#x20;
* **Orthogonal Fusion** — penalizes redundancy when Pearson correlation between signals is high; shown to improve UF performance in our experiments.&#x20;


## Results


## License

Suggested: MIT License. Add `LICENSE` file with MIT text.


## Contributing

Contributions welcome — please open issues for bug reports and feature requests. If you add new judge models or reproduce on larger backbones, please attach reproducible config files and a small README describing compute used.

