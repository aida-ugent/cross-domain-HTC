# Cross-Domain Resources for Text Classification with Hierarchical Labels

This repository is the companion repository for the paper ``Your Next State-of-the-Art Could Come from Another Domain: A Cross-Domain Analysis of Hierarchical Text Classification.''

[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)


## 📊 Datasets

We provide a collection of seven diverse datasets for hierarchical text classification, spanning legal, scientific, medical, and patent domains. Each dataset comes with gold-standard taxonomies, making them ideal for developing and evaluating hierarchical text classification methods.


| Dataset | Domain | Documents | Labels | Hierarchy Depth | Avg Length |
|---------|--------|-----------|---------|----------------|------------|
| EurLex-3985 | Legal | 19,306 | 3,985 | 2 | 2,635 |
| EurLex-DC-410 | Legal | 19,340 | 410 | 2 | 2,635 |
| WOS-141 | Scientific | 46,985 | 141 | 2 | 200 |
| SciHTC-83 | Scientific | 186,160 | 83 | 6 | 145 |
| SciHTC-800 | Scientific | 186,160 | 800 | 6 | 145 |
| MIMIC3-3681 | Medical | 52,712 | 3,681 | 3* | 1,514 |
| USPTO2M-632 | Patent | 1,998,408 | 632 | 2* | 117 |

\* Expanded hierarchy for certain methods (see documentation)


### 🚀 Getting Started
Please fill the [Consent Form](https://forms.gle/PAAA8z2JYMNjeKRo9) to get access to the datasets.


## 💻 Code

Please see [src/README.md](src/README.md) for more details.

## 📚 Citation
