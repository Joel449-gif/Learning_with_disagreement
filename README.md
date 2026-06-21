# LeWiDi Irony Detection — Learning With Disagreement

Soft-label irony detection on social media replies, built for the **LeWiDi (Learning With
Disagreement)** shared task paradigm. Instead of collapsing annotator disagreement into a single
hard label, this project predicts a full probability distribution over "Ironic" vs "Not Ironic" —
directly modelling human disagreement as signal rather than noise.

## Problem

Standard classifiers force every example into one label, discarding the fact that human annotators
often genuinely disagree — especially for subjective tasks like irony detection. LeWiDi reframes
this as a **soft-label regression problem**: given a post-reply exchange, predict the distribution
of annotator judgements (e.g. `[Not Ironic: 0.3, Ironic: 0.7]`), evaluated using **Manhattan
Distance (L1 loss)** between predicted and true soft labels.

## Approach

The notebook benchmarks four progressively more sophisticated approaches on the same dataset:

| Stage | Method | Key Idea |
|---|---|---|
| 1 | **Classical ML** | TF-IDF features + Linear Regression / SVR / Random Forest / MLP Regressor |
| 2 | **BERT fine-tuning** | `bert-base-uncased` with a regression head, trained with L1 loss |
| 3 | **Hyperparameter search** | Grid search over learning rate, batch size, epochs, and max sequence length |
| 4 | **LLMs** | Zero-shot prompting with **Mistral-7B-Instruct**, and fine-tuned **DistilBERT** regression |

All models are evaluated with the same metric — **Manhattan Distance** between predicted and
ground-truth soft-label vectors — enabling direct comparison across classical ML, transformer
fine-tuning, and large language model prompting.

## Dataset

Uses the `MP_dev.json` split from the LeWiDi shared task (Multi-Perspective irony detection),
containing post/reply pairs with soft labels derived from multiple human annotators.

> Dataset not included in this repo due to shared-task licensing — see the
> [LeWiDi shared task page](https://le-wi-di.github.io/) for access.

## Results Summary

| Model | Manhattan Distance (lower is better) |
|---|---|
| Random Forest Regressor (TF-IDF) | *fill in your result* |
| BERT (base, default hyperparameters) | *fill in your result* |
| BERT (best tuned hyperparameters) | *fill in your result* |
| Mistral-7B-Instruct (zero-shot) | *fill in your result* |
| DistilBERT (fine-tuned) | *fill in your result* |

*(Pull final numbers from your notebook's printed outputs and fill in before publishing.)*

## Tech Stack

- **Languages:** Python
- **ML/DL:** scikit-learn, PyTorch, HuggingFace Transformers
- **Models:** BERT, DistilBERT, Mistral-7B-Instruct (via `transformers` + `accelerate` + `bitsandbytes`)
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn

## Repository Structure

```
.
├── LeWiDi_irony.ipynb     # Main notebook: all 4 modelling stages
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

```bash
git clone https://github.com/Joel449-gif/LeWiDi-irony-detection.git
cd LeWiDi-irony-detection
pip install -r requirements.txt
```

Set your HuggingFace token as an environment variable before running LLM-related cells:

```bash
export HF_TOKEN="your_token_here"
```

Never hardcode API tokens directly in notebook cells.

## Key Learnings

- Modelling annotator disagreement as a soft-label regression task rather than discarding it via majority voting.
- Comparing classical ML, transformer fine-tuning, and zero-shot LLM prompting on the same evaluation metric.
- Practical hyperparameter tuning for transformer models under compute constraints.
- Prompt engineering for structured numeric output extraction from LLMs (regex-based parsing of percentage predictions).

## Author

**Joel Joseph**
[GitHub](https://github.com/Joel449-gif) · [LinkedIn](https://www.linkedin.com/in/joel-joseph-362b69200/)
