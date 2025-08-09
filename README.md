# Cross-Prompt Automated Essay Scoring Using Multi-Task and Holistic Neural Models

## Project Overview

This project focuses on developing robust cross-prompt Automated Essay Scoring (AES) models that generalize well across multiple essay prompts. The goal is to predict essay quality by scoring either a holistic score or multiple trait-based scores from extracted essay features.

Two main modeling approaches are explored:

- **Approach A:** A holistic scoring model using 86 input features to predict a single overall score. It leverages shallow and deep fully-connected neural networks.
- **Approach B:** A multi-task learning model that predicts all nine scores simultaneously — including one holistic and eight trait-specific scores.

---

## Dataset and Evaluation

- Essays cover multiple prompts, each with different scoring traits (e.g., "holistic," "content," "organization").
- Scores are normalized per prompt based on predefined score ranges.
- Leave-one-out prompt cross-validation ensures the model is evaluated on unseen prompts, improving generalization.
- 7-fold cross-validation on training data optimizes hyperparameters and mitigates overfitting.
- Evaluation metric: **Quadratic Weighted Kappa (QWK)**, which accounts for ordinal score agreement.

---

## Methodology

### Data Preparation

- Feature extraction from essays combined with normalized trait scores.
- Score normalization (`scale_score`) and rescaling (`rescale_score`) accommodate diverse score ranges per prompt.
- Non-applicable traits for specific prompts are masked during training and loss calculation.

### Model Architectures

- Fully-connected neural networks with configurable hidden layers and units.
- He initialization (`nn.init.kaiming_normal_`) ensures stable training with ReLU activations.
- Two approaches:
  - Approach A predicts a single holistic score.
  - Approach B predicts all trait scores simultaneously with a masking mechanism to ignore irrelevant traits.

### Training

- Custom loss function based on **Mean Squared Error (MSE)**, applied selectively per trait.
- Early stopping based on validation QWK prevents overfitting.
- Grid search over hyperparameters: hidden layers, units, learning rate, and batch size.
- Separate models trained per prompt, with data from other prompts used for training.

### Deployment Model

- Trained on combined data from all prompts.
- Single unified model for practical deployment scenarios.

---

## Results and Analysis

| Prompt  | Best QWK (Approach A) | Best QWK (Approach B) |
|---------|-----------------------|-----------------------|
| 1       | 0.695                 | 0.399                 |
| 2       | 0.560                 | 0.358                 |
| 3       | 0.616                 | 0.508                 |
| 4       | 0.621                 | 0.529                 |
| 5       | 0.734                 | 0.415                 |
| 6       | 0.541                 | 0.376                 |
| 7       | 0.725                 | 0.099                 |
| 8       | 0.662                 | 0.369                 |
| Deployment | 0.843               | 0.696                 |

- Approach A generally outperformed Approach B in QWK scores, indicating holistic scoring is easier for the model than multi-trait scoring.
- Best batch size across models: 32, balancing stable gradients and efficient training.
- Cross-prompt validation and scaling were crucial to robust model generalization.
- Performance is competitive with state-of-the-art AES models, with room for improvements in complex prompts.

---

## Usage

1. Clone the repository.
2. Prepare dataset following the described scaling and masking procedure.
3. Run training scripts for either Approach A or B.
4. Use leave-one-out prompt splits to evaluate model performance on unseen prompts.
5. Fine-tune hyperparameters via provided grid search mechanisms.
6. Use the deployment model for unified essay scoring across prompts.

---

## Key Functions

- `scale_score(raw_score, prompt, trait)`: Normalizes raw scores to [0,1].
- `rescale_score(scaled_score, prompt, trait)`: Converts scaled scores back to original range.
- `applicable_traits_mask`: Masks out non-applicable traits per prompt.
- Custom MSE loss function adapted for trait masking.
- `train_and_evaluate`: Handles model training, validation, and early stopping.
- Grid search for hyperparameter tuning.

---

## Future Work

- Incorporate additional linguistic and semantic features.
- Explore transformer-based architectures for richer representations.
- Investigate data augmentation techniques to handle low-resource prompts.
- Improve multi-task learning strategies to better capture trait dependencies.
