# Dynamic Event Extraction (EE) and Automated Timeline Generation

## Overview

This repository documents an end-to-end Natural Language Processing (NLP) project focused on transforming massive streams of unstructured news data into structured, time-ordered knowledge graphs. The core implementation involves fine-tuning a Transformer model (RoBERTa) to perform complex sequence labeling tasks necessary for Event Extraction (EE).

The project's key challenge was successfully generalizing a model, trained on high-quality supervised data (MAVEN), to a noisy, high-volume real-world source (GDELT), ultimately structuring events for automated chronological visualization.

## Technical Architecture

The architecture is built around a chained deep learning pipeline:

- Base Model: RoBERTa Transformer (chosen for superior generalization through dynamic masking).

- Task 1: Event Trigger Detection (ETD): Sequence labeling to identify the core verb/phrase that marks an event.

- Task 2: Event Argument Detection (EAD): Multi-classification to link pre-identified entities (Person, Org, Loc) to semantic roles (Agent, Target, Location).

## Key Design Considerations

- Pre-filtering with Ontology: Named Entity Recognition (NER) was performed in the preprocessing step to drastically reduce the search space for the downstream EAD model, enhancing efficiency and schema compliance.

- Temporal Normalization: All temporal expressions were tagged and normalized using Timex3 standards, ensuring the final knowledge tuple is correctly formatted for chronological visualization.

- Resource Management: Aggressive sequence length limiting ($\le 256$ tokens) was implemented to manage memory constraints during fine-tuning on limited GPU resources (NVIDIA T4).

## Data Sources

| Source | Role | Size | Notes |
|--------|------|------|-------|
| MAVEN | Training/Supervised Learning | ~120,000 | Used for fine-tuning the base RoBERTa model. |
| GDELT | External Evaluation/Generalization | ~100,000 | Real-world, noisy news corpus used to test model robustness. |

## Installation and Setup

### Prerequisites

- Python 3.8+

- NVIDIA GPU (Recommended for training)

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/Dynamic-Event-Extraction-EE-and-Automated-Timeline-Generation.git
cd Dynamic-Event-Extraction-EE-and-Automated-Timeline-Generation
```


2. Install Dependencies:
   
```bash
# Install PyTorch, Hugging Face transformers, spaCy, and other dependencies
pip install torch transformers scikit-learn pandas tqdm
# Download the large spaCy language model for NER and segmentation
python -m spacy download en_core_web_lg
```

3. Data Acquisition (MAVEN):

- Download the MAVEN dataset from its official source and place it in a data/raw/ directory.

4. Data Acquisition (GDELT):

- The Init.ipynb notebook contains scripts for programmatically accessing and sampling GDELT data archives. Execution requires local processing and filtering.

## Running the Pipeline

The project workflow is structured across several Jupyter Notebooks:

1. Init.ipynb: Handles initial GDELT data download, cleaning, and preliminary sampling.

2. advanced_preprocessor.ipynb: Executes the cleaning, sentence segmentation, NER, and temporal tagging pipeline. This notebook generates the feature-engineered data used for model training.

3. Final_Data_Preparation,_Splitting,_Training,_and_Evaluation.ipynb: Contains the code for data tokenization, dataset splitting, RoBERTa fine-tuning (using PyTorch/Hugging Face Trainer), and final performance evaluation.

## Results and Performance

The model's performance on the generalization test set (GDELT-derived data) highlighted a key challenge in complex NLP systems.

| Metric | Value | Interpretation |
|:-------|:------|:---------------|
| **Micro-Average F1** | $51.3\%$ | Aggregate overall accuracy on all classes |
| **Macro-Average F1** | $33.7\%$ | Performance on rare event classes |
| **Conclusion** | **Significant gap** | Demonstrates successful functional extraction but severe Class Imbalance, leading to poor recall on low-frequency event types. |

## License

This project is licensed under the MIT License. See the LICENSE.md file for details.
