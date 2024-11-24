
# Graph-to-Text Generation with GNN and GPT-2

## Overview

Welcome to the **Graph-to-Text Generation** project! This project integrates Graph Neural Networks (GNNs) with GPT-2 to generate coherent and contextually relevant text based on structured graph inputs. It uses GNNs for graph data embeddings and GPT-2 for text generation, achieving high-quality outputs.

## Features

- **Graph Neural Networks (GNNs)**: Processes graph structures to generate node embeddings.
- **GPT-2 Integration**: Generates high-quality natural language text from graph embeddings.
- **Evaluation Metrics**: Includes BLEU and ROUGE for output quality assessment.
- **Flexible Usage**: Training and inference scripts provided for seamless implementation.

## Requirements

- **Python**: 3.11
- **Libraries**: Refer to the `requirements.txt` file for a complete list of dependencies.

## Installation

### Virtual Environment Setup

1. Create a virtual environment:
    ```bash
    python3.11 -m venv llm_project_env
    source llm_project_env/bin/activate
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

Place your training and testing data in the `data/` directory:
- `data/train_seen.source` and `data/train_seen.target`
- `data/test_seen.source` and `data/test_seen.target`

## Usage

### Training

Train the model with the provided `train.py` script:
```bash
python train.py
```
- Saves model checkpoints (e.g., `best_model_epoch_10.pt`).

### Inference

Run inference with the `inference.py` script:
```bash
python inference.py
```
- Requires `node_to_idx.json` for consistent node mappings.
- Outputs predictions in `test_predictions.txt`.

## Evaluation Metrics

- **BLEU**: Assesses textual similarity with reference text.
- **ROUGE**: Measures the recall of n-grams and longest common subsequences.

## Contributing

Feel free to fork the repository, make changes, and submit a pull request!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [SacreBLEU](https://github.com/mjpost/sacreBLEU)
- [ROUGE](https://github.com/pltrdy/rouge)
