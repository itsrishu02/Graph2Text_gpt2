
# Graph to Text - Fine-tuning and Decoding with HuggingFace

This project leverages the HuggingFace framework. Please refer to the [HuggingFace documentation](https://huggingface.co) for further details on installation and dependencies.

## Environments and Dependencies
- **Python**: 3.6
- **Transformers**: 3.3.1
- **PyTorch Lightning**: 0.9.0
- **Torch**: 1.4.0
- **Parsimonious**: 0.8.1

## Dataset
The WebNLG dataset is used in this project. Ensure the dataset is downloaded and accessible.

## Preprocessing
Before fine-tuning, preprocess the WebNLG dataset to convert it into the format required for the model. To preprocess, run:

```bash
./preprocess_WEBNLG.sh <dataset_folder>
```

Replace `<dataset_folder>` with the path to the WebNLG dataset.

## Fine-tuning
To fine-tune the model on the WebNLG dataset, execute:

```bash
./finetune_graph2text.sh <model> <gpu_id>
```

- `<model>` options: `gpt2`.
- `<gpu_id>`: Specify the GPU ID to be used for training.

### Example
```bash
./finetune_graph2text.sh gpt2 0
```

## Decoding
To decode using a fine-tuned model on the WebNLG dataset, run:

```bash
./test_graph2text.sh <model> <checkpoint> <gpu_id>
```

- `<model>`: The model used for fine-tuning.
- `<checkpoint>`: Path to the fine-tuned model checkpoint.
- `<gpu_id>`: Specify the GPU ID to be used for decoding.

### Example
```bash
./test_graph2text.sh  gpt2 webnlg-gpt2-base.ckpt 0
```

## Notes
1. Ensure all required dependencies are installed before running any scripts.
2. Follow the provided examples to set up your fine-tuning and decoding workflows successfully.

For any issues, please refer to the [HuggingFace documentation](https://huggingface.co) or open an issue in the repository.
