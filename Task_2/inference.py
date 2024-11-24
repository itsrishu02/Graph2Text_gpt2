import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge import Rouge
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import re
import sacrebleu


def ensure_feature_dim(graph_data, required_dim, device):
    """
    Adjusts the feature dimension of the graph data to match the required dimension by padding or truncating.

    Args:
        graph_data (torch_geometric.data.Data): The graph data object.
        required_dim (int): The desired feature dimension.
        device (str): The device to place tensors on.

    Returns:
        torch_geometric.data.Data: The updated graph data object.
    """
    current_dim = graph_data.x.size(1)
    if current_dim < required_dim:
        padding = torch.zeros((graph_data.x.size(0), required_dim - current_dim), device=device)
        graph_data.x = torch.cat([graph_data.x, padding], dim=1)
    elif current_dim > required_dim:
        graph_data.x = graph_data.x[:, :required_dim]
    return graph_data

def parse_graph_from_source(source_file, device):
    """
    Parses a source file to create a graph suitable for PyTorch Geometric.
    Assumes each line contains triples in the format <H>head<R>relation<T>tail.

    Args:
        source_file (str): Path to the source file.
        device (str): The device to place tensors on.

    Returns:
        torch_geometric.data.Data: The graph data object.
    """
    with open(source_file, 'r') as f:
        lines = f.readlines()

    edge_index = []
    node_to_idx = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        triples = line.split('<H>')
        for triple in triples:
            if not triple.strip():
                continue

            try:
                triple = f"<H>{triple.strip()}"
                h, rest = triple.split('<R>')
                r, t = rest.split('<T>')

                h = h.strip('<H>').strip()
                r = r.strip()
                t = t.strip()

                if h not in node_to_idx:
                    node_to_idx[h] = len(node_to_idx)
                if t not in node_to_idx:
                    node_to_idx[t] = len(node_to_idx)

                edge_index.append([node_to_idx[h], node_to_idx[t]])

            except ValueError:
                print(f"Skipping malformed triple: {triple}")
                continue

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    num_nodes = len(node_to_idx)

    # Use node indices as features
    node_indices = torch.arange(num_nodes, dtype=torch.long, device=device)

    # Initialize node features (can be customized)
    graph_data = Data(x=node_indices, edge_index=edge_index, num_nodes=num_nodes)

    return graph_data

def process_target_data(target_file, tokenizer, max_length=64, device='cpu'):
    """
    Tokenizes the target text data using GPT-2's tokenizer.

    Args:
        target_file (str): Path to the target file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        max_length (int): Maximum sequence length.
        device (str): The device to place tensors on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input IDs and labels tensors.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(target_file, 'r') as f:
        target_texts = [line.strip() for line in f if line.strip()]

    tokenized = tokenizer(
        target_texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized.input_ids.to(device)
    labels = input_ids.clone()

    return input_ids, labels


class GraphEmbeddingModel(nn.Module):
    """
    A two-layer Graph Convolutional Network (GCN) that generates embeddings for graph nodes.
    """
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim):
        super(GraphEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, data):
        # Embed the node indices
        x = self.embedding(data.x)
        edge_index = data.edge_index

        # Pass through GNN layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GraphToTextModel(nn.Module):
    """
    Integrates the GNN with GPT-2 to generate text from graph embeddings.
    Includes an optional auxiliary classification task to enhance GNN training.
    """
    def __init__(self, gnn_model, lm_model, hidden_dim, num_classes=None):
        super(GraphToTextModel, self).__init__()
        self.gnn_model = gnn_model
        self.lm_model = lm_model
        self.graph_to_hidden = nn.Linear(gnn_model.output_dim, hidden_dim)

        # Auxiliary task layer (optional)
        self.auxiliary_classifier = nn.Linear(gnn_model.output_dim, num_classes) if num_classes else None

    def forward(self, graph_data, lm_inputs):
        # Forward pass through GNN
        graph_embeddings = self.gnn_model(graph_data)

        # Optional: Add Gaussian noise to embeddings (regularization)
        graph_embeddings = graph_embeddings + torch.randn_like(graph_embeddings) * 0.01

        auxiliary_loss = 0.0
        if self.auxiliary_classifier:
            node_labels = torch.randint(
                0, self.auxiliary_classifier.out_features, 
                (graph_embeddings.size(0),), device=graph_embeddings.device
            )
            aux_logits = self.auxiliary_classifier(graph_embeddings)
            auxiliary_loss = nn.CrossEntropyLoss()(aux_logits, node_labels)

        # Global pooling for graph-level embedding
        if hasattr(graph_data, 'batch'):
            graph_embedding = global_mean_pool(graph_embeddings, graph_data.batch)
        else:
            graph_embedding = global_mean_pool(graph_embeddings, torch.zeros(graph_embeddings.size(0), dtype=torch.long, device=graph_embeddings.device))

        # Integrate graph embeddings into GPT-2
        graph_embedding = self.graph_to_hidden(graph_embedding)  # (batch_size, hidden_dim)
        batch_size = lm_inputs['input_ids'].shape[0]
        seq_len = lm_inputs['input_ids'].shape[1]
        graph_embedding_expanded = graph_embedding.unsqueeze(1).expand(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)

        # Get GPT-2 input embeddings
        lm_inputs_embeds = self.lm_model.transformer.wte(lm_inputs['input_ids'])  # (batch_size, seq_len, hidden_dim)

        # Combine graph embeddings with GPT-2 embeddings
        lm_inputs_embeds = lm_inputs_embeds + graph_embedding_expanded  # (batch_size, seq_len, hidden_dim)

        # Forward pass through GPT-2
        lm_outputs = self.lm_model(inputs_embeds=lm_inputs_embeds, labels=lm_inputs['labels'])

        # Attach auxiliary loss if present
        if self.auxiliary_classifier:
            lm_outputs.loss += auxiliary_loss

        return lm_outputs


def load_model_for_inference(checkpoint_path, gnn_model, lm_model, hidden_dim, num_classes=None, device='cpu'):
    """
    Loads the saved checkpoint into the model with appropriate device mapping.

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        gnn_model (nn.Module): Initialized GNN model.
        lm_model (nn.Module): Initialized GPT-2 model.
        hidden_dim (int): Hidden dimension for the GraphToTextModel.
        num_classes (int or None): Number of classes for auxiliary task. Default is None.
        device (str): Device to map the model to ('cpu' or 'cuda').

    Returns:
        GraphToTextModel: Loaded model ready for inference.
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize the combined model
    model = GraphToTextModel(gnn_model, lm_model, hidden_dim=hidden_dim, num_classes=num_classes)
    
    # Load the state dict with map_location to handle device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move the model to the specified device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def generate_predictions(model, test_graph_data, test_input_ids, tokenizer, batch_size=16, device='cpu'):
    """
    Generates predictions for the test data.

    Args:
        model (GraphToTextModel): The trained model.
        test_graph_data (torch_geometric.data.Data): Parsed graph data.
        test_input_ids (torch.Tensor): Tokenized input IDs for the test set.
        tokenizer (PreTrainedTokenizer): Tokenizer used to decode predictions.
        batch_size (int): Batch size for inference.
        device (str): Device for computation.

    Returns:
        List[str]: Predicted texts.
    """
    predictions = []
    test_loader = torch.utils.data.DataLoader(test_input_ids, batch_size=batch_size)
    
    # Ensure graph data is on the correct device
    test_graph_data = test_graph_data.to(device)

    with torch.no_grad():
        for batch_input_ids in tqdm(test_loader, desc="Generating Predictions", dynamic_ncols=True):
            batch_input_ids = batch_input_ids.to(device)

            # Forward pass through the model
            outputs = model(test_graph_data, {"input_ids": batch_input_ids, "labels": batch_input_ids})
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            batch_predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

            # Decode predictions
            decoded_predictions = tokenizer.batch_decode(batch_predictions, skip_special_tokens=True)
            predictions.extend(decoded_predictions)

    return predictions


def evaluate_predictions(predictions, targets):
    """
    Evaluates the predictions against the targets using BLEU and ROUGE.

    Args:
        predictions (List[str]): Generated texts.
        targets (List[str]): Ground truth texts.

    Returns:
        Dict[str, float]: Evaluation metrics (BLEU, ROUGE-L).
    """
    print("Evaluating predictions...")
    rouge_evaluator = Rouge()

    # Calculate BLEU scores
    bleu_scores = [
        sacrebleu.corpus_bleu([pred], [[target]]).score
        for pred, target in zip(predictions, targets)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    # Calculate ROUGE scores
    rouge_scores = rouge_evaluator.get_scores(predictions, targets, avg=True)
    avg_rouge_l = rouge_scores['rouge-l']['f']

    return {"BLEU": avg_bleu, "ROUGE-L": avg_rouge_l}


def run_inference(
    checkpoint_path, test_source_file, test_target_file, hidden_dim,
    tokenizer, batch_size=16, device='cpu', output_file="test_predictions.txt",
    num_classes=None
):

    # Parse test data
    print("Parsing test graph data...")
    test_graph_data = parse_graph_from_source(test_source_file, device)

    # Dynamically determine the number of nodes
    num_nodes = test_graph_data.num_nodes
    print(f"Number of nodes in the graph: {num_nodes}")

    # Define GNN model parameters dynamically based on the graph
    gnn_model_params = {
        "num_nodes": num_nodes,
        "embedding_dim": 128,
        "hidden_dim": 768,  # Typically matches GPT-2 hidden size
        "output_dim": 128,
    }

    print("Processing test target data...")
    test_input_ids, test_labels = process_target_data(test_target_file, tokenizer, max_length=64, device=device)

    # Initialize GNN model
    print("Initializing GNN model...")
    gnn_model = GraphEmbeddingModel(**gnn_model_params).to(device)

    # Initialize GPT-2 model
    print("Initializing GPT-2 model...")
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Load the trained model with proper device mapping
    model = load_model_for_inference(
        checkpoint_path, gnn_model, lm_model, hidden_dim, num_classes=num_classes, device=device
    )

    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, test_graph_data, test_input_ids, tokenizer, batch_size, device)

    # Load ground truth targets
    with open(test_target_file, "r") as f:
        targets = [line.strip() for line in f if line.strip()]

    # Evaluate predictions
    metrics = evaluate_predictions(predictions, targets)

    # Save predictions and metrics
    with open(output_file, "w") as f_output:
        f_output.write("Predictions and Targets:\n")
        f_output.write("-" * 80 + "\n")
        for pred, target in zip(predictions, targets):
            f_output.write(f"Prediction: {pred.strip()}\nTarget: {target.strip()}\n" + "-" * 80 + "\n")
        f_output.write("\nEvaluation Metrics:\n")
        for metric, value in metrics.items():
            f_output.write(f"{metric}: {value:.4f}\n")

    print("Inference completed!")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Define file paths and parameters
    checkpoint_path = "/home/scai/mtech/aib232070/scratch/LLM_Project/best_model_epoch_10.pt"  # Replace with your actual model path
    test_source = "/home/scai/mtech/aib232070/scratch/LLM_Project/plms-graph2text/webnlg/data/webnlg/test_unseen.source"
    test_target = "/home/scai/mtech/aib232070/scratch/LLM_Project/plms-graph2text/webnlg/data/webnlg/test_unseen.target"
    output_file = "/home/scai/mtech/aib232070/scratch/LLM_Project/test_predictions_both.txt"  # File to save predictions

    # Hidden dimension should match the one used during training
    hidden_dim = 768  # Typically matches GPT-2 hidden size

    # Tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run inference
    run_inference(
        checkpoint_path=checkpoint_path,
        test_source_file=test_source,
        test_target_file=test_target,
        hidden_dim=hidden_dim,
        tokenizer=tokenizer,
        batch_size=16,
        device=device,
        output_file=output_file,
        num_classes=None 
    )
