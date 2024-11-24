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

# ================================
# 1. Tokenizer Definition
# ================================

def regex_tokenize(text):
    """
    Tokenizes input text using regular expressions.

    Args:
        text (str): The input text to tokenize.

    Returns:
        List[str]: A list of tokens.
    """
    pattern = r"\b\w+(?:'\w+)?\b"
    tokens = re.findall(pattern, text.lower())
    return tokens

# ================================
# 2. Data Parsing Functions
# ================================

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

# ================================
# 3. Model Definitions
# ================================

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
            # Placeholder for meaningful auxiliary labels
            # Replace with actual labels if available
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

# ================================
# 4. Training Function
# ================================

def train_model(
    train_source_file, train_target_file, val_source_file, val_target_file,
    epochs=50, lr_gnn=1e-3, lr_lm=1e-4, max_length=64, device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=16, accumulation_steps=4, output_file="predictions.txt", num_classes=None
):
    """
    Trains the Graph-to-Text model using the provided training and validation data.

    Args:
        train_source_file (str): Path to the training source file.
        train_target_file (str): Path to the training target file.
        val_source_file (str): Path to the validation source file.
        val_target_file (str): Path to the validation target file.
        epochs (int): Number of training epochs.
        lr_gnn (float): Learning rate for GNN parameters.
        lr_lm (float): Learning rate for Language Model (GPT-2) parameters.
        max_length (int): Maximum sequence length for tokenization.
        device (str): Device to use ('cuda' or 'cpu').
        batch_size (int): Batch size for training.
        accumulation_steps (int): Gradient accumulation steps.
        output_file (str): Path to the output file for logging predictions and targets.
        num_classes (int or None): Number of classes for the auxiliary task. Set to None to disable.
    """
    # Data Parsing
    print("Parsing training graph data...")
    train_graph_data = parse_graph_from_source(train_source_file, device)
    print("Parsing validation graph data...")
    val_graph_data = parse_graph_from_source(val_source_file, device)

    # Tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process Target Data
    print("Processing training target data...")
    train_input_ids, train_labels = process_target_data(train_target_file, tokenizer, max_length, device)
    print("Processing validation target data...")
    val_input_ids, val_labels = process_target_data(val_target_file, tokenizer, max_length, device)

    # Create DataLoaders
    print("Creating DataLoaders...")
    train_dataset = TensorDataset(train_input_ids, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model Initialization
    print("Initializing models...")
    num_nodes = train_graph_data.num_nodes
    embedding_dim = 128
    hidden_dim = 768  # Typically matches GPT-2's hidden size
    output_dim = 128  # GNN output dimension

    gnn_model = GraphEmbeddingModel(num_nodes, embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model = GraphToTextModel(gnn_model, lm_model, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    # Ensure all model parameters require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} has requires_grad=False. Enabling gradients.")
            param.requires_grad = True

    # Optimizer
    print("Setting up optimizer...")
    # Include all parameters in the optimizer with appropriate learning rates
    optimizer = optim.Adam([
        {'params': gnn_model.parameters(), 'lr': lr_gnn},
        {'params': lm_model.parameters(), 'lr': lr_lm},
        {'params': model.graph_to_hidden.parameters(), 'lr': lr_gnn},  # Assuming same lr as GNN
    ], lr=lr_gnn)  # Default lr for any remaining parameters

    # If using auxiliary task, include its parameters
    if model.auxiliary_classifier:
        optimizer.add_param_group({'params': model.auxiliary_classifier.parameters(), 'lr': lr_gnn})

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize Evaluation Metrics
    best_bleu = 0.0  # For checkpointing best model based on BLEU
    rouge_evaluator = Rouge()

    with open(output_file, "w") as f_output:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Training Phase
            model.train()
            optimizer.zero_grad()
            train_loss = 0.0

            for batch_idx, (batch_input_ids, batch_labels) in enumerate(tqdm(train_loader, desc="Training")):
                batch_input_ids = batch_input_ids.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                outputs = model(train_graph_data, {"input_ids": batch_input_ids, "labels": batch_labels})

                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()

                # Gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                train_loss += loss.item()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                # # Debugging: Check if GNN parameters have gradients
                # if batch_idx % 100 == 0:  # Adjust the frequency as needed
                #     print(f"Batch {batch_idx}: Checking GNN gradients...")
                #     for name, param in gnn_model.named_parameters():
                #         if param.grad is not None:
                #             print(f"Gradient for {name}: {param.grad.norm().item():.4f}")
                #         else:
                #             print(f"Gradient for {name}: None")

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # Validation Phase
            model.eval()
            val_loss = 0.0
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch_input_ids, batch_labels in tqdm(val_loader, desc="Validation"):
                    batch_input_ids = batch_input_ids.to(device)
                    batch_labels = batch_labels.to(device)

                    # Forward pass
                    outputs = model(val_graph_data, {"input_ids": batch_input_ids, "labels": batch_labels})

                    loss = outputs.loss
                    val_loss += loss.item()

                    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                    predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

                    # Decode predictions and targets
                    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    decoded_targets = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)

                    all_predictions.extend(decoded_predictions)
                    all_targets.extend(decoded_targets)

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")

            # Compute Evaluation Metrics
            print("Calculating evaluation metrics...")
            bleu_scores = [
                sacrebleu.corpus_bleu([pred], [[target]]).score
                for pred, target in zip(all_predictions, all_targets)
            ]
            rouge_scores = rouge_evaluator.get_scores(all_predictions, all_targets, avg=True)

            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            avg_rouge_l = rouge_scores['rouge-l']['f']  # Example: ROUGE-L F1 score

            print(f"Validation BLEU Score: {avg_bleu:.2f}")
            print(f"Validation ROUGE-L Score: {avg_rouge_l:.4f}")

            # Log predictions and targets
            f_output.write(f"Epoch {epoch + 1} Validation Predictions and Targets:\n")
            f_output.write("-" * 80 + "\n")
            for pred, target in zip(all_predictions, all_targets):
                output_line = f"Prediction: {pred.strip()}\nTarget: {target.strip()}\n" + "-" * 80 + "\n"
                f_output.write(output_line)

            # Checkpointing
            if avg_bleu > best_bleu:
                best_bleu = avg_bleu
                checkpoint_path = f"best_model_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New best model saved to {checkpoint_path}")

            # Step the scheduler
            scheduler.step()

    print("\nTraining complete.")

# ================================
# 5. Main Execution
# ================================

if __name__ == "__main__":
    # Define file paths
    train_source = "/home/scai/mtech/aib232070/scratch/LLM_Project/plms-graph2text/webnlg/data/webnlg/train.source"
    train_target = "/home/scai/mtech/aib232070/scratch/LLM_Project/plms-graph2text/webnlg/data/webnlg/train.target"
    val_source = "/home/scai/mtech/aib232070/scratch/LLM_Project/plms-graph2text/webnlg/data/webnlg/val.source"
    val_target = "/home/scai/mtech/aib232070/scratch/LLM_Project/plms-graph2text/webnlg/data/webnlg/val.target"
    output_file = "/home/scai/mtech/aib232070/scratch/LLM_Project/epoch_predictions.txt"

    # Optional: Define number of classes for auxiliary task
    # Set to None if not using auxiliary task
    num_classes = None  # Example: 10

    # Execute training
    train_model(
        train_source_file=train_source,
        train_target_file=train_target,
        val_source_file=val_source,
        val_target_file=val_target,
        epochs=50,
        lr_gnn=1e-3,
        lr_lm=1e-4,
        max_length=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=16,
        accumulation_steps=4,
        output_file=output_file,
        num_classes=num_classes
    )
