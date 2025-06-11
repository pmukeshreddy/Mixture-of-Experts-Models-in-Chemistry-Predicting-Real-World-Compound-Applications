import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
from rdkit import Chem

# --- Ensure all model and helper classes from above are defined correctly ---
# (GenricSmilesTokenizer, PositionalEncoding, Expert, MoELayer, MoETransformerEncoderLayer,
# MoETransformerforChemApp, get_atom_featues, get_bond_features, smiles_to_graph_data,
# MolecularDataset, GCN, GNN_MoE_Model)

if __name__ == '__main__':
    # --- 1. Configuration ---\n    
    CSV_FILE_PATH = '/kaggle/input/chemical-part-1-data/part_1.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # GNN Hyperparameters
    # Get feature length from a sample atom
    NUM_NODE_FEATURES = len(get_atom_featues(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))
    print(f"Number of atom features: {NUM_NODE_FEATURES}")
    GNN_NUM_LAYERS = 3
    GNN_HIDDEN_CHANNELS = 128
    GNN_OUTPUT_DIM = 64 # This MUST match MOE_D_MODEL if there's no projection layer

    # MoE Transformer Hyperparameters
    temp_tokenizer = GenricSmilesTokenizer()
    MOE_VOCAB_SIZE = temp_tokenizer.vocab_size
    MOE_D_MODEL = 64  # This is the d_model for the Transformer part
    MOE_NHEAD = 4
    MOE_NUM_ENCODER_LAYERS = 2
    MOE_NUM_EXPERTS = 4
    MOE_EXPERT_HIDDEN_DIM = 128
    MOE_TOP_K = 2
    MOE_DROPOUT_RATE = 0.1
    MOE_MAX_SEQ_LEN_MODEL = 50
    MOE_LOAD_BALANCE_COEFF = 0.01

    # Training Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # --- 2. Data Preprocessing ---
    all_smiles, all_labels, label_names = preprocess_data(CSV_FILE_PATH)
    NUM_DISTINCT_LABELS = len(label_names)
    print(f"Total valid molecules: {len(all_smiles)}")
    if not all_smiles:
        print("No valid SMILES strings found. Exiting.")
        exit()

    # Split data
    train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
        all_smiles, all_labels, test_size=0.2, random_state=42
    )
    val_smiles, test_smiles, val_labels, test_labels = train_test_split(
        temp_smiles, temp_labels, test_size=0.5, random_state=42
    )
    print(f"Train samples: {len(train_smiles)}, Val samples: {len(val_smiles)}, Test samples: {len(test_smiles)}")

    # Create PyTorch Geometric Datasets using the corrected InMemoryDataset class
    train_dataset = MolecularDataset(root='data/train', smiles_list=train_smiles, labels_list=train_labels)
    val_dataset = MolecularDataset(root='data/val', smiles_list=val_smiles, labels_list=val_labels)
    test_dataset = MolecularDataset(root='data/test', smiles_list=test_smiles, labels_list=test_labels)

    # Use the DataLoader from torch_geometric.loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Model Initialization ---
    gnn_model = GCN(
        num_node_features=NUM_NODE_FEATURES,
        num_gnn_features=GNN_NUM_LAYERS,
        hidden_channels=GNN_HIDDEN_CHANNELS,
        gnn_output_dim=GNN_OUTPUT_DIM,
        drop_out=MOE_DROPOUT_RATE
    ).to(DEVICE)

    # Corrected class name
    moe_transformer_model = MoETransformerforChemApp(
        vocab_size=MOE_VOCAB_SIZE,
        d_model=MOE_D_MODEL,
        nhead=MOE_NHEAD,
        num_encoder_layers=MOE_NUM_ENCODER_LAYERS,
        moe_num_experts=MOE_NUM_EXPERTS,
        moe_expert_hidden_dim=MOE_EXPERT_HIDDEN_DIM,
        moe_top_k=MOE_TOP_K,
        num_labels=NUM_DISTINCT_LABELS,
        dropout_rate=MOE_DROPOUT_RATE,
        max_seq_len=MOE_MAX_SEQ_LEN_MODEL,
        moe_load_balance_coeff=MOE_LOAD_BALANCE_COEFF
    ).to(DEVICE)

    combined_model = GNN_MoE_Model(
        gnn=gnn_model,
        moe_transformer=moe_transformer_model,
        gnn_output_dim=GNN_OUTPUT_DIM,
        transformer_d_model=MOE_D_MODEL
    ).to(DEVICE)

    num_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in combined model: {num_params:,}")

    # --- 4. Training Setup ---
    optimizer = torch.optim.AdamW(combined_model.parameters(), lr=LEARNING_RATE)
    criterion_task = nn.BCEWithLogitsLoss()

    # --- 5. Training Loop ---
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        combined_model.train()
        total_epoch_loss = 0
        total_epoch_task_loss = 0
        total_epoch_aux_loss = 0
        
        for batch_data in train_loader:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            
            logits, aux_loss = combined_model(batch_data)
            
            # The labels 'y' are already correctly batched by the PyG DataLoader
            task_loss = criterion_task(logits, batch_data.y)
            combined_loss = task_loss + aux_loss

            combined_loss.backward()
            optimizer.step()
            
            total_epoch_loss += combined_loss.item()
            total_epoch_task_loss += task_loss.item()
            total_epoch_aux_loss += aux_loss.item()

        avg_task_loss = total_epoch_task_loss / len(train_loader)
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} --- Avg Task Loss: {avg_task_loss:.4f}")

        # Validation step
        combined_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(DEVICE)
                logits, aux_loss = combined_model(batch_data)
                task_loss = criterion_task(logits, batch_data.y)
                val_loss = task_loss + aux_loss
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Average Validation Combined Loss: {avg_val_loss:.4f}")
        print("-" * 30)

    print("Training finished.")
