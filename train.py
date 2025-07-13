import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

#from lstm_model import LSTMForecast
#from transformer_model import TransformerForecast
from models.hybrid_tf_model import HybridTimeFrequencyTransformer

from dataset.power_dataset import PowerDataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import joblib

# Load target scaler
target_scaler = joblib.load('scalers/target_scaler.pkl')

# Fix random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

# ==== Configuration ====
INPUT_LEN = 90
OUTPUT_LEN = 90
BATCH_SIZE = 8
EPOCHS = 500
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REPEATS = 5
PATIENCE = 10  # Early stopping patience

def plot_prediction_curve(y_true, y_pred, output_len, save_path=None, title="Prediction vs Ground Truth", units="kW"):
    days = np.arange(output_len)
    plt.figure(figsize=(12, 6))
    plt.plot(days, y_true, label='Ground Truth')
    plt.plot(days, y_pred, label='Prediction')
    plt.xlabel("Future Days")
    plt.ylabel(f"Global_active_power ({units})")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[+] Image saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_loss_curves(train_losses, test_losses, save_path, loss_name="SmoothL1Loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel(loss_name)
    plt.title("Train vs Test Loss Curve")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[+] Loss curve saved to {save_path}")
    plt.close()

# ==== EarlyStopping class ====
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def step(self, metric):
        if self.best_score is None or metric < self.best_score - self.delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

# ==== Training function ====
def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    
    return total_loss / total_samples

# ==== Enhanced evaluation function (returns normalized and raw value metrics) ====
def evaluate(model, data_loader, device, criterion, target_scaler):
    model.eval()
    all_preds = []
    all_targets = []
    total_smoothl1 = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    # Raw value (kW) loss statistics
    total_smoothl1_orig = 0.0
    total_mse_orig = 0.0
    total_mae_orig = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Calculate losses on normalized data
            if criterion:
                smoothl1_loss = criterion(outputs, targets)
                total_smoothl1 += smoothl1_loss.item() * inputs.size(0)
            
            # Calculate MSE and MAE on normalized data
            mse = F.mse_loss(outputs, targets).item()
            mae = F.l1_loss(outputs, targets).item()
            
            total_mse += mse * inputs.size(0)
            total_mae += mae * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Convert to numpy for inverse normalization
            batch_preds = outputs.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            all_preds.append(batch_preds)
            all_targets.append(batch_targets)
            
            # ==== Calculate raw value (kW) losses ====
            # Inverse normalization: flatten, transform, then reshape
            batch_preds_orig = target_scaler.inverse_transform(
                batch_preds.reshape(-1, 1)
            ).reshape(batch_preds.shape)
            
            batch_targets_orig = target_scaler.inverse_transform(
                batch_targets.reshape(-1, 1)
            ).reshape(batch_targets.shape)
            
            # Convert to Tensor for loss calculation
            preds_orig_tensor = torch.tensor(batch_preds_orig, device=device)
            targets_orig_tensor = torch.tensor(batch_targets_orig, device=device)
            
            # Calculate raw value losses
            smoothl1_loss_orig = F.smooth_l1_loss(preds_orig_tensor, targets_orig_tensor).item()
            mse_orig = F.mse_loss(preds_orig_tensor, targets_orig_tensor).item()
            mae_orig = F.l1_loss(preds_orig_tensor, targets_orig_tensor).item()
            
            total_smoothl1_orig += smoothl1_loss_orig * inputs.size(0)
            total_mse_orig += mse_orig * inputs.size(0)
            total_mae_orig += mae_orig * inputs.size(0)
    
    # Calculate average losses (normalized values)
    avg_smoothl1 = total_smoothl1 / total_samples if criterion else None
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    # Calculate average losses (raw values - kW)
    avg_smoothl1_orig = total_smoothl1_orig / total_samples
    avg_mse_orig = total_mse_orig / total_samples
    avg_mae_orig = total_mae_orig / total_samples
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return (avg_smoothl1, avg_mse, avg_mae, 
            avg_smoothl1_orig, avg_mse_orig, avg_mae_orig,
            all_preds, all_targets)

# ==== Load data ====
train_df = pd.read_csv('data/cleaned2_train.csv', index_col=0)
test_df = pd.read_csv('data/cleaned2_test.csv', index_col=0)

train_dataset = PowerDataset(train_df, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
test_dataset = PowerDataset(test_df, input_len=INPUT_LEN, output_len=OUTPUT_LEN)

# ==== Multi-round training ====
# Normalized value metrics
all_smoothl1, all_mse, all_mae = [], [], []

# Raw value (kW) metrics
all_smoothl1_orig, all_mse_orig, all_mae_orig = [], [], []  

# For storing denormalized results of each round
all_preds_inv_all_exps = []
all_targets_inv_all_exps = []

for i in range(N_REPEATS):
    print(f"\n🚀 Round {i+1}/{N_REPEATS} training")

    seed = 42 + i  # Use different but deterministic seed for each round
    set_seed(seed)  # Reset all random states
    
    # Recreate data loaders for each round
    train_loader = DataLoader(train_dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=True,
                             worker_init_fn=lambda id: set_seed(seed + id))
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    # Initialize model
    """
    model = TransformerForecast(
        input_dim=13,
        d_model=16,   
        nhead=4,      
        num_layers=3, 
        output_len=OUTPUT_LEN,
        dropout=0.1  
    ).to(DEVICE)
    """

    """
    model = HybridTimeFrequencyTransformer(
        input_dim=13,
        d_model=16,
        nhead=4,
        num_layers=3,
        output_len=OUTPUT_LEN,
        dropout=0.1
    ).to(DEVICE)
    """

    """
    model = LSTMForecast(
        input_dim=13,
        hidden_dim=256,
        num_layers=3,
        output_len=OUTPUT_LEN,
        dropout_rate=0.2
    ).to(DEVICE)
    """

    # 参数组合
    model = HybridTimeFrequencyTransformer(
        input_dim=13,             # 输入变量数（例如：电压、电流、功率等）
        input_len=INPUT_LEN,             # 输入时间步（如过去90天）
        d_model=32,               # Transformer隐空间维度
        nhead=4,                  # 多头注意力头数
        num_layers=3,             # Transformer层数
        output_len=OUTPUT_LEN,            # 预测时间步（如未来90天）

        dropout=0.2,              # dropout概率
        
        freq_method='phase',      # 使用频域的“相位谱”，结构性更强
        use_freq_pooling=False,    # 使用频域平均池化（推荐，参数量小）
        retain_freq_ratio=0.25,   # 保留前25%频率分量（低频），避免噪声

        time_pooling='last',      # 时域池化方式（可选：'mean', 'max', 'last'）
        pos_encoder_enabled=True, # 是否启用位置编码

        fusion_strategy='concat', # 时频融合方式（'concat', 'add', 'gate'）
        hidden_ratio=2            # MLP隐藏层倍数，默认即可
    ).to(DEVICE)




    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    early_stopper = EarlyStopping(patience=PATIENCE)

    best_test_smoothl1_orig = float('inf')  # Use raw value (kW) SmoothL1Loss for early stopping
    best_model_path = f"checkpoints/best_model_exp{i+1}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    train_losses = []
    test_smoothl1_losses = []  # Normalized values
    test_smoothl1_losses_orig = []  # Raw values (kW)

    for epoch in range(EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Evaluate model (get normalized and raw value metrics)
        (test_smoothl1, test_mse, test_mae, 
         test_smoothl1_orig, test_mse_orig, test_mae_orig, 
         _, _) = evaluate(
            model, test_loader, DEVICE, criterion, target_scaler
        )
        
        train_losses.append(train_loss)
        test_smoothl1_losses.append(test_smoothl1)
        test_smoothl1_losses_orig.append(test_smoothl1_orig)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}")
        print(f"    Norm: SmoothL1={test_smoothl1:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
        print(f"    Raw: SmoothL1={test_smoothl1_orig:.4f} kW, MSE={test_mse_orig:.4f} kW², MAE={test_mae_orig:.4f} kW")

        # Save model based on raw value (kW) SmoothL1Loss (business relevant metric)
        if test_smoothl1_orig < best_test_smoothl1_orig:
            best_test_smoothl1_orig = test_smoothl1_orig
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model (SmoothL1Loss={test_smoothl1_orig:.4f} kW): {best_model_path}")

        # Early stopping based on raw value (kW) SmoothL1Loss
        if early_stopper.step(test_smoothl1_orig):
            print(f"🛑 Early stopping at epoch {epoch+1} (based on raw value loss)")
            break

    # === Load best model for final evaluation ===
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation (only care about predictions and targets)
    _, _, _, _, _, _, all_preds, all_targets = evaluate(
        model, test_loader, DEVICE, criterion, target_scaler
    )
    
    # Get raw value metrics (only take the last evaluation)
    (_, _, _, 
     test_smoothl1_orig, test_mse_orig, test_mae_orig, 
     all_preds, all_targets) = evaluate(
        model, test_loader, DEVICE, criterion, target_scaler
    )
    
    # Record final metrics for this round
    all_smoothl1_orig.append(test_smoothl1_orig)
    all_mse_orig.append(test_mse_orig)
    all_mae_orig.append(test_mae_orig)

    print(f"\n🔥 Test set raw performance (kW):")
    print(f"SmoothL1Loss = {test_smoothl1_orig:.4f} kW")
    print(f"MSE = {test_mse_orig:.4f} kW²")
    print(f"MAE = {test_mae_orig:.4f} kW")

    # Denormalize predictions and targets (already in kW)
    all_preds_inv = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    all_targets_inv = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)

    # Collect denormalized predictions and targets for each round
    all_preds_inv_all_exps.append(all_preds_inv)
    all_targets_inv_all_exps.append(all_targets_inv)

    # === Visualize predictions (raw values, unit: kW) ===
    sample_id = 0
    plot_prediction_curve(
        all_targets_inv.reshape(-1, OUTPUT_LEN)[sample_id],
        all_preds_inv.reshape(-1, OUTPUT_LEN)[sample_id],
        output_len=OUTPUT_LEN,
        save_path=f"plots_best/pred_vs_true_len{OUTPUT_LEN}_exp{i+1}_sample{sample_id}.png",
        title=f"Prediction vs Ground Truth (Sample {sample_id}) [Raw-kW]",
        units="kW"
    )

    # === Visualize both loss curves ===
    plot_loss_curves(
        train_losses,
        test_smoothl1_losses,
        save_path=f"plots_best/loss_curve_norm_exp{i+1}.png",
        loss_name="SmoothL1Loss (Norm)"
    )
    
    plot_loss_curves(
        train_losses,
        test_smoothl1_losses_orig,
        save_path=f"plots_best/loss_curve_orig_exp{i+1}.png",
        loss_name="SmoothL1Loss (kW)"
    )

# ==== Final results ====
print("\n📊 5-round average metrics (normalized values):")
print(f"SmoothL1Loss: {np.mean(all_smoothl1):.4f} ± {np.std(all_smoothl1):.4f}")
print(f"MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
print(f"MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")

print("\n🔥 5-round average metrics (raw values-kW):")
print(f"SmoothL1Loss: {np.mean(all_smoothl1_orig):.4f} kW ± {np.std(all_smoothl1_orig):.4f} kW")
print(f"MSE: {np.mean(all_mse_orig):.4f} kW² ± {np.std(all_mse_orig):.4f} kW²")
print(f"MAE: {np.mean(all_mae_orig):.4f} kW ± {np.std(all_mae_orig):.4f} kW")

print("\n✅ All experiments completed!")