import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import random
import joblib
import itertools
import gc
import sys
from datetime import datetime

# 确保导入自定义模型和数据集
sys.path.append('.')
from hybrid_tf_model import HybridTimeFrequencyTransformer
from dataset.power_dataset import PowerDataset

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
OUTPUT_LEN = 365
EPOCHS = 500  # 最大训练轮数
N_REPEATS = 5  # 每个组合的重复次数
PATIENCE = 10    # 早停耐心值
MAX_COMBINATIONS = 600  # 最大测试组合数量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Hyperparameter Search Space ====
GRID_SEARCH_SPACE = {
    'batch_size': [32, 64, 8],
    'd_model': [32, 64, 96],
    'nhead': [4, 8],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2],
    'learning_rate': [1e-4, 3e-4],
    'freq_method': ['amplitude', 'phase', 'both'],
    'use_freq_pooling': [True, False],
    'retain_freq_ratio': [0.25, 0.5],
    'time_pooling': ['mean', 'max', 'last'],
    'pos_encoder_enabled': [True, False],
    'fusion_strategy': ['concat', 'gate'],
    'hidden_ratio': [1, 2]
}

# 生成所有可能的组合
all_combinations = list(itertools.product(*GRID_SEARCH_SPACE.values()))
keys = list(GRID_SEARCH_SPACE.keys())
hyperparam_combinations = [dict(zip(keys, values)) for values in all_combinations]

# 过滤无效组合：确保nhead能被d_model整除
valid_combinations = []
for combo in hyperparam_combinations:
    d_model = combo['d_model']
    nhead = combo['nhead']
    if d_model % nhead == 0:
        valid_combinations.append(combo)

print(f"总可能组合数: {len(hyperparam_combinations)}")
print(f"有效组合数 (nhead/d_model检查后): {len(valid_combinations)}")

# 随机采样组合以限制数量
if len(valid_combinations) > MAX_COMBINATIONS:
    print(f"随机采样 {MAX_COMBINATIONS} 个组合进行测试 (总有效组合: {len(valid_combinations)})")
    valid_combinations = random.sample(valid_combinations, MAX_COMBINATIONS)
else:
    print(f"测试所有 {len(valid_combinations)} 个有效组合")

# ==== Load data ====
print("加载数据...")
train_df = pd.read_csv('data/cleaned2_train.csv', index_col=0)
test_df = pd.read_csv('data/cleaned2_test.csv', index_col=0)

train_dataset = PowerDataset(train_df, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
test_dataset = PowerDataset(test_df, input_len=INPUT_LEN, output_len=OUTPUT_LEN)

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

# ==== Evaluation function ====
def evaluate(model, data_loader, device, criterion, target_scaler):
    model.eval()
    total_smoothl1_orig = 0.0
    total_mse_orig = 0.0
    total_mae_orig = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Convert to numpy for inverse normalization
            batch_preds = outputs.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            
            # ==== Calculate raw value (kW) losses ====
            # Inverse normalization
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
            total_samples += inputs.size(0)
    
    # Calculate average losses (raw values - kW)
    avg_smoothl1_orig = total_smoothl1_orig / total_samples
    avg_mse_orig = total_mse_orig / total_samples
    avg_mae_orig = total_mae_orig / total_samples
    
    return avg_smoothl1_orig, avg_mse_orig, avg_mae_orig

# ==== Training and Evaluation for One Combination ====
def train_and_evaluate(hyperparams, combo_id, total_combinations):
    # 在开始训练前显示超参数
    print(f"\n=== 测试参数组合 {combo_id+1}/{total_combinations}: {hyperparams} ===")
    
    repetition_results = []
    
    for repetition in range(N_REPEATS):
        seed = 42 + repetition
        set_seed(seed)
        
        # 使用当前batch_size创建DataLoader
        batch_size = hyperparams['batch_size']
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=lambda id: set_seed(seed + id)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        model = HybridTimeFrequencyTransformer(
            input_dim=13,
            input_len=INPUT_LEN,  # 添加输入序列长度
            d_model=hyperparams['d_model'],
            nhead=hyperparams['nhead'],
            num_layers=hyperparams['num_layers'],
            output_len=OUTPUT_LEN,
            dropout=hyperparams['dropout'],
            freq_method=hyperparams['freq_method'],
            use_freq_pooling=hyperparams['use_freq_pooling'],
            retain_freq_ratio=hyperparams['retain_freq_ratio'],
            time_pooling=hyperparams['time_pooling'],
            pos_encoder_enabled=hyperparams['pos_encoder_enabled'],
            fusion_strategy=hyperparams['fusion_strategy'],
            hidden_ratio=hyperparams['hidden_ratio']
        ).to(DEVICE)


        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=1e-5
        )
        criterion = nn.SmoothL1Loss()
        early_stopper = EarlyStopping(patience=PATIENCE)
        
        # Training storage
        best_test_smoothl1_orig = float('inf')
        best_test_mse_orig = float('inf')
        best_test_mae_orig = float('inf')
        
        # Training loop
        for epoch in range(EPOCHS):  # 使用配置中的EPOCHS
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
            # Evaluate
            smoothl1_orig, mse_orig, mae_orig = evaluate(
                model, test_loader, DEVICE, criterion, target_scaler
            )
            
            # Early stopping and model saving
            if smoothl1_orig < best_test_smoothl1_orig:
                best_test_smoothl1_orig = smoothl1_orig
                best_test_mse_orig = mse_orig
                best_test_mae_orig = mae_orig
            
            if early_stopper.step(smoothl1_orig):
                break
        
        # Print repetition result
        print(f"🚀 第 {repetition+1}/{N_REPEATS} 轮训练")
        print(f"参数组合 {hyperparams} - 第 {repetition+1} 轮: "
              f"SmoothL1Loss={best_test_smoothl1_orig:.6f}, "
              f"MSE={best_test_mse_orig:.6f}, "
              f"MAE={best_test_mae_orig:.6f}")
        
        # Save final metrics for this repetition
        repetition_results.append({
            'smoothl1': best_test_smoothl1_orig,
            'mse': best_test_mse_orig,
            'mae': best_test_mae_orig
        })
        
        # Clean up memory
        del model, optimizer, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    # Compute average metrics across repetitions
    smoothl1_avgs = [r['smoothl1'] for r in repetition_results]
    mse_avgs = [r['mse'] for r in repetition_results]
    mae_avgs = [r['mae'] for r in repetition_results]
    
    avg_smoothl1 = np.mean(smoothl1_avgs)
    avg_mse = np.mean(mse_avgs)
    avg_mae = np.mean(mae_avgs)
    
    print(f"参数组合 {hyperparams} 平均性能: "
          f"SmoothL1Loss={avg_smoothl1:.6f}, "
          f"MSE={avg_mse:.6f}, "
          f"MAE={avg_mae:.6f}")
    
    return avg_smoothl1, avg_mse, avg_mae

# ==== Main Grid Search Execution ====
def main():
    total_combinations = len(valid_combinations)
    best_combo = None
    best_score = float('inf')
    best_mse = float('inf')
    best_mae = float('inf')
    
    print(f"开始网格搜索，共 {total_combinations} 个组合...")
    
    for combo_id, hyperparams in enumerate(valid_combinations):
        avg_smoothl1, avg_mse, avg_mae = train_and_evaluate(
            hyperparams, combo_id, total_combinations
        )
        
        # Update best combination (based on SmoothL1 loss)
        if avg_smoothl1 < best_score:
            best_score = avg_smoothl1
            best_mse = avg_mse
            best_mae = avg_mae
            best_combo = hyperparams
            print(f"🔥 新最佳参数组合: {hyperparams} "
                  f"(SmoothL1Loss={best_score:.6f}, "
                  f"MSE={best_mse:.6f}, "
                  f"MAE={best_mae:.6f})")
    
    # Final report
    if best_combo:
        print(f"\n🏆 最佳参数组合: {best_combo}")
        print(f"  性能指标:")
        print(f"    SmoothL1Loss: {best_score:.6f}")
        print(f"    MSE: {best_mse:.6f}")
        print(f"    MAE: {best_mae:.6f}")
    else:
        print("\n没有找到最佳参数组合")

if __name__ == "__main__":
    set_seed(42)  # 确保随机采样的可重复性
    main()
    print("\n✅ 网格搜索完成!")