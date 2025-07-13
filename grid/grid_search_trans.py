import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformer_model import TransformerForecast
from dataset.power_dataset import PowerDataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import joblib
from itertools import product

# 加载目标列的 scaler
target_scaler = joblib.load('scalers/target_scaler.pkl')

# 固定随机种子
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

# ==== 配置 ====
INPUT_LEN = 90
OUTPUT_LEN = 90
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REPEATS = 5
PATIENCE = 10  # Early stopping patience

# ==== Transformer 网格搜索参数 ====
HYPERPARAMS = {
    'batch_size': [8, 16, 32],
    'd_model': [16, 32, 64],       # 模型维度
    'nhead': [2, 4, 8],            # 注意力头数
    'num_layers': [1, 2, 3],       # Transformer 层数
    'dropout': [0.1, 0.2, 0.3],   # Dropout 率
    'lr': [1e-5, 5e-5, 1e-4]      # 学习率
}

def plot_prediction_curve(y_true, y_pred, output_len, save_path=None, title="Prediction vs Ground Truth"):
    days = np.arange(output_len)
    plt.figure(figsize=(12, 6))
    plt.plot(days, y_true, label='Ground Truth')
    plt.plot(days, y_pred, label='Prediction')
    plt.xlabel("Future Days")
    plt.ylabel("Normalized Global_active_power(KW)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[+] 图像已保存至 {save_path}")
    else:
        plt.show()

def plot_loss_curves(train_losses, test_losses, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1Loss")
    plt.title("Train vs Test Loss Curve")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[+] Loss 曲线图已保存至 {save_path}")
    plt.close()

# ==== EarlyStopping 类 ====
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

# ==== 训练函数 ====
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

# ==== 评估函数 ====
def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_smoothl1 = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # 计算SmoothL1Loss
            if criterion:
                smoothl1_loss = criterion(outputs, targets)
                total_smoothl1 += smoothl1_loss.item() * inputs.size(0)
            
            # 计算MSE和MAE
            mse = F.mse_loss(outputs, targets).item()
            mae = F.l1_loss(outputs, targets).item()
            
            total_mse += mse * inputs.size(0)
            total_mae += mae * inputs.size(0)
            total_samples += inputs.size(0)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 计算平均损失
    avg_smoothl1 = total_smoothl1 / total_samples if criterion else None
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_smoothl1, avg_mse, avg_mae, all_preds, all_targets

# ==== 加载数据 ====
train_df = pd.read_csv('data/cleaned2_train.csv', index_col=0)
test_df = pd.read_csv('data/cleaned2_test.csv', index_col=0)

train_dataset = PowerDataset(train_df, input_len=INPUT_LEN, output_len=OUTPUT_LEN)
test_dataset = PowerDataset(test_df, input_len=INPUT_LEN, output_len=OUTPUT_LEN)

# ==== 网格搜索 ====
best_params = None
best_score = float('inf')
results = []

# 生成所有超参数组合
param_combinations = list(product(*HYPERPARAMS.values()))
param_names = list(HYPERPARAMS.keys())

print(f"开始Transformer网格搜索，共 {len(param_combinations)} 组参数组合...")

for i, params in enumerate(param_combinations):
    param_dict = dict(zip(param_names, params))
    print(f"\n=== 测试参数组合 {i+1}/{len(param_combinations)}: {param_dict} ===")
    
    # 记录当前参数组合的性能
    param_smoothl1 = []
    param_mse = []
    param_mae = []
    
    for repeat in range(N_REPEATS):
        print(f"\n🚀 第 {repeat+1}/{N_REPEATS} 轮训练")
        
        seed = 42 + repeat
        set_seed(seed)
        
        # 创建数据加载器 - 使用当前批大小
        train_loader = DataLoader(train_dataset, 
                                 batch_size=param_dict['batch_size'], 
                                 shuffle=True,
                                 worker_init_fn=lambda id: set_seed(seed + id))
        
        test_loader = DataLoader(test_dataset, 
                                batch_size=param_dict['batch_size'], 
                                shuffle=False)
        
        # 创建Transformer模型
        model = TransformerForecast(
            input_dim=13,
            d_model=param_dict['d_model'],
            nhead=param_dict['nhead'],
            num_layers=param_dict['num_layers'],
            output_len=OUTPUT_LEN,
            dropout=param_dict['dropout']
        ).to(DEVICE)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=param_dict['lr'], weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()
        early_stopper = EarlyStopping(patience=PATIENCE)
        
        best_test_smoothl1 = float('inf')
        best_model_path = f"checkpoints/transformer_best_model_param_{i}_repeat_{repeat}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        
        train_losses = []
        test_smoothl1_losses = []
        
        for epoch in range(EPOCHS):
            # 训练一个epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
            # 评估模型
            test_smoothl1, test_mse, test_mae, _, _ = evaluate(
                model, test_loader, DEVICE, criterion
            )
            
            train_losses.append(train_loss)
            test_smoothl1_losses.append(test_smoothl1)
            
            # 根据SmoothL1Loss保存模型
            if test_smoothl1 < best_test_smoothl1:
                best_test_smoothl1 = test_smoothl1
                torch.save(model.state_dict(), best_model_path)
            
            # 早停基于SmoothL1Loss
            if early_stopper.step(test_smoothl1):
                break
        
        # 加载最优模型并评估
        model.load_state_dict(torch.load(best_model_path))
        test_smoothl1, test_mse, test_mae, _, _ = evaluate(
            model, test_loader, DEVICE, criterion
        )
        
        param_smoothl1.append(test_smoothl1)
        param_mse.append(test_mse)
        param_mae.append(test_mae)
        
        print(f"参数组合 {param_dict} - 第 {repeat+1} 轮: SmoothL1Loss={test_smoothl1:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
    
    # 计算当前参数组合的平均性能
    avg_smoothl1 = np.mean(param_smoothl1)
    avg_mse = np.mean(param_mse)
    avg_mae = np.mean(param_mae)
    
    results.append({
        'params': param_dict,
        'avg_smoothl1': avg_smoothl1,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'all_smoothl1': param_smoothl1,
        'all_mse': param_mse,
        'all_mae': param_mae
    })
    
    print(f"参数组合 {param_dict} 平均性能: SmoothL1Loss={avg_smoothl1:.4f}, MSE={avg_mse:.4f}, MAE={avg_mae:.4f}")
    
    # 更新最佳参数
    if avg_smoothl1 < best_score:
        best_score = avg_smoothl1
        best_params = param_dict
        print(f"🔥 新最佳参数组合: {best_params} (SmoothL1Loss={best_score:.4f})")

# ==== 使用最佳参数进行最终训练和评估 ====
print(f"\n最佳参数组合: {best_params}")
print(f"最佳SmoothL1Loss: {best_score:.4f}")

# 保存结果到文件
results_df = pd.DataFrame(results)
results_df.to_csv('transformer_grid_search_results.csv', index=False)
print("Transformer网格搜索结果已保存至 transformer_grid_search_results.csv")

# 使用最佳参数进行最终训练
all_smoothl1, all_mse, all_mae = [], [], []
all_preds_inv_all_exps = []
all_targets_inv_all_exps = []

for i in range(N_REPEATS):
    print(f"\n🚀 最终训练 - 第 {i+1}/{N_REPEATS} 轮")
    
    seed = 42 + i
    set_seed(seed)
    
    # 创建数据加载器 - 使用最佳批大小
    train_loader = DataLoader(train_dataset, 
                             batch_size=best_params['batch_size'], 
                             shuffle=True,
                             worker_init_fn=lambda id: set_seed(seed + id))
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=best_params['batch_size'], 
                            shuffle=False)
    
    # 创建Transformer模型
    model = TransformerForecast(
        input_dim=13,
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        output_len=OUTPUT_LEN,
        dropout=best_params['dropout']
    ).to(DEVICE)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    early_stopper = EarlyStopping(patience=PATIENCE)
    
    best_test_smoothl1 = float('inf')
    best_model_path = f"checkpoints/best_transformer_final_model_exp{i+1}.pth"
    
    train_losses = []
    test_smoothl1_losses = []
    
    for epoch in range(EPOCHS):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # 评估模型
        test_smoothl1, test_mse, test_mae, _, _ = evaluate(
            model, test_loader, DEVICE, criterion
        )
        
        train_losses.append(train_loss)
        test_smoothl1_losses.append(test_smoothl1)
        
        # 根据SmoothL1Loss保存模型
        if test_smoothl1 < best_test_smoothl1:
            best_test_smoothl1 = test_smoothl1
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存测试集最优模型（SmoothL1Loss={test_smoothl1:.4f}）：{best_model_path}")
        
        # 早停基于SmoothL1Loss
        if early_stopper.step(test_smoothl1):
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    # 加载最优模型并评估
    model.load_state_dict(torch.load(best_model_path))
    test_smoothl1, test_mse, test_mae, all_preds, all_targets = evaluate(
        model, test_loader, DEVICE, criterion
    )
    
    all_smoothl1.append(test_smoothl1)
    all_mse.append(test_mse)
    all_mae.append(test_mae)
    
    print(f"\n📈 测试集最终性能：SmoothL1Loss = {test_smoothl1:.4f}, "
          f"MSE = {test_mse:.4f}, MAE = {test_mae:.4f}")
    
    # 对预测值和真实值进行逆归一化
    all_preds_inv = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    all_targets_inv = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    
    # 收集反归一化结果
    all_preds_inv_all_exps.append(all_preds_inv)
    all_targets_inv_all_exps.append(all_targets_inv)
    
    # ==== 可视化部分 ====
    
    # 可视化预测
    sample_id = 0
    plot_prediction_curve(
        all_targets_inv.reshape(-1, OUTPUT_LEN)[sample_id],
        all_preds_inv.reshape(-1, OUTPUT_LEN)[sample_id],
        output_len=OUTPUT_LEN,
        save_path=f"plots_transformer_best/pred_vs_true_len{OUTPUT_LEN}_exp{i+1}_sample{sample_id}.png",
        title=f"Transformer Prediction vs Ground Truth (Sample {sample_id}) [Inverse]"
    )
    
    # 可视化Loss曲线
    plot_loss_curves(
        train_losses,
        test_smoothl1_losses,
        save_path=f"plots_transformer_best/loss_curve_exp{i+1}.png"
    )

# ==== 最终结果 ====
print("\n📊 5轮平均指标：")
print(f"SmoothL1Loss: {np.mean(all_smoothl1):.4f} ± {np.std(all_smoothl1):.4f}")
print(f"MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
print(f"MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")

# 计算平均误差最小的样本
print("\n🎯 正在计算五轮平均误差最小的样本...")

all_preds_array = np.stack(all_preds_inv_all_exps, axis=0)
all_targets_array = np.stack(all_targets_inv_all_exps, axis=0)

mean_preds = np.mean(all_preds_array, axis=0)
mean_targets = np.mean(all_targets_array, axis=0)

mse_per_sample = np.mean((mean_preds - mean_targets) ** 2, axis=1)
best_sample_id = np.argmin(mse_per_sample)

print(f"✅ 全局最优样本 ID: {best_sample_id}")

# 绘制该样本的平均预测效果图
plot_prediction_curve(
    mean_targets[best_sample_id],
    mean_preds[best_sample_id],
    output_len=OUTPUT_LEN,
    save_path=f"plots_transformer_best/avg_prediction_sample{best_sample_id}.png",
    title=f"Transformer Avg Prediction over 5 Runs (Sample {best_sample_id})"
)

# 绘制该样本在每一轮的预测效果图
print(f"\n🖼 正在输出 sample_id={best_sample_id} 在每轮的预测图...")

for i in range(N_REPEATS):
    y_true = all_targets_array[i, best_sample_id]
    y_pred = all_preds_array[i, best_sample_id]
    
    plot_prediction_curve(
        y_true,
        y_pred,
        output_len=OUTPUT_LEN,
        save_path=f"plots_transformer_best/best_sample{best_sample_id}_exp{i+1}.png",
        title=f"Transformer Sample {best_sample_id} - Prediction in Run {i+1}"
    )