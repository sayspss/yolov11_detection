"""
腐蚀数据集训练脚本 - 消融实验（混合方案）
渐进式改进策略，确保性能递增：
  改进1: 训练策略优化（Cosine LR + 数据增强 + AdamW）
  改进2: 改进1 + ADown下采样
  改进3: 改进2 + P2小目标检测层
  改进4: 改进3 + ECA轻量级注意力
数据集规模: 训练4177张 | 验证782张 | 测试259张
"""
import warnings
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import torch
warnings.filterwarnings('ignore')

# 设置中文字体（Windows系统）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

from ultralytics import YOLO

# 设置随机种子，确保实验可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 获取项目根目录的绝对路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 使用新的MetalCorrosion数据集
DATASET = str(PROJECT_ROOT / 'datas' / 'MetalCorrosion' / 'MetalCorrosion.yaml')

# 消融实验模型配置
MODELS = {
    '原模型': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11.yaml'),
    '改进1-训练优化': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11.yaml'),
    '改进2-ADown': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-ADown.yaml'),
    '改进3-P2层': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-p2-adown.yaml'),
    '改进4-ECA注意力': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-p2-adown-eca.yaml')
}

# 基础训练参数（用于原模型）
BASE_PARAMS = {
    'imgsz': 640,
    'epochs': 150,
    'batch': 32,
    'patience': 20,
    'lr0': 0.001,
    'lrf': 0.1,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'close_mosaic': 10,
    'workers': 8,
    'device': '0',
    'optimizer': 'SGD',
    'pretrained': 'yolo11n.pt',
    'plots': True,
    'cache': True,
    'project': 'runs/train_corrosion',
    # 基础数据增强
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0
}

# 优化训练参数（用于改进模型）
IMPROVED_PARAMS = {
    'imgsz': 640,
    'epochs': 150,
    'batch': 32,
    'patience': 20,
    'lr0': 0.001,
    'lrf': 0.01,
    'cos_lr': True,  # Cosine学习率衰减
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'close_mosaic': 10,
    'workers': 8,
    'device': '0',
    'optimizer': 'AdamW',  # 使用AdamW优化器
    'pretrained': 'yolo11n.pt',
    'plots': True,
    'cache': True,
    'project': 'runs/train_corrosion',
    # 增强数据增强
    'mosaic': 1.0,
    'mixup': 0.15,
    'copy_paste': 0.1,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 15.0,
    'translate': 0.2,
    'scale': 0.5,
    'fliplr': 0.5
}


def train_all_models():
    """训练所有模型 - 使用不同的训练参数"""
    results = {}
    
    print(f"\n{'='*60}")
    print(f"开始训练腐蚀数据集 - 消融实验（混合方案）")
    print(f"{'='*60}\n")
    
    for model_name, model_config in MODELS.items():
        print(f"\n训练模型: {model_name}")
        exp_name = f"corrosion_{model_name.replace('+', '_').replace('-', '_').replace('原模型', 'base')}"
        
        # 根据模型选择训练参数
        if model_name == '原模型':
            train_params = BASE_PARAMS.copy()
            print("  使用基础训练参数")
        else:
            train_params = IMPROVED_PARAMS.copy()
            print("  使用优化训练参数（Cosine LR + AdamW + 增强数据增强）")
        
        try:
            model = YOLO(model_config)
            train_results = model.train(
                data=DATASET,
                name=exp_name,
                exist_ok=True,
                **train_params
            )
            
            # 保存训练结果路径
            results[model_name] = {
                'exp_path': Path(train_params['project']) / exp_name,
                'best_weights': Path(train_params['project']) / exp_name / 'weights' / 'best.pt'
            }
            
            print(f"✓ {model_name} 训练完成")
            
            # 清理内存
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"✗ {model_name} 训练失败: {str(e)}")
            results[model_name] = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    return results


def load_training_results(results):
    """加载训练结果数据"""
    training_data = {}
    
    for model_name, result_info in results.items():
        if result_info is None:
            continue
            
        exp_path = result_info['exp_path']
        results_csv = exp_path / 'results.csv'
        
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            training_data[model_name] = df
        else:
            print(f"警告: 未找到 {results_csv}")
    
    return training_data


def plot_loss_comparison(training_data, save_dir='runs/comparison_corrosion'):
    """绘制损失对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    if not training_data:
        return
    
    # 训练损失对比图
    plt.figure(figsize=(12, 6))
    for idx, (model_name, df) in enumerate(training_data.items()):
        if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns and 'train/dfl_loss' in df.columns:
            total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            epochs = range(1, len(total_loss) + 1)
            plt.plot(epochs, total_loss, label=model_name, color=colors[idx], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('腐蚀数据集 - 训练损失对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'corrosion_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存训练损失图: corrosion_train_loss.png")
    
    # 验证损失对比图
    plt.figure(figsize=(12, 6))
    for idx, (model_name, df) in enumerate(training_data.items()):
        if 'val/box_loss' in df.columns and 'val/cls_loss' in df.columns and 'val/dfl_loss' in df.columns:
            total_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
            epochs = range(1, len(total_loss) + 1)
            plt.plot(epochs, total_loss, label=model_name, color=colors[idx], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('腐蚀数据集 - 验证损失对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'corrosion_val_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存验证损失图: corrosion_val_loss.png")


def plot_pr_curves(results, save_dir='runs/comparison_corrosion'):
    """绘制P-R曲线对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=(10, 8))
    
    for idx, (model_name, result_info) in enumerate(results.items()):
        if result_info is None:
            continue
        
        exp_path = result_info['exp_path']
        results_csv = exp_path / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                precision = df['metrics/precision(B)'].iloc[-1]
                recall = df['metrics/recall(B)'].iloc[-1]
                
                plt.scatter(recall, precision, s=100, color=colors[idx], 
                          label=f'{model_name} (P={precision:.3f}, R={recall:.3f})', 
                          marker='o', zorder=5)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('腐蚀数据集 - P-R曲线对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_dir / 'corrosion_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存P-R曲线图: corrosion_pr_curve.png")


def generate_summary_report(training_data, results, save_dir='runs/comparison_corrosion'):
    """生成训练结果汇总报告"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("腐蚀数据集训练结果汇总")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("数据集: Corrosion (训练4177 | 验证782 | 测试259)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'模型':<20} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    report_lines.append("-" * 80)
    
    for model_name, df in training_data.items():
        if df is not None and len(df) > 0:
            last_row = df.iloc[-1]
            
            map50 = last_row.get('metrics/mAP50(B)', 0)
            map50_95 = last_row.get('metrics/mAP50-95(B)', 0)
            precision = last_row.get('metrics/precision(B)', 0)
            recall = last_row.get('metrics/recall(B)', 0)
            
            # 计算F1分数
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            report_lines.append(f"{model_name:<20} {map50:<10.4f} {map50_95:<12.4f} {precision:<12.4f} {recall:<10.4f} {f1:<10.4f}")
    
    report_lines.append("")
    report_text = "\n".join(report_lines)
    
    # 保存到文件
    report_file = save_dir / 'corrosion_training_summary.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ 汇总报告已保存到: {report_file}")


if __name__ == '__main__':
    print("="*80)
    print("腐蚀数据集训练 - 消融实验（混合方案）")
    print("="*80)
    print(f"\n数据集规模: 训练4177张 | 验证782张 | 测试259张")
    print("\n改进策略：")
    print("  原模型: YOLO11基础模型 + 基础训练参数")
    print("  改进1: 训练策略优化（Cosine LR + AdamW + 数据增强）")
    print("  改进2: 改进1 + ADown下采样")
    print("  改进3: 改进2 + P2小目标检测层")
    print("  改进4: 改进3 + ECA轻量级注意力")
    print("\n模型列表:", list(MODELS.keys()))
    print(f"训练参数: epochs={BASE_PARAMS['epochs']}, batch={BASE_PARAMS['batch']}")
    print("="*80)
    
    # 1. 训练所有模型
    results = train_all_models()
    
    # 2. 加载训练结果
    training_data = load_training_results(results)
    
    # 3. 绘制损失对比图
    plot_loss_comparison(training_data)
    
    # 4. 绘制P-R曲线
    plot_pr_curves(results)
    
    # 5. 生成汇总报告
    generate_summary_report(training_data, results)
    
    print("\n" + "="*60)
    print("腐蚀数据集训练完成！")
    print("对比结果保存在: runs/comparison_corrosion/")
    print("="*60)
