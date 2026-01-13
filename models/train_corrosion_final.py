"""
腐蚀数据集训练脚本 - 完成剩余训练
1. 改进3-P2层: 从150 epochs继续训练到300 epochs
2. 改进4-ECA注意力: 训练完整的300 epochs
与 train_corrosion_optimized.py 完全相同的参数和画图
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

# 只训练改进3和改进4
MODELS = {
    '改进3-P2层': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-p2-adown.yaml'),
    '改进4-ECA注意力': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-p2-adown-eca.yaml')
}

# 与 train_corrosion_optimized.py 完全相同的优化训练参数
IMPROVED_PARAMS = {
    'imgsz': 640,
    'epochs': 300,  # 训练到300 epochs
    'batch': 16,
    'patience': 20,
    'lr0': 0.001,
    'lrf': 0.1,
    'cos_lr': True,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'close_mosaic': 10,
    'workers': 8,
    'device': '3',
    'optimizer': 'AdamW',
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


def train_remaining_models():
    """训练剩余的模型"""
    results = {}
    
    print(f"\n{'='*60}")
    print(f"完成腐蚀数据集训练 - 改进3和改进4")
    print(f"{'='*60}\n")
    print("训练计划:")
    print("  改进3-P2层: 从150 epochs继续训练到300 epochs")
    print("  改进4-ECA注意力: 训练完整的300 epochs")
    print(f"{'='*60}\n")
    
    for model_name, model_config in MODELS.items():
        print(f"\n训练模型: {model_name}")
        exp_name = f"corrosion_{model_name.replace('+', '_').replace('-', '_')}"
        
        train_params = IMPROVED_PARAMS.copy()
        print("  使用优化训练参数（与train_corrosion_optimized.py相同）")
        print(f"  epochs={train_params['epochs']}, batch={train_params['batch']}")
        
        # 检查是否已有训练结果
        exp_path = Path(train_params['project']) / exp_name
        last_weights = exp_path / 'weights' / 'last.pt'
        
        try:
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 如果有last.pt，从断点继续训练
            if last_weights.exists():
                print(f"  ✓ 检测到已有训练结果: {last_weights}")
                print(f"  ✓ 将从断点继续训练到300 epochs")
                model = YOLO(last_weights)  # 直接加载last.pt
                train_results = model.train(
                    data=DATASET,
                    resume=True,  # 继续训练
                    **train_params
                )
            else:
                print(f"  ⊘ 未找到已有训练结果，从头开始训练")
                model = YOLO(model_config)
                train_results = model.train(
                    data=DATASET,
                    name=exp_name,
                    exist_ok=True,
                    **train_params
                )
            
            # 保存训练结果路径
            results[model_name] = {
                'exp_path': exp_path,
                'best_weights': exp_path / 'weights' / 'best.pt'
            }
            
            print(f"✓ {model_name} 训练完成")
            
            # 清理内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"✗ {model_name} 训练失败: {str(e)}")
            results[model_name] = None
            
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


def load_all_results():
    """加载所有5个模型的训练结果"""
    all_models = {
        '原模型': 'corrosion_base',
        '改进1-训练优化': 'corrosion_改进1_训练优化',
        '改进2-ADown': 'corrosion_改进2_ADown',
        '改进3-P2层': 'corrosion_改进3_P2层',
        '改进4-ECA注意力': 'corrosion_改进4_ECA注意力'
    }
    
    training_data = {}
    base_dir = Path('runs/train_corrosion')
    
    for model_name, exp_name in all_models.items():
        results_csv = base_dir / exp_name / 'results.csv'
        
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            training_data[model_name] = df
            print(f"✓ 加载 {model_name} 结果")
        else:
            print(f"⊘ 未找到 {model_name} 结果")
    
    return training_data


def plot_loss_comparison(training_data, save_dir='runs/comparison_corrosion'):
    """绘制损失对比图 - 与train_corrosion_optimized.py完全相同"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    if not training_data:
        return
    
    # 训练损失对比图
    plt.figure(figsize=(12, 6))
    for idx, (model_name, df) in enumerate(training_data.items()):
        if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns and 'train/dfl_loss' in df.columns:
            total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            epochs = range(1, len(total_loss) + 1)
            plt.plot(epochs, total_loss, label=model_name, color=colors[idx % len(colors)], linewidth=2)
    
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
            plt.plot(epochs, total_loss, label=model_name, color=colors[idx % len(colors)], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('腐蚀数据集 - 验证损失对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'corrosion_val_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存验证损失图: corrosion_val_loss.png")


def plot_pr_curves(training_data, save_dir='runs/comparison_corrosion'):
    """绘制P-R曲线对比图 - 与train_corrosion_optimized.py完全相同"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plt.figure(figsize=(10, 8))
    
    for idx, (model_name, df) in enumerate(training_data.items()):
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            precision = df['metrics/precision(B)'].iloc[-1]
            recall = df['metrics/recall(B)'].iloc[-1]
            
            plt.scatter(recall, precision, s=100, color=colors[idx % len(colors)], 
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


def generate_summary_report(training_data, save_dir='runs/comparison_corrosion'):
    """生成训练结果汇总报告 - 与train_corrosion_optimized.py完全相同"""
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
    print("腐蚀数据集训练 - 完成剩余训练")
    print("="*80)
    print(f"\n数据集规模: 训练4177张 | 验证782张 | 测试259张")
    print("\n训练计划:")
    print("  改进3-P2层: 从150 epochs继续训练到300 epochs")
    print("  改进4-ECA注意力: 训练完整的300 epochs")
    print(f"\n训练参数: epochs={IMPROVED_PARAMS['epochs']}, batch={IMPROVED_PARAMS['batch']}")
    print("="*80)
    
    # 1. 训练剩余模型
    results = train_remaining_models()
    
    # 2. 加载所有5个模型的结果
    print("\n" + "="*60)
    print("加载所有训练结果")
    print("="*60)
    training_data = load_all_results()
    
    # 3. 绘制损失对比图
    if len(training_data) >= 3:
        print("\n生成对比图...")
        plot_loss_comparison(training_data)
        
        # 4. 绘制P-R曲线
        plot_pr_curves(training_data)
        
        # 5. 生成汇总报告
        generate_summary_report(training_data)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("结果保存在: runs/train_corrosion/")
    print("对比图保存在: runs/comparison_corrosion/")
    print("="*60)
