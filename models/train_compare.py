"""
训练对比脚本：对两个数据集训练四个模型并生成对比图
- 4张训练/验证损失对比图（每个数据集的train loss和val loss各一张）
- 2张P-R曲线对比图（每个数据集一张）
"""
import warnings
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置中文字体（Windows系统）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用默认字体
    pass

from ultralytics import YOLO

# 获取项目根目录的绝对路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 配置参数 - 使用相对路径
DATASETS = {
    'corrosion': str(PROJECT_ROOT / 'datas' / 'corrosion' / 'data.yaml'),
    'cracks': str(PROJECT_ROOT / 'datas' / 'cracks' / 'data.yaml')
}

MODELS = {
    '原模型': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11.yaml'),
    '原模型+1改进': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-ADown.yaml'),
    '原模型+2改进': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-p2-adown.yaml'),
    '原模型+3改进': str(PROJECT_ROOT / 'models' / '模型配置' / 'yolo11-p2-adown-siam.yaml')
}

TRAIN_PARAMS = {
    'imgsz': 640,
    'epochs': 100,
    'batch': 8,
    'lr0': 0.002,
    'lrf': 0.25,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'close_mosaic': 10,
    'workers': 4,
    'device': '0',
    'optimizer': 'SGD',
    'pretrained': False,
    'plots': False,  # 关闭训练过程中的图片和图表生成，减少无用文件
    'project': 'runs/train'
}


def train_all_models():
    """训练所有模型组合"""
    results = {}
    
    for dataset_name, dataset_path in DATASETS.items():
        results[dataset_name] = {}
        print(f"\n{'='*60}")
        print(f"开始训练数据集: {dataset_name}")
        print(f"{'='*60}\n")
        
        for model_name, model_config in MODELS.items():
            print(f"\n训练模型: {model_name}")
            exp_name = f"{dataset_name}_{model_name.replace('+', '_')}"
            
            try:
                # 加载模型，可以选择: 'n', 's', 'm', 'l', 'x'
                model = YOLO(model_config)
                # 如果想用更大的模型，修改这里: 'n'->nano, 's'->small, 'm'->medium
                # model.model.yaml['scale'] = 's'  # 使用small规模
                train_results = model.train(
                    data=dataset_path,
                    name=exp_name,
                    exist_ok=True,  # 允许覆盖已存在的实验
                    **TRAIN_PARAMS
                )
                
                # 保存训练结果路径
                results[dataset_name][model_name] = {
                    'exp_path': Path(TRAIN_PARAMS['project']) / exp_name,
                    'best_weights': Path(TRAIN_PARAMS['project']) / exp_name / 'weights' / 'best.pt'
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
                results[dataset_name][model_name] = None
                
                # 即使失败也清理内存
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    return results


def load_training_results(results):
    """加载训练结果数据"""
    training_data = {}
    
    for dataset_name, models in results.items():
        training_data[dataset_name] = {}
        
        for model_name, result_info in models.items():
            if result_info is None:
                continue
                
            exp_path = result_info['exp_path']
            results_csv = exp_path / 'results.csv'
            
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                # 去除列名中的空格
                df.columns = df.columns.str.strip()
                training_data[dataset_name][model_name] = df
            else:
                print(f"警告: 未找到 {results_csv}")
    
    return training_data


def plot_loss_comparison(training_data, save_dir='runs/comparison'):
    """绘制损失对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for dataset_name, models_data in training_data.items():
        if not models_data:
            continue
        
        # 训练损失对比图
        plt.figure(figsize=(12, 6))
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns and 'train/dfl_loss' in df.columns:
                # 计算总训练损失
                total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
                epochs = range(1, len(total_loss) + 1)
                plt.plot(epochs, total_loss, label=model_name, color=colors[idx], linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title(f'{dataset_name} - 训练损失对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_train_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存训练损失图: {dataset_name}_train_loss.png")
        
        # 验证损失对比图
        plt.figure(figsize=(12, 6))
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'val/box_loss' in df.columns and 'val/cls_loss' in df.columns and 'val/dfl_loss' in df.columns:
                # 计算总验证损失
                total_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
                epochs = range(1, len(total_loss) + 1)
                plt.plot(epochs, total_loss, label=model_name, color=colors[idx], linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title(f'{dataset_name} - 验证损失对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_val_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存验证损失图: {dataset_name}_val_loss.png")


def plot_pr_curves(results, save_dir='runs/comparison'):
    """绘制P-R曲线对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for dataset_name, models in results.items():
        plt.figure(figsize=(10, 8))
        
        for idx, (model_name, result_info) in enumerate(models.items()):
            if result_info is None:
                continue
            
            # 查找PR曲线数据
            exp_path = result_info['exp_path']
            pr_curve_file = exp_path / 'PR_curve.png'
            
            # 尝试从results.csv读取precision和recall
            results_csv = exp_path / 'results.csv'
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                
                # 获取最后一个epoch的precision和recall
                if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                    precision = df['metrics/precision(B)'].iloc[-1]
                    recall = df['metrics/recall(B)'].iloc[-1]
                    
                    # 绘制点
                    plt.scatter(recall, precision, s=100, color=colors[idx], 
                              label=f'{model_name} (P={precision:.3f}, R={recall:.3f})', 
                              marker='o', zorder=5)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{dataset_name} - P-R曲线对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存P-R曲线图: {dataset_name}_pr_curve.png")


def generate_summary_report(training_data, results, save_dir='runs/comparison'):
    """生成训练结果汇总报告"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("训练结果汇总报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for dataset_name, models_data in training_data.items():
        report_lines.append(f"\n数据集: {dataset_name}")
        report_lines.append("-" * 80)
        report_lines.append(f"{'模型':<20} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<10}")
        report_lines.append("-" * 80)
        
        for model_name, df in models_data.items():
            if df is not None and len(df) > 0:
                # 获取最后一个epoch的指标
                last_row = df.iloc[-1]
                
                map50 = last_row.get('metrics/mAP50(B)', 0)
                map50_95 = last_row.get('metrics/mAP50-95(B)', 0)
                precision = last_row.get('metrics/precision(B)', 0)
                recall = last_row.get('metrics/recall(B)', 0)
                
                report_lines.append(f"{model_name:<20} {map50:<10.4f} {map50_95:<12.4f} {precision:<12.4f} {recall:<10.4f}")
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # 保存到文件
    report_file = save_dir / 'training_summary.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ 汇总报告已保存到: {report_file}")


if __name__ == '__main__':
    print("开始训练和对比流程...")
    print(f"数据集: {list(DATASETS.keys())}")
    print(f"模型: {list(MODELS.keys())}")
    print(f"训练参数: epochs={TRAIN_PARAMS['epochs']}, batch={TRAIN_PARAMS['batch']}")
    
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
    print("所有训练和可视化完成！")
    print("结果保存在: runs/comparison/")
    print("="*60)
