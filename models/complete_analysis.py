"""
完整的目标检测分析脚本
包含：样本集分布图、训练/测试损失对比、P-R曲线、消融实验、结果可视化
"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from collections import Counter
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def find_latest_experiment(base_name, train_dir='runs/train'):
    """查找最新的实验结果"""
    train_dir = Path(train_dir)
    
    # 查找所有匹配的文件夹
    matching_dirs = []
    
    # 精确匹配
    exact_match = train_dir / base_name
    if exact_match.exists():
        matching_dirs.append(exact_match)
    
    # 带数字后缀的匹配
    for d in train_dir.glob(f"{base_name}[0-9]*"):
        matching_dirs.append(d)
    
    if not matching_dirs:
        return None
    
    # 返回最新修改的文件夹
    latest = max(matching_dirs, key=lambda p: p.stat().st_mtime)
    return str(latest)


def auto_detect_results():
    """自动检测最新的训练结果"""
    results = {}
    
    model_names = ['原模型', '原模型_1改进', '原模型_2改进', '原模型_3改进']
    
    for dataset_name in ['corrosion', 'cracks']:
        results[dataset_name] = {}
        
        for idx, model_name in enumerate(model_names):
            base_name = f"{dataset_name}_{model_name}"
            latest_path = find_latest_experiment(base_name)
            
            if latest_path:
                # 转换回显示名称
                display_name = ['原模型', '原模型+1改进', '原模型+2改进', '原模型+3改进'][idx]
                results[dataset_name][display_name] = latest_path
                print(f"✓ 找到: {dataset_name} - {display_name} -> {Path(latest_path).name}")
            else:
                print(f"✗ 未找到: {base_name}")
    
    return results


# 配置
DATASETS = {
    'corrosion': str(PROJECT_ROOT / 'datas' / 'corrosion'),
    'cracks': str(PROJECT_ROOT / 'datas' / 'cracks')
}

# 自动检测训练结果（优先使用）
print("正在自动检测训练结果...")
RESULTS_PATHS = auto_detect_results()

# 如果自动检测失败，使用手动配置
if not any(RESULTS_PATHS.values()):
    print("\n自动检测失败，使用手动配置...")
    RESULTS_PATHS = {
        'corrosion': {
            '原模型': 'runs/train/corrosion_原模型',
            '原模型+1改进': 'runs/train/corrosion_原模型_1改进',
            '原模型+2改进': 'runs/train/corrosion_原模型_2改进',
            '原模型+3改进': 'runs/train/corrosion_原模型_3改进'
        },
        'cracks': {
            '原模型': 'runs/train/cracks_原模型',
            '原模型+1改进': 'runs/train/cracks_原模型_1改进',
            '原模型+2改进': 'runs/train/cracks_原模型_2改进',
            '原模型+3改进': 'runs/train/cracks_原模型_3改进'
        }
    }


def count_images_in_dataset(dataset_path):
    """统计数据集中的图片数量"""
    dataset_path = Path(dataset_path)
    counts = {}
    
    # 对于corrosion数据集
    if (dataset_path / 'images').exists():
        for split in ['train', 'valid', 'test']:
            img_dir = dataset_path / 'images' / split
            if img_dir.exists():
                counts[split] = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    # 对于cracks数据集
    else:
        for split in ['train', 'valid', 'test']:
            img_dir = dataset_path / split / 'images'
            if img_dir.exists():
                counts[split] = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    
    return counts


def plot_dataset_distribution(save_dir='runs/analysis'):
    """1. 绘制样本集分布图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for idx, (dataset_name, dataset_path) in enumerate(DATASETS.items()):
        counts = count_images_in_dataset(dataset_path)
        
        splits = list(counts.keys())
        values = list(counts.values())
        
        ax = axes[idx]
        bars = ax.bar(splits, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Dataset Split', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=13, fontweight='bold')
        ax.set_title(f'{dataset_name.upper()} Dataset Distribution', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加总数
        total = sum(values)
        ax.text(0.5, 0.95, f'Total: {total} images', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 样本集分布图已保存: dataset_distribution.png")


def load_training_results(results_paths):
    """加载训练结果"""
    data = {}
    
    for dataset_name, models in results_paths.items():
        data[dataset_name] = {}
        
        for model_name, exp_path in models.items():
            csv_path = Path(exp_path) / 'results.csv'
            
            print(f"尝试加载: {csv_path}")
            print(f"  文件存在: {csv_path.exists()}")
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                data[dataset_name][model_name] = df
                print(f"✓ 加载成功: {dataset_name} - {model_name} (共 {len(df)} 行)")
            else:
                print(f"✗ 未找到: {csv_path}")
    
    return data


def plot_loss_comparison(data, save_dir='runs/analysis'):
    """2&3. 绘制训练损失和验证损失对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', '--', '-.', ':']
    
    for dataset_name, models_data in data.items():
        if not models_data:
            continue
        
        # 训练损失
        plt.figure(figsize=(12, 7))
        for idx, (model_name, df) in enumerate(models_data.items()):
            if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
                total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
                epochs = range(1, len(total_loss) + 1)
                plt.plot(epochs, total_loss, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
        
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Training Loss', fontsize=13, fontweight='bold')
        plt.title(f'{dataset_name.upper()} - Training Loss Comparison', 
                 fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_train_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练损失图: {dataset_name}_train_loss.png")
        
        # 验证损失
        plt.figure(figsize=(12, 7))
        for idx, (model_name, df) in enumerate(models_data.items()):
            if all(col in df.columns for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
                total_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
                epochs = range(1, len(total_loss) + 1)
                plt.plot(epochs, total_loss, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
        
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Validation Loss', fontsize=13, fontweight='bold')
        plt.title(f'{dataset_name.upper()} - Validation Loss Comparison', 
                 fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_val_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 验证损失图: {dataset_name}_val_loss.png")


def plot_pr_curves(data, save_dir='runs/analysis'):
    """4. 绘制P-R曲线对比 - 使用训练过程中的所有数据点"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for dataset_name, models_data in data.items():
        if not models_data:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 左图：训练过程中的 Precision 变化
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'metrics/precision(B)' in df.columns:
                epochs = range(1, len(df) + 1)
                precision = df['metrics/precision(B)']
                ax1.plot(epochs, precision, label=model_name, 
                        color=colors[idx], linewidth=2.5, marker=markers[idx],
                        markevery=max(1, len(df)//10), markersize=8)
        
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax1.set_title('Precision over Training', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 右图：训练过程中的 Recall 变化
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'metrics/recall(B)' in df.columns:
                epochs = range(1, len(df) + 1)
                recall = df['metrics/recall(B)']
                ax2.plot(epochs, recall, label=model_name, 
                        color=colors[idx], linewidth=2.5, marker=markers[idx],
                        markevery=max(1, len(df)//10), markersize=8)
        
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=13, fontweight='bold')
        ax2.set_title('Recall over Training', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'{dataset_name.upper()} - Precision & Recall Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ P-R曲线图: {dataset_name}_pr_curve.png")
        
        # 额外生成一个 P-R 散点图（最终性能对比）
        plt.figure(figsize=(10, 9))
        
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                last_row = df.iloc[-1]
                precision = last_row['metrics/precision(B)']
                recall = last_row['metrics/recall(B)']
                
                plt.scatter(recall, precision, s=400, color=colors[idx], 
                          marker=markers[idx], zorder=5, edgecolors='black', linewidths=2.5,
                          label=model_name, alpha=0.85)
                
                # 添加数值标注
                offset_x = [20, -80, 20, -80][idx]
                offset_y = [20, 20, -30, -30][idx]
                plt.annotate(f'{model_name}\nP={precision:.3f}, R={recall:.3f}', 
                           xy=(recall, precision), 
                           xytext=(offset_x, offset_y), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.6', 
                           facecolor=colors[idx], alpha=0.2, edgecolor=colors[idx], linewidth=2),
                           arrowprops=dict(arrowstyle='->', color=colors[idx], lw=2))
        
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title(f'{dataset_name.upper()} - Final Performance Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, framealpha=0.95, loc='best', edgecolor='black', shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_pr_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ P-R散点图: {dataset_name}_pr_scatter.png")


def generate_ablation_study(data, save_dir='runs/analysis'):
    """5. 生成消融实验表格"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建消融实验表格
    ablation_data = []
    
    improvements = {
        '原模型': ['Baseline', '-', '-', '-'],
        '原模型+1改进': ['✓', 'ADown', '-', '-'],
        '原模型+2改进': ['✓', 'ADown', 'P2 Layer', '-'],
        '原模型+3改进': ['✓', 'ADown', 'P2 Layer', 'Siamese']
    }
    
    for dataset_name, models_data in data.items():
        for model_name, df in models_data.items():
            if df is not None and len(df) > 0:
                last_row = df.iloc[-1]
                
                map50 = last_row.get('metrics/mAP50(B)', 0)
                map50_95 = last_row.get('metrics/mAP50-95(B)', 0)
                precision = last_row.get('metrics/precision(B)', 0)
                recall = last_row.get('metrics/recall(B)', 0)
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                improvement_list = improvements.get(model_name, ['-', '-', '-', '-'])
                
                ablation_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Baseline': improvement_list[0],
                    'ADown': improvement_list[1],
                    'P2 Layer': improvement_list[2],
                    'Siamese': improvement_list[3],
                    'mAP50': f'{map50:.4f}',
                    'mAP50-95': f'{map50_95:.4f}',
                    'Precision': f'{precision:.4f}',
                    'Recall': f'{recall:.4f}',
                    'F1-Score': f'{f1:.4f}'
                })
    
    # 保存为CSV
    df_ablation = pd.DataFrame(ablation_data)
    csv_file = save_dir / 'ablation_study.csv'
    df_ablation.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 保存为格式化的文本表格
    txt_file = save_dir / 'ablation_study.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("消融实验结果 (Ablation Study Results)\n")
        f.write("="*120 + "\n\n")
        
        for dataset_name in data.keys():
            f.write(f"\n数据集: {dataset_name.upper()}\n")
            f.write("-"*120 + "\n")
            f.write(f"{'Model':<25} {'Baseline':<10} {'ADown':<10} {'P2 Layer':<12} {'Siamese':<10} "
                   f"{'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-"*120 + "\n")
            
            dataset_rows = [row for row in ablation_data if row['Dataset'] == dataset_name]
            for row in dataset_rows:
                f.write(f"{row['Model']:<25} {row['Baseline']:<10} {row['ADown']:<10} "
                       f"{row['P2 Layer']:<12} {row['Siamese']:<10} {row['mAP50']:<10} "
                       f"{row['mAP50-95']:<10} {row['Precision']:<10} {row['Recall']:<10} "
                       f"{row['F1-Score']:<10}\n")
            f.write("\n")
    
    print(f"✓ 消融实验表格已保存:")
    print(f"  - {csv_file}")
    print(f"  - {txt_file}")
    
    # 可视化消融实验
    plot_ablation_visualization(df_ablation, save_dir)


def plot_ablation_visualization(df_ablation, save_dir):
    """可视化消融实验结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for dataset_name in df_ablation['Dataset'].unique():
            dataset_data = df_ablation[df_ablation['Dataset'] == dataset_name]
            values = [float(v) for v in dataset_data[metric]]
            models = dataset_data['Model'].tolist()
            
            x = np.arange(len(models))
            width = 0.35
            offset = -width/2 if dataset_name == 'corrosion' else width/2
            
            ax.bar(x + offset, values, width, label=dataset_name, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ablation_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 消融实验可视化: ablation_visualization.png")


def visualize_detection_results(results_paths, save_dir='runs/analysis'):
    """6. 展示四个模型的检测结果"""
    save_dir = Path(save_dir) / 'detection_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n开始收集检测结果图...")
    
    for dataset_name, models in results_paths.items():
        dataset_dir = save_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        for model_name, exp_path in models.items():
            exp_path = Path(exp_path)
            
            # 查找验证结果图片
            val_batch_path = exp_path / 'val_batch0_pred.jpg'
            if val_batch_path.exists():
                # 复制到结果目录
                import shutil
                dest_path = dataset_dir / f"{model_name.replace('+', '_')}_result.jpg"
                shutil.copy(val_batch_path, dest_path)
                print(f"✓ 复制检测结果: {dest_path.name}")
    
    print(f"\n✓ 检测结果已保存到: {save_dir}")


def generate_summary_report(data, save_dir='runs/analysis'):
    """生成完整的分析报告"""
    save_dir = Path(save_dir)
    
    report_file = save_dir / 'complete_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("目标检测完整分析报告\n")
        f.write("="*100 + "\n\n")
        
        f.write("生成的文件列表:\n")
        f.write("-"*100 + "\n")
        f.write("1. dataset_distribution.png - 样本集分布图\n")
        f.write("2. corrosion_train_loss.png - 腐蚀数据集训练损失对比\n")
        f.write("3. corrosion_val_loss.png - 腐蚀数据集验证损失对比\n")
        f.write("4. cracks_train_loss.png - 裂缝数据集训练损失对比\n")
        f.write("5. cracks_val_loss.png - 裂缝数据集验证损失对比\n")
        f.write("6. corrosion_pr_curve.png - 腐蚀数据集P-R曲线\n")
        f.write("7. cracks_pr_curve.png - 裂缝数据集P-R曲线\n")
        f.write("8. ablation_study.csv - 消融实验数据表\n")
        f.write("9. ablation_study.txt - 消融实验格式化表格\n")
        f.write("10. ablation_visualization.png - 消融实验可视化\n")
        f.write("11. detection_results/ - 四个模型的检测结果图\n")
        f.write("\n")
        
        f.write("\n性能指标汇总:\n")
        f.write("="*100 + "\n")
        
        for dataset_name, models_data in data.items():
            f.write(f"\n数据集: {dataset_name.upper()}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Model':<25} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-"*100 + "\n")
            
            for model_name, df in models_data.items():
                if df is not None and len(df) > 0:
                    last_row = df.iloc[-1]
                    
                    map50 = last_row.get('metrics/mAP50(B)', 0)
                    map50_95 = last_row.get('metrics/mAP50-95(B)', 0)
                    precision = last_row.get('metrics/precision(B)', 0)
                    recall = last_row.get('metrics/recall(B)', 0)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    f.write(f"{model_name:<25} {map50:<12.4f} {map50_95:<12.4f} "
                           f"{precision:<12.4f} {recall:<12.4f} {f1:<12.4f}\n")
            f.write("\n")
    
    print(f"\n✓ 完整分析报告: {report_file}")


if __name__ == '__main__':
    print("="*100)
    print("开始完整的目标检测分析流程")
    print("="*100)
    
    save_dir = Path('runs/analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 样本集分布图
    print("\n[1/6] 生成样本集分布图...")
    plot_dataset_distribution(save_dir)
    
    # 2. 加载训练结果
    print("\n[2/6] 加载训练结果...")
    data = load_training_results(RESULTS_PATHS)
    
    if not any(data.values()):
        print("\n错误: 未找到训练结果，请先运行 train_compare.py 训练模型")
        exit(1)
    
    # 3. 训练和验证损失对比
    print("\n[3/6] 生成损失对比图...")
    plot_loss_comparison(data, save_dir)
    
    # 4. P-R曲线
    print("\n[4/6] 生成P-R曲线...")
    plot_pr_curves(data, save_dir)
    
    # 5. 消融实验
    print("\n[5/6] 生成消融实验表格...")
    generate_ablation_study(data, save_dir)
    
    # 6. 检测结果可视化
    print("\n[6/6] 收集检测结果图...")
    visualize_detection_results(RESULTS_PATHS, save_dir)
    
    # 7. 生成汇总报告
    print("\n生成完整分析报告...")
    generate_summary_report(data, save_dir)
    
    print("\n" + "="*100)
    print("✓ 所有分析完成！")
    print(f"结果保存在: {save_dir.absolute()}")
    print("="*100)
