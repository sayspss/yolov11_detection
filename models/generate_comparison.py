"""
生成对比图脚本 - 基于已训练好的模型结果
从 runs/train_corrosion 和 runs/train_cracks 读取结果，生成对比图
"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 获取项目根目录（models的上一级目录）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 配置训练结果路径 - 指向项目根目录下的 runs 文件夹（只包含改进1-4）
RESULTS_PATHS = {
    'corrosion': {
        '改进1-训练优化': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进1_训练优化',
        '改进2-ADown': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进2_ADown',
        '改进3-P2层': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进3_P2层',
        '改进4-ECA注意力': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进4_ECA注意力'
    },
    'cracks': {
        '改进1-训练优化': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进1_训练优化',
        '改进2-ADown': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进2_ADown',
        '改进3-P2层': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进3_P2层',
        '改进4-ECA注意力': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进4_ECA注意力'
    }
}


def load_training_results(results_paths):
    """加载训练结果数据"""
    data = {}
    
    for dataset_name, models in results_paths.items():
        data[dataset_name] = {}
        
        for model_name, exp_path in models.items():
            csv_path = Path(exp_path) / 'results.csv'
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                data[dataset_name][model_name] = df
                print(f"✓ 加载成功: {dataset_name} - {model_name} (共 {len(df)} 轮)")
            else:
                print(f"✗ 未找到: {csv_path}")
    
    return data


def plot_loss_comparison(data, save_dir=None):
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'runs' / 'comparison'
    """绘制损失对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':']
    
    for dataset_name, models_data in data.items():
        if not models_data:
            continue
        
        # 训练损失对比图
        plt.figure(figsize=(14, 7))
        for idx, (model_name, df) in enumerate(models_data.items()):
            if all(col in df.columns for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']):
                total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
                epochs = range(1, len(total_loss) + 1)
                plt.plot(epochs, total_loss, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
        
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Training Loss', fontsize=13, fontweight='bold')
        plt.title(f'{dataset_name.upper()} - 训练损失对比', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_train_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存训练损失图: {dataset_name}_train_loss.png")
        
        # 验证损失对比图
        plt.figure(figsize=(14, 7))
        for idx, (model_name, df) in enumerate(models_data.items()):
            if all(col in df.columns for col in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']):
                total_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
                epochs = range(1, len(total_loss) + 1)
                plt.plot(epochs, total_loss, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
        
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Validation Loss', fontsize=13, fontweight='bold')
        plt.title(f'{dataset_name.upper()} - 验证损失对比', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_val_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存验证损失图: {dataset_name}_val_loss.png")


def plot_pr_curves(data, save_dir=None):
    """绘制P-R曲线对比图 - 显示训练过程中的完整曲线"""
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'runs' / 'comparison'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    for dataset_name, models_data in data.items():
        if not models_data:
            continue
        
        # 绘制完整的 P-R 曲线（训练过程中的变化）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左图：Precision 随训练变化
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'metrics/precision(B)' in df.columns:
                epochs = range(1, len(df) + 1)
                precision = df['metrics/precision(B)']
                ax1.plot(epochs, precision, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx],
                        marker=markers[idx], markevery=max(1, len(df)//15), markersize=6)
        
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax1.set_title('Precision 变化曲线', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1.05])
        
        # 右图：Recall 随训练变化
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'metrics/recall(B)' in df.columns:
                epochs = range(1, len(df) + 1)
                recall = df['metrics/recall(B)']
                ax2.plot(epochs, recall, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx],
                        marker=markers[idx], markevery=max(1, len(df)//15), markersize=6)
        
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=13, fontweight='bold')
        ax2.set_title('Recall 变化曲线', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1.05])
        
        plt.suptitle(f'{dataset_name.upper()} - Precision & Recall 对比', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存 Precision & Recall 曲线图: {dataset_name}_precision_recall_curves.png")
        
        # 绘制真正的 P-R 曲线（Recall 为 X 轴，Precision 为 Y 轴）
        plt.figure(figsize=(12, 10))
        
        for idx, (model_name, df) in enumerate(models_data.items()):
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                precision = df['metrics/precision(B)'].values
                recall = df['metrics/recall(B)'].values
                
                # 绘制完整的训练轨迹曲线
                plt.plot(recall, precision, label=model_name, 
                        color=colors[idx], linewidth=2.5, linestyle=line_styles[idx],
                        marker=markers[idx], markevery=max(1, len(df)//15), markersize=8, alpha=0.7)
                
                # 标记最终点
                final_precision = precision[-1]
                final_recall = recall[-1]
                plt.scatter(final_recall, final_precision, s=300, color=colors[idx], 
                          marker=markers[idx], zorder=10, edgecolors='black', linewidths=2.5, alpha=1.0)
                
                # 添加最终点的数值标注
                offset_positions = [(15, 15), (-80, 15), (15, -30), (-80, -30)]
                offset_x, offset_y = offset_positions[idx]
                plt.annotate(f'P={final_precision:.3f}\nR={final_recall:.3f}', 
                           xy=(final_recall, final_precision), 
                           xytext=(offset_x, offset_y), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=colors[idx], alpha=0.3, edgecolor=colors[idx], linewidth=1.5),
                           arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5))
        
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title(f'{dataset_name.upper()} - P-R 曲线对比（训练轨迹）', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, framealpha=0.95, loc='best', edgecolor='black', shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{dataset_name}_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存 P-R 曲线图: {dataset_name}_pr_curve.png")


def generate_summary_report(data, save_dir=None):
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'runs' / 'comparison'
    """生成训练结果汇总报告"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("训练结果汇总报告")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    for dataset_name, models_data in data.items():
        epochs_info = "300轮" if dataset_name == 'corrosion' else "200轮"
        report_lines.append(f"\n数据集: {dataset_name.upper()} ({epochs_info})")
        report_lines.append("-" * 100)
        report_lines.append(f"{'模型':<25} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        report_lines.append("-" * 100)
        
        for model_name, df in models_data.items():
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
                
                report_lines.append(f"{model_name:<25} {map50:<12.4f} {map50_95:<12.4f} "
                                  f"{precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # 保存到文件
    report_file = save_dir / 'training_summary.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ 汇总报告已保存到: {report_file}")


if __name__ == '__main__':
    print("=" * 100)
    print("生成对比图 - 基于已训练好的模型")
    print("=" * 100)
    print(f"\n项目根目录: {PROJECT_ROOT}")
    print("\n数据源:")
    print(f"  - 腐蚀数据集: {PROJECT_ROOT / 'runs' / 'train_corrosion'} (300轮)")
    print(f"  - 裂缝数据集: {PROJECT_ROOT / 'runs' / 'train_cracks'} (200轮)")
    print(f"\n输出目录: {PROJECT_ROOT / 'runs' / 'comparison'}")
    print("=" * 100)
    
    # 1. 加载训练结果
    print("\n[1/3] 加载训练结果...")
    data = load_training_results(RESULTS_PATHS)
    
    if not any(data.values()):
        print("\n错误: 未找到训练结果CSV文件")
        exit(1)
    
    # 2. 绘制损失对比图
    print("\n[2/3] 生成损失对比图...")
    plot_loss_comparison(data)
    
    # 3. 绘制P-R曲线
    print("\n[3/3] 生成P-R曲线...")
    plot_pr_curves(data)
    
    # 4. 生成汇总报告
    print("\n生成汇总报告...")
    generate_summary_report(data)
    
    print("\n" + "=" * 100)
    print("✓ 所有对比图生成完成！")
    print(f"结果保存在: {PROJECT_ROOT / 'runs' / 'comparison'}")
    print("=" * 100)
