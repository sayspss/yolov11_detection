"""
检查训练结果文件
"""
import pandas as pd
from pathlib import Path

print("="*80)
print("检查训练结果文件")
print("="*80)

# 腐蚀数据集
print("\n【腐蚀数据集】")
print("-"*80)
corrosion_results = {
    '原模型': 'models/runs/train_corrosion/corrosion_base/results.csv',
    '改进1-训练优化': 'models/runs/train_corrosion/corrosion_改进1_训练优化/results.csv',
    '改进2-ADown': 'models/runs/train_corrosion/corrosion_改进2_ADown/results.csv',
    '改进3-P2层': 'models/runs/train_corrosion/corrosion_改进3_P2层/results.csv',
    '改进4-ECA注意力': 'models/runs/train_corrosion/corrosion_改进4_ECA注意力/results.csv',
}

for model_name, csv_path in corrosion_results.items():
    csv_file = Path(csv_path)
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            # 获取最后一行的指标
            last_row = df.iloc[-1]
            map50 = last_row.get('metrics/mAP50(B)', 0)
            map50_95 = last_row.get('metrics/mAP50-95(B)', 0)
            
            print(f"✓ {model_name:<20} | Epochs: {len(df):>3} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}")
        except Exception as e:
            print(f"✗ {model_name:<20} | 读取失败: {e}")
    else:
        print(f"⊘ {model_name:<20} | 文件不存在")

# 裂缝数据集
print("\n【裂缝数据集】")
print("-"*80)
cracks_results = {
    '原模型': 'models/runs/train_cracks/cracks_原模型/results.csv',
    '改进1-训练优化': 'models/runs/train_cracks/cracks_改进1_训练优化/results.csv',
    '改进2-ADown': 'models/runs/train_cracks/cracks_改进2_ADown/results.csv',
    '改进3-P2层': 'models/runs/train_cracks/cracks_改进3_P2层/results.csv',
    '改进4-ECA注意力': 'models/runs/train_cracks/cracks_改进4_ECA注意力/results.csv',
}

for model_name, csv_path in cracks_results.items():
    csv_file = Path(csv_path)
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            # 获取最后一行的指标
            last_row = df.iloc[-1]
            map50 = last_row.get('metrics/mAP50(B)', 0)
            map50_95 = last_row.get('metrics/mAP50-95(B)', 0)
            
            print(f"✓ {model_name:<20} | Epochs: {len(df):>3} | mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}")
        except Exception as e:
            print(f"✗ {model_name:<20} | 读取失败: {e}")
    else:
        print(f"⊘ {model_name:<20} | 文件不存在")

print("\n" + "="*80)
print("检查完成")
print("="*80)
