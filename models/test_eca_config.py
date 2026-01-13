"""
测试ECA配置文件是否能正常加载
"""
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

print("="*60)
print("测试ECA配置文件")
print("="*60)

# 测试配置文件
configs = [
    ('yolo11.yaml', '原模型'),
    ('yolo11-ADown.yaml', 'ADown模型'),
    ('yolo11-p2-adown.yaml', 'P2+ADown模型'),
    ('yolo11-p2-adown-eca.yaml', 'P2+ADown+ECA模型'),
]

for config_file, name in configs:
    config_path = PROJECT_ROOT / 'models' / '模型配置' / config_file
    print(f"\n测试 {name}: {config_file}")
    
    try:
        model = YOLO(str(config_path))
        print(f"  ✓ 配置加载成功")
        print(f"  模型参数: {sum(p.numel() for p in model.model.parameters()):,}")
        del model
    except Exception as e:
        print(f"  ✗ 配置加载失败: {str(e)}")

print("\n" + "="*60)
print("测试完成")
print("="*60)
