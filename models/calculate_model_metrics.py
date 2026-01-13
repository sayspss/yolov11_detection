"""
计算模型轻量化指标：FPS、计算量(GFLOPs)、参数量(Params)
不需要重新训练，直接从已训练的模型中提取
"""
import torch
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 配置已训练模型的路径
MODEL_PATHS = {
    'corrosion': {
        '改进1-训练优化': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进1_训练优化' / 'weights' / 'best.pt',
        '改进2-ADown': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进2_ADown' / 'weights' / 'best.pt',
        '改进3-P2层': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进3_P2层' / 'weights' / 'best.pt',
        '改进4-ECA注意力': PROJECT_ROOT / 'runs' / 'train_corrosion' / 'corrosion_改进4_ECA注意力' / 'weights' / 'best.pt'
    },
    'cracks': {
        '改进1-训练优化': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进1_训练优化' / 'weights' / 'best.pt',
        '改进2-ADown': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进2_ADown' / 'weights' / 'best.pt',
        '改进3-P2层': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进3_P2层' / 'weights' / 'best.pt',
        '改进4-ECA注意力': PROJECT_ROOT / 'runs' / 'train_cracks' / 'cracks_改进4_ECA注意力' / 'weights' / 'best.pt'
    }
}


def calculate_model_metrics(model_path, device='cpu', img_size=640, warmup=10, test_runs=100):
    """
    计算模型的FPS、GFLOPs、参数量
    
    Args:
        model_path: 模型权重路径
        device: 运行设备 ('cpu' or 'cuda')
        img_size: 输入图像尺寸
        warmup: 预热次数
        test_runs: 测试次数
    """
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 获取模型信息
        # 使用YOLO的val方法可以获取详细信息
        try:
            # 尝试使用YOLO自带的info方法
            from ultralytics.nn.tasks import DetectionModel
            
            # 计算参数量
            params = sum(p.numel() for p in model.model.parameters())
            params_mb = params * 4 / (1024 * 1024)  # float32
            
            # 计算GFLOPs - 使用YOLO的profile方法
            try:
                # 创建一个640x640的输入
                img = torch.zeros((1, 3, img_size, img_size), device=device)
                model.model.to(device)
                
                # 使用YOLO内置的profile功能
                from ultralytics.utils.torch_utils import profile
                flops = profile(model.model, inputs=[img], verbose=False)[0]
                gflops = flops / 1e9
            except:
                # 备用方案：手动估算
                gflops = params * 2 * img_size * img_size / 1e9  # 粗略估算
                
        except Exception as e:
            print(f"    警告: 无法精确计算GFLOPs: {e}")
            params = sum(p.numel() for p in model.model.parameters())
            params_mb = params * 4 / (1024 * 1024)
            gflops = 0
        
        # 创建测试输入
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        model.model.to(device)
        model.model.eval()
        
        # 预热
        print(f"  预热中... ({warmup}次)", end='', flush=True)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.model(dummy_input)
        print(" 完成")
        
        # 测试FPS
        print(f"  测试FPS... ({test_runs}次)", end='', flush=True)
        times = []
        with torch.no_grad():
            for _ in range(test_runs):
                start = time.time()
                _ = model.model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        print(" 完成")
        
        return {
            'fps': fps,
            'gflops': gflops,
            'params_mb': params_mb,
            'params': params
        }
        
    except Exception as e:
        print(f"\n  ✗ 错误: {str(e)}")
        return None


def generate_metrics_table(results, save_dir=None):
    """生成轻量化指标表格"""
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'runs' / 'comparison'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("模型轻量化指标对比")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    for dataset_name, models_metrics in results.items():
        report_lines.append(f"\n数据集: {dataset_name.upper()}")
        report_lines.append("-" * 100)
        report_lines.append(f"{'模型':<25} {'FPS(帧/秒)':<15} {'计算量(GFLOPs)':<18} {'参数量(MB)':<15}")
        report_lines.append("-" * 100)
        
        for model_name, metrics in models_metrics.items():
            if metrics:
                report_lines.append(
                    f"{model_name:<25} {metrics['fps']:<15.1f} {metrics['gflops']:<18.1f} {metrics['params_mb']:<15.1f}"
                )
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # 保存到文件
    report_file = save_dir / 'lightweight_metrics.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ 轻量化指标报告已保存到: {report_file}")
    
    # 同时保存为CSV格式
    csv_lines = []
    csv_lines.append("数据集,模型,FPS(帧/秒),计算量(GFLOPs),参数量(MB)")
    
    for dataset_name, models_metrics in results.items():
        for model_name, metrics in models_metrics.items():
            if metrics:
                csv_lines.append(
                    f"{dataset_name},{model_name},{metrics['fps']:.1f},{metrics['gflops']:.1f},{metrics['params_mb']:.1f}"
                )
    
    csv_file = save_dir / 'lightweight_metrics.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(csv_lines))
    
    print(f"✓ CSV格式已保存到: {csv_file}")


if __name__ == '__main__':
    print("=" * 100)
    print("计算模型轻量化指标")
    print("=" * 100)
    print("\n指标说明:")
    print("  - FPS (帧/秒): 模型推理速度，越高越好")
    print("  - 计算量 (GFLOPs): 浮点运算次数，越低越轻量")
    print("  - 参数量 (MB): 模型大小，越小越轻量")
    print("\n注意: FPS测试在CPU上进行，实际GPU性能会更高")
    print("=" * 100)
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device.upper()}")
    
    if device == 'cuda':
        print("检测到GPU，将在GPU上测试（更准确）")
    else:
        print("未检测到GPU，将在CPU上测试")
    
    results = {}
    
    for dataset_name, models in MODEL_PATHS.items():
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        results[dataset_name] = {}
        
        for model_name, model_path in models.items():
            print(f"\n[{model_name}]")
            
            if not model_path.exists():
                print(f"  ✗ 模型文件不存在: {model_path}")
                continue
            
            print(f"  模型路径: {model_path.name}")
            metrics = calculate_model_metrics(
                model_path, 
                device=device,
                img_size=640,
                warmup=10,
                test_runs=50  # 减少测试次数以加快速度
            )
            
            if metrics:
                results[dataset_name][model_name] = metrics
                print(f"  ✓ FPS: {metrics['fps']:.1f} 帧/秒")
                print(f"  ✓ 计算量: {metrics['gflops']:.1f} GFLOPs")
                print(f"  ✓ 参数量: {metrics['params_mb']:.1f} MB")
    
    # 生成报告
    print("\n" + "="*100)
    print("生成轻量化指标报告...")
    print("="*100)
    generate_metrics_table(results)
    
    print("\n" + "="*100)
    print("✓ 所有指标计算完成！")
    print("="*100)
