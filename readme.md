项目详细说明文档

📁 项目结构

```
'your path'
├── datas/                          # 数据集目录
│   ├── corrosion/                  # 腐蚀数据集（旧版）
│   │   ├── data.yaml
│   │   ├── images/
│   │   └── labels/
│   ├── cracks/                     # 裂缝数据集
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── MetalCorrosion/             # 金属腐蚀数据集（新版）
│       ├── MetalCorrosion.yaml
│       ├── train/                  # 训练集 4177张
│       ├── valid/                  # 验证集 782张
│       └── test/                   # 测试集 259张
│
├── models/                         # 模型训练与分析脚本
│   ├── 模型配置/                   # YOLO模型配置文件
│   │   ├── yolo11.yaml            # 基础YOLO11模型
│   │   ├── yolo11-ADown.yaml      # 改进1: ADown下采样
│   │   ├── yolo11-p2-adown.yaml   # 改进2: P2层 + ADown
│   │   └── yolo11-p2-adown-eca.yaml  # 改进3: P2层 + ADown + ECA注意力
│   │
│   ├── train_corrosion_optimized.py   # 腐蚀数据集训练脚本（300轮）
│   ├── train_cracks_optimized.py      # 裂缝数据集训练脚本（200轮）
│   ├── generate_comparison.py         # 生成对比图脚本
│   ├── complete_analysis.py           # 完整分析脚本
│   ├── train_compare.py               # 训练对比脚本
│   ├── detect.py                      # 目标检测脚本
│   ├── val.py                         # 验证脚本
│   └── runs/                          # models下的训练结果（旧）
│
├── runs/                           # 训练结果目录（新）
│   ├── train_corrosion/            # 腐蚀数据集训练结果（300轮）
│   │   ├── corrosion_改进1_训练优化/
│   │   ├── corrosion_改进2_ADown/
│   │   ├── corrosion_改进3_P2层/
│   │   └── corrosion_改进4_ECA注意力/
│   │
│   ├── train_cracks/               # 裂缝数据集训练结果（200轮）
│   │   ├── cracks_改进1_训练优化/
│   │   ├── cracks_改进2_ADown/
│   │   ├── cracks_改进3_P2层/
│   │   └── cracks_改进4_ECA注意力/
│   │
│   ├── comparison_corrosion/       # 腐蚀数据集对比结果
│   ├── comparison_cracks/          # 裂缝数据集对比结果
│   └── comparison/                 # 综合对比结果（改进1-4）
│       ├── corrosion_train_loss.png
│       ├── corrosion_val_loss.png
│       ├── corrosion_pr_curve.png
│       ├── cracks_train_loss.png
│       ├── cracks_val_loss.png
│       ├── cracks_pr_curve.png
│       └── training_summary.txt
│
├── requirements.txt                # Python依赖包
├── yolo11n.pt                      # YOLO11预训练权重（nano版本）
├── yolo11m.pt                      # YOLO11预训练权重（medium版本）
└── 项目结构说明.md                 # 本文档
```

---

数据预处理配置

### 1. 腐蚀数据集 (train_corrosion_optimized.py)

#### 数据集信息
- **数据集**: MetalCorrosion
- **规模**: 训练集 4177张 | 验证集 782张 | 测试集 259张
- **训练轮数**: 300 epochs
- **批次大小**: 32
- **图像尺寸**: 640×640

#### 基础数据预处理（原模型）
```python
'mosaic': 1.0,          # Mosaic数据增强（拼接4张图）
'mixup': 0.0,           # 不使用Mixup
'copy_paste': 0.0       # 不使用Copy-Paste
```

#### 优化数据预处理（改进1-4）
```python
# 数据增强
'mosaic': 1.0,          # Mosaic数据增强
'mixup': 0.15,          # Mixup增强（15%概率）
'copy_paste': 0.1,      # Copy-Paste增强（10%概率）

# 颜色空间增强
'hsv_h': 0.015,         # 色调抖动
'hsv_s': 0.7,           # 饱和度抖动
'hsv_v': 0.4,           # 明度抖动

# 几何变换
'degrees': 15.0,        # 旋转角度 ±15°
'translate': 0.2,       # 平移范围 ±20%
'scale': 0.5,           # 缩放范围 ±50%
'fliplr': 0.5           # 水平翻转概率 50%
```

---

2. 裂缝数据集 (train_cracks_optimized.py)

#### 数据集信息
- **数据集**: CRACKS
- **训练轮数**: 200 epochs
- **批次大小**: 16
- **图像尺寸**: 640×640

#### 基础数据预处理（原模型）
```python
'mosaic': 1.0,          # Mosaic数据增强
'mixup': 0.0,           # 不使用Mixup
'copy_paste': 0.0       # 不使用Copy-Paste
```

#### 优化数据预处理（改进1-4）
```python
# 数据增强
'mosaic': 1.0,          # Mosaic数据增强
'mixup': 0.15,          # Mixup增强（15%概率）
'copy_paste': 0.1,      # Copy-Paste增强（10%概率）

# 颜色空间增强
'hsv_h': 0.015,         # 色调抖动
'hsv_s': 0.7,           # 饱和度抖动
'hsv_v': 0.4,           # 明度抖动

# 注意：裂缝数据集未使用几何变换增强
```

---

模型改进策略（渐进式消融实验）

### 改进1: 训练策略优化
**改进内容**:
- ✅ **Cosine学习率衰减** (`cos_lr=True`)
  - 学习率从 `lr0=0.001` 平滑衰减到 `lrf=0.01`
  - 相比线性衰减，能更好地收敛
  
- ✅ **AdamW优化器** (替代SGD)
  - 自适应学习率
  - 更好的权重衰减机制
  - 适合深度网络训练

- ✅ **增强数据增强**
  - Mixup (15%)
  - Copy-Paste (10%)
  - HSV颜色空间增强
  - 几何变换（腐蚀数据集）

**模型架构**: 基础 YOLO11 (yolo11.yaml)

---

改进2: ADown下采样
**改进内容**:
- ✅ **继承改进1的所有训练策略**
- ✅ **ADown下采样模块**
  - 替换传统的卷积下采样
  - 更高效的特征提取
  - 减少信息损失
  - 降低计算复杂度

**模型架构**: yolo11-ADown.yaml

**改进点**:
- 在主干网络的下采样层使用ADown
- 保留更多空间信息
- 提升小目标检测能力

---

改进3: P2小目标检测层
**改进内容**:
- ✅ **继承改进1+2的所有优化**
- ✅ **P2检测层** (高分辨率特征层)
  - 原始YOLO11: P3, P4, P5 三层检测
  - 改进后: P2, P3, P4, P5 四层检测
  - P2层分辨率更高（160×160）

**模型架构**: yolo11-p2-adown.yaml

**改进点**:
- 专门针对小目标检测
- 增加浅层特征的利用
- 提升对细小腐蚀点和裂缝的检测能力

---

改进4: ECA轻量级注意力
**改进内容**:
- ✅ **继承改进1+2+3的所有优化**
- ✅ **ECA注意力机制** (Efficient Channel Attention)
  - 轻量级通道注意力
  - 自适应特征加权
  - 几乎不增加计算量

**模型架构**: yolo11-p2-adown-eca.yaml

**改进点**:
- 在关键特征层添加ECA模块
- 增强重要特征通道
- 抑制无关特征
- 提升模型判别能力

---
 训练参数对比

| 参数 | 原模型 | 改进1-4 |
|------|--------|---------|
| **优化器** | SGD | AdamW |
| **学习率策略** | 线性衰减 | Cosine衰减 |
| **初始学习率** | 0.001 | 0.001 |
| **最终学习率** | 0.1 | 0.01 |
| **Mosaic** | 1.0 | 1.0 |
| **Mixup** | 0.0 | 0.15 |
| **Copy-Paste** | 0.0 | 0.1 |
| **HSV增强** | ❌ | ✅ |
| **几何变换** | ❌ | ✅ (腐蚀) |

---

使用指南

### 1. 训练模型
```bash
# 训练腐蚀数据集（300轮）
python models/train_corrosion_optimized.py

# 训练裂缝数据集（200轮）
python models/train_cracks_optimized.py
```

### 2. 生成对比图
```bash
# 生成改进1-4的对比分析图
python models/generate_comparison.py
```

生成的对比图包括：
- 训练损失对比图
- 验证损失对比图
- Precision & Recall 变化曲线
- P-R 曲线（训练轨迹）
- 训练结果汇总报告

### 3. 模型验证
```bash
# 验证模型性能
python models/val.py --weights runs/train_corrosion/corrosion_改进4_ECA注意力/weights/best.pt
```

### 4. 目标检测
```bash
# 使用训练好的模型进行检测
python models/detect.py --weights runs/train_corrosion/corrosion_改进4_ECA注意力/weights/best.pt --source test_images/
```

---

实验结果

### 评估指标
- **mAP50**: 在IoU=0.5时的平均精度
- **mAP50-95**: 在IoU=0.5-0.95时的平均精度
- **Precision**: 精确率（检测正确的比例）
- **Recall**: 召回率（找到目标的比例）
- **F1-Score**: 精确率和召回率的调和平均

### 查看结果
训练完成后，结果保存在：
- `runs/comparison/training_summary.txt` - 详细性能指标
- `runs/comparison/*.png` - 可视化对比图

---

环境要求

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖
- Python >= 3.8
- PyTorch >= 1.8
- ultralytics (YOLO11)
- opencv-python
- pandas
- matplotlib
- numpy
```

---

注意事项

1. **GPU要求**: 建议使用NVIDIA GPU进行训练，通过device设置，本地跑用cuda:'0'
   - 腐蚀数据集: 使用GPU 0 (batch=32)
   - 裂缝数据集: 使用GPU 2 (batch=16)

2. **训练时间**:
   - 腐蚀数据集 (300轮): 约10-15小时
   - 裂缝数据集 (200轮): 约8-12小时

3. **存储空间**: 每个训练实验约需要 500MB-1GB 空间

4. **随机种子**: 所有实验使用固定随机种子 (seed=42) 确保可复现性

---

项目信息

- **项目路径**: 'your path'
- **任务类型**: 目标检测 (金属腐蚀与裂缝检测)
- **基础模型**: YOLO11

---

改进总结

本项目采用**渐进式消融实验**策略，逐步验证每个改进的有效性：

1. **改进1** 奠定基础：优化训练策略和数据增强
2. **改进2** 提升特征：引入ADown高效下采样
3. **改进3** 增强小目标：添加P2高分辨率检测层
4. **改进4** 精细化特征：集成ECA轻量级注意力

每个改进都在前一个改进的基础上进行，确保性能递增，最终实现最优检测效果。
