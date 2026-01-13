"""
一键后处理：补固定刻度损失图、整条 P-R 曲线、GFLOPs 表格、2×2 可视化拼接
运行时机：train_compare.py 跑完后
"""
import re
import json
import csv
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT  = Path(__file__).parent.absolute()
SUPP_DIR      = PROJECT_ROOT / "runs/comparison/supplement"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['corrosion', 'cracks']
MODELS   = ['原模型', '原模型+1改进', '原模型+2改进', '原模型+3改进']
COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ---------- 1. 重绘固定刻度损失图 ----------
def redo_loss_plots():
    for ds in DATASETS:
        csv_files = sorted(Path("runs/train").glob(f"{ds}_*/results.csv"))
        plt.figure(figsize=(6,4))
        for idx, cf in enumerate(csv_files):
            df = pd.read_csv(cf)
            df.columns = df.columns.str.strip()
            train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
            plt.plot(train_loss, label=MODELS[idx], color=COLORS[idx], lw=2)
        plt.ylim(0, 7); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'{ds} - 训练损失（固定刻度）')
        plt.legend(); plt.grid(alpha=.3)
        plt.savefig(SUPP_DIR/f'{ds}_train_loss_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ 重绘固定刻度损失图完成')

# ---------- 2. 整条 P-R 曲线 ----------
def whole_pr_curves():
    for ds in DATASETS:
        plt.figure(figsize=(6,5))
        for idx, m in enumerate(MODELS):
            exp = list(Path("runs/train").glob(f"{ds}_{m.replace('+','_')}*"))[0]
            best = exp/"weights"/"best.pt"
            model = YOLO(best)
            # 用 val set 跑一遍拿到 COCO eval json
            res = model.val(data=f'datas/{ds}/data.yaml', save_json=True, project='runs/temp', name='tmp')
            json_p = Path("runs/temp/tmp/predictions.json")  # COCO json
            if not json_p.exists(): continue
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            cocoGt = COCO(f'datas/{ds}/val/_annotations.coco.json')
            cocoDt = cocoGt.loadRes(str(json_p))
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate(); cocoEval.accumulate()
            stats = cocoEval.eval['precision']   # [TxRxKxAxM]
            # 取 IoU=0.5, 所有类别求平均
            pr = stats[0, :, :, 0, 0].mean(axis=1)  # [R,]
            rec = np.linspace(0, 1, len(pr))
            plt.plot(rec, pr, lw=2, color=COLORS[idx], label=f'{m}')
        plt.xlim(0,1); plt.ylim(0,1); plt.grid(alpha=.3)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{ds} - P-R 曲线（整条）')
        plt.legend(); plt.savefig(SUPP_DIR/f'{ds}_whole_pr.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('✓ 整条 P-R 曲线完成')

# ---------- 3. GFLOPs / 参数量表格 ----------
def gflop_table():
    rows = []
    for ds in DATASETS:
        for idx, m in enumerate(MODELS):
            exp = list(Path("runs/train").glob(f"{ds}_{m.replace('+','_')}*"))[0]
            best = exp/"weights"/"best.pt"
            model = YOLO(best)
            info = model_info(model.model, detailed=False, verbose=False)  # 返回 (params, gf)
            rows.append([ds, m, f'{info[0]/1e6:.2f}M', f'{info[1]:.2f}'])
    df = pd.DataFrame(rows, columns=['Dataset','Model','Params','GFLOPs'])
    df.to_csv(SUPP_DIR/'gflops.csv', index=False, float_format='%.3f')
    print('✓ GFLOPs 表格已保存至', SUPP_DIR/'gflops.csv')

# ---------- 4. 2×2 可视化拼接 ----------
def concat_pred_images():
    """取同一张验证图，把四个模型预测画在一起"""
    val_img = Path('datas/corrosion/val/0001.jpg')  # 换成你数据里任意一张
    if not val_img.exists(): return
    imgs = []
    for m in MODELS:
        exp = list(Path("runs/train").glob(f"corrosion_{m.replace('+','_')}*"))[0]
        best = exp/"weights"/"best.pt"
        model = YOLO(best)
        res = model.predict(val_img, save=False, line_width=2)
        im_array = res[0].plot()  # BGR numpy
        im = Image.fromarray(im_array[:,:,::-1])
        imgs.append(im.resize((640,640)))
    canvas = Image.new('RGB', (1280,1280))
    for i, im in enumerate(imgs):
        canvas.paste(im, ((i%2)*640, (i//2)*640))
        draw = ImageDraw.Draw(canvas)
        # 写模型名
        draw.text(((i%2)*640+5, (i//2)*640+5), MODELS[i], fill='red')
    canvas.save(SUPP_DIR/'corrosion_2x2_pred.jpg', quality=95)
    print('✓ 2×2 拼接图完成')

# ---------- main ----------
if __name__ == '__main__':
    redo_loss_plots()
    whole_pr_curves()
    gflop_table()
    concat_pred_images()
    print('\n===== 所有补充图/表已输出至 =====')
    print(SUPP_DIR)