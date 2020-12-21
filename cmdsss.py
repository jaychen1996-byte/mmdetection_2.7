# 快速开始
"""
# YOLOv3
python demo/image_demo.py ./demo/demo.jpg ./configs/detr/detr_r50_8x2_150e_coco.py ./checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth

#SSD算法300尺寸
python demo/image_demo.py demo/demo.jpg configs/ssd/ssd300_coco.py checkpoints/ssd300_coco_20200307-a92d2092.pth
"""

# mmdetection的标准格式是coco格式（可以把voc、cityscapes等等转换为coco格式）

# 测试模型
# 测试模型并立即显示
"""
python tools/test.py configs/yolo/yolov3_d53_320_273e_coco.py checkpoints/yolov3_d53_320_273e_coco-421362b6.pth --show
"""

# 测试模型并保存
"""
python tools/test.py configs/yolo/yolov3_d53_320_273e_coco.py checkpoints/yolov3_d53_320_273e_coco-421362b6.pth  --show-dir yolov3_d53_320_273e_coco_results
"""

# 测试模型查看mAP
"""
python tools/test.py configs/yolo/yolov3_d53_320_273e_coco.py checkpoints/yolov3_d53_320_273e_coco-421362b6.pth  --eval mAP
"""

# 在COCO数据集上进行训练
"""
#训练SSD算法
python tools/train.py configs/ssd/ssd300_coco.py
"""

"test git"