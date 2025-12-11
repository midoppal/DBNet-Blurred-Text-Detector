# DINO
## Train
python dino_train.py

## Inference:
- python dino_eval.py \
--images /workspace/DBNet-Blurred-Text-Detector/datasets/icdar2015/test_images \
--gts /workspace/DBNet-Blurred-Text-Detector/datasets/icdar2015/test_gts \
--head_ckpt /workspace/DBNet-Blurred-Text-Detector/fine_tuned_models/dino_text_head.ckpt \
--model_name vit_small_patch14_dinov2.lvd142m \
--img_size 700 \
--iou_thr 0.3 --match_mode pairwise  \
--viz_out /workspace/DBNet-Blurred-Text-Detector/datasets/dino_text_boxes

- Make sure ckpt model is in in the right path
#DBNet
## Train
- python train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./pretrained/ic15_resnet50.pth --num_gpus 1

## Inference
- python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./pretrained/ic15_resnet50.pth
- Replace "./pretrained/ic15_resnet50.pth" with finetuned model

