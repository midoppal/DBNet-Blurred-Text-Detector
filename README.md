# DINO
## Train
python dino_train.py

## Inference:
- python dino_produce_maps
- python dino_heatmap_pixel_eval.py   --images datasets/icdar2015/test_images   --gts datasets/icdar2015/test_gts   --heatmaps datasets/dino_maps   --img_ext .jpg

- Make sure all file paths are correct
  
#DBNet
## Train
- python train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./pretrained/ic15_resnet50.pth --num_gpus 1

## Inference
- python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./pretrained/ic15_resnet50.pth
- Replace "./pretrained/ic15_resnet50.pth" with finetuned model

