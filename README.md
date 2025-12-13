# DINO

## Train
python dino_train.py

## Inference
Note: Remember to update all file paths (datasets, checkpoints, and output directories) to match your local setup before running inference.

### Generate probability maps  
  dino_produce_maps.py runs inference on the test images and saves per-pixel text probability maps.
 -  python dino_produce_maps.py

### Pixel-level evaluation with threshold sweep  
  dino_heatmap_pixel_eval.py evaluates the generated probability maps by sweeping over binarization thresholds and computing precision, recall, and F-measure.
  - python dino_heatmap_pixel_eval.py \
    --images datasets/icdar2015/test_images \
    --gts datasets/icdar2015/test_gts \
    --heatmaps datasets/dino_maps \
    --img_ext .jpg

---

# DBNet

## Train
- python train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
  --resume ./pretrained/ic15_resnet50.pth \
  --num_gpus 1

## Inference
- python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
  --resume ./pretrained/ic15_resnet50.pth

Note: Replace ./pretrained/ic15_resnet50.pth with the path to your fine-tuned DBNet model checkpoint.

# Models
- https://drive.google.com/drive/folders/1MnveYGSlHHK5PDudmO004dLQl9fDSXDD?usp=drive_link

# Note
- blur_every_length.py produces the test sets for varying blur levels

Remember to change paths
