#!python3
import argparse
import os
import torch
import yaml
from tqdm import tqdm
import numpy as np
import cv2
import time 

from trainer import Trainer
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.image_dataset import ImageDataset
from training.checkpoint import Checkpoint
from training.learning_rate import (
    ConstantLearningRate, PriorityLearningRate, FileMonitorLearningRate
)
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config
import time

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--result_dir', type=str, default='./results/', help='path to save results')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--start_iter', type=int,
                        help='Begin counting iterations starting from this value (should be used with resume)')
    parser.add_argument('--start_epoch', type=int,
                        help='Begin counting epoch starting from this value (should be used with resume)')
    parser.add_argument('--max_size', type=int, help='max length of label')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--no-verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    
    parser.add_argument('--save_prob_maps', action='store_true', default=True,
                        help='Save per-image probability maps as PNGs')
    parser.add_argument('--save_prob_maps_gray', action='store_true', default=False,
                        help='Also save raw grayscale [0..255] PNG alongside heatmap')
    
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    parser.add_argument('--speed', action='store_true', dest='test_speed',
                        help='Test speed only')
    parser.add_argument('--dest', type=str,
                        help='Specify which prediction will be used for decoding.')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='Run with debug mode, which hacks dataset num_samples to toy number')
    parser.add_argument('--no-debug', action='store_false',
                        dest='debug', help='Run without debug mode')
    parser.add_argument('-d', '--distributed', action='store_true',
                        dest='distributed', help='Use distributed training')
    parser.add_argument('--local_rank', dest='local_rank', default=0,
                        type=int, help='Use distributed training')
    parser.add_argument('-g', '--num_gpus', dest='num_gpus', default=1,
                        type=int, help='The number of accessible gpus')
    parser.set_defaults(debug=False, verbose=False)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Eval(experiment, experiment_args, cmd=args, verbose=args['verbose']).eval(args['visualize'])


class Eval:
    def __init__(self, experiment, args, cmd=dict(), verbose=False):
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = cmd.get(
            'resume', os.path.join(
                self.logger.save_dir(model_saver.dir_path),
                'final'))
        self.verbose = verbose

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            self.logger.warning("Checkpoint not found: " + path)
            return
        self.logger.info("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        self.logger.info("Resumed from " + path)

    def report_speed(self, model, batch, times=100):
        data = {k: v[0:1]for k, v in batch.items()}
        if  torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time() 
        for _ in range(times):
            pred = model.forward(data)
        for _ in range(times):
            output = self.structure.representer.represent(batch, pred, is_output_polygon=False) 
        time_cost = (time.time() - start) / times
        self.logger.info('Params: %s, Inference speed: %fms, FPS: %f' % (
            str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            time_cost * 1000, 1 / time_cost))
        
        return time_cost
        
    def format_output(self, batch, output, img_root='.'):
        batch_boxes, batch_scores = output
        os.makedirs('predictions', exist_ok=True)  # ✨ Create output folder for images

        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)

            boxes = batch_boxes[index]
            scores = batch_scores[index]

            # ✨ Load original image (assumes filename is a valid image path)
            # ✨ Construct full path to the image
            img_path = os.path.join(img_root, filename)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"[WARNING] Could not load image: {img_path}")
                continue

            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).astype(int)
                        result = ",".join([str(x) for x in box.flatten()])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
                        
                        # ✨ Draw polygons
                        cv2.polylines(image_bgr, [box.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i, :, :].reshape(-1).astype(int)
                        result = ",".join([str(x) for x in box])
                        res.write(result + ',' + str(score) + "\n")

                        # ✨ Draw rectangles as polygons
                        cv2.polylines(image_bgr, [box.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

            # ✨ Save output image with bounding boxes
            out_img_name = os.path.splitext(os.path.basename(filename))[0] + '_pred.jpg'
            out_img_path = os.path.join('predictions', out_img_name)
            cv2.imwrite(out_img_path, image_bgr)
            # print(f"[INFO] Saved prediction image to {out_img_path}")

            # ✨ Also save ground truth boxes
            os.makedirs('ground_truth', exist_ok=True)
            image_gt = cv2.imread(img_path)
            if image_gt is not None:
                if 'polygons' in batch:
                    gt_polygons = batch['polygons'][index]
                    for gt_box in gt_polygons:
                        gt_box = np.array(gt_box).reshape(-1, 1, 2).astype(int)
                        cv2.polylines(image_gt, [gt_box], isClosed=True, color=(0, 0, 255), thickness=2)  # red
                gt_img_name = os.path.splitext(os.path.basename(filename))[0] + '_gt.jpg'
                gt_img_path = os.path.join('ground_truth', gt_img_name)
                cv2.imwrite(gt_img_path, image_gt)
                # print(f"[INFO] Saved ground truth image to {gt_img_path}")
            else:
                print(f"[WARNING] Could not load image for ground truth: {img_path}")

    def _walk_tensors(self, obj, prefix="pred"):
        """
        Recursively yield (name, tensor) for any torch.Tensor with HxW spatial shape.
        Supports dicts, lists/tuples, and plain tensors.
        """
        import torch
        if torch.is_tensor(obj):
            yield (prefix, obj)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                for nk, nv in self._walk_tensors(v, f"{prefix}.{k}"):
                    yield (nk, nv)
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                for nk, nv in self._walk_tensors(v, f"{prefix}[{idx}]"):
                    yield (nk, nv)

    def _save_map(self, arr01, out_png):
        import cv2, numpy as np
        arr01 = np.clip(arr01, 0.0, 1.0)
        arr255 = (arr01 * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(arr255, cv2.COLORMAP_JET)
        cv2.imwrite(out_png, heat)

    def save_all_spatial_maps(self, batch, pred, img_root='.', save_gray=False):
        """
        For each image in batch:
        - Find every tensor in `pred` that has shape [N, C, H, W] or [N, H, W]
        - For each channel, save a heatmap (and optional grayscale) resized to original image size if we can
        - Filenames include the tensor 'path' (key/index chain) so you can identify the real prob map
        """
        import os, cv2, numpy as np, torch

        out_dir = 'probability_map_predictions'
        os.makedirs(out_dir, exist_ok=True)

        B = batch['image'].size(0)

        # Try reading originals once to get target sizes
        imgs = []
        sizes = []
        for i in range(B):
            filename = batch['filename'][i]
            img_path = os.path.join(img_root, filename)
            img = cv2.imread(img_path)
            imgs.append(img)
            if img is not None:
                sizes.append(img.shape[:2])
            else:
                # fallback to batch['shape'] if present, else None
                H = W = None
                if 'shape' in batch:
                    try:
                        shp = batch['shape'][i]
                        if hasattr(shp, 'cpu'): shp = shp.cpu().numpy()
                        shp = np.array(shp).reshape(-1)
                        if shp.size >= 2:
                            H, W = int(shp[0]), int(shp[1])
                    except:
                        pass
                sizes.append((H, W) if (H and W) else None)

        saved = 0
        # Walk every tensor in pred, try to interpret as spatial maps
        for name, ten in self._walk_tensors(pred, "pred"):
            if not torch.is_tensor(ten):
                continue
            # Expect [N,C,H,W] or [N,H,W]
            d = ten.dim()
            if d not in (3, 4):
                continue

            if d == 4:
                N, C, H, W = ten.shape
            else:
                # [N,H,W] -> treat C=1
                N, H, W = ten.shape
                C = 1
                ten = ten.unsqueeze(1)

            # Per-image in batch
            for i in range(min(B, ten.shape[0])):
                base = os.path.splitext(os.path.basename(batch['filename'][i]))[0]
                target_size = sizes[i]
                img = imgs[i]

                # Per channel
                for c in range(min(C, 4)):  # cap to 4 channels per tensor to avoid explosion
                    m = ten[i, c].detach().float().cpu()
                    # If looks like logits, sigmoid; else clamp
                    if (m.min() < 0) or (m.max() > 1):
                        m = torch.sigmoid(m)
                    m = torch.clamp(m, 0.0, 1.0)
                    m_np = m.numpy()

                    # Resize to original size if known
                    if target_size is not None and all(target_size):
                        Ht, Wt = target_size
                        m_np_r = cv2.resize(m_np, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
                    else:
                        m_np_r = m_np

                    safe_name = (
                        name.replace('/', '_')
                            .replace('.', '_')
                            .replace('[', '_').replace(']', '')
                    )
                    # Save heatmap overlay if we have the image and sizes match
                    m255 = (np.clip(m_np_r, 0, 1) * 255.0).astype(np.uint8)
                    heat = cv2.applyColorMap(m255, cv2.COLORMAP_JET)

                    out_heat = os.path.join(out_dir, f"{base}__{safe_name}__ch{c}_heat.png")

                    if img is not None and img.shape[:2] == heat.shape[:2]:
                        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0.0)
                        cv2.imwrite(out_heat, overlay)
                    else:
                        cv2.imwrite(out_heat, heat)

                    if save_gray:
                        out_gray = os.path.join(out_dir, f"{base}__{safe_name}__ch{c}_gray.png")
                        cv2.imwrite(out_gray, m255)

                    saved += 1

        # print(f"[INFO] Saved {saved} map image(s) to {out_dir}")

        
    def eval(self, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        vis_images = dict()
        with torch.no_grad():
            for _, data_loader in self.data_loaders.items():
                raw_metrics = []
                for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    if self.args['test_speed']:
                        time_cost = self.report_speed(model, batch, times=50)
                        continue
                    pred = model.forward(batch, training=False)
                    output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])


                    # self.format_output(batch, output)
                    img_root = os.path.join('datasets\icdar2015', 'test_images')  # fallback if not found
                    self.format_output(batch, output, img_root)

                    # NEW: save probability maps if requested
                    if self.args.get('save_prob_maps', False):
                        self.save_all_spatial_maps(
                            batch, pred,
                            img_root=os.path.join('datasets', 'icdar2015', 'test_images'),
                            save_gray=self.args.get('save_prob_maps_gray', False)
                        )


                    raw_metric = self.structure.measurer.validate_measure(batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                    raw_metrics.append(raw_metric)

                    if visualize and self.structure.visualizer:
                        vis_image = self.structure.visualizer.visualize(batch, output, pred)
                        self.logger.save_image_dict(vis_image)
                        vis_images.update(vis_image)
                metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
                for key, metric in metrics.items():
                    self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))

if __name__ == '__main__':
    # start = time.time()
    main()
    # end = time.time()
    # print(f"Total runtime: {end - start:.2f} seconds")
