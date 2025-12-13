import os

import torch
from tqdm import tqdm

from experiment import Experiment
from data.data_loader import DistributedSampler

import math
VAL_SCORE_MODE = os.environ.get("VAL_SCORE_MODE", "avg_blur").lower()
VAL_SCORE_KEYS = os.environ.get("VAL_SCORE_KEYS", "").strip()

def _compute_val_score(all_metrics, logger=None):

    def get(k): return all_metrics[k].avg if k in all_metrics else None

    
    if VAL_SCORE_KEYS:
        keys = [k.strip() for k in VAL_SCORE_KEYS.split(",") if k.strip()]
        vals = [get(k) for k in keys]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if vals:
            return sum(vals) / len(vals)


    f1_items = [(k, m.avg) for k, m in all_metrics.items() if k.endswith("/fmeasure") and math.isfinite(m.avg)]

    if VAL_SCORE_MODE == "avg_blur":
        blur_vals = [v for k, v in f1_items if "_blur_" in k or "blur_" in k]
        if blur_vals:
            return sum(blur_vals) / len(blur_vals)
    elif VAL_SCORE_MODE == "clean":
        clean_vals = [v for k, v in f1_items if "clean" in k]
        if clean_vals:
            return sum(clean_vals) / len(clean_vals)

   
    if f1_items:
        return sum(v for _, v in f1_items) / len(f1_items)

    # Last resort
    if logger:
        logger.info("[val-score] No fmeasure metrics found; defaulting to 0.0")
    return 0.0
# END Validation scoring and best checkpoint 

# Validation history dump helpers 
import math, csv

VAL_DUMP_DIR = os.environ.get("VAL_DUMP_DIR", "val_history")

def _dump_validation_metrics(all_metrics, epoch, step, run_root, logger=None):
    """
    Writes a per-step snapshot folder + appends a long CSV for all loaders.
    all_metrics: dict like {"icdar2015_clean/precision": Metric, ...}
    """
    import os
    base = os.path.join(run_root, VAL_DUMP_DIR)
    os.makedirs(base, exist_ok=True)

    snap_dir = os.path.join(base, f"step_{step:09d}_epoch_{epoch:03d}")
    os.makedirs(snap_dir, exist_ok=True)

  
    per_loader = {}
    for k, m in all_metrics.items():
        if "/" in k:
            loader, metric = k.split("/", 1)
        else:
            loader, metric = "default", k
        per_loader.setdefault(loader, {})[metric] = m

   
    for loader, mdict in per_loader.items():
        p = mdict.get("precision"); r = mdict.get("recall"); f = mdict.get("fmeasure")
        with open(os.path.join(snap_dir, f"{loader}.txt"), "w", encoding="utf-8") as fh:
            if p is not None: fh.write(f"precision : {p.avg:.6f} ({int(p.count)})\n")
            if r is not None: fh.write(f"recall    : {r.avg:.6f} ({int(r.count)})\n")
            if f is not None: fh.write(f"fmeasure  : {f.avg:.6f} ({int(f.count)})\n")

  
    csv_path = os.path.join(base, "metrics_log.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["step", "epoch", "loader", "precision", "recall", "fmeasure", "count"])
        for loader, mdict in per_loader.items():
            p = mdict.get("precision"); r = mdict.get("recall"); fm = mdict.get("fmeasure")
            cnt = None
            for mm in (p, r, fm):
                if mm is not None:
                    cnt = int(mm.count); break
            w.writerow([
                step, epoch, loader,
                f"{p.avg:.6f}"  if p  is not None else "",
                f"{r.avg:.6f}"  if r  is not None else "",
                f"{fm.avg:.6f}" if fm is not None else "",
                cnt if cnt is not None else ""
            ])
    if logger:
        logger.info(f"[val-dump] wrote {snap_dir} and appended {csv_path}")



import torch.nn as nn


FREEZE_BACKBONE_EPOCHS = int(os.environ.get("FREEZE_BACKBONE_EPOCHS", "10"))


FREEZE_STAGES = [
    s.strip() for s in os.environ.get("FREEZE_STAGES", "conv1,bn1,layer1,layer2").split(",")
    if s.strip()
]


BACKBONE_LR_MULT = float(os.environ.get("BACKBONE_LR_MULT", "1"))

def _get_model_root(module):
    """unwrap DataParallel/DistributedDataParallel and common wrappers"""
    return module.module if hasattr(module, "module") else module

def _get_backbone(module):
    """
    Retrieve the backbone submodule from different wrapper layouts.
    Tries a few common names, then falls back to a heuristic scan.
    """
    m = _get_model_root(module)


    for name in ("backbone", "body", "encoder", "feature_extractor", "resnet"):
        if hasattr(m, name):
            return getattr(m, name)


    if hasattr(m, "model"):
        mm = _get_model_root(m.model)
        for name in ("backbone", "body", "encoder", "feature_extractor", "resnet"):
            if hasattr(mm, name):
                return getattr(mm, name)

    
    for name, child in m.named_children():
        if name in ("backbone", "body", "encoder", "feature_extractor", "resnet"):
            return child
    return None

def _toggle_requires_grad(module, trainable: bool):
    for p in module.parameters():
        p.requires_grad = trainable

def set_backbone_stages_trainable(model, stages, trainable: bool, freeze_bn: bool = True, logger=None):
    """
    Freeze/unfreeze ONLY the requested 'stages' inside the backbone.
    On torchvision ResNet, valid stage names: conv1, bn1, layer1, layer2, layer3, layer4.
    """
    bb = _get_backbone(model)
    if bb is None:
        if logger: logger.info("[freeze] backbone not found; skipping")
        return

    
    name_to_child = dict(bb.named_children())

    changed = []
    for name in stages:
        if name in name_to_child:
            _toggle_requires_grad(name_to_child[name], trainable)
            changed.append(name)

    if ("conv1" in stages) and hasattr(bb, "bn1"):
        _toggle_requires_grad(bb.bn1, trainable)
        if "bn1" not in changed:
            changed.append("bn1")

    if freeze_bn:
        def _maybe_set_bn_mode(mod, stage_trainable):
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval() if not stage_trainable else mod.train()

        for name in changed:
            tgt = getattr(bb, name, None)
            if tgt is None:
                continue
            tgt.apply(lambda m: _maybe_set_bn_mode(m, trainable))

    if logger:
        state = "UNFROZEN" if trainable else "FROZEN"
        def _count_params(mod):
            total = sum(p.numel() for p in mod.parameters())
            train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            return train, total

        msg_bits = []
        for name in changed:
            mod = getattr(bb, name, None)
            if mod is not None:
                tr, tot = _count_params(mod)
                msg_bits.append(f"{name}({tr}/{tot})")
        msg = ", ".join(msg_bits) if msg_bits else "none"
        logger.info(f"[freeze] stages {state}: {msg}")



class Trainer:
    def __init__(self, experiment: Experiment):
        self.init_device()

        self.experiment = experiment
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver
        self._stages_frozen = False

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.logger.info(f"[paths] run_dir={os.path.dirname(self.model_saver.dir_path)} "
                 f"model_dir={self.model_saver.dir_path}")
        self.current_lr = 0

        self.total = 0

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)
        return model

    def update_learning_rate(self, optimizer, epoch, step):
        """
        Apply scheduler LR to each param group, honoring optional 'lr_mult' on groups.
        If a group has group['lr_mult'], its lr becomes base_lr * lr_mult.
        """
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            mult = group.get('lr_mult', 1.0)
            group['lr'] = lr * mult
        self.current_lr = lr

    def train(self):
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        self.logger.info(f"[paths] run_dir={self.logger.log_dir}  model_dir={self.model_saver.dir_path}")
        model = self.init_model()
        
        train_data_loader = self.experiment.train.data_loader
        if self.experiment.validation:
            validation_loaders = self.experiment.validation.data_loaders
        
        self.steps = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta
        else:
            epoch, iter_delta = 0, 0  # safety default
        
        # Build optimizer with optional backbone LR group

        if BACKBONE_LR_MULT != 1.0:
            bb_params, head_params = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
    
                if ("backbone." in name) or (".backbone." in name) or name.startswith("backbone"):
                    bb_params.append(p)
                else:
                    head_params.append(p)

            if len(bb_params) == 0: 
                bb = _get_backbone(model)
                if bb is not None:
                    bb_ids = {id(p) for p in bb.parameters()}
                    for p in model.parameters():
                        (bb_params if id(p) in bb_ids else head_params).append(p)
                else:
                    bb_params, head_params = [], list(model.parameters())

            param_groups = []
            if bb_params:
                param_groups.append({"params": bb_params,  "lr_mult": BACKBONE_LR_MULT})
            if head_params:
                param_groups.append({"params": head_params, "lr_mult": 1.0})

            optimizer = self.experiment.train.scheduler.create_optimizer(param_groups)
            self.logger.info(f"[optimizer] using backbone LR mult = {BACKBONE_LR_MULT}")
        else:
            optimizer = self.experiment.train.scheduler.create_optimizer(model.parameters())
            self.logger.info("[optimizer] single LR for all params (BACKBONE_LR_MULT=1.0)")
   

        self.logger.report_time('Init')

        # Initial backbone freeze window 
        def _maybe_stage_freeze(epoch_now):
            if FREEZE_BACKBONE_EPOCHS <= 0 or not FREEZE_STAGES:
                return
          
            if epoch_now < FREEZE_BACKBONE_EPOCHS and not self._stages_frozen:
                set_backbone_stages_trainable(model, FREEZE_STAGES, trainable=False, freeze_bn=True, logger=self.logger)
                self._stages_frozen = True
                self.logger.info(f"[freeze] freezing stages {FREEZE_STAGES} for epochs [0..{FREEZE_BACKBONE_EPOCHS-1}]")
            
            elif epoch_now >= FREEZE_BACKBONE_EPOCHS and self._stages_frozen:
                set_backbone_stages_trainable(model, FREEZE_STAGES, trainable=True, freeze_bn=True, logger=self.logger)
                self._stages_frozen = False
                self.logger.info(f"[freeze] unfroze stages at epoch {epoch_now}")

    
        _maybe_stage_freeze(epoch)
    

        model.train()
        while True:
            # Unfreeze when we reach the target epoch 
            _maybe_stage_freeze(epoch)

            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                self.logger.report_time("Data loading")

                if self.experiment.validation and\
                        self.steps % self.experiment.validation.interval == 0 and\
                        self.steps > self.experiment.validation.exempt:
                
                    self.logger.info(f"[val] firing at step={self.steps} (interval={self.experiment.validation.interval})")
                    val_metrics = self.validate(validation_loaders, model, epoch, self.steps)
                    run_root = os.path.dirname(self.model_saver.dir_path)  # parent of 'model' dir for this run
                    _dump_validation_metrics(val_metrics, epoch, self.steps, run_root, logger=self.logger)
                    if not hasattr(self, "best_val_score"):
                        self.best_val_score = -1.0
                    score = _compute_val_score(val_metrics, logger=self.logger)
                    self.logger.info(f"[val-score] step={self.steps} score={score:.6f} (mode={VAL_SCORE_MODE or 'avg_blur'})")
                    if score > self.best_val_score:
                        self.best_val_score = score
                        tag = "best_blur" if VAL_SCORE_MODE == "avg_blur" and not VAL_SCORE_KEYS else "best_val"
                        self.model_saver.save_checkpoint(model, tag)
                        self.logger.info(f"[best] improved to {score:.6f}; saved checkpoint '{tag}.pth'")
                        ##
                self.logger.report_time('Validating ')
                if self.logger.verbose:
                    torch.cuda.synchronize()

                self.train_step(model, optimizer, batch,
                                epoch=epoch, step=self.steps)
                if self.logger.verbose:
                    torch.cuda.synchronize()
                self.logger.report_time('Forwarding ')

                self.model_saver.maybe_save_model(
                    model, epoch, self.steps, self.logger)

                self.steps += 1
                self.logger.report_eta(self.steps, self.total, epoch)

            epoch += 1
            if epoch > self.experiment.train.epochs:
                self.model_saver.save_checkpoint(model, 'final')

                if self.experiment.validation:
                    val_metrics = self.validate(validation_loaders, model, epoch, self.steps)
                    run_root = os.path.dirname(self.model_saver.dir_path)
                    _dump_validation_metrics(val_metrics, epoch, self.steps, run_root, logger=self.logger)
                    score = _compute_val_score(val_metrics, logger=self.logger)
                    self.logger.info(f"[val-score] final score={score:.6f}")
                    if not hasattr(self, "best_val_score") or score > self.best_val_score:
                        self.best_val_score = score
                        tag = "best_blur" if VAL_SCORE_MODE == "avg_blur" and not VAL_SCORE_KEYS else "best_val"
                        self.model_saver.save_checkpoint(model, tag)
                        self.logger.info(f"[best] final improved to {score:.6f}; saved checkpoint '{tag}.pth'")
                        ##
                self.logger.info('Training done')
                break
            iter_delta = 0

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        optimizer.zero_grad()

        results = model.forward(batch, training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()
        loss.backward()
        optimizer.step()

        if step % self.experiment.logger.log_interval == 0:
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                    step, epoch, loss.item(), self.current_lr))
            self.logger.add_scalar('loss', loss, step)
            self.logger.add_scalar('learning_rate', self.current_lr, step)
            for name, metric in metrics.items():
                self.logger.add_scalar(name, metric.mean(), step)
                self.logger.info('%s: %6f' % (name, metric.mean()))

            self.logger.report_time('Logging')

    def validate(self, validation_loaders, model, epoch, step):
        all_matircs = {}
        model.eval()
        for name, loader in validation_loaders.items():
            if self.experiment.validation.visualize:
                metrics, vis_images = self.validate_step(
                    loader, model, True)
                self.logger.images(
                    os.path.join('vis', name), vis_images, step)
            else:
                metrics, vis_images = self.validate_step(loader, model, False)
            for _key, metric in metrics.items():
                key = name + '/' + _key
                if key in all_matircs:
                    all_matircs[key].update(metric.val, metric.count)
                else:
                    all_matircs[key] = metric

        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        self.logger.metrics(epoch, self.steps, all_matircs)
        model.train()
        return all_matircs

    def validate_step(self, data_loader, model, visualize=False):
        raw_metrics = []
        vis_images = dict()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred)

            res = self.structure.measurer.validate_measure(batch, output)
            if isinstance(res, tuple):
                if len(res) == 2:
                    raw_metric, interested = res
                else:
                    raw_metric = res[0]
                    interested = None
            else:
                raw_metric, interested = res, None

            raw_metrics.append(raw_metric)

            if visualize and self.structure.visualizer and interested is not None:
                vis_image = self.structure.visualizer.visualize(batch, output, interested)
                vis_images.update(vis_image)

        metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
        return metrics, vis_images

    def to_np(self, x):
        return x.cpu().data.numpy()
