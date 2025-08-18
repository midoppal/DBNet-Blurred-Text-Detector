from concern.config import Configurable, State
import os
import torch


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']

    # def restore_model(self, model, device, logger):
    #     if self.resume is None:
    #         return

    #     if not os.path.exists(self.resume):
    #         self.logger.warning("Checkpoint not found: " +
    #                             self.resume)
    #         return

    #     logger.info("Resuming from " + self.resume)
    #     state_dict = torch.load(self.resume, map_location=device)
    #     model.load_state_dict(state_dict, strict=False)
    #     logger.info("Resumed from " + self.resume)
    def restore_model(self, model, device, logger):
        if self.resume is None:
            return
        if not os.path.exists(self.resume):
            logger.warning("Checkpoint not found: " + self.resume)
            return

        logger.info("Resuming from " + self.resume)

        cpu = torch.device('cpu')

        # 1) ensure model is on CPU while loading (prevents storage swap across devices)
        model.to(cpu)

        # 2) force all storages to CPU (lambda handles legacy pickles)
        ckpt = torch.load(self.resume, map_location=lambda storage, loc: storage.cpu())

        # 3) extract state dict (handles multiple formats)
        if isinstance(ckpt, dict):
            state = (ckpt.get('state_dict')
                    or ckpt.get('model')
                    or ckpt.get('model_state')
                    or ckpt.get('model_state_dict')
                    or ckpt)
        else:
            state = ckpt

        # (optional) strip DataParallel prefix
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.info(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

        # 4) now move to the requested device
        model.to(device)
        print("after  load: model device =", next(model.parameters()).device)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
