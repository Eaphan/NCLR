import os
import yaml
import logging
import warnings
import importlib

import torch
import datasets

import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import confusion_matrix

from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from utils.utils import wblue, wgreen
from utils.callbacks import CustomProgressBar
from transforms import get_transforms, get_input_channels
from utils.confusion_matrix import ConfusionMatrix


def logs_file(filepath, epoch, log_data):
    if not os.path.exists(filepath):
        log_str = "epoch"
        for key, value in log_data.items():
            log_str += f", {key}"
        log_str += "\n"
        with open(filepath, "a+") as logs:
            logs.write(log_str)
            logs.flush()

    # write the logs
    log_str = f"{epoch}"
    for key, value in log_data.items():
        log_str += f", {value}"
    log_str += "\n"
    with open(filepath, "a+") as logs:
        logs.write(log_str)
        logs.flush()


def isnan(x):
    return x != x


def mean(ls, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    ls = iter(ls)
    if ignore_nan:
        ls = ifilterfalse(isnan, ls)
    try:
        n = 1
        acc = next(ls)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(ls, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LightningDownstreamTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.config = config

        if config["network"]["backbone_params"] is None:
            config["network"]["backbone_params"] = {}
        config["network"]["backbone_params"]["in_channels"] = get_input_channels(config["inputs"])
        config["network"]["backbone_params"]["out_channels"] = config["downstream"]["num_classes"]

        backbone_name = "networks.backbone."
        if config["network"]["framework"] is not None:
            backbone_name += config["network"]["framework"]
        model_module = importlib.import_module(backbone_name)
        model = getattr(model_module, config["network"]["backbone"])
        self.net = model(**config["network"]["backbone_params"])

        if config["downstream"]["checkpoint_name"] is not None:
            logging.info("Loading the weights from pretrained network")
            ckpt = torch.load(
                os.path.join(
                    config["downstream"]["checkpoint_dir"],
                    config["downstream"]["checkpoint_name"]
                )
            )
            if 'state_dict' in ckpt:
                # ckpt = {k[8:]: v for k, v in ckpt['state_dict'].items()}
                ckpt = {k.replace('model_points.', '').replace('encoder.', ''): v for k, v in ckpt['state_dict'].items()}
                if 'final.kernel' in ckpt:
                    del ckpt['final.kernel'], ckpt['final.bias']                
                if 'final.0.weight' in ckpt:
                    del ckpt['final.0.weight'], ckpt['final.0.bias']
                elif 'classifier.0.bias' in ckpt:
                    del ckpt['classifier.0.weight'], ckpt['classifier.0.bias']
            elif 'model_state' in ckpt:
                ckpt = {k.replace("pc_encoder.backbone_3d.", "").replace("backbone_3d.", ""): v for k, v in ckpt['model_state'].items()}
                ckpt = {k:v for k, v in ckpt.items() if k in self.net.state_dict()}
                # del ckpt['stem.0.kernel']
            elif 'model' in ckpt:
                ckpt = ckpt['model']
            for k in self.net.state_dict():
                if k not in ckpt.keys():
                    print("  key missing", k)
            self.net.load_state_dict(ckpt, strict=False)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config["downstream"]["ignore_index"])
        self.num_classes = config["downstream"]["num_classes"]

    def forward(self, data):
        return self.net(data)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"])(self.parameters(), **self.config["optimizer_params"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config["training"]["max_epochs"],
            # eta_min=0,
        )
        return [optimizer], [scheduler]

    def compute_confusion_matrix(self, output_data):
        outputs = output_data["predictions"].squeeze(-1)
        occupancies = output_data["occupancies"].float()

        output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
        target_np = occupancies.cpu().numpy().astype(int)
        cm = confusion_matrix(
            target_np.ravel(), output_np.ravel(), labels=list(range(2))
        )
        return cm

    def compute_loss(self, predictions, targets, prefix):
        loss = self.criterion(predictions, targets)
        # log also the total loss
        self.log(prefix + "/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def compute_log_data(self, outputs, cm, prefix):
        # iou_per_class = cm.get_per_class_iou()
        miou = cm.get_mean_iou()
        fiou = cm.get_freqweighted_iou()
        self.log(prefix + "/miou", miou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(prefix + "/fiou", fiou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        log_data = {}
        log_data["miou"] = miou
        log_data["fiou"] = fiou
        log_data["steps"] = self.global_step

        return log_data

    def get_description_string(self, log_data):
        from datetime import datetime
        current_time = datetime.now()
        desc = f"Time: {current_time}, Epoch {self.current_epoch} |"
        for key, value in log_data.items():
            if "iou" in key:
                desc += f"{key}:{value*100:.2f} |"
            elif "steps" in key:
                desc += f"{key}:{value} |"
            else:
                desc += f"{key}:{value:.3e} |"
        return desc

    def on_train_epoch_start(self) -> None:
        self.train_cm = ConfusionMatrix(self.num_classes, 0)
        return super().on_train_epoch_start()

    def training_step(self, train_batch, batch_idx):
        torch.cuda.empty_cache()
        predictions = self.forward(train_batch)
        loss = self.compute_loss(predictions, train_batch["y"], prefix="train")

        targets = train_batch["y"].cpu().numpy()
        predictions = torch.nn.functional.softmax(predictions[:, 1:], dim=1).max(dim=1)[1].cpu().numpy() + 1
        self.train_cm.update(targets, predictions)

        miou = self.train_cm.get_mean_iou()
        self.log("iou", miou, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, outputs):
        log_data = self.compute_log_data(outputs, self.train_cm, prefix="train")

        os.makedirs(self.logger.log_dir, exist_ok=True)
        logs_file(os.path.join(self.logger.log_dir, "logs_train.csv"), self.current_epoch, log_data)

        if (self.global_step > 0) and (not self.config["interactive_log"]):
            desc = "Train " + self.get_description_string(log_data)
            print(wblue(desc))

    def on_validation_epoch_start(self) -> None:
        self.val_cm = ConfusionMatrix(self.num_classes, 0)
        return super().on_validation_epoch_start()

    def validation_step(self, val_batch, batch_idx):
        torch.cuda.empty_cache()
        predictions = self.forward(val_batch)
        loss = self.compute_loss(predictions, val_batch["y"], prefix="val")

        targets = val_batch["y"].cpu().numpy()
        predictions = torch.nn.functional.softmax(predictions[:, 1:], dim=1).max(dim=1)[1].cpu().numpy() + 1
        self.val_cm.update(targets, predictions)

        miou = self.val_cm.get_mean_iou()
        self.log("iou", miou, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def validation_epoch_end(self, outputs):
        if self.global_step > 0:

            log_data = self.compute_log_data(outputs, self.val_cm, prefix="val")

            os.makedirs(self.logger.log_dir, exist_ok=True)
            logs_file(os.path.join(self.logger.log_dir, "logs_val.csv"), self.current_epoch, log_data)

            if (not self.config["interactive_log"]):
                desc = "Val " + self.get_description_string(log_data)
                print(wgreen(desc))


def get_savedir_name(config):
    savedir = f"{config['network']['backbone']}"
    savedir += f"_{config['dataset_name']}"
    savedir += f"_{config['manifold_points']}"
    savedir += f"_{config['train_split']}Split"
    savedir += f"_{config['downstream']['skip_ratio']}"
    savedir += f"_{config['downstream']['seed_offset']}"
    savedir += f"_{config['downstream']['checkpoint_name']}"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"

    return savedir


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):

    warnings.filterwarnings("ignore", category=UserWarning)

    config = OmegaConf.to_container(config)["cfg"]

    config["training"]["max_epochs"] = config["downstream"]["max_epochs"]
    config["training"]["batch_size"] = config["downstream"]["batch_size"]
    config["training"]["val_interval"] = config["downstream"]["val_interval"]

    if config["downstream"]["checkpoint_dir"] is not None:
        config["save_dir"] = config["downstream"]["checkpoint_dir"]

    logging.getLogger().setLevel(config["logging"])

    logging.info("Getting the dataset")
    DatasetClass = getattr(datasets, config["dataset_name"])

    train_transforms = get_transforms(config, train=True, downstream=True)
    test_transforms = get_transforms(config, train=False, downstream=True)

    # build the dataset
    train_dataset = DatasetClass(
        config["dataset_root"],
        split=config["train_split"],
        transform=train_transforms,
        skip_ratio=config['downstream']["skip_ratio"],
        seed_offset=config['downstream']["seed_offset"]
    )
    val_dataset = DatasetClass(
        config["dataset_root"],
        split=config["val_split"],
        transform=test_transforms,
    )

    logging.info(f"Batch size - {config['training']['batch_size']}")

    # build the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["threads"],
        follow_batch=["voxel_coords", "voxel_proj_yx"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["threads"],
        follow_batch=["voxel_coords", "voxel_proj_yx"]
    )

    logging.info("Creating trainer")

    savedir_root = get_savedir_name(config)
    # savedir_root = os.path.join(config["save_dir"], config["name"], savedir_root)
    savedir_root = os.path.join(config["save_dir"], "Downstream", savedir_root)

    logging.info(f"Savedir_root - {savedir_root}")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir_root)
    trainer = pl.Trainer(
        # accelerator=config["device"],
        # devices=config["num_device"],
        gpus=config["num_device"],
        check_val_every_n_epoch=config["training"]["val_interval"],
        logger=tb_logger,
        max_epochs=config["training"]["max_epochs"],
        resume_from_checkpoint=config["resume"],
        callbacks=[
            CustomProgressBar(refresh_rate=int(config["interactive_log"]))
        ]
    )

    logging.info(f"Saving at {trainer.logger.log_dir}")
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(trainer.logger.log_dir, "config.yaml"), "w"), default_flow_style=False)

    model = LightningDownstreamTrainer(config)
    trainer.fit(model, train_loader, val_loader, ckpt_path=config["resume"])

    model_path = os.path.join(savedir_root, 'final_model.ckpt')
    trainer.save_checkpoint(model_path)


if __name__ == "__main__":
    main()
