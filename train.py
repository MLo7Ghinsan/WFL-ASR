import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import json
import yaml
import torch
import random
import argparse
import numpy as np
import soundfile as sf
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from model import BIOPhonemeTagger, FocalLoss
from utils import decode_bio_tags, visualize_prediction, load_phoneme_list
import pytorch_optimizer as optim


def collate_fn(batch):
    input_values, label_ids, wavs, segments_gt, wav_paths, lang_ids = zip(*batch)
    label_lengths = torch.tensor([len(x) for x in label_ids])
    padded_input = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
    return padded_input, padded_labels, wavs, segments_gt, wav_paths, torch.tensor(lang_ids, dtype=torch.long), label_lengths

class PhonemeDataset(Dataset):
    def __init__(self, dataset_path, label_list, max_seq_len=None, aug_cfg=None):
        with open(dataset_path, "r") as f: self.samples = json.load(f)
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_seq_len = max_seq_len
        self.aug_cfg = aug_cfg or {"enable": False}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        wav, sr = sf.read(sample["wav_path"])
        if sr != 16000: 
            wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()

        if self.aug_cfg.get("enable", False) and random.random() < self.aug_cfg.get("prob", 0.5):
            wav *= random.uniform(*self.aug_cfg.get("volume_range", [0.9, 1.1]))
            if self.aug_cfg.get("noise_std", 0) > 0:
                wav += np.random.normal(0, self.aug_cfg["noise_std"], wav.shape)
            wav = np.clip(wav, -1.0, 1.0)

        label_ids = torch.tensor([self.label2id.get(tag, self.label2id["O"]) for tag in sample["bio_tags"]], dtype=torch.long)
        wav_tensor = torch.tensor(wav, dtype=torch.float32)
        if self.max_seq_len: wav_tensor = wav_tensor[:self.max_seq_len]
        
        return wav_tensor, label_ids, wav, sample["phoneme_segments"], sample["wav_path"], sample["lang_id"]

class WFLDataModule(pl.LightningDataModule):
    def __init__(self, config, label_list):
        super().__init__()
        self.config = config
        self.label_list = label_list
        self.save_dir = config["output"]["save_dir"]
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]

    def setup(self, stage=None):
        dataset_path = os.path.join(self.save_dir, "dataset.json")
        train_dataset = PhonemeDataset(
            dataset_path,
            self.label_list,
            max_seq_len=self.config["data"]["max_seq_len"],
            aug_cfg=self.config.get("augmentation"),
        )
        val_dataset = PhonemeDataset(
            dataset_path,
            self.label_list,
            max_seq_len=self.config["data"]["max_seq_len"],
            aug_cfg={"enable": False},
        )

        paths = [s["wav_path"] for s in train_dataset.samples]
        indices = {p: i for i, p in enumerate(paths)}
        val_count = self.config["data"]["num_val_files"]
        split_path = os.path.join(self.save_dir, "data_split.json")

        if len(indices) != len(paths):
            raise ValueError("Duplicate audio paths in dataset")
        if not 0 < val_count < len(paths):
            raise ValueError("Invalid num_val_files")

        if os.path.exists(split_path):
            with open(split_path, encoding="utf-8") as f:
                split = json.load(f)
        else:
            train, val = random_split(
                sorted(paths),
                [len(paths) - val_count, val_count],
                generator=torch.Generator().manual_seed(
                    self.config["data"].get("split_seed", 42)
                ),
            )
            split = {"train_paths": list(train), "val_paths": list(val)}
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split, f, indent=2, ensure_ascii=False)

        saved = split["train_paths"] + split["val_paths"]
        if (
            len(saved) != len(paths)
            or set(saved) != set(paths)
            or len(split["val_paths"]) != val_count
        ):
            raise ValueError("Split mismatch. Restore the dataset/config or rename data_split.json to create a new split")

        self.train_ds = Subset(
            train_dataset, [indices[p] for p in split["train_paths"]]
        )
        self.val_ds = Subset(
            val_dataset, [indices[p] for p in split["val_paths"]]
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, 
                          collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, 
                          collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

class WFLModel(pl.LightningModule):
    def __init__(self, config, label_list):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) 
        self.config = config
        self.label_list = label_list
        self.id2label = {i: l for i, l in enumerate(label_list)}
        
        self.model = BIOPhonemeTagger(config, label_list)
        
        if config.get("finetune", {}).get("freeze_backbone", False):
            print(">>> Fine-tuning mode: Freezing Conformer backbone.")
            for param in self.model.conformer.parameters():
                param.requires_grad = False
                
        self.criterion = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=-100)
        self.offset_weight = config["model"].get("subframe_loss_weight", 5.0)
        self.frame_duration = config["data"].get("frame_duration", 0.02)
        
        total_val = config["data"]["num_val_files"]
        self.num_vis_samples = min(total_val, 8) 
        
        # MFCC
        self.envelope_loss_weight = config["model"].get("envelope_loss_weight", 0.5)
        hop_length = int(config["data"]["sample_rate"] * self.frame_duration)
        n_mfcc = config["model"].get("envelope_dim", 20)
        
        self.envelope_extractor = torchaudio.transforms.MFCC(
            sample_rate=config["data"]["sample_rate"],
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": hop_length,
                "n_mels": 80,
                "mel_scale": "htk",
            }
        )

    def forward(self, x, lang_ids, max_len=None):
        return self.model(x, lang_ids, max_label_len=max_len)

    def calculate_loss(self, logits, offsets, pred_env, target_env, labels, segs_gt, lengths):
        cls_loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        total_offset_loss = torch.tensor(0.0, device=self.device)
        if offsets is not None:
            target_map = torch.zeros_like(offsets) 
            mask_map = torch.zeros_like(offsets)   
            
            for b_idx in range(len(segs_gt)):
                for start_t, end_t, _ in segs_gt[b_idx]:
                    s_f = int(start_t / self.frame_duration)
                    e_f = int(end_t / self.frame_duration)
                    
                    if s_f < lengths[b_idx]:
                        target_map[b_idx, s_f, 0] = (start_t / self.frame_duration) - s_f
                        mask_map[b_idx, s_f, 0] = 1.0
                    if e_f < lengths[b_idx]:
                        target_map[b_idx, e_f, 1] = (end_t / self.frame_duration) - e_f
                        mask_map[b_idx, e_f, 1] = 1.0
            
            diff = torch.abs(offsets - target_map) * mask_map
            total_offset_loss = (diff.sum() / (mask_map.sum() + 1e-8)) * self.offset_weight

        # env l1
        env_loss = torch.tensor(0.0, device=self.device)
        if target_env is not None and pred_env is not None:
            mask = (labels != -100).unsqueeze(-1).expand_as(pred_env)
            min_len = min(pred_env.size(1), target_env.size(1))
            
            p_env = pred_env[:, :min_len, :] * mask[:, :min_len, :]
            t_env = target_env[:, :min_len, :] * mask[:, :min_len, :]
            
            env_loss = torch.nn.functional.l1_loss(p_env, t_env, reduction='sum') / (mask.sum() + 1e-8)
            env_loss = env_loss * self.envelope_loss_weight

        total_loss = cls_loss + total_offset_loss + env_loss
        return total_loss, cls_loss, total_offset_loss, env_loss

    def training_step(self, batch, batch_idx):
        inputs, labels, wavs, segs_gt, _, langs, lengths = batch
        max_len = torch.max(lengths) if lengths.numel() > 0 else 0
        
        wav_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(w, dtype=torch.float32) for w in wavs], batch_first=True
        ).to(self.device)
        
        with torch.no_grad():
            target_env = self.envelope_extractor(wav_tensor).transpose(1, 2)

        logits, offsets, pred_env = self(inputs, langs, max_len)
        loss, cls_loss, off_loss, env_loss = self.calculate_loss(
            logits, offsets, pred_env, target_env, labels, segs_gt, lengths
        )
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("train/cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=inputs.size(0))
        self.log("train/off_loss", off_loss, on_step=False, on_epoch=True, batch_size=inputs.size(0))
        self.log("train/env_loss", env_loss, on_step=False, on_epoch=True, batch_size=inputs.size(0))
        return loss

    def on_validation_epoch_start(self):
        self.val_vis_count = 0

    def validation_step(self, batch, batch_idx):
        inputs, labels, wavs, segs_gt, _, langs, lengths = batch
        max_len = torch.max(lengths) if lengths.numel() > 0 else 0
        
        wav_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(w, dtype=torch.float32) for w in wavs], batch_first=True
        ).to(self.device)
        
        with torch.no_grad():
            target_env = self.envelope_extractor(wav_tensor).transpose(1, 2)

        logits, offsets, pred_env = self(inputs, langs, max_len)
        loss, _, _, _ = self.calculate_loss(logits, offsets, pred_env, target_env, labels, segs_gt, lengths)
        
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        acc = (preds == labels)[mask].float().mean() * 100.0
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))

        for i in range(len(wavs)):
            if self.val_vis_count < self.num_vis_samples:
                self._log_visualization(wavs[i], preds[i], lengths[i], offsets[i], segs_gt[i], sample_idx=self.val_vis_count)
                self.val_vis_count += 1

        return loss
    
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        loss = metrics.get("val/loss", 0.0)
        acc = metrics.get("val/acc", 0.0)
        print(f"\n[Epoch {self.current_epoch}] Validation Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

    def _log_visualization(self, wav, pred_ids, length, offset_tensor, gt_segments, sample_idx=0):
        pred_ids = pred_ids[:length].cpu().numpy()
        pred_tags = [self.id2label[p] for p in pred_ids]
        curr_offsets = offset_tensor[:length].cpu()
        
        vis_segs = decode_bio_tags(pred_tags, self.frame_duration, offsets=curr_offsets)
        fig = visualize_prediction(wav, 16000, vis_segs, gt_segments)
        
        if self.logger:
            self.logger.experiment.add_figure(f"val/prediction_{sample_idx}", fig, global_step=self.global_step)

    def configure_optimizers(self):
        opt_name = self.config["training"].get("optimizer", "AdamW")
        lr = self.config["training"]["learning_rate"]
        decay = self.config["training"].get("weight_decay", 1e-4)
        
        try:
            opt_cls = getattr(optim, opt_name)
        except AttributeError:
            opt_cls = getattr(torch.optim, opt_name)
            
        optimizer = opt_cls(self.parameters(), lr=lr, weight_decay=decay)
        
        step_size = self.config["training"].get("lr_decay_every_n_epochs", 10)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=self.config["training"]["lr_decay_gamma"]
        )
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="checkpoints_micro/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to .ckpt file to resume training from")
    args = parser.parse_args()

    with open(args.config, "r") as f: 
        config = yaml.safe_load(f)
    
    pl.seed_everything(42)

    save_dir = config["output"]["save_dir"]
    phoneme_path = os.path.join(save_dir, "phonemes.txt")
    if not os.path.exists(phoneme_path):
        raise FileNotFoundError(f"Phoneme list not found at {phoneme_path}. Run preprocess.py first.")
        
    label_list = load_phoneme_list(phoneme_path)
    
    data_module = WFLDataModule(config, label_list)
    
    ft_cfg = config.get("finetune", {})
    if ft_cfg.get("enabled", False) and ft_cfg.get("checkpoint_path"):
        ckpt = ft_cfg["checkpoint_path"]
        print(f"Loading weights for fine-tuning from: {ckpt}")
        model = WFLModel.load_from_checkpoint(ckpt, config=config, label_list=label_list)
    else:
        model = WFLModel(config, label_list)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="model-ep{epoch:02d}-val_loss{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=config["training"]["max_checkpoints"],
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    max_epochs = config["training"].get("max_epochs", 100)
    check_val_every_n_epoch = config["training"].get("check_val_every_n_epoch", 1)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger(save_dir=config["training"]["log_dir"], name="lightning_logs"),
        accelerator="auto",
        devices=1,
        precision="32",
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )

    print(f"Starting Training for {max_epochs} epochs (Validation every {check_val_every_n_epoch} epochs)...")
    trainer.fit(model, data_module, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
