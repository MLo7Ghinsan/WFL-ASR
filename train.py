import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import math
import random

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_optimizer as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from infer import continuous_segments, forced_align_viterbi, viterbi_decode
from model import BIOPhonemeTagger, FocalLoss
from utils import (
    ctc_greedy_decode,
    decode_bio_tags,
    load_langs,
    load_phoneme_list,
    visualize_prediction,
)
from finetune_utils import load_wfl_config, read_checkpoint, transfer_wfl_weights


def collate_fn(batch):
    input_values, label_ids, wavs, segments_gt, wav_paths, lang_ids = zip(*batch)
    label_lengths = torch.tensor([len(x) for x in label_ids])
    padded_input = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        label_ids, batch_first=True, padding_value=-100
    )
    return (
        padded_input,
        padded_labels,
        wavs,
        segments_gt,
        wav_paths,
        torch.tensor(lang_ids, dtype=torch.long),
        label_lengths,
    )


class PhonemeDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        label_list,
        max_seq_len=None,
        aug_cfg=None,
        frame_duration=0.02,
    ):
        with open(dataset_path, "r") as f:
            self.samples = json.load(f)
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_seq_len = max_seq_len
        self.frame_duration = frame_duration
        self.aug_cfg = aug_cfg or {"enable": False}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        wav, sr = sf.read(sample["wav_path"], dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, 16000
            ).numpy()

        if self.max_seq_len:
            wav = wav[: self.max_seq_len]

        duration = len(wav) / 16000
        num_frames = math.ceil(len(wav) / (16000 * self.frame_duration))
        tags = sample["bio_tags"][:num_frames]
        segments = [
            (s, min(e, duration), ph)
            for s, e, ph in sample["phoneme_segments"]
            if 0 <= s < min(e, duration)
        ]

        if self.aug_cfg.get("enable", False) and random.random() < self.aug_cfg.get(
            "prob", 0.5
        ):
            wav *= random.uniform(*self.aug_cfg.get("volume_range", [0.9, 1.1]))
            if self.aug_cfg.get("noise_std", 0) > 0:
                wav += np.random.normal(0, self.aug_cfg["noise_std"], wav.shape)
            wav = np.clip(wav, -1.0, 1.0)

        label_ids = torch.tensor(
            [-100 if tag == "O" else self.label2id[tag] for tag in tags],
            dtype=torch.long,
        )
        wav_tensor = torch.tensor(wav, dtype=torch.float32)
        return (
            wav_tensor,
            label_ids,
            wav,
            segments,
            sample["wav_path"],
            sample["lang_id"],
        )


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
        max_seq_len = self.config["data"].get("max_seq_len")
        if self.config["model"].get("encoder_type", "whisper").lower() == "whisper":
            max_seq_len = min(max_seq_len or 480000, 480000)

        train_dataset = PhonemeDataset(
            dataset_path,
            self.label_list,
            max_seq_len=max_seq_len,
            frame_duration=self.config["data"].get("frame_duration", 0.02),
            aug_cfg=self.config.get("augmentation"),
        )
        val_dataset = PhonemeDataset(
            dataset_path,
            self.label_list,
            max_seq_len=max_seq_len,
            frame_duration=self.config["data"].get("frame_duration", 0.02),
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
            raise ValueError(
                "Split mismatch. Restore the dataset/config or rename data_split.json to create a new split"
            )

        self.train_ds = Subset(
            train_dataset, [indices[p] for p in split["train_paths"]]
        )
        self.val_ds = Subset(val_dataset, [indices[p] for p in split["val_paths"]])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class WFLModel(pl.LightningModule):
    def __init__(self, config, label_list, whisper_state=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "whisper_state"])
        self.config = config
        self.label_list = label_list
        self.id2label = {i: l for i, l in enumerate(label_list)}
    
        self.model = BIOPhonemeTagger(config, label_list, whisper_state=whisper_state)

        ft_cfg = config.get("finetune", {})

        if ft_cfg.get("enabled", False):
            if ft_cfg.get("freeze_conformer", False):
                self.model.conformer_layers.requires_grad_(False)
            if ft_cfg.get("freeze_dilated_conv", False):
                self.model.dilated_conv_stack.requires_grad_(False)
            if ft_cfg.get("freeze_boundary_head", False):
                self.model.boundary_offset_head.requires_grad_(False)

        self.criterion = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=-100)
        self.offset_weight = config["model"].get("subframe_loss_weight", 5.0)
        self.ctc_weight = config["model"].get("ctc_loss_weight", 0.5)
        self.frame_duration = config["data"].get("frame_duration", 0.02)

        total_val = config["data"]["num_val_files"]
        self.num_vis_samples = min(total_val, 8)

    def forward(self, x, lang_ids, lengths):
        return self.model(x, lang_ids, lengths=lengths)

    def calculate_loss(self, logits, offsets, labels, segs_gt, lengths):
        cls_loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )

        total_offset_loss = torch.tensor(0.0, device=self.device)
        if offsets is not None:
            target_map = torch.zeros_like(offsets)
            mask_map = torch.zeros_like(offsets)

            for b_idx in range(len(segs_gt)):
                for start_t, end_t, _ in segs_gt[b_idx]:
                    s_f = int(start_t / self.frame_duration)
                    e_f = int(end_t / self.frame_duration)

                    if s_f < lengths[b_idx]:
                        target_map[b_idx, s_f, 0] = start_t / self.frame_duration - s_f
                        mask_map[b_idx, s_f, 0] = 1.0
                    if e_f < lengths[b_idx]:
                        target_map[b_idx, e_f, 1] = end_t / self.frame_duration - e_f
                        mask_map[b_idx, e_f, 1] = 1.0

            diff = torch.abs(offsets - target_map) * mask_map
            total_offset_loss = (
                diff.sum() / (mask_map.sum() + 1e-8)
            ) * self.offset_weight

        total_loss = cls_loss + total_offset_loss
        return total_loss, cls_loss, total_offset_loss

    def calculate_ctc_loss(self, ctc_logits, segs_gt, lengths):
        # target = gt phoneme sequence
        phone2id = self.model.ctc_phone2id
        targets = [[phone2id[ph] for _, _, ph in segs] for segs in segs_gt]
        target_lengths = torch.tensor(
            [len(t) for t in targets], dtype=torch.long, device=self.device
        )
        flat_targets = torch.tensor(
            [i for t in targets for i in t], dtype=torch.long, device=self.device
        )
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
        loss = F.ctc_loss(
            log_probs,
            flat_targets,
            lengths,
            target_lengths,
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )
        return loss / lengths.sum().clamp_min(1)

    def _forced_align_tags(self, logits, phones, insert_forced=True):
        post = self.config.get("postprocess", {})
        fa_args = post.get("forced_alignment_args", {})
        forced = []
        if insert_forced:
            label2id = self.model.label2id
            forced = [
                p for p in (post.get("forced_predict_fa") or [])
                if f"B-{p}" in label2id and f"I-{p}" in label2id
            ]
        return forced_align_viterbi(
            logits,
            self.id2label,
            phones,
            self_loop_penalty=fa_args.get("self_loop_penalty", -4.6),
            forward_penalty=fa_args.get("forward_penalty", -0.6),
            skip_penalty=fa_args.get("skip_penalty", -2.3),
            forced_pred_phones=forced,
            forced_pred_penalty=fa_args.get("forced_predict_penalty", 0.0),
        )

    def _ctc_align(self, logits, offsets, phones, duration):
        tags = self._forced_align_tags(logits, phones)
        return continuous_segments(
            decode_bio_tags(tags, self.frame_duration, offsets.detach().cpu()),
            duration,
        )

    def _boundary_errors(self, logits, offsets, gt_segments):
        phones = [ph for _, _, ph in gt_segments]
        if len(phones) < 2 or len(phones) > logits.size(0):
            return None
        tags = self._forced_align_tags(logits, phones, insert_forced=False)
        pred = decode_bio_tags(tags, self.frame_duration, offsets.detach().cpu())
        if [ph for _, _, ph in pred] != phones:
            return None
        return [abs(p[0] - g[0]) for p, g in zip(pred[1:], gt_segments[1:])]

    def training_step(self, batch, batch_idx):
        inputs, labels, wavs, segs_gt, _, langs, lengths = batch
        logits, offsets, ctc_logits = self(inputs, langs, lengths)
        loss, cls_loss, off_loss = self.calculate_loss(
            logits, offsets, labels, segs_gt, lengths
        )

        total_loss = loss
        if ctc_logits is not None:
            ctc_loss = self.calculate_ctc_loss(ctc_logits, segs_gt, lengths)
            total_loss = loss + self.ctc_weight * ctc_loss
            self.log(
                "train/ctc_loss",
                ctc_loss,
                on_step=False,
                on_epoch=True,
                batch_size=inputs.size(0),
            )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        self.log(
            "train/cls_loss",
            cls_loss,
            on_step=False,
            on_epoch=True,
            batch_size=inputs.size(0),
        )
        self.log(
            "train/off_loss",
            off_loss,
            on_step=False,
            on_epoch=True,
            batch_size=inputs.size(0),
        )
        return total_loss

    def on_validation_epoch_start(self):
        self.val_vis_count = 0

    @staticmethod
    def _collapse_phones(phones):
        collapsed = []
        for phone in phones:
            if not collapsed or phone != collapsed[-1]:
                collapsed.append(phone)
        return collapsed

    @staticmethod
    def _edit_distance(reference, prediction):
        row = list(range(len(prediction) + 1))
        for i, ref in enumerate(reference, 1):
            next_row = [i]
            for j, pred in enumerate(prediction, 1):
                next_row.append(
                    min(
                        row[j] + 1,
                        next_row[j - 1] + 1,
                        row[j - 1] + (ref != pred),
                    )
                )
            row = next_row
        return row[-1]

    def validation_step(self, batch, batch_idx):
        inputs, labels, wavs, segs_gt, _, langs, lengths = batch
        logits, offsets, ctc_logits = self(inputs, langs, lengths)
        loss, cls_loss, off_loss = self.calculate_loss(
            logits, offsets, labels, segs_gt, lengths
        )

        if ctc_logits is not None:
            self.log(
                "val/ctc_loss",
                self.calculate_ctc_loss(ctc_logits, segs_gt, lengths),
                on_step=False,
                on_epoch=True,
                batch_size=inputs.size(0),
            )

        valid = labels != -100
        frame_count = int(valid.sum().item())
        if frame_count:
            acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean() * 100
            self.log(
                "val/acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=frame_count,
            )
        for name, value in (
            ("loss", loss),
            ("cls_loss", cls_loss),
            ("off_loss", off_loss),
        ):
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=inputs.size(0),
            )

        errors, ctc_errors, phone_count = 0, 0, 0
        boundary_errors = []
        for i, wav in enumerate(wavs):
            length = int(lengths[i].item())
            tags = viterbi_decode(
                logits[i, :length],
                self.id2label,
                viterbi_bias=self.config.get("validation", {}).get("viterbi_bias", 5),
            )
            pred_segments = continuous_segments(
                decode_bio_tags(
                    tags, self.frame_duration, offsets[i, :length].detach().cpu()
                ),
                len(wav) / 16000,
            )
            reference = self._collapse_phones([ph for _, _, ph in segs_gt[i]])
            prediction = self._collapse_phones([ph for _, _, ph in pred_segments])
            errors += self._edit_distance(reference, prediction)
            phone_count += len(reference)

            boundary_errors.extend(
                self._boundary_errors(
                    logits[i, :length], offsets[i, :length], segs_gt[i]
                ) or []
            )

            ctc_phones = None
            if ctc_logits is not None:
                ctc_phones = ctc_greedy_decode(
                    ctc_logits[i, :length], self.model.ctc_vocab
                )
                ctc_errors += self._edit_distance(
                    reference, self._collapse_phones(ctc_phones)
                )

            if self.val_vis_count < self.num_vis_samples:
                self._log_visualization(
                    wav, pred_segments, segs_gt[i], self.val_vis_count
                )
                if ctc_phones:
                    self._log_visualization(
                        wav,
                        self._ctc_align(
                            logits[i, :length],
                            offsets[i, :length],
                            ctc_phones,
                            len(wav) / 16000,
                        ),
                        segs_gt[i],
                        self.val_vis_count,
                        tag="ctc_align",
                    )
                self.val_vis_count += 1

        if phone_count:
            self.log(
                "val/per",
                logits.new_tensor(100.0 * errors / phone_count),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=phone_count,
            )
            if ctc_logits is not None:
                self.log(
                    "val/ctc_per",
                    logits.new_tensor(100.0 * ctc_errors / phone_count),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=phone_count,
                )

        if boundary_errors:
            errors_ms = 1000.0 * torch.tensor(
                boundary_errors, dtype=torch.float32, device=self.device
            )
            self.log(
                "val/boundary_mae_ms",
                errors_ms.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=len(boundary_errors),
            )
            self.log(
                "val/boundary_acc_20ms",
                (errors_ms <= 20.0).float().mean() * 100,
                on_step=False,
                on_epoch=True,
                batch_size=len(boundary_errors),
            )
        return loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        ctc_part = (
            f" | CTC PER: {metrics['val/ctc_per']:.2f}%"
            if "val/ctc_per" in metrics
            else ""
        )
        boundary_part = (
            f" | Boundary error: {metrics['val/boundary_mae_ms']:.1f} ms "
            f"({metrics['val/boundary_acc_20ms']:.1f}% within 20 ms)"
            if "val/boundary_mae_ms" in metrics
            else ""
        )
        print(
            f"\n[Epoch {self.current_epoch}] "
            f"Loss: {metrics.get('val/loss', 0.0):.4f} | "
            f"Frame accuracy: {metrics.get('val/acc', 0.0):.2f}% | "
            f"PER: {metrics.get('val/per', float('nan')):.2f}%"
            f"{ctc_part}{boundary_part}"
        )

    def _log_visualization(
        self, wav, pred_segments, gt_segments, sample_idx=0, tag="prediction"
    ):
        fig = visualize_prediction(wav, 16000, pred_segments, gt_segments)
        try:
            if self.logger:
                self.logger.experiment.add_figure(
                    f"val/{tag}_{sample_idx}",
                    fig,
                    global_step=self.global_step,
                )
        finally:
            plt.close(fig)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["wfl_labels"] = list(self.label_list)

        lang_path = os.path.join(
            self.config["output"]["save_dir"],
            "langs.txt",
        )

        if os.path.exists(lang_path):
            checkpoint["wfl_lang2id"] = load_langs(lang_path)
            
    def configure_optimizers(self):
        opt_name = self.config["training"].get("optimizer", "AdamW")
        lr = self.config["training"]["learning_rate"]
        decay = self.config["training"].get("weight_decay", 1e-4)

        try:
            opt_cls = getattr(optim, opt_name)
        except AttributeError:
            opt_cls = getattr(torch.optim, opt_name)

        ft_cfg = self.config.get("finetune", {})

        if ft_cfg.get("enabled", False):
            rates = ft_cfg.get("learning_rates", {})
            groups = []

            def add_group(module, name):
                params = [
                    p for p in module.parameters()
                    if p.requires_grad
                ]

                if params:
                    groups.append({
                        "params": params,
                        "lr": float(rates.get(name, lr)),
                    })

            add_group(self.model.classifier, "classifier")
            add_group(self.model.lang_emb, "language_embeddings")
            add_group(self.model.lang_proj, "language_projection")

            add_group(self.model.conformer_layers, "acoustic_stack")
            add_group(self.model.dilated_conv_stack, "acoustic_stack")
            add_group(self.model.boundary_offset_head, "boundary_head")
            if self.model.ctc_proj is not None:
                add_group(self.model.ctc_layers, "ctc_head")
                add_group(self.model.ctc_proj, "ctc_head")

            if hasattr(self.model, "layer_weights"):
                groups.append({
                    "params": [self.model.layer_weights],
                    "lr": float(rates.get("layer_mixing", lr)),
                })

            optimizer = opt_cls(
                groups,
                weight_decay=decay,
            )

        else:
            optimizer = opt_cls(
                (p for p in self.parameters() if p.requires_grad),
                lr=lr,
                weight_decay=decay,
            )

        step_size = self.config["training"].get("lr_decay_every_n_epochs", 10)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=self.config["training"]["lr_decay_gamma"],
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints_micro/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to .ckpt file to resume training from",
    )
    args = parser.parse_args()

    config = load_wfl_config(args.config)

    pl.seed_everything(42)

    save_dir = config["output"]["save_dir"]
    phoneme_path = os.path.join(save_dir, "phonemes.txt")
    if not os.path.exists(phoneme_path):
        raise FileNotFoundError(
            f"Phoneme list not found at {phoneme_path}. Run preprocess.py first."
        )

    label_list = load_phoneme_list(phoneme_path)

    data_module = WFLDataModule(config, label_list)

    ft_cfg = config.get("finetune", {})

    if ft_cfg.get("enabled", False):
        if args.resume:
            raise ValueError(
                "Use either fine-tuning or --resume, not both."
            )

        source_path = ft_cfg["checkpoint_path"]
        print(f"Fine-tuning WFL from: {source_path}")

        source_ckpt = read_checkpoint(source_path)
        source_state = source_ckpt["state_dict"]

        model = WFLModel(
            config,
            label_list,
            whisper_state=source_state,
        )

        transfer_wfl_weights(
            model,
            source_ckpt,
            source_path,
            save_dir,
            label_list,
        )

        del source_ckpt, source_state

    else:
        model = WFLModel(config, label_list)

    best_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-ep{epoch:02d}-{val/loss:.4f}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=config["training"]["max_checkpoints"],
    )

    latest_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="latest",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    max_epochs = config["training"].get("max_epochs", 100)
    check_val_every_n_epoch = config["training"].get("check_val_every_n_epoch", 1)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[best_checkpoint, latest_checkpoint, lr_monitor],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=config["training"]["log_dir"], name="lightning_logs"
        ),
        accelerator={"auto": "auto", "cpu": "cpu", "gpu": "gpu", "cuda": "gpu"}[
            str(config["model"].get("device", "auto")).lower()
        ],
        devices=1,
        precision=config["training"].get("precision", "32"),
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    print(
        f"Starting Training for {max_epochs} epochs (Validation every {check_val_every_n_epoch} epochs)..."
    )
    trainer.fit(model, data_module, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
