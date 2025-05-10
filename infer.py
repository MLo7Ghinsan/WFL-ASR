import sys
import os
import torch
import yaml
import soundfile as sf
import torchaudio
import numpy as np
from model import BIOPhonemeTagger
from utils import decode_bio_tags, save_lab, load_phoneme_list, merge_adjacent_segments, load_langs
from scipy.ndimage import median_filter

frame_duration = 0.02  # ~20ms per frame
MAX_SEGMENT_DURATION = 30.0

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def split_audio(audio, sr, max_duration=MAX_SEGMENT_DURATION):
    total_samples = len(audio)
    samples_per_segment = int(max_duration * sr)
    segments = []

    for start in range(0, total_samples, samples_per_segment):
        end = min(start + samples_per_segment, total_samples)
        segments.append(audio[start:end])

    return segments

def align_phoneme_list(segments_pred, forced_list):
    result = []
    pred_idx = 0
    forced_idx = 0
    used_preds = set()

    pred_map = [None] * len(forced_list)
    for f_i, f_ph in enumerate(forced_list):
        for p_i in range(pred_idx, len(segments_pred)):
            _, _, p_ph = segments_pred[p_i]
            if p_ph == f_ph and p_i not in used_preds:
                pred_map[f_i] = p_i
                used_preds.add(p_i)
                pred_idx = p_i + 1
                break
    pred_ptr = 0
    for f_i, f_ph in enumerate(forced_list):
        if pred_map[f_i] is None:
            while pred_ptr < len(segments_pred) and pred_ptr in used_preds:
                pred_ptr += 1
            if pred_ptr < len(segments_pred):
                pred_map[f_i] = pred_ptr
                used_preds.add(pred_ptr)
                pred_ptr += 1

    for f_i, f_ph in enumerate(forced_list):
        p_i = pred_map[f_i]
        if p_i is not None and p_i < len(segments_pred):
            s, e, _ = segments_pred[p_i]
            result.append((s, e, f_ph))
    return result

def process_segments(model, segments, sr, config, device, lang_id=None):
    all_segments = []
    current_time = 0.0

    for segment in segments:
        if len(segment) > 0:
            segment = segment / (max(abs(segment)) + 1e-8)

        input_values = torch.tensor(segment, dtype=torch.float32).to(device)

        logits_list = []
        offsets_list = []

        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))

        if lang_id is not None:
            # Check for OOR
            try:
              for lid in lang2id.values():
                if lang_id <= lid:
                  pass
                else:
                  raise ValueError(f"Error: Language ID ({lang_id}) is higher than the latest ID ({lid}) of this model.\n Languages and Codes available: {lang2id}")
                  break
            except ValueError as e:
              print(e)
              sys.exit(0)

            lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
            output = model(input_values, lang_tensor)
            logits, offsets = output if isinstance(output, tuple) else (output, None)
            logits_list.append(logits)
            if offsets is not None:
                offsets_list.append(offsets)
        else:            
            for lid in lang2id.values():
                lang_tensor = torch.tensor([lid], dtype=torch.long).to(device)
                output = model(input_values, lang_tensor)
                logits, offsets = output if isinstance(output, tuple) else (output, None)
                logits_list.append(logits)
                if offsets is not None:
                    offsets_list.append(offsets)

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        avg_offsets = torch.mean(torch.stack(offsets_list), dim=0).squeeze(0) if offsets_list else None

        pred_ids = torch.argmax(avg_logits, dim=-1).squeeze(0).cpu().numpy()
        smoothed_ids = median_filter(pred_ids, size=config["postprocess"]["median_filter"])

        pred_tags = [model.id2label[i] for i in smoothed_ids]
        segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration, offsets=avg_offsets)

        shifted_segments = [(start + current_time, end + current_time, ph) for start, end, ph in segments_pred]
        all_segments.extend(shifted_segments)
        current_time += len(segment) / sr

    return all_segments

def infer_audio(audio_path, config_path="config.yaml", checkpoint_path="best_model.pt", output_lab_path=None, device="cuda", lang_id=None):
    config = load_config(config_path)
    phoneme_txt = audio_path.replace(".wav", ".txt")
    forced = None

    labels = load_phoneme_list(os.path.join(config["output"]["save_dir"], "phonemes.txt"))
    model = BIOPhonemeTagger(config, labels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    if os.path.exists(phoneme_txt):
        forced = []
        with open(phoneme_txt, "r", encoding="utf-8") as f:
            for line in f:
                forced.extend(line.strip().split())
        print(f"Loaded forced phoneme list with {len(forced)} phonemes.")

    audio, sr = sf.read(audio_path)
    if sr != config["data"]["sample_rate"]:
        audio = torchaudio.functional.resample(torch.tensor(audio), orig_freq=sr, new_freq=config["data"]["sample_rate"]).numpy()
        sr = config["data"]["sample_rate"]

    if len(audio) > 0:
        audio = audio / (max(abs(audio)) + 1e-8)

    if len(audio) / sr > MAX_SEGMENT_DURATION:
        print(f"Audio is too long ({len(audio)/sr:.1f}s), splitting...")
        segments = split_audio(audio, sr)
        segments_pred = process_segments(model, segments, sr, config, device, lang_id)
    else:
        inp = torch.tensor(audio, dtype=torch.float32).to(device)
        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))

        logits_list = []
        offsets_list = []

        if lang_id is not None:
            # Check for OOR
            try:
              for lid in lang2id.values():
                if lang_id <= lid:
                  pass
                else:
                  raise ValueError(f"Error: Language ID ({lang_id}) is higher than the latest ID ({lid}) of this model.\n Languages and Codes available: {lang2id}")
                  break
            except ValueError as e:
              print(e)
              sys.exit(0)
            
            lt = torch.tensor([lang_id], dtype=torch.long).to(device)
            out = model(inp, lt)
            logits, offsets = out if isinstance(out, tuple) else (out, None)
            logits_list.append(logits)
            if offsets is not None:
                offsets_list.append(offsets)
        else:            
            for lid in lang2id.values():
                lt = torch.tensor([lid], dtype=torch.long).to(device)
                out = model(inp, lt)
                logits, offsets = out if isinstance(out, tuple) else (out, None)
                logits_list.append(logits)
                if offsets is not None:
                    offsets_list.append(offsets)

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        avg_offsets = torch.mean(torch.stack(offsets_list), dim=0).squeeze(0) if offsets_list else None

        pred_ids = torch.argmax(avg_logits, dim=-1).squeeze(0).cpu().numpy()
        smoothed_ids = median_filter(pred_ids, size=config["postprocess"]["median_filter"])
        tags = [labels[i] for i in smoothed_ids]
        segments_pred = decode_bio_tags(tags, frame_duration=frame_duration, offsets=avg_offsets)

    if config["postprocess"]["merge_segments"] != "none":
        segments_pred = merge_adjacent_segments(segments_pred, mode=config["postprocess"]["merge_segments"])

    if forced is not None:
        aligned = align_phoneme_list(segments_pred, forced)
        if "SP" not in forced and "AP" not in forced:
            before = [s for s in segments_pred if s[2] in ("SP", "AP") and s[1] <= aligned[0][0]]
            after = [s for s in segments_pred if s[2] in ("SP", "AP") and s[0] >= aligned[-1][1]]
            segments_pred = before + aligned + after
        else:
            segments_pred = aligned

    if output_lab_path:
        dir_path = os.path.dirname(output_lab_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        save_lab(output_lab_path, segments_pred)
        print(f"Predictions saved to: {output_lab_path}")

    return segments_pred

def infer_folder(folder_path: str, config_path: str = "config.yaml", checkpoint_path: str = "best_model.pt", output_dir: str = "outputs", device: str = "cuda", lang_id: int = None):
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    os.makedirs(output_dir, exist_ok=True)

    for wav_file in wav_files:
        full_audio_path = os.path.join(folder_path, wav_file)
        output_lab_path = os.path.join(output_dir, wav_file.replace(".wav", ".lab"))

        print(f"\nInferencing: {wav_file}")
        segments = infer_audio(
            audio_path=full_audio_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_lab_path=output_lab_path,
            device=device,
            lang_id=lang_id
        )

        print("Predicted segments:")
        for seg in segments:
            start, end, ph = seg
            print(f"({round(start, 2)}, {round(end, 2)}, {ph})")

if __name__ == "__main__":
    import click
    from pathlib import Path
    @click.command(help='Infer with WFL')
    @click.argument('path', metavar='PATH')
    @click.option('--checkpoint', '-ckpt', type=str, required=True, help='Path to WFL Checkpoint.')
    @click.option('--config', '-c', type=str, required=True, help='Path to Config file.')
    @click.option('--output', '-o', type=str, required=False, default=".", help='Path to output labels.')
    @click.option('--lang-id', '-l', type=int, required=False, default=None, help='Language ID.')
    def main(path, checkpoint, config, output, lang_id) -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        folder = False
        inf_path = Path(path)
        checkpoint_path = Path(checkpoint)
        config_path = Path(config)
        if output == ".":
            output_path = inf_path
        else:
            output_path = output
        if not inf_path.exists():
            print(f"Unable to locate folder {str(inf_path)}")
            sys.exit(1)
        if inf_path.is_dir():
            folder = True
        if lang_id <= -1:
          lang_id = None

        if folder:
            infer_folder(
                folder_path=str(inf_path),
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                output_dir=str(output_path),
                device=device,
                lang_id=lang_id
            )
        else:
            segments = infer_audio(
                audio_path=str(inf_path),
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                output_lab_path=str(output_path),
                device=device,
                lang_id=lang_id
            )
            print("Predicted segments:")
            for seg in segments:
                start, end, ph = seg
                print(f"({round(start, 2)}, {round(end, 2)}, {ph})")

    main()
