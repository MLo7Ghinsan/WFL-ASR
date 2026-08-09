import os
import torch
import yaml
import click
import soundfile as sf
import torchaudio
import numpy as np

from model import BIOPhonemeTagger
from utils import (
    decode_bio_tags,
    save_lab,
    load_phoneme_list,
    load_langs,
    load_phoneme_merge_map,
    canonical_to_lang,
    load_phones_txt,
    forced_align_bio,
    merge_adjacent_segments,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_wavs(path):
    if os.path.isfile(path) and path.lower().endswith(".wav"):
        return [path]
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".wav")
        ]
    raise ValueError(f"--input must be a .wav file or a directory: {path}")


def find_matching_txt(wav_path):
    base, _ = os.path.splitext(wav_path)
    txt_path = base + ".txt"
    return txt_path if os.path.isfile(txt_path) else None


def constrained_decode(logits, id2label):
    preds = []
    prev_tag = "O"
    prev_ph = None
    probs = torch.softmax(logits, dim=-1)

    for t in range(logits.shape[0]):
        step_probs = probs[t].clone()

        for i in range(logits.shape[-1]):
            label = id2label[i]
            if label.startswith("I-"):
                ph = label[2:]
                if not (prev_ph == ph and prev_tag in [f"B-{ph}", f"I-{ph}"]):
                    step_probs[i] = 0.0

        if torch.sum(step_probs) == 0:
            step_probs = probs[t].clone()

        best_id = torch.argmax(step_probs).item()
        best_tag = id2label[best_id]

        preds.append(best_tag)
        prev_tag = best_tag
        prev_ph = best_tag[2:] if best_tag != "O" else None

    return preds


def apply_hard_silence(segments, audio, sr, threshold, min_duration, silence_phoneme):
    if len(audio) == 0:
        return segments

    frame_length = int(sr * 0.01)
    if frame_length < 1:
        frame_length = 1

    pad_len = (frame_length - (len(audio) % frame_length)) % frame_length
    padded_audio = np.pad(np.abs(audio), (0, pad_len), mode="constant")

    frames = padded_audio.reshape(-1, frame_length)
    frame_max = np.max(frames, axis=1)

    is_silent_frame = frame_max < threshold

    silence_intervals = []
    in_silence = False
    start_frame = 0

    for i, silent in enumerate(is_silent_frame):
        if silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not silent and in_silence:
            in_silence = False
            duration = (i - start_frame) * 0.01
            if duration >= min_duration:
                silence_intervals.append((start_frame * 0.01, i * 0.01))

    if in_silence:
        duration = (len(is_silent_frame) - start_frame) * 0.01
        if duration >= min_duration:
            silence_intervals.append((start_frame * 0.01, len(is_silent_frame) * 0.01))

    if not silence_intervals:
        return segments

    temp_segments = segments.copy()
    
    for sil_start, sil_end in silence_intervals:
        next_temp_segments = []
        for s_start, s_end, s_label in temp_segments:
            if s_end <= sil_start or s_start >= sil_end:
                next_temp_segments.append((s_start, s_end, s_label))
                continue

            if s_start < sil_start:
                next_temp_segments.append((s_start, sil_start, s_label))
            if s_end > sil_end:
                next_temp_segments.append((sil_end, s_end, s_label))

        temp_segments = next_temp_segments

    for s, e in silence_intervals:
        temp_segments.append((s, e, silence_phoneme))

    temp_segments.sort(key=lambda x: x[0])
    return temp_segments


def process_audio(
    model,
    audio,
    sr,
    config,
    device,
    lang_id=None,
    merge_map=None,
    lang_name=None,
    phones=None,
    no_use_offset=False,
):
    original_duration = len(audio) / sr
    pad_sec = 0.5
    pad_samples = int(pad_sec * sr)
    audio = np.pad(audio, (0, pad_samples), mode='constant')

    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    total_len = len(audio)

    MAX_SEC = 28.0
    CHUNK_SIZE = int(MAX_SEC * sr)

    if lang_id is not None:
        lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
    else:
        lang_tensor = torch.zeros(1, dtype=torch.long).to(device)

    accumulated_logits = []
    accumulated_offsets = []

    for start in range(0, total_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_len)
        chunk = audio[start:end]

        if len(chunk) < 1600:
            if len(chunk) == 0:
                continue
            pad_res = 1600 - len(chunk)
            chunk = np.pad(chunk, (0, pad_res), mode='constant')

        input_values = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, offsets, _ = model(input_values, lang_tensor)

            logits = logits.squeeze(0).cpu()
            if offsets is not None:
                offsets = offsets.squeeze(0).cpu()

            frame_dur = config["data"]["frame_duration"]
            expected_frames = int(len(audio[start:end]) / sr / frame_dur)

            if logits.size(0) > expected_frames:
                logits = logits[:expected_frames]
                if offsets is not None:
                    offsets = offsets[:expected_frames]

            accumulated_logits.append(logits)
            if offsets is not None:
                accumulated_offsets.append(offsets)

    full_logits = torch.cat(accumulated_logits, dim=0)

    full_offsets = None
    if accumulated_offsets and not no_use_offset:
        full_offsets = torch.cat(accumulated_offsets, dim=0)

    if phones:
        pred_tags = forced_align_bio(full_logits, model.id2label, phones)
    else:
        pred_tags = constrained_decode(full_logits, model.id2label)

    segments = decode_bio_tags(pred_tags, config["data"]["frame_duration"], full_offsets)

    all_segments = []
    for s, e, ph in segments:
        if merge_map and lang_name:
            ph = canonical_to_lang(ph, lang_name, merge_map)
        all_segments.append((s, e, ph))

    valid_segments = []
    for s, e, ph in all_segments:
        if s >= original_duration:
            continue
        
        if e > original_duration:
            e = original_duration
        
        valid_segments.append((s, e, ph))

    if not valid_segments:
        return [(0.0, original_duration, "sil")]

    valid_segments.sort(key=lambda x: x[0])
    final_segments = []

    if valid_segments and valid_segments[0][0] > 0.0:
        final_segments.append((0.0, valid_segments[0][0], "SP"))

    for i, (s, e, ph) in enumerate(valid_segments):
        if final_segments:
            prev_s, prev_e, prev_ph = final_segments[-1]
            if prev_e < s:
                final_segments[-1] = (prev_s, s, prev_ph)

        final_segments.append((s, e, ph))

    if final_segments:
        last_s, last_e, last_ph = final_segments[-1]
        if last_e < original_duration:
            final_segments[-1] = (last_s, original_duration, last_ph)

    return final_segments


@click.command()
@click.option("--input", "-i", "input_path", default="infer_test", help="Path to a .wav file or folder containing .wav files")
@click.option("--checkpoint", "-ckpt", default="checkpoints_no_env/model.ckpt", help="Path to WFL .ckpt file")
@click.option("--config", "-c", default="checkpoints_no_env/config.yaml", help="Path to config file")
@click.option("--lang-id", "-l", type=int, default=None, help="Language ID (int) used during training. Example: `-l 0`")
@click.option("--no_use_offset", is_flag=True, help="Disable offset head refinement (offsets ON by default).")
# long silence stuff
@click.option("--silence-phoneme", default="SP", help="The phoneme label to use for hard-coded silence (default: SP)")
@click.option("--silence-threshold", default=0.005, type=float, help="Amplitude threshold (0.0-1.0) to consider as silence")
@click.option("--min-silence-duration", default=0.5, type=float, help="Minimum duration (seconds) required to trigger hard silence")
def main(input_path, checkpoint, config, lang_id, no_use_offset, silence_phoneme, silence_threshold, min_silence_duration):
    cfg = load_config(config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available else "cpu"
    print(f"Running on: {device}")

    save_dir = cfg["output"]["save_dir"]
    phonemes_path = os.path.join(save_dir, "phonemes.txt")

    if not os.path.exists(phonemes_path):
        print(f"Error: {phonemes_path} not found.")
        return

    labels = load_phoneme_list(phonemes_path)
    merge_map = load_phoneme_merge_map(os.path.join(save_dir, "phoneme_merge_map.json"))

    lang_name = None
    if lang_id is not None:
        lang_path = os.path.join(save_dir, "langs.txt")
        if os.path.exists(lang_path):
            lang2id = load_langs(lang_path)
            id2lang = {v: k for k, v in lang2id.items()}
            lang_name = id2lang.get(lang_id)
            print(f"Language: {lang_name} (ID: {lang_id})")

    print("Loading model...")
    model = BIOPhonemeTagger(cfg, labels).to(device)
    model.eval()

    # weights_only=False because I dont like the the 'untrusted-models' warning
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_data["state_dict"] if "state_dict" in checkpoint_data else checkpoint_data

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    files = collect_wavs(input_path)
    print(f"Found {len(files)} files.")

    for wav_path in files:
        print(f"Processing: {wav_path}")

        # auto-detect .txt
        txt_path = find_matching_txt(wav_path)
        phones = None
        if txt_path:
            try:
                phones = load_phones_txt(txt_path)
                if phones:
                    print(f"  Forced-align enabled (found: {os.path.basename(txt_path)})")
                else:
                    phones = None
            except Exception as e:
                print(f"  Warning: failed to read {txt_path}: {e}")
                phones = None

        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            continue

        if sr != cfg["data"]["sample_rate"]:
            audio_t = torch.tensor(audio, dtype=torch.float32)
            if audio_t.dim() > 1:
                audio_t = audio_t.mean(dim=1)
            audio = torchaudio.functional.resample(audio_t, sr, cfg["data"]["sample_rate"]).numpy()
            sr = cfg["data"]["sample_rate"]

        segments = process_audio(
            model,
            audio,
            sr,
            cfg,
            device,
            lang_id=lang_id,
            merge_map=merge_map,
            lang_name=lang_name,
            phones=phones,
            no_use_offset=no_use_offset,
        )

        if cfg.get("postprocess", {}).get("merge_segments", "right") != "none":
            segments = merge_adjacent_segments(segments, cfg["postprocess"]["merge_segments"])

        # Apply hard silence ONLY if we are NOT using forced alignment
        # Forced alignment already knows where silence is based on the text "SP" tag if its in the txt
        # adding heuristic silence on top of forced alignment usually breaks things so yea no
        if phones is None:
            segments = apply_hard_silence(
                segments,
                audio,
                sr,
                threshold=silence_threshold,
                min_duration=min_silence_duration,
                silence_phoneme=silence_phoneme
            )

        out_path = wav_path.replace(".wav", ".lab")
        save_lab(out_path, segments)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
