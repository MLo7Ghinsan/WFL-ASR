import math
import os

import click
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from librosa.sequence import _viterbi, viterbi
from librosa.util import tiny
from numba.cuda.stubs import const

from model import BIOPhonemeTagger
from utils import (
    canonical_to_lang,
    decode_bio_tags,
    forced_align_bio,
    load_langs,
    load_phoneme_list,
    load_phoneme_merge_map,
    load_phones_txt,
    merge_adjacent_segments,
    save_lab,
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


def bio_inputs(logits, id2label):
    labels = [id2label[i] for i in range(len(id2label))]
    scores = logits.detach().double().cpu().numpy()

    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Expected logits shaped (frames, labels).")
    if not np.isfinite(scores).all():
        raise ValueError("Decoder received non-finite logits.")
    if any(
        tag != "O" and not (tag.startswith(("B-", "I-")) and len(tag) > 2)
        for tag in labels
    ):
        raise ValueError("Expected O and valid B-/I- labels.")

    keep = [i for i, tag in enumerate(labels) if tag != "O"]
    scores = scores[:, keep]
    labels = [labels[i] for i in keep]

    starts = np.array([tag.startswith("B-") for tag in labels], dtype=bool)
    if not starts.any():
        raise ValueError("Decoder requires at least one B- label.")

    allowed = np.array(
        [
            [
                nxt.startswith("B-")
                or (nxt.startswith("I-") and prev in (f"B-{nxt[2:]}", nxt))
                for nxt in labels
            ]
            for prev in labels
        ],
        dtype=bool,
    )

    return scores, labels, allowed, starts


def constrained_decode(logits, id2label):
    scores, labels, allowed, valid = bio_inputs(logits, id2label)
    preds = []
    for frame in scores:
        best = int(np.argmax(np.where(valid, frame, -np.inf)))
        preds.append(labels[best])
        valid = allowed[best]
    return preds


def viterbi_decode(logits, id2label, viterbi_bias=1):
    if not np.isfinite(viterbi_bias) or viterbi_bias < 1:
        raise ValueError("viterbi_bias must be finite and at least 1.")

    scores, labels, allowed, starts = bio_inputs(logits, id2label)
    if scores.shape[0] == 0:
        return []

    log_probs = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)

    transitions = np.where(allowed, 0.0, -np.inf)
    for j, tag in enumerate(labels):
        if tag.startswith("I-"):
            transitions[allowed[:, j], j] = np.log(viterbi_bias)

    initial = np.where(starts, -np.log(starts.sum()), -np.inf)
    path, _ = _viterbi(log_probs, transitions, initial)

    return [labels[int(i)] for i in path]


def forced_align_viterbi(logits, id2label, phones):
    label2id = {label: id for id, label in id2label.items()}
    # turn to log probs
    log_probs = (
        torch.log_softmax(logits, dim=-1).detach().double().cpu().numpy().transpose()
    )

    # make target sequence
    target_seq = []
    for phn in phones:
        target_seq.extend([f"B-{phn}", f"I-{phn}"])
    target_seq_idx = [label2id[phn] for phn in target_seq]

    # forward pass setup
    K = len(target_seq)
    _, T = log_probs.shape
    scores = np.full((K, T), -np.inf)
    pointers = np.zeros((K, T), dtype=int)

    scores[0, 0] = log_probs[target_seq_idx[0], 0]

    for t in range(1, T):
        for s in range(K):
            state_idx = target_seq_idx[s]

            candidates = []

            if target_seq[s].startswith("B-"):
                # disallow self transition for B phonemes
                candidates.append((s, -np.inf))
            else:
                candidates.append((s, scores[s, t - 1] - 4.6))

            if s > 0:
                candidates.append((s - 1, scores[s - 1, t - 1] - 0.6))

            if (
                s > 1
                and target_seq[s - 1].startswith("I-")
                and target_seq[s].startswith("B-")
            ):
                # I phoneme skip
                candidates.append((s - 2, scores[s - 2, t - 1] - 2.3))

            best_prev_state, best_score = max(candidates, key=lambda x: x[1])

            scores[s, t] = best_score + log_probs[state_idx, t]
            pointers[s, t] = best_prev_state

    path = np.zeros(T, dtype=int)
    path[T - 1] = (
        K - 2
        if target_seq[K - 1].startswith("I-")
        and scores[K - 2, T - 1] > scores[K - 1, T - 1]
        else K - 1
    )
    for t in range(T - 2, -1, -1):
        path[t] = pointers[path[t + 1], t + 1]

    return [target_seq[p] for p in path]


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


def continuous_segments(segments, duration):
    if duration <= 0:
        return []
    starts = []
    for s, _, ph in sorted(segments, key=lambda seg: seg[0]):
        s = float(s)
        s = max(0.0, s)
        if s >= duration or (starts and s <= starts[-1][0]):
            continue
        starts.append((s, ph))

    if not starts:
        return []
    starts[0] = (0.0, starts[0][1])
    return [
        (s, starts[i + 1][0] if i + 1 < len(starts) else duration, ph)
        for i, (s, ph) in enumerate(starts)
    ]


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
    decoder="constrained",
    viterbi_bias=5,
):
    if len(audio) == 0:
        return []
    original_duration = len(audio) / sr

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
            chunk = np.pad(chunk, (0, pad_res), mode="constant")

        expected_frames = math.ceil(
            (end - start) / sr / config["data"]["frame_duration"]
        )
        input_values = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([expected_frames], device=device)

        with torch.no_grad():
            logits, offsets = model(input_values, lang_tensor, lengths=lengths)

        accumulated_logits.append(logits.squeeze(0).cpu())
        if offsets is not None:
            accumulated_offsets.append(offsets.squeeze(0).cpu())

    full_logits = torch.cat(accumulated_logits, dim=0)

    full_offsets = None
    if accumulated_offsets and not no_use_offset:
        full_offsets = torch.cat(accumulated_offsets, dim=0)

    if phones:
        if decoder == "constrained":
            pred_tags = forced_align_bio(full_logits, model.id2label, phones)
        elif decoder == "viterbi":
            pred_tags = forced_align_viterbi(full_logits, model.id2label, phones)
    else:
        if decoder == "constrained":
            pred_tags = constrained_decode(full_logits, model.id2label)
        elif decoder == "viterbi":
            pred_tags = viterbi_decode(
                full_logits, model.id2label, viterbi_bias=viterbi_bias
            )

    segments = decode_bio_tags(
        pred_tags, config["data"]["frame_duration"], full_offsets
    )

    all_segments = []
    for s, e, ph in segments:
        if merge_map and lang_name:
            ph = canonical_to_lang(ph, lang_name, merge_map)
        all_segments.append((s, e, ph))

    return continuous_segments(all_segments, original_duration)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    default="infer_test",
    help="Path to a .wav file or folder containing .wav files",
)
@click.option(
    "--checkpoint",
    "-ckpt",
    default="checkpoints_no_env/model.ckpt",
    help="Path to WFL .ckpt file",
)
@click.option(
    "--config",
    "-c",
    default="checkpoints_no_env/config.yaml",
    help="Path to config file",
)
@click.option(
    "--lang-id",
    "-l",
    type=int,
    default=None,
    help="Language ID (int) used during training. Example: `-l 0`",
)
@click.option(
    "--no_use_offset",
    is_flag=True,
    help="Disable offset head refinement (offsets ON by default).",
)
# long silence stuff
@click.option(
    "--silence-phoneme",
    default="SP",
    help="The phoneme label to use for hard-coded silence (default: SP)",
)
@click.option(
    "--silence-threshold",
    default=0.005,
    type=float,
    help="Amplitude threshold (0.0-1.0) to consider as silence",
)
@click.option(
    "--min-silence-duration",
    default=0.5,
    type=float,
    help="Minimum duration (seconds) required to trigger hard silence",
)
@click.option(
    "--decoder-type",
    "-d",
    default="viterbi",
    type=str,
    help="Decoder type for no transcription inference [constrained|viterbi] (default: viterbi)",
)
@click.option(
    "--viterbi-bias",
    default=5,
    type=float,
    help="Amount of bias (>=1) added to frames of the same phoneme for viterbi decoding. (default: 5)",
)
def main(
    input_path,
    checkpoint,
    config,
    lang_id,
    no_use_offset,
    silence_phoneme,
    silence_threshold,
    min_silence_duration,
    decoder_type,
    viterbi_bias,
):
    cfg = load_config(config)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
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
    state_dict = (
        checkpoint_data["state_dict"]
        if "state_dict" in checkpoint_data
        else checkpoint_data
    )

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
                    print(
                        f"  Forced-align enabled (found: {os.path.basename(txt_path)})"
                    )
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
            audio = torchaudio.functional.resample(
                audio_t, sr, cfg["data"]["sample_rate"]
            ).numpy()
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
            decoder=decoder_type,
            viterbi_bias=viterbi_bias,
        )

        if cfg.get("postprocess", {}).get("merge_segments", "right") != "none":
            segments = merge_adjacent_segments(
                segments, cfg["postprocess"]["merge_segments"]
            )

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
                silence_phoneme=silence_phoneme,
            )

        segments = continuous_segments(segments, len(audio) / sr)
        out_path = wav_path.replace(".wav", ".lab")
        save_lab(out_path, segments)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
