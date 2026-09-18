import os
import glob
import json
import yaml
import math
import soundfile as sf
from tqdm import tqdm
import argparse

def load_config(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def to_bio_tags(phonemes, num_frames, frame_duration, audio_duration):
    if not phonemes or num_frames < 1:
        raise ValueError("Empty labels or audio.")

    tags = ["O"] * num_frames
    previous_end, previous_start_frame = 0.0, -1

    for start, end, ph in phonemes:
        if not 0 <= start < end <= audio_duration + 1e-7:
            raise ValueError(f"Invalid timestamps: {start} {end} {ph}")
        if start < previous_end:
            raise ValueError(f"Overlapping phoneme: {ph}")

        s_idx = int(start / frame_duration)
        e_idx = min(int(end / frame_duration), num_frames - 1)

        if s_idx >= num_frames:
            raise ValueError(f"Phoneme starts outside the frame grid: {ph} ... EXCUSE ME HOW???\nAchievement unlocked: How Did We Get Here?")
        if s_idx == previous_start_frame:
            raise ValueError(f"Multiple phoneme starts in frame {s_idx}: {ph} ... EXCUSE ME HOW???\nAchievement unlocked: How Did We Get Here?")

        tags[s_idx] = f"B-{ph}"
        for i in range(s_idx + 1, e_idx + 1):
            tags[i] = f"I-{ph}"

        previous_end, previous_start_frame = end, s_idx

    return tags

def preprocess(data_dir, config):
    frame_dur = config["data"].get("frame_duration", 0.02)
    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # lang map
    lang_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    lang2id = {l: i for i, l in enumerate(lang_dirs)}
    
    # phoneme merging map
    merge_map, reverse_map = {}, {}
    for group in config.get("training", {}).get("merged_phoneme_groups", []):
        canon = group[0]
        for item in group[1:]:
            if "/" in item:
                l, p = item.split("/", 1)
                merge_map.setdefault(l, {})[p] = canon
                reverse_map.setdefault(canon, {})[l] = p

    dataset, phoneme_set = [], set()
    lang_phonemes = {lang: set() for lang in lang_dirs} # tracking...
    
    for lang in lang_dirs:
        wavs = glob.glob(os.path.join(data_dir, lang, "*.wav"))
        for wav in tqdm(wavs, desc=lang):
            lab = wav.replace(".wav", ".lab")
            if not os.path.exists(lab): continue
            
            # read audio
            try:
                with sf.SoundFile(wav) as f:
                    dur = len(f) / f.samplerate
                num_frames = math.ceil(dur / frame_dur)
            except Exception as e:
                print(f"[ERROR] Could not read {wav}: {e}")
                continue
            
            # parse Lab
            segs = []
            with open(lab, "r", encoding="utf-8") as lf:
                for line_number, line in enumerate(lf, 1):
                    p = line.strip().split()
                    if not p:
                        continue
                    if len(p) != 3:
                        raise ValueError(f"{lab}:{line_number}: Invalid label line")
                    ph = p[2]
                    # merge
                    ph = merge_map.get(lang, {}).get(ph, ph)
                    start, end = float(p[0]) / 1e7, float(p[1]) / 1e7
                    if start == end or start >= dur:
                        continue
                    end = min(end, dur)
                    segs.append((start, end, ph))
                    phoneme_set.add(ph)
                    lang_phonemes[lang].add(ph)
            
            if not segs or num_frames < 1:
                print(f"[SKIP] No usable labels or audio: {lab}")
                continue

            try:
                tags = to_bio_tags(segs, num_frames, frame_dur, dur)
            except ValueError as e:
                print(f"[SKIP] {lab}: {e}")
                continue
            dataset.append({
                "wav_path": wav, "bio_tags": tags, 
                "phoneme_segments": segs, "lang_id": lang2id[lang]
            })

    # save paths
    dataset_path = os.path.join(save_dir, "dataset.json")
    langs_path = os.path.join(save_dir, "langs.txt")
    phonemes_path = os.path.join(save_dir, "phonemes.txt")
    lang_phonemes_path = os.path.join(save_dir, "lang_phonemes.json")
    merge_map_path = os.path.join(save_dir, "phoneme_merge_map.json")
    config_path = os.path.join(save_dir, "config.yaml")

    # save files
    with open(dataset_path, "w") as f: json.dump(dataset, f, indent=2)
    
    with open(langs_path, "w", encoding="utf-8") as f:
        for l, i in lang2id.items(): f.write(f"{l},{i}\n")
    
    all_tags = sorted(["O"] + [f"B-{p}" for p in phoneme_set] + [f"I-{p}" for p in phoneme_set])
    with open(phonemes_path, "w", encoding="utf-8") as f: f.write("\n".join(all_tags))

    with open(lang_phonemes_path, "w", encoding="utf-8") as f:
        json.dump({k: sorted(list(v)) for k, v in lang_phonemes.items()}, f, indent=2, ensure_ascii=False)
    
    if reverse_map:
        with open(merge_map_path, "w", encoding="utf-8") as f: json.dump(reverse_map, f, indent=2, ensure_ascii=False)

    config["model"]["num_languages"] = len(lang2id)
    with open(config_path, "w") as f: yaml.dump(config, f, sort_keys=False)

    if merge_map:
        print("\nApplied merged phoneme groups:")
        for lang, mapping in merge_map.items():
            for src, tgt in mapping.items():
                print(f"  {lang}/{src} -> {tgt}")

    print(f"\nProcessed {len(dataset)} samples.")
    print(f"\nGenerated {len(all_tags)} labels -> {phonemes_path}")
    print(f"\nSaved language mapping -> {langs_path}")
    print(f"\nSaved language phoneme list -> {lang_phonemes_path}")
    if reverse_map:
        print(f"\nSaved phoneme merge map -> {merge_map_path}")
    
    print("\nPhoneme usage by language:")
    for lang, phonemes in lang_phonemes.items():
        sorted_phs = sorted(list(phonemes))
        print(f"  {lang}: {sorted_phs}")
        
    print(f"\nSaved updated config -> {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for WFL-ASR")
    parser.add_argument("--config", type=str, default="configs/small.yaml", help="Path to config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        exit(1)

    print(f"Using config: {args.config}")
    config = load_config(args.config)
    preprocess(config["data"]["data_dir"], config)
