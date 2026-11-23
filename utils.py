import os, json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def decode_bio_tags(tags, frame_duration=0.02, offsets=None):
    segments = []
    curr_ph, start_idx = None, None

    def finalize(end_idx):
        nonlocal curr_ph, start_idx
        s_offset = offsets[start_idx, 0] if offsets is not None else 0.0
        s_time = (start_idx + float(s_offset)) * frame_duration

        last_frame = end_idx - 1
        if last_frame < 0:
            last_frame = 0

        # get offset for the end of the last frame
        if offsets is not None:
            if last_frame < len(offsets):
                e_offset = offsets[last_frame, 1]
            else:
                e_offset = 1.0
        else:
            e_offset = 1.0

        e_time = (last_frame + float(e_offset)) * frame_duration
        segments.append((s_time, e_time, curr_ph))
        curr_ph, start_idx = None, None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if curr_ph:
                finalize(i)
            curr_ph = tag[2:]
            start_idx = i
        elif tag == "O":
            if curr_ph:
                finalize(i)
        elif tag.startswith("I-"):
            ph = tag[2:]
            if ph != curr_ph:
                if curr_ph:
                    finalize(i)
                curr_ph = ph
                start_idx = i

    if curr_ph:
        finalize(len(tags))
    return segments


def save_lab(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, p in segments:
            f.write(f"{int(s*1e7)} {int(e*1e7)} {p}\n")


def load_phoneme_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_phones_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    text = text.replace("\n", " ").replace("\t", " ")
    return [tok for tok in text.split(" ") if tok.strip()]


def load_langs(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            k, v = l.strip().split(",")
            d[k] = int(v)
    return d


def load_phoneme_merge_map(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def canonical_to_lang(ph, lang, m_map):
    return m_map.get(ph, {}).get(lang, ph)


def merge_adjacent_segments(segs, mode="right"):
    if not segs or mode == "none":
        return segs
    merged = [segs[0]]
    for s, e, p in segs[1:]:
        ls, le, lp = merged[-1]
        if p == lp:
            merged[-1] = (ls, e, lp)
        else:
            merged.append((s, e, p))
    return merged


def _to_numpy_2d(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (T, V) logits, got shape {x.shape}")
    return x.astype(np.float64, copy=False)


def _log_softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    x2 = x - m
    lse = np.log(np.sum(np.exp(x2), axis=1, keepdims=True)) + m
    return x - lse


def forced_predict_slot(phones, forced_phones, label2id):
    forced = []
    for p in forced_phones or []:
        if f"B-{p}" not in label2id or f"I-{p}" not in label2id:
            raise ValueError(f"forced_predict_fa phoneme '{p}' is not in the moodel")
        if p not in forced:
            forced.append(p)

    n = len(phones)
    slots = []
    for k in range(n + 1):
        neighbors = {phones[k - 1] if k > 0 else None, phones[k] if k < n else None}
        slots.append([p for p in forced if p not in neighbors])
    return slots    

def forced_align_bio(
    logits,
    id2label,
    phones,
    *,
    allow_end_in_last_phone=True,
    forced_pred_phones=None,
    forced_pred_penalty=0.0

):
    x = _to_numpy_2d(logits)
    T, V = x.shape
    logp = _log_softmax(x)

    # Build label2id
    if isinstance(id2label, dict):
        label2id = {lab: i for i, lab in id2label.items()}
    else:
        label2id = {lab: i for i, lab in enumerate(id2label)}

    o_id = label2id.get("O", None)
    if o_id is None:
        raise ValueError("Label set must contain 'O'")

    phones = phones or []
    N = len(phones)

    b_ids = []
    i_ids = []
    for p in phones:
        b = label2id.get(f"B-{p}")
        ii = label2id.get(f"I-{p}")
        if b is None or ii is None:
            raise ValueError(f"Phoneme '{p}' missing from label set (need B-{p} and I-{p}).")
        b_ids.append(b)
        i_ids.append(ii)

    def idx_O(k): return k
    def idx_B(k): return (N + 1) + k
    def idx_I(k): return (N + 1) + N + k

    slots = forced_predict_slot(phones, forced_pred_phones, label2id)
    opt = [(k, p) for k, slot in enumerate(slots) for p in slot]
    S0 = (N + 1) + 2 * N

    def idx_OB(j): return S0 + 2 * j
    def idx_OI(j): return S0 + 2 * j + 1

    S = S0 + 2 * len(opt)

    emit = np.full((T, S), -np.inf, dtype=np.float64)
    for k in range(N + 1):
        emit[:, idx_O(k)] = logp[:, o_id]
    for k in range(N):
        emit[:, idx_B(k)] = logp[:, b_ids[k]]
        emit[:, idx_I(k)] = logp[:, i_ids[k]]
    for j, (_, p) in enumerate(opt):
        emit[:, idx_OB(j)] = logp[:, label2id[f"B-{p}"]] + forced_pred_penalty
        emit[:, idx_OI(j)] = logp[:, label2id[f"I-{p}"]]

    trans_from = [[] for _ in range(S)]

    # O_k -> O_k, O_k -> B_k
    for k in range(N + 1):
        trans_from[idx_O(k)].append(idx_O(k))
        if k < N:
            trans_from[idx_O(k)].append(idx_B(k))

    # B_k -> I_k
    for k in range(N):
        trans_from[idx_B(k)].append(idx_I(k))

    # I_k -> I_k, I_k -> O_{k+1}, I_k -> B_{k+1}
    for k in range(N):
        trans_from[idx_I(k)].append(idx_I(k))
        trans_from[idx_I(k)].append(idx_O(k + 1))
        if k + 1 < N:
            trans_from[idx_I(k)].append(idx_B(k + 1))

    # forced o in gap k
    for j, (k, _) in enumerate(opt):
        trans_form[idx_OB(j)].append(idx_OI(j))
        trans_form[idx_OI(j)].append(idx_OI(j))
        trans_form[idx_O(k)].append(idx_OB(j))
        if k > 0:
            trans_form[idx_I(k - 1)].append(idx_OB(j))
        for j2, (k2, _) in enumerate(opt):
            if k2 == k and j2 != j:
                trans_form[idx_OI(j2)].append(idx_OB(j))
        if k < N:
            trans_form[idx_OI(j)].append(idx_B(k))

    dp = np.full((T, S), -np.inf, dtype=np.float64)
    back = np.full((T, S), -1, dtype=np.int32)

    dp[0, idx_O(0)] = emit[0, idx_O(0)]

    for t in range(1, T):
        prev = dp[t - 1]
        cur = dp[t]
        for s_prev in range(S):
            ps = prev[s_prev]
            if ps == -np.inf:
                continue
            for s_next in trans_from[s_prev]:
                sc = ps + emit[t, s_next]
                if sc > cur[s_next]:
                    cur[s_next] = sc
                    back[t, s_next] = s_prev

    end_states = [idx_O(N)]
    if allow_end_in_last_phone and N > 0:
        end_states.append(idx_I(N - 1))
        end_states.extend(idx_OI(j) for j, (k, _) in enumerate(opt) if k == N)
    end_state = max(end_states, key=lambda s: dp[T - 1, s])

    path = [end_state]
    for t in range(T - 1, 0, -1):
        p = back[t, path[-1]]
        if p < 0:
            p = idx_O(0)
        path.append(p)
    path.reverse()

    # States -> tags
    tags = []
    for s in path:
        if s <= N:
            tags.append("O")
        elif s < (N + 1) + N:
            k = s - (N + 1)
            tags.append(f"B-{phones[k]}")
        elif s < S0:
            k = s - ((N + 1) + N)
            tags.append(f"I-{phones[k]}")
        else:
            j, is_inside = divmod(s - S0, 2)
            tags.append(f"{'I' if is_inside else 'B'}-{opt[j][1]}")
    return tags


def visualize_prediction(wav, sr, pred, gt=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.linspace(0, len(wav) / sr, len(wav)), wav, color="lightblue", alpha=0.85, linewidth=1)

    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(0, len(wav) / sr)
    ax.set_yticks([])

    # pred
    y_pred = 0.8
    for s, e, p in pred:
        ax.axvline(s, color="red", alpha=0.6, linestyle="--", linewidth=1)
        if e - s > 0.02:
            ax.text((s + e) / 2, y_pred, p, color="red", ha="center", va="center", fontsize=12, fontweight="bold")

    # gt
    y_gt = -0.8
    if gt:
        for item in gt:
            s, e, p = item[:3]
            ax.axvline(s, color="green", alpha=0.6, linewidth=1)
            if e - s > 0.02:
                ax.text((s + e) / 2, y_gt, p, color="green", ha="center", va="center", fontsize=12, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color="red", marker="o", linestyle="none", label="Pred"),
        Line2D([0], [0], color="green", marker="o", linestyle="none", label="GT"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig
