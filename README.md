# WFL-ASR: Whisper for Phoneme Labeling

**WFL-ASR** is a deep learning model designed for automatic phoneme segmentation using frame-level BIO tagging. It uses Whisper as audio encoders, and is structured for flexible and efficient training on phoneme-aligned (HTK) datasets.

---

## How It Works

This model performs **frame-level phoneme labeling** using the BIO tag format (`B-`, `I-`, `O`).

### 1. Label Preprocessing
- `.lab` files define phoneme segments using HTK format.
- Each segment is converted into BIO tags aligned to time frames based on `frame_duration` (hardcoded to 20ms for Whisper compatibility).
- Tags are stored along with the audio path in a training JSON.

### 2. Feature Extraction
- **Whisper** encoders process the audio waveform into frame-wise feature vectors.
  - Whisper uses fixed 20ms frame stride.

### 3. Classification
- A linear layer maps each time step to a BIO tag.

### 4. Inference and Postprocessing
- Predict BIO tags from audio.
- Optional smoothing (median filtering) and merging for better boundary clarity.
- Convert tags back to `.lab` segments.

---

## Features

- Whisper encoder support
- Frame-level BIO tag training
- HTK-compatible `.lab` output format
- Optional waveform augmentation via the `augmentation` config section

---

## Augmentation Options

The `config.yaml` file now includes an optional `augmentation` section used during training. When enabled it randomly applies volume scaling and Gaussian noise:

```yaml
augmentation:
  enable: true
  noise_std: 0.005      # standard deviation of Gaussian noise
  prob: 0.5             # probability to augment a sample
  volume_range: [0.9, 1.1]  # random scaling of audio volume
```

Disable augmentation by setting `enable: false`.

---

### Phoneme Merging
Phonemes can be merged across languages by defining `merged_phoneme_groups` in
`config.yaml`. Each group starts with a merge label such as `merged_1` (can be anything) followed
by language specific phonemes:

```yaml
training:
   # define phonemes group that has the same sound (like-phoneme) throughout the dataset across labeling systems
  merged_phoneme_groups:
    - ["merged_1", "en/ah", "ja/a"]
    - ["merged_2", "en/ih", "ja/i"]
    - ["custom_var", "en/AP", "ja/AP"]
    - ["CustomVar", "en/SP", "ja/SP"]
```

During preprocessing these phonemes are replaced with the merged label. For
TensorBoard visualisation and inference, the labels are mapped back to the original phoneme for
the sample's language
