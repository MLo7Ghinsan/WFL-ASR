import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel, WavLMModel, Wav2Vec2FeatureExtractor

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=4, ff_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion, dropout)
        self.ff2 = FeedForwardModule(dim, ff_expansion, dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.ln1(x + attn_out)

        x_ln = self.ln2(x)
        x_conv = self.conv(x_ln.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        x = x + 0.5 * self.ff2(x)
        return x

class BIOPhonemeTagger(nn.Module):
    def __init__(self, config, label_list):
        super().__init__()
        encoder_type = config["model"]["encoder_type"].lower()
        model_name = config["model"]["whisper_model"] if encoder_type == "whisper" else config["model"]["wavlm_model"]
        
        self.encoder_type = encoder_type
        self.freeze_encoder = config["model"].get("freeze_encoder", False)

        if encoder_type == "whisper":
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            self.encoder = WhisperModel.from_pretrained(model_name).encoder
            hidden_size = self.encoder.config.d_model
        elif encoder_type == "wavlm":
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.encoder = WavLMModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
        else:
            raise ValueError("Unsupported encoder type. Use 'whisper' or 'wavlm'.")

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=config["model"].get("bilstm_num_layer", 1),
            batch_first=True,
            bidirectional=True
        )

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=hidden_size,
                heads=config["model"].get("conformer_heads", 4),
                ff_expansion=config["model"].get("conformer_ff_expansion", 4),
                conv_kernel=config["model"].get("conformer_kernel_size", 31),
                dropout=config["model"].get("conformer_dropout", 0.1)
            )
            for _ in range(config["model"].get("num_conformer_layers", 2))
        ])

        self.classifier = nn.Linear(hidden_size, len(label_list))
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def forward(self, input_values):
        real_len = input_values.size(0)
        input_values = input_values.unsqueeze(0)  # [1, T]

        features = self.feature_extractor(input_values.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = features["input_features"].to(input_values.device)

        if self.encoder_type == "whisper":
            encoder_outputs = self.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            real_duration = real_len / 16000
            num_frames = int(real_duration / 0.02)
            hidden_states = hidden_states[:, :num_frames, :]
        else:
            hidden_states = self.encoder(input_values).last_hidden_state

        lstm_out, _ = self.bilstm(hidden_states)

        for layer in self.conformer_layers:
            lstm_out = layer(lstm_out)

        logits = self.classifier(lstm_out)
        return logits

    def decode_predictions(self, logits):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    def id_to_label(self, ids):
        return [[self.id2label[i.item()] for i in seq] for seq in ids]
