"""
Model Architecture Module (with Bahdanau Attention)
Upgraded version of Encoder-Decoder LSTM for translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config
import random


# ============================
#   Bahdanau Attention
# ============================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden):
        """
        encoder_outputs: [batch, src_len, hidden]
        hidden: [1, batch, hidden] (last hidden state from decoder)
        """
        # Prepare hidden for addition
        # Ensure hidden is 3D: [batch, 1, hidden]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)  # add time dimension
        else:
            hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden]
        
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention = torch.softmax(self.V(energy), dim=1)


        # Compute context vector
        context = torch.sum(attention * encoder_outputs, dim=1)  # [batch, hidden]
        return context, attention


# ============================
#   Encoder
# ============================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Config.PAD_IDX)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, (hidden, cell)


# ============================
#   Decoder (with Attention)
# ============================
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Config.PAD_IDX)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_size,  # concat context
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        """
        input_token: [batch, 1]
        hidden, cell: [num_layers, batch, hidden]
        encoder_outputs: [batch, src_len, hidden]
        """
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, embed]
        context, attn_weights = self.attention(encoder_outputs, hidden[-1])  # [batch, hidden]
        context = context.unsqueeze(1)
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch, 1, embed + hidden]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        prediction = self.fc_out(torch.cat((output, context.squeeze(1)), dim=1))
        prediction = F.log_softmax(prediction, dim=1)
        return prediction, hidden, cell, attn_weights

    def forward_sequence(self, tgt, encoder_hidden, encoder_outputs, teacher_forcing_ratio=0.5):
        batch_size, seq_len = tgt.size()
        outputs = torch.zeros(batch_size, seq_len - 1, self.vocab_size).to(tgt.device)
        input_token = tgt[:, 0].unsqueeze(1)
        hidden, cell = encoder_hidden

        for t in range(1, seq_len):
            output, hidden, cell, _ = self.forward(input_token, hidden, cell, encoder_outputs)
            outputs[:, t - 1] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs


# ============================
#   Encoder-Decoder Model
# ============================
class EncoderDecoderLSTM_Attn(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embedding_dim=256, hidden_size=512,
                 num_layers=2, dropout=0.3):
        super(EncoderDecoderLSTM_Attn, self).__init__()

        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.08, 0.08)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        outputs = self.decoder.forward_sequence(tgt, encoder_hidden, encoder_outputs, teacher_forcing_ratio)
        return outputs

    def translate(self, src, src_len, max_length=50,
                  eos_idx=Config.EOS_IDX, sos_idx=Config.SOS_IDX):
        self.eval()
        with torch.no_grad():
            if src.dim() == 1:
                src = src.unsqueeze(0)
            if src_len.dim() > 1:
                src_len = src_len.view(-1)

            encoder_outputs, encoder_hidden = self.encoder(src, src_len)
            hidden, cell = encoder_hidden
            input_token = torch.tensor([[sos_idx]], dtype=torch.long).to(src.device)
            predictions = []

            for _ in range(max_length):
                output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
                predicted_idx = output.argmax(1).item()
                predictions.append(predicted_idx)
                if predicted_idx == eos_idx:
                    break
                input_token = torch.tensor([[predicted_idx]], dtype=torch.long).to(src.device)

        return predictions


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
