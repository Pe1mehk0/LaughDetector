# laughdetector/nn/tagger.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, features):
        features_permuted = features.permute(2, 0, 1).contiguous()

        # Pass through LSTM
        lstm_out, _ = self.lstm(features_permuted)

        tag_space_flat = self.hidden2tag(lstm_out.reshape(-1, self.hidden_dim * 2))
        tag_scores = tag_space_flat.view(features.size(0), features.size(2), self.tagset_size)

        # Apply log_softmax along the last dimension (tagset_size)
        return F.log_softmax(tag_scores, dim=-1)