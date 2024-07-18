from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast, GradScaler
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  # Softmax over the last dimension
        
    def forward(self, x):
        # x shape: [batch_size, channels, length]
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Compute the attention scores
        scores = torch.bmm(query.permute(0, 2, 1), key)  # Batch matrix-matrix product
        attention = self.softmax(scores)

        # Apply the attention weights to the values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        return out


class TemporalProcessLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TemporalProcessLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm1d(out_features)

        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)        

        return x

class VideoProcessLayer(nn.Module):
    def __init__(self, in_features, temporal_out_features, num_classes):
        super(VideoProcessLayer, self).__init__()
        self.temporal_processLayer1 = TemporalProcessLayer(in_features, temporal_out_features)
        self.temporal_processLayer2 = TemporalProcessLayer(512, 512)
        self.temporal_processLayer3 = TemporalProcessLayer(512, 512)
        self.temporal_processLayer4 = TemporalProcessLayer(512, 512)
        self.temporal_processLayer5 = TemporalProcessLayer(512, 512)
        self.temporal_processLayer6 = TemporalProcessLayer(512, 512)
        self.temporal_processLayer7 = TemporalProcessLayer(512, 512)

        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, features, sequence_length]

        x = self.temporal_processLayer1(x)
        x = self.temporal_processLayer2(x)
        x = self.temporal_processLayer3(x)
        x = self.temporal_processLayer4(x)
        x = self.temporal_processLayer5(x)
        x = self.temporal_processLayer6(x)
        x = self.temporal_processLayer7(x)

        x = x.permute(0, 2, 1)  # Reshape to [batch_size, features, sequence_length]
        return x
        
class VideoModel(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)

        in_dim = 512 + 1 if self.args.border else 512
        in_dim_processLayer = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)        
        self.v_cls = nn.Linear(2560, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)
        self.VideoProcessLayer = VideoProcessLayer(in_dim,512,self.args.n_class)
        
    def forward(self, v, border=None):
        f_v = self.video_cnn(v)
        f_v = self.dropout(f_v)
        f_v = f_v.float()


        # calculate using bigru
        if self.args.border:
            border = border[:, :, None]
            f_v = torch.cat([f_v, border], -1)

        # bigru
        h_bigru, _ = self.gru(f_v)
        h_processLayer = self.VideoProcessLayer(f_v)

        h = torch.cat([h_bigru,h_processLayer], dim=-1)

        y_v = self.v_cls(self.dropout(h.mean(1)))
            

        return y_v

