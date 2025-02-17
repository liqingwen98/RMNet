import torch
from torch import nn
from GCRTcall.encoder import ConformerBlock
import torch.nn.functional as F
import torch.autograd as autograd

class BiLSTM(nn.Module):    # for train
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5, device=0):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_rate,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        
        h0 = autograd.Variable(torch.randn(self.num_layers * 2, x.size(0), self.hidden_size))
        c0 = autograd.Variable(torch.randn(self.num_layers * 2, x.size(0), self.hidden_size))
        h0, c0 = h0.cuda(self.device), c0.cuda(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        return out

class ReadLevelRMNet(nn.Module):

    def __init__(self, dropout_rate=0.1, device=0):
        super(ReadLevelRMNet, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.seq_len = 5
        
        self.embed = nn.Embedding(4, 55+65)
        
        self.encoder1 = nn.ModuleList([ConformerBlock(
            encoder_dim=128,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=31,
            half_step_residual=True,
            layer_idx=i
        ) for i in range(4)])
        
        self.bilstm = BiLSTM(128, 128, 2, dropout_rate=dropout_rate, device=device)
        
        self.encoder2 = nn.ModuleList([ConformerBlock(
            encoder_dim=256,
            num_attention_heads=8,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=31,
            half_step_residual=True,
            layer_idx=i
        ) for i in range(4)])
        
        self.bigru = nn.GRU(input_size=256,
                             num_layers=2,
                             bidirectional=True,    
                             hidden_size=256,
                             batch_first=True)
        
        self.cls = nn.Sequential(nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.Dropout(p=dropout_rate),
                                nn.Linear(256,  128),
                                nn.Dropout(p=dropout_rate),
                                nn.ReLU(),
                                nn.Linear(128,  1),
                                )

    def forward(self, features):
        
        kmer = features[:, 0, :]
        
        mean_sig = features[:, 1, :]
        median_sig = features[:, 2, :]
        std_sig = features[:, 3, :]
        len_sig = features[:, 4, :]
        
        qual = features[:, 5, :]
        mis = features[:, 6, :]
        ins = features[:, 7, :]
        dele = features[:, 8, :]

        kmer_embed = self.embed(kmer.long())
        
        mean_sig = torch.reshape(mean_sig, (-1, self.seq_len, 1)).float()
        median_sig = torch.reshape(median_sig, (-1, self.seq_len, 1)).float()
        std_sig = torch.reshape(std_sig, (-1, self.seq_len, 1)).float()
        len_sig = torch.reshape(len_sig, (-1, self.seq_len, 1)).float()
        
        qual = torch.reshape(qual, (-1, self.seq_len, 1)).float()
        mis = torch.reshape(mis, (-1, self.seq_len, 1)).float()
        ins = torch.reshape(ins, (-1, self.seq_len, 1)).float()
        dele = torch.reshape(dele, (-1, self.seq_len, 1)).float()
        
        y = torch.cat((kmer_embed, 
                        mean_sig, median_sig, std_sig,  
                        qual, mis, ins, dele, 
                        len_sig, 
                        ), 2)
        
        for layer in self.encoder1:
            y = layer(y, None)
            
        y = self.bilstm(y)
        for layer in self.encoder2:
            y = layer(y, None)
        
        y, h_n = self.bigru(y)
        h_n = h_n.permute(1, 0, 2)
        repre = torch.cat((h_n[:, 0, :], 
                           h_n[:, 1, :],
                           h_n[:, 2, :],
                           h_n[:, 3, :],), dim=-1)
        out_ = self.cls(repre)
        out_ = self.sigmoid(out_)
        
        return out_

class SiteLevelRMNet(nn.Module):

    def __init__(self, dropout_rate=0.1, device=0):
        super(SiteLevelRMNet, self).__init__()
        self.read_level_model = ReadLevelRMNet(dropout_rate, device=device)
    def forward(self, features):
        coverage = features.shape[1]
        features = features.view(-1, features.shape[2], features.shape[3])
        probs = self.read_level_model(features).view(-1, coverage)
        return (torch.sum(probs >= 0.5, dim=1))/coverage