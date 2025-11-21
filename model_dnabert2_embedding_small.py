from torch import nn
import torch
import numpy as np
from config import (EMBEDDING_MATRIX_PATH)


NUMBER_WORDS = 4097
NUMBER_POS = 70



EMBEDDING_DIM = 768
CNN_KERNEL_SIZE = 40
POOL_KERNEL_SIZE = 20
OUT_CHANNELs = 64


embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)

class BalancedBCELoss(nn.Module):

    def __init__(self, lambda1=0.2, lambda2=0.02, tau=0.6):
        super(BalancedBCELoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = tau
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):

        # make sure predictions and targets are 1D tensors
        if predictions.dim() > 1 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        bce_loss = self.bce_loss(predictions, targets)

        total_loss = bce_loss
        
        return total_loss


class EPIModel(nn.Module):
    def __init__(self):
        super(EPIModel, self).__init__()

        #embedding
        self.embedding_en = nn.Embedding(4097, 768)
        self.embedding_pr = nn.Embedding(4097, 768)

        self.embedding_en.weight = nn.Parameter(embedding_matrix)
        self.embedding_pr.weight = nn.Parameter(embedding_matrix)

        self.embedding_en.requires_grad = True
        self.embedding_pr.requires_grad = True

        self.enhancer_sequential = nn.Sequential(nn.Conv1d(in_channels=768, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
                                                 nn.BatchNorm1d(64),
                                                 nn.Dropout(p=0.5)
                                                 )
        self.promoter_sequential = nn.Sequential(nn.Conv1d(in_channels=768, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
                                                 nn.BatchNorm1d(64),
                                                 nn.Dropout(p=0.5)
                                                 )
        

        self.l1GRU = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, num_layers=2)
        self.l2GRU = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, num_layers=2)

        self.MTHEAD = nn.MultiheadAttention(embed_dim=64,num_heads=8)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4)
        # self.transformerencoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.batchnorm1d = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(nn.Linear(64 * 563, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(128, 1),           
               )
   
       
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight)
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
        
        
    def forward(self, enhancer_ids, promoter_ids, enhancer_features, promoter_features):
        SAMPLE_SIZE = enhancer_ids.size(0)

        enhancer_embedding = self.embedding_en(enhancer_ids)
        promoter_embedding = self.embedding_pr(promoter_ids)


        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))

        
        enhancers_output, _ = self.l1GRU(enhancers_output.permute(2, 0, 1))
        promoters_output, _ = self.l2GRU(promoters_output.permute(2, 0, 1))
        
        enhancers_output, _ = self.MTHEAD(enhancers_output, enhancers_output, enhancers_output)
        promoters_output, _ = self.MTHEAD(promoters_output, promoters_output, promoters_output)

        stacked_tensor = torch.cat((enhancers_output, promoters_output), dim=0).permute(1, 2, 0)
        output = self.batchnorm1d(stacked_tensor)
        output = self.dropout(output)

        fc_input_dim = output.flatten(start_dim=1).shape[1]
        
        expected_dim = 64 * 563
        if fc_input_dim < expected_dim:
            flattened_output = output.flatten(start_dim=1)
            padding = torch.zeros(flattened_output.size(0), expected_dim - fc_input_dim, device=flattened_output.device)
            flattened_output = torch.cat([flattened_output, padding], dim=1)
        else:
            flattened_output = output.flatten(start_dim=1)
        
        result = self.fc(flattened_output)

        return torch.sigmoid(result), output.flatten(start_dim=1)