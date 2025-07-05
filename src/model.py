import torch
import torch.nn as nn
import math

class positionalEncoding(nn.Module):
    def __init__(self, dModel: int, maxLen:int=5000):
        super().__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(
            torch.arange(0, dModel, 2, dtype=torch.float)*(-math.log(10000.0)/dModel)
        )
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:g                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            x + positional encodings: same shape
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class T9Labeler(nn.Module):
    """
    Transformer-based sequence labeling model for T9:
    - Encodes an input digit token sequence via Transformer Encoder
    - Predicts a character label at each input position via a linear classifier

    Input:
        - input_ids: LongTensor of shape (batch_size, seq_len) with digit token IDs
        - attention_mask: BoolTensor of shape (batch_size, seq_len), True for real tokens
    Output:
        - logits: FloatTensor of shape (batch_size, seq_len, char_vocab_size)
    """
    def __init__(
        self,
        digitVocabSize: int,
        charVocabSize: int,
        dModel:int = 512,  #嵌入层维度
        nhead: int = 8,   #注意力头的数量
        numLayers: int = 6,  #transformer堆叠数量
        dimFeedForward: int = 2048,  #前馈层维度
        dropout: float = 0.1,
        maxLen: int = 512
    ):
        super().__init__()
        #Embedding layer for digs
        self.embedding = nn.Embedding(digitVocabSize, dModel)
        #Positional encoding to add sequence order information
        self.posEncoder = positionalEncoding(dModel, maxLen=maxLen)
        #Transformer encoder layers
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=dModel,
            nhead=nhead,
            dim_feedforward=dimFeedForward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        #Stack numLayers of encoder layers
        self.transformerEncoder = nn.TransformerEncoder(
            encoderLayer,
            num_layers=numLayers
        )
        self.dropout = nn.Dropout(dropout)
        #classifier: map dModel vectors to charVocabSize logits
        self.classifier = nn.Linear(dModel, charVocabSize)

    def forward(
        self,
        inputIDs: torch.Tensor,
        attentionMask: torch,Tensor = None
    )-> torch.Tensor:
        x = self.embedding(inputIDs)
        x = x * math.sqrt(self.embedding.embedding_dim)
        x = self.posEncoder(x)
        x = self.transformerEncoder(
            x,
            src_key_padding_mask=~attentionMask if attentionMask is not None else None
        )
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits