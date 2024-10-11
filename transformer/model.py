import torch
import torch.nn as nn
import math

'''
vocab_size : size of vacabulary
d_model : dimension of embedding
register_buffer : save the positional encoding in a buffer. 
                   when you have a tensor that you want to keep inside a model but not update through backpropagation
                   It is not a parameter but will be saved when we save the model
'''
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
'''
Dropout layer is added to reduce overfitting
max_len is the maximum length of the input sequence allowed
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # creaat a position tensor of shape (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # create a div_term tensor of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # add a batch dimension, final shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        #save the positional encoding in a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape(1),:]).requires_grad_(False)
        return self.dropout(x)

'''
Layer Normalization is used to normalize the input to each neuron in a layer
eps is a small number to avoid division by zero
alpha and bias are learnable parameters
'''
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
'''
x.mean(dim): This computes the mean of the elements along a specified dimension (dim).
In this case, -1 refers to the last dimension.
keepdim=True: This ensures that the reduced dimension (where the mean is calculated) is retained in the result as a dimension of size 1, 
instead of being squeezed out. This is useful when you want to maintain the original shape of the tensor, 
but with the calculated values in the corresponding dimension.
For example, if x is a tensor of shape [batch_size, num_features], 
after calculating the mean along the last dimension (num_features), the shape will remain [batch_size, 1] rather than [batch_size].
'''

'''
d_ff is the dimension of the feedforward layer (In paper 2048 nodes were used)
'''
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        ##(Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
'''
h is the number of heads
d_k is the dimension of the key and value vectors
'''
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model:int, h : int, dropout: float) -> None:
        super().__init__() 
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 # d_model must be divisible by h
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def attention(self, query, key, value, mask , dropout : nn.Dropout):
        d_k = query.shape[-1]
        
        ## scaling by d_k
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply the softmax function
        attention_scores = torch.softmax(attention_scores, dim=-1) ## (Batch, h, seq_len, seq_len)
        
        # Apply dropout
        if dropout is not None:
            attention_scores = self.dropout(attention_scores)
        
        return torch.matmul(attention_scores, value) , attention_scores
        
        
    
    def forward(self, q, k, v, mask=None):
        
        # Linear transformation of Query, Key and Value
        
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        query = self.w_q(q)
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v)
        
        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Calculate the attention scores
        x , self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        ## (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
        
        
'''
sublayer is the output  of the next layer i.e. FeedForwardBlock or MultiHeadAttention
'''

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock , dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feedforward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection
        self.residual_connection_2 = ResidualConnection
    
    def forward(self, x, src_mask):
        x = self.residual_connection_1(x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connection_2(x, self.feedforward_block)
        return x

class Encoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(dropout = dropout)
        self.residual_connection_2 = ResidualConnection(dropout = dropout)
        self.residual_connection_3 = ResidualConnection(dropout = dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection_1(x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection_2(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection_3(x, self.feedforward_block)
        return x

class Decoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed:InputEmbeddings ,src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self,encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int,tgt_vocab_size: int,src_max_len: int, tgt_max_len: int,d_model: int = 512, N: int = 6, h: int =8,dropout:float = 0.1, d_ff: int = 2048):
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, dropout, src_max_len)
    tgt_pos = PositionalEncoding(d_model, dropout, tgt_max_len)
    
    ## Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    ## Initialize the weights using Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    