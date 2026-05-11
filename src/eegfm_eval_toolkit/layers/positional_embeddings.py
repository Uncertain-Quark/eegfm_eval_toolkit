import torch
import torch.nn as nn
import math

class SinPosEmbeddings(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            max_len: int = 5000
    ):
        """
        Args:
            d_model: The dimension of the embeddings (must match the input x dimension).
            max_len: The maximum length of the sequence.
        """
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        # 1. Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create a vector of positions (0, 1, ... max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Calculate the division term: 10000^(2i/d_model)
        # We use exp(log(...)) for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. Register as a buffer (part of state_dict but not a trainable parameter)
        # This handles device movement (CPU->GPU) automatically.
        self.register_buffer('pos_embeddings', pe)

    def forward(self, x, positions=None):
        """
        Adds positional embeddings to the input tensor.
        
        :param x: data tensor of shape (B, L, d_model)
        :param positions: Optional tensor of shape (B, L) representing specific 
                          timesteps. If None, assumes sequential (0, 1, 2...)
        :return: Tensor of shape (B, L, d_model)
        """
        batch_size, seq_len, _ = x.size()

        if positions is None:
            # If no specific positions are provided, slice the pre-computed 
            # embeddings from 0 to seq_len and broadcast batch dim
            # Output shape: (1, L, d_model)
            x_pos_embed = self.pos_embeddings[:seq_len, :].unsqueeze(0)
        else:
            # If specific positions are provided (e.g., for diffusion timesteps),
            # gather them from the buffer.
            # positions shape: (B, L) -> output shape: (B, L, d_model)
            x_pos_embed = self.pos_embeddings[positions]

        return x_pos_embed
    
    