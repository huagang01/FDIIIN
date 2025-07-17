import torch
import torch.nn as nn

class NumEmbedding(nn.Module):
    """
    Numerical feature embedding module.
    Args:
        n: Number of numerical features.
        d_in: Input dimension (usually 1).
        d_out: Output embedding dimension.
        bias: Whether to include bias term.
    Input shape: [batch_size, num_features, d_in]
    Output shape: [batch_size, num_features, d_out]
    """
    def __init__(self, n, d_in, d_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(torch.Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x_num):
        assert x_num.ndim == 3
        x = torch.einsum("bfi,fij->bfj", x_num, self.weight)
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class CatEmbedding(nn.Module):
    """
    Categorical feature embedding module.
    Args:
        categories: List of unique values per categorical feature.
        d_embed: Embedding dimension.
    Input shape: [batch_size, num_features]
    Output shape: [batch_size, num_features, d_embed]
    """
    def __init__(self, categories, d_embed):
        super().__init__()
        self.embedding = nn.Embedding(sum(categories), d_embed)
        self.offsets = nn.Parameter(torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x_cat):
        x = x_cat + self.offsets[None]
        return self.embedding(x)

class CatLinear(nn.Module):
    """
    Linear transformation for categorical features (equivalent to one-hot + linear layer).
    Args:
        categories: List of unique values per categorical feature.
        d_out: Output dimension.
    Input shape: [batch_size, num_features]
    Output shape: [batch_size, d_out]
    """
    def __init__(self, categories, d_out=1):
        super().__init__()
        self.fc = nn.Embedding(sum(categories), d_out)
        self.bias = nn.Parameter(torch.zeros((d_out,)))
        self.offsets = nn.Parameter(torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x_cat):
        x = x_cat + self.offsets[None]
        return torch.sum(self.fc(x), dim=1) + self.bias
