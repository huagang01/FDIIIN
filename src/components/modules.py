import torch
import torch.nn as nn
from itertools import combinations

class SENetAttention(nn.Module):
    """
    Squeeze-and-Excitation attention mechanism.
    Args:
        num_fields: Number of feature fields.
        reduction_ratio: Reduction ratio for bottleneck.
    Input shape: [batch_size, num_fields, d_embed]
    Output shape: [batch_size, num_fields, d_embed]
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        Z = torch.mean(x, dim=-1)
        A = self.excitation(Z)
        return x * A.unsqueeze(-1)

class BilinearInteraction(nn.Module):
    """
    Bilinear feature interaction module.
    Args:
        num_fields: Number of feature fields.
        d_embed: Embedding dimension.
        bilinear_type: Type of bilinear interaction (field_all / field_each / field_interaction).
    Input shape: [batch_size, num_fields, d_embed]
    Output shape: [batch_size, num_pairs, d_embed]
    """
    def __init__(self, num_fields, d_embed, bilinear_type="field_interaction"):
        super().__init__()
        self.bilinear_type = bilinear_type
        if bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(d_embed, d_embed, bias=False)
        elif bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=False) for _ in range(num_fields)])
        elif bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=False)
                                                 for _ in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        else:
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron (MLP).
    Args:
        d_in: Input dimension.
        d_layers: List of hidden layer sizes.
        dropout: Dropout probability.
        d_out: Output dimension.
    Input shape: [batch_size, d_in]
    Output shape: [batch_size, d_out]
    """
    def __init__(self, d_in, d_layers, dropout, d_out=1):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.extend([nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p=dropout)])
            d_in = d
        layers.append(nn.Linear(d_layers[-1], d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
