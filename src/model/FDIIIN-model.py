import torch
import torch.nn as nn

from components.inputs import NumEmbedding, CatEmbedding
from components.modules import SENetAttention, BilinearInteraction, MultiLayerPerceptron

class FDIIIN_model(nn.Module):
    """
    FDIIIN Model.
    Args:
        d_numerical: Number of numerical features.
        categories: List of unique values per categorical feature.
        d_embed: Embedding dimension.
        mlp_layers: Layer sizes for final MLP.
        mlp_dropout: Dropout for final MLP.
        fdin_mlp_layers: Layer sizes for FDIN MLP.
        fdin_mlp_dropout: Dropout for FDIN MLP.
        reduction_ratio: SENet reduction ratio.
        bilinear_type: Bilinear interaction type.
        seia_reduction_ratio: FIIN SENet reduction ratio.
        n_classes: Number of output classes.
    """
    def __init__(self, d_numerical, categories, d_embed, mlp_layers, mlp_dropout,
                 fdin_mlp_layers, fdin_mlp_dropout, reduction_ratio=3,
                 bilinear_type="field_interaction", seia_reduction_ratio=3, n_classes=1):
        super().__init__()
        self.d_numerical = d_numerical or 0
        self.categories = categories or []
        self.d_embed = d_embed
        self.n_classes = n_classes
        self.num_fields = self.d_numerical + len(self.categories)

        self.num_embedding = NumEmbedding(self.d_numerical, 1, d_embed) if self.d_numerical else None
        self.cat_embedding = CatEmbedding(self.categories, d_embed) if self.categories else None

        self.senet = SENetAttention(self.num_fields, reduction_ratio)
        self.bilinear = BilinearInteraction(self.num_fields, d_embed, bilinear_type)
        self.fiin_senet = SENetAttention(int(self.num_fields * (self.num_fields - 1) / 2), seia_reduction_ratio)

        self.fdin_mlp = MultiLayerPerceptron(self.num_fields * d_embed, fdin_mlp_layers, fdin_mlp_dropout,
                                              self.num_fields * d_embed)
        self.mlp = MultiLayerPerceptron(self.num_fields * (self.num_fields - 1) * d_embed, mlp_layers,
                                        mlp_dropout, n_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_num, x_cat = x

        # Embedding
        x_embedding = []
        if self.num_embedding:
            x_embedding.append(self.num_embedding(x_num[..., None]))
        if self.cat_embedding:
            x_embedding.append(self.cat_embedding(x_cat))
        x_embedding = torch.cat(x_embedding, dim=1)

        # FDIN: dynamic and static importance perception
        dynamic_embedding = self.senet(x_embedding)
        static_embedding = self.fdin_mlp(self.flatten(x_embedding)).reshape(-1, self.num_fields, self.d_embed)
        dual_embedding = dynamic_embedding + static_embedding

        # Bilinear interaction
        bilinear_out = self.bilinear(x_embedding)
        dual_bilinear_out = self.bilinear(dual_embedding)

        # FIIN: adaptive interaction importance modeling
        fiin_bilinear_out = self.fiin_senet(bilinear_out) + bilinear_out
        fiin_dual_ffm_out = self.fiin_senet(dual_bilinear_out) + dual_bilinear_out

        # Output
        fiin_concat = self.flatten(torch.cat([fiin_bilinear_out, fiin_dual_ffm_out], dim=1))
        out = self.mlp(fiin_concat)
        if self.n_classes == 1:
            out = out.squeeze(-1)
        return out