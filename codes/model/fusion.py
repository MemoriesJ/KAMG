import torch

from util import Type


class FusionType(Type):
    """Standard names for embedding type
    The following keys are defined:
    * `EMBEDDING`: Return the embedding after lookup.
    * `REGION_EMBEDDING`: Return the region embedding.
        Reference: A New Method of Region Embedding for Text Classification
    """
    ATTACH = 'attach'
    CONCATENATION = 'concatenation'

    @classmethod
    def str(cls):
        return ",".join([cls.ATTACH, cls.CONCATENATION])


class FusionConcatenation(torch.nn.Module):

    def __init__(self, in_features, out_features, bias):
        super(FusionConcatenation, self).__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )

    def forward(self, tuple_of_tensors):
        out = torch.cat(tuple_of_tensors, dim=-1)
        return self.linear(out)


class Fusion(torch.nn.Module):

    def __init__(self, config):
        super(Fusion, self).__init__()

        if config.fusion.fusion_type == FusionType.ATTACH:
            self.fuse = lambda x: torch.cat(x, dim=-1)

        if config.fusion.fusion_type == FusionType.CONCATENATION:
            self.fuse = FusionConcatenation(
                in_features=config.fusion.in_features,
                out_features=config.fusion.out_features,
                bias=config.fusion.bias
            )

    def forward(self, tuple_of_tensors):
        return self.fuse(tuple_of_tensors)
