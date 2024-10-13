import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import BertPreTrainedModel, BertModel
from transformers.trainer_pt_utils import nested_numpify, nested_concat, distributed_concat
# from path_transformer import PathTransformer
from .prefix_bert import MenBert

def _pad_across_processes(tensor, rank, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_pad_across_processes(t, rank, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _pad_across_processes(v, rank, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor

    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = _nested_gather(size, rank).cpu()

    max_size = max(s[1] for s in sizes)
    if tensor.shape[1] == max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor

def _nested_gather(tensors, rank):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if rank != -1:
        tensors = distributed_concat(tensors)
    return tensors

def gather_data(feature, rank):
    gather_feature_host = None
    gather_feature = _pad_across_processes(feature, rank)
    gather_feature = _nested_gather(gather_feature, rank)
    gather_feature_host = gather_feature if gather_feature_host is None else nested_concat(gather_feature_host,
                                                                gather_feature,
                                                                padding_index=-100)
    return gather_feature_host

class PLM(BertPreTrainedModel):

    def __init__(self, config, args=None):
        super().__init__(config)
        self.args = args
        self.pool_method = 'mean'  # mean  first
        self.bert = MenBert.from_pretrained(args.plm_name)#.embeddings.word_embeddings

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None):

        output = self.bert(input_ids=input_ids,
                           inputs_embeds=inputs_embeds,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           use_memory=False).last_hidden_state

        return output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)

        return output



