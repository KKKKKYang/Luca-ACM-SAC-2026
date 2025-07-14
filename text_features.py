# text_features.py
from sentence_transformers import SentenceTransformer
import torch
from typing import List

# 1) load your heavy model once
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model  = SentenceTransformer("all-MiniLM-L6-v2", device=_device)

# 2) decide on a small RED_D (_reduced_ dimension)
RED_D = 8

# 3) build a fixed random projection matrix
#    (we do this on the same device so that torch.cat never
#     complains about cpu vs cuda later)
_orig_D = _model.get_sentence_embedding_dimension()
_proj   = torch.randn(_orig_D, RED_D, device=_device)

def get_sentence_embedding_dimension() -> int:
    """
    Returns the (reduced) dimensionality of each text embedding.
    """
    return RED_D

def encode_op_descriptions(descriptions: List[str]):
    """
    descriptions: a list of length num_ops of strings describing each op.
    returns: a torch.Tensor of shape [num_ops, RED_D]
    """
    # 1) get the full embedding [num_ops, _orig_D]
    emb = _model.encode(descriptions, convert_to_tensor=True).to(_device)
    # 2) project it down [num_ops, RED_D]
    return emb @ _proj
