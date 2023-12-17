from typing import List, Dict
import torch
from torch import Tensor
import pickle
from sentence_transformers import SentenceTransformer

""" to be used once we have user embs """


class UserUtils:
    def __init__(self, userID: str = None, embs: List[Tensor] = []) -> None:
        self.userID = userID
        self.embs = embs

    def get_user_embs_mean(self, bStore: bool = True) -> Tensor:
        if isinstance(self.embs, list):
            emb = torch.mean(torch.stack(self.embs), dim=0)
        else:
            emb = torch.mean(self.embs, dim=0)
        if bStore:
            with open(f"{self.userID}_emb.pkl", "wb") as f:
                pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
        return emb

    def compare_two_users(self, emb1: Tensor, emb2: Tensor) -> Dict:
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        cosim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        dot = torch.dot(emb1, emb2)
        euclidean = torch.norm(emb1 - emb2)
        magnitude1, magnitude2 = self.calc_magnitude(emb1), self.calc_magnitude(emb2)
        return {"cosim": cosim, "dot": dot, "euclidean": euclidean, "magnitude1": magnitude1, "magnitude2": magnitude2}

    def normalize(self, x: Tensor) -> Tensor:
        return x / torch.norm(x)

    def calc_magnitude(self, x: Tensor) -> Tensor:
        return torch.norm(x)
