from sentence_transformers import SentenceTransformer, util
import pickle
import logging
from typing import List, Tuple
import pandas as pd
import torch
import re

logging.basicConfig(level=logging.INFO)



class UserEmbedder:
    def __init__(self, modelName: str = "all-mpnet-base-v2", userID: str = None, observations: List = []) -> None:
        self.model = SentenceTransformer(modelName, device="cuda")
        self.model.max_seq_length = 384
        self.userID = userID
        tooLong = 0
        for observation in observations:
            if isinstance(observation, str) and len(observation) > self.model.max_seq_length:
                logging.warning(f"Observation '{observation[:20]}...' is {len(observation)} chars long, so longer than 384 characters.")
                tooLong += 1
        print(f"Found {tooLong} observations that are too long to be embedded.")

    def gen_embs_from_observations(self, observations: List[str], bStore: bool = True) -> List:
        embs = self.model.encode(observations, convert_to_tensor=True)
        if bStore and self.userID is not None:
            with open(f"{self.userID}_embs.pkl", "wb") as f:
                pickle.dump(embs, f, protocol=pickle.HIGHEST_PROTOCOL)

        return embs


def load_msgs_from_csv(csvPath: str, userID: str) -> List[str]:
    df = pd.read_csv(csvPath, sep=",", on_bad_lines="skip")
    df = df[df["Username"] == userID]
    msgs = df["Content"].apply(str).tolist()
    l = len(msgs)
    msgs = [re.sub(r'```.*?```|`.*?`|^>.*?$', '', msg, flags=re.DOTALL|re.MULTILINE) for msg in msgs]
    print(f"Filtered out {l - len(msgs)} messages!")
    return msgs



if __name__ == "__main__":
    
    userID = "sarahjkcc"
    user2ID = "simtoon"

    csvPath = "/home/simtoon/git/ACARISv2/datasets/sarah/sarah.csv"
    msgs = load_msgs_from_csv(csvPath=csvPath, userID=userID)
    msgs2 = load_msgs_from_csv(csvPath=csvPath, userID=user2ID)
    print(f"Loaded {len(msgs)} messages from {csvPath} for user {userID}.\nFirst five: {msgs[:5]}")
    print(f"Loaded {len(msgs2)} messages from {csvPath} for user {user2ID}.\nFirst five: {msgs2[:5]}")
    print(f"Collectively, {len(msgs) + len(msgs2)} messages were loaded.")

    embedder = UserEmbedder(userID=userID, observations=msgs)
    embedder2 = UserEmbedder(userID=user2ID, observations=msgs2)

    embs = embedder.gen_embs_from_observations(msgs, bStore=False)
    embs2 = embedder.gen_embs_from_observations(msgs2, bStore=False)

    from userutils import UserUtils

    userutil = UserUtils(userID=userID, embs=embs)
    userutil2 = UserUtils(userID=user2ID, embs=embs2)

    emb1Mean = userutil.get_user_embs_mean(bStore=False)
    emb2Mean = userutil2.get_user_embs_mean(bStore=False)

    print(userutil.compare_two_users(emb1Mean, emb2Mean))


    mean1 = torch.mean(embs, dim=0)
    mean2 = torch.mean(embs2, dim=0)
    all = msgs + msgs2
    allEmbs = torch.cat((embs, embs2))
    res = mean2 + mean1
    
    sims = torch.nn.functional.cosine_similarity(res.unsqueeze(0), allEmbs)
    mostSimIdx = sims.argmax().item()
    print("Most similar message to the mixed embeddings: ", all[mostSimIdx])

    sims1 = torch.nn.functional.cosine_similarity(mean1.unsqueeze(0), allEmbs)
    mostSimIdx1 = sims1.argmax().item()
    print("Most similar message to the first embedding: ", all[mostSimIdx1])

    sims2 = torch.nn.functional.cosine_similarity(mean2.unsqueeze(0), allEmbs)
    mostSimIdx2 = sims2.argmax().item()
    print("Most similar message to the second embedding: ", all[mostSimIdx2])

    dotSims = res @ allEmbs.T
    mostSimDotIdx = dotSims.argmax().item()
    print("Most similar message to the mixed embeddings using dot: ", all[mostSimDotIdx])

    dotSims1 = mean1 @ allEmbs.T
    mostSimDotIdx1 = dotSims1.argmax().item()
    print("Most similar message to the first embedding using dot: ", all[mostSimDotIdx1])

    dotSims2 = mean2 @ allEmbs.T
    mostSimDotIdx2 = dotSims2.argmax().item()
    print("Most similar message to the second embedding using dot: ", all[mostSimDotIdx2])

    cosines = util.cos_sim(embs, embs2)
    dots = util.dot_products(embs, embs2)

    mostSimPairIdx = torch.argmax(cosines)
    mostSimPairIdx1, mostSimPairIdx2 = mostSimPairIdx // len(msgs2), mostSimPairIdx % len(msgs2)
    print(f"Most similar pair: {mostSimPairIdx1} ||| {mostSimPairIdx2}")