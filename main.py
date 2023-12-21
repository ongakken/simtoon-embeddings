from user_embedder import UserEmbedder
from user_utils import UserUtils
import torch
from sentence_transformers import util
import numpy as np
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO)



np.random.seed(69)

embedder = UserEmbedder()

csvPath = "/home/simtoon/git/ACARIS/datasets/train.csv"
msgs, userID = embedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="uid", msgCol="content", sep="|", limitToUsers=["simtoon", "Reknez#9257"])

print(f"Collectively, there are {sum(len(userMsgs) for userMsgs in msgs)} messages from {len(msgs)} users.\n{userID[0]} has {len(msgs[0])} messages, {userID[1]} has {len(msgs[1])} messages.")

embs = embedder.gen_embs_from_observations(msgs[0], bStore=True, userID=userID[0])
embs2 = embedder.gen_embs_from_observations(msgs[1], bStore=True, userID=userID[1])

userutils = UserUtils()

emb1Mean = userutils.get_user_embs_mean(embs=embs)
emb2Mean = userutils.get_user_embs_mean(embs=embs2)

print(userutils.compare_two_users(embs, embs2))

mean1 = torch.mean(embs, dim=0)
mean2 = torch.mean(embs2, dim=0)
all = msgs[0] + msgs[1]
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

sent = "intimate"
sentEmb = embedder.gen_embs_from_observations([sent], bStore=False, userID=None)
sentSims = torch.nn.functional.cosine_similarity(sentEmb, allEmbs)
# get top 5 most similar sentences
mostSimSentIndices = sentSims.argsort(descending=True)[:10].cpu().numpy()
coloredSents = [colored(sentence, "red") if userID[0] in sentence else colored(sentence, "blue") for sentence in np.array(all)[mostSimSentIndices]]
print(f"10 most similar sentences to \"{sent}\": {' | '.join(coloredSents)}")

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
dots = util.dot_score(embs, embs2)

mostSimPairIdx = torch.argmax(cosines)
mostSimPairIdx1, mostSimPairIdx2 = mostSimPairIdx // len(msgs[1]), mostSimPairIdx % len(msgs[1])
print(f"Most similar pair: {all[mostSimPairIdx1]} ||| {all[mostSimPairIdx2]}")

userutils.plot_embs((embs, userID[0], msgs[0]), (embs2, userID[1], msgs[1]))
userutils.plot_embs_3D((embs, userID[0]), (embs2, userID[1]))

userutils.plot_mean_embs(emb1Mean, emb2Mean, (userID[0], userID[1]))
userutils.plot_mean_embs_3D(emb1Mean, emb2Mean, (userID[0], userID[1]))

userutils.plot_vector_field_3D((embs, userID[0]), (embs2, userID[1]))
userutils.plot_vector_field_3D_mean([emb1Mean, emb2Mean])

# userutils.plot_topics((embs, userID), (embs2, userID[1]))