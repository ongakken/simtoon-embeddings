from user_embedder import UserEmbedder
from user_utils import UserUtils
import torch
from sentence_transformers import util
import numpy as np
import logging
from termcolor import colored
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

np.random.seed(69)

embedder = UserEmbedder()
userutils = UserUtils()

# csvPath = "/home/simtoon/git/ACARIS/datasets/train.csv"
# msgs, userID = embedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="uid", msgCol="content", sep="|", limitToUsers=["simtoon", "Reknez#9257"])

# csvPath = "/home/simtoon/git/ACARISv2/datasets/sarah/sarah.csv"
# msgs, userID = embedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="Username", msgCol="Content", sep=",")

# csvPath = "/home/simtoon/git/ACARISv2/datasets/m7/smtn-m7_msgs.csv"
# msgs, userID = embedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="Username", msgCol="Content", sep=",")

datPath = "/home/simtoon/git/ACARISv2/datasets/messages.dat"
msgs, userID, timestamps = embedder.load_msgs_from_dat(datPath=datPath, limitToUsers=["simtoon1011#0", "simmiefairy#0"])

# txtPath = "/home/simtoon/git/ACARISv2/datasets/allan/DMs.txt"
# msgs, userID = embedder.load_direct_msgs_from_copied_discord_txt(txtPath=txtPath)

for i, user in enumerate(userID):
    print(f"{user}: time: {timestamps[i][:5]} sample messages: {msgs[i][:5]}")

msgs[0] = [msg for msg in msgs[0] if len(embedder.tokenizer.tokenize(msg)) <= embedder.model.max_seq_length]
msgs[1] = [msg for msg in msgs[1] if len(embedder.tokenizer.tokenize(msg)) <= embedder.model.max_seq_length]

print(f"Collectively, there are {sum(len(userMsgs) for userMsgs in msgs)} messages from {len(msgs)} users.\n{userID[0]} has {len(msgs[0])} messages, {userID[1]} has {len(msgs[1])} messages.")

print(f"First 3 messages from {userID[0]}: {msgs[0][:3]}\nFirst 3 messages from {userID[1]}: {msgs[1][:3]}")

messageUserIDPairs = [(msg, userID[0]) for msg in msgs[0]] + [(msg, userID[1]) for msg in msgs[1]]

print(f"First and last 3 messages: {messageUserIDPairs[:3]} ... {messageUserIDPairs[-3:]}")

embs = embedder.gen_embs_from_observations(msgs[0], bStore=True, userID=userID[0])
embs2 = embedder.gen_embs_from_observations(msgs[1], bStore=True, userID=userID[1])

userutils.get_bert_topics((embs, userID[0]), (embs2, userID[1]), msgs[0], msgs[1])
userutils.plot_topics((embs, userID[0]), (embs2, userID[1]))
breakpoint()

embsReduced, embs2Reduced = userutils.get_user_embs(embs, embs2, True, 2)  # ! these are reduced, but not meaned. meaning is done below

print(userutils.compare_two_users(embs, embs2))

all = msgs[0] + msgs[1]
allEmbs = torch.cat((embs, embs2)).cpu()

embsMean = torch.mean(embs, dim=0).cpu()
embs2Mean = torch.mean(embs2, dim=0).cpu()

res = (embsMean + embs2Mean) / 2

print(f"{userID[0]}'s mean embedding message: {all[torch.nn.functional.cosine_similarity(embsMean.unsqueeze(0), embs.cpu()).argmax().item()]}\n{userID[1]}'s mean embedding message: {all[torch.nn.functional.cosine_similarity(embs2Mean.unsqueeze(0), embs2.cpu()).argmax().item()]}")

sims = torch.nn.functional.cosine_similarity(res.unsqueeze(0), allEmbs)
mostSimIdx = sims.argmax().item()
print("Most similar message to the mixed embeddings: ", all[mostSimIdx])

sims1 = torch.nn.functional.cosine_similarity(embsMean.unsqueeze(0), embs2.cpu())
mostSimIdx1 = sims1.argmax().item()
print("Most similar message to the first embedding: ", all[mostSimIdx1])

sims2 = torch.nn.functional.cosine_similarity(embs2Mean.unsqueeze(0), embs.cpu())
mostSimIdx2 = sims2.argmax().item()
print("Most similar message to the second embedding: ", all[mostSimIdx2])

sent = "intimacy"
sentEmb = embedder.gen_embs_from_observations([sent], bStore=False, userID=None).cpu()
sentSims = torch.nn.functional.cosine_similarity(sentEmb, allEmbs)
mostSimSentIndices = sentSims.argsort(descending=True)[:10].cpu().numpy()
assert len(messageUserIDPairs) == allEmbs.shape[0], "Len of combined messages and embeddings is not the same!"
coloredSents = []
for idx in mostSimSentIndices:
    sentence, user = messageUserIDPairs[idx]
    color = "red" if str(user) == str(userID[0]) else "blue"
    coloredSents.append(colored(sentence, color))
print(f"10 most similar messages to \"{sent}\": {' | '.join(coloredSents)}")

dotSims = res @ allEmbs.T
mostSimDotIdx = dotSims.argmax().item()
print("Most similar message to the mixed embeddings using dot: ", all[mostSimDotIdx])

dotSims1 = embsMean @ allEmbs.T
mostSimDotIdx1 = dotSims1.argmax().item()
print("Most similar message to the first embedding using dot: ", all[mostSimDotIdx1])

dotSims2 = embs2Mean @ allEmbs.T
mostSimDotIdx2 = dotSims2.argmax().item()
print("Most similar message to the second embedding using dot: ", all[mostSimDotIdx2])

cosines = util.cos_sim(embs, embs2)
dots = util.dot_score(embs, embs2)

mostSimPairIdx = torch.argmax(cosines)
mostSimPairIdx1, mostSimPairIdx2 = mostSimPairIdx // len(msgs[1]), mostSimPairIdx % len(msgs[1])
print(f"Most similar pair: {all[mostSimPairIdx1]} ||| {all[mostSimPairIdx2]}")

userutils.plot_embs((embsReduced, userID[0], msgs[0]), (embs2Reduced, userID[1], msgs[1]))
# userutils.plot_embs_3D((embs, userID[0]), (embs2, userID[1]))

userutils.plot_mean_embs(torch.mean(embsReduced, dim=0), torch.mean(embs2Reduced, dim=0), (userID[0], userID[1]))
# userutils.plot_mean_embs_3D(mean1, mean2, (userID[0], userID[1]))

# userutils.plot_vector_field_3D((embs, userID[0]), (embs2, userID[1]))
# userutils.plot_vector_field_3D_mean([mean1, mean2])

# userutils.plot_topics((embs, userID), (embs2, userID[1]))