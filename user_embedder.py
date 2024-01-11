from sentence_transformers import SentenceTransformer
import pickle
import logging
from typing import List, Tuple
import pandas as pd
import re
import os
from collections import defaultdict
import torch



class UserEmbedder:
    def __init__(self, modelName: str = "all-mpnet-base-v2") -> None:
        self.model = SentenceTransformer(modelName, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.max_seq_length = 384


    def gen_embs_from_observations(self, observations: List[str], userID: str = None, bStore: bool = True) -> List:
        tooLong = 0
        for observation in observations:
            if isinstance(observation, str) and len(observation) > self.model.max_seq_length:
                logging.warning(f"Observation '{observation[:20]}...' is {len(observation)} chars long, so longer than 384 characters.")
                tooLong += 1
        print(f"Found {tooLong} observations that are too long to be embedded.")


        if userID is not None:
            if os.path.exists(f"{userID}_embs.pkl"):
                with open(f"{userID}_embs.pkl", "rb") as f:
                    embs = pickle.load(f)
                    logging.info(f"Loaded {len(embs)} embeddings from {userID}_embs.pkl")
                return embs

        embs = self.model.encode(observations, convert_to_tensor=True)
        if bStore and userID is not None:
            with open(f"{userID}_embs.pkl", "wb") as f:
                pickle.dump(embs, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"Stored {len(embs)} embeddings in {userID}_embs.pkl")

        return embs

    def clean_msg(self, msg: str) -> str:
        cleanedMsg = re.sub(r'```.*?```|`.*?`|^>.*?$', '', msg, flags=re.DOTALL|re.MULTILINE)
        cleanedMsg = re.sub(r'http\S+|www\S+', '', cleanedMsg)
        cleanedMsg = re.sub(r'-----BEGIN PGP PUBLIC KEY BLOCK-----.+?-----END PGP PUBLIC KEY BLOCK-----|ssh-rsa [^\s]+', '', cleanedMsg, flags=re.DOTALL)
        cleanedMsg = re.sub(r'-----BEGIN PGP MESSAGE-----.+?-----END PGP MESSAGE-----', '', cleanedMsg, flags=re.DOTALL)
        cleanedMsg = re.sub(r'-----BEGIN PGP SIGNED MESSAGE-----.+?-----END PGP SIGNED MESSAGE-----', '', cleanedMsg, flags=re.DOTALL)
        cleanedMsg = re.sub(r'(?im)^(?:!.*?$|^@Clyde.*?$|-----BEGIN PGP SIGNED MESSAGE-----|notifywhenonline|-----BEGIN PGP MESSAGE-----|posttosimtoonapi|givegame|queryownedgames).*$', '', cleanedMsg)
        return cleanedMsg


    def load_msgs_from_csv(self, csvPath: str, usernameCol: str, msgCol: str, sep: str = ",", limitToUsers: List[str] = None) -> Tuple[List[List[str]], List[str]]:
        df = pd.read_csv(csvPath, sep=sep, on_bad_lines="skip")
        userIDs = list(set(df[usernameCol].dropna().astype(str).tolist()))
        userIDs = [user for user in userIDs if user != "SYS"]
        if limitToUsers is not None:
            userIDs = [user for user in userIDs if user in limitToUsers]
        logging.info(f"Found {len(userIDs)} unique users in {csvPath}.")
        groupedMsgs = df.groupby(usernameCol)[msgCol].apply(list)
        msgsPerUser = []
        for userID in userIDs:
            userMsgs = groupedMsgs.get(userID, [])
            cleanedMsgs = []
            for msg in userMsgs:
                if isinstance(msg, str):
                    cleanedMsg = self.clean_msg(msg)
                    if cleanedMsg.strip():
                        cleanedMsgs.append(cleanedMsg)
            msgsPerUser.append(cleanedMsgs)
        logging.info(f"Loaded messages from {csvPath} for users {userIDs}.")
        originalMsgCount = sum(len(userMsgs) for userMsgs in msgsPerUser)
        filteredMsgCount = sum(len(cleanedMsgs) for cleanedMsgs in msgsPerUser)
        logging.info(f"Filtered out {originalMsgCount - filteredMsgCount} messages!")
        return msgsPerUser, userIDs

    def load_msgs_from_dat(self, datPath: str, limitToUsers: List[str] = None) -> Tuple[List[List[str]], List[str]]:
        userIDs = set()
        msgsPerUser = defaultdict(list)
        with open(datPath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                msgs = line.split("\n")
                for msg in msgs:
                    if "::" in msg:
                        parts = msg.split("::")
                        if len(parts) >= 2:
                            userMsg = parts[-1]
                            if ":" in userMsg:
                                userID, msg = userMsg.split(":", 1)
                                userID = userID.strip()
                                if userID != "SYS" and (limitToUsers is None or userID in limitToUsers):
                                    userIDs.add(userID)
                                    cleanedMsg = self.clean_msg(msg.strip())
                                    if cleanedMsg:
                                        msgsPerUser[userID].append(cleanedMsg)
                                    else:
                                        logging.info(f"Skipping message '{msg}' from user {userID} because it is empty after cleaning.")
                                else:
                                    logging.info(f"Skipping message '{msg}' from user {userID} because user is not in {limitToUsers}.")
                            else:
                                logging.info(f"Skipping message '{msg}' from user {userID} because it does not contain a colon.")
                        else:
                            logging.info(f"Skipping message '{msg}' from user {userID} because it is empty after splitting on a colon.")
        userIDs = list(userIDs)
        logging.info(f"Found {len(userIDs)} unique users in {datPath}.")
        msgsPerUser = [msgsPerUser[userID] for userID in userIDs]
        logging.info(f"Loaded messages from {datPath} for users {userIDs}.")
        originalMsgCount = sum(len(userMsgs) for userMsgs in msgsPerUser)
        filteredMsgCount = sum(len(cleanedMsgs) for cleanedMsgs in msgsPerUser)
        logging.info(f"Filtered out {originalMsgCount - filteredMsgCount} messages!")
        return msgsPerUser, userIDs