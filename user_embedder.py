from sentence_transformers import SentenceTransformer
import pickle
import logging
from typing import List, Tuple
import pandas as pd
import re
import os
from collections import defaultdict
import torch
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class UserEmbedder:
    def __init__(self, modelName: str = "all-mpnet-base-v2") -> None:
        self.model = SentenceTransformer(modelName, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.max_seq_length = 384
        self.tokenizer = self.model.tokenizer


    def gen_embs_from_observations(self, observations: List[str], userID: str = None, bStore: bool = True) -> List:
        tooLong = 0
        for observation in observations:
            if isinstance(observation, str) and len(self.tokenizer.tokenize(observation)) > self.model.max_seq_length:
                logging.warning(f"Observation '{observation[:20]}...' is {len(observation)} chars long, so longer than 384 characters.")
                tooLong += 1
        print(f"Found {tooLong} observations that are too long to be embedded.")


        if userID is not None:
            if os.path.exists(f"{userID}_embs.pt"):
                with open(f"{userID}_embs.pt", "rb") as f:
                    embs = torch.load(f, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    logging.info(f"Loaded {len(embs)} embeddings from {userID}_embs.pt")
                return embs

        embs = self.model.encode(observations, convert_to_tensor=True)
        if bStore and userID is not None:
            with open(f"{userID}_embs.pt", "wb") as f:
                torch.save(embs, f)
                logging.info(f"Stored {len(embs)} embeddings in {userID}_embs.pt")

        return embs

    def clean_msg(self, msg: str) -> str:
        cleanedMsg = re.sub(r'```.*?```|`.*?`|^>.*?$', '', msg, flags=re.DOTALL|re.MULTILINE)
        cleanedMsg = re.sub(r'http\S+|www\S+', '', cleanedMsg)
        cleanedMsg = re.sub(r'-----BEGIN PGP PUBLIC KEY BLOCK-----.+?-----END PGP PUBLIC KEY BLOCK-----|ssh-rsa [^\s]+', '', cleanedMsg, flags=re.DOTALL)
        cleanedMsg = re.sub(r'-----BEGIN PGP MESSAGE-----.+?-----END PGP MESSAGE-----', '', cleanedMsg, flags=re.DOTALL)
        cleanedMsg = re.sub(r'-----BEGIN PGP SIGNED MESSAGE-----.+?-----END PGP SIGNED MESSAGE-----', '', cleanedMsg, flags=re.DOTALL)
        cleanedMsg = re.sub(r'(?im)^(?:!.*?$|^@Clyde.*?$|-----BEGIN PGP SIGNED MESSAGE-----|notifywhenonline|-----BEGIN PGP MESSAGE-----|posttosimtoonapi|givegame|queryownedgames).*$', '', cleanedMsg)
        cleanedMsg = re.sub(r'(?im)^\$.*?\$|^\$\$.*?\$\$', '', cleanedMsg)
        cleanedMsg = re.sub(r"^(Screenshot|Image|File).*", "", cleanedMsg, flags=re.IGNORECASE)
        cleanedMsg = re.sub(r".*\.pdf$", "", cleanedMsg, flags=re.IGNORECASE)
        return cleanedMsg


    def load_msgs_from_csv(self, csvPath: str, usernameCol: str, msgCol: str, sep: str = ",", limitToUsers: List[str] = None) -> Tuple[List[List[str]], List[str], List[str]]:
        df = pd.read_csv(csvPath, sep=sep, on_bad_lines="skip")
        userIDs = list(set(df[usernameCol].dropna().astype(str).tolist()))
        userIDs = [user for user in userIDs if user != "SYS"]
        userIDs = [user for user in userIDs if "#" not in user or user.endswith("#0")]
        if limitToUsers is not None:
            userIDs = [user for user in userIDs if user in limitToUsers]
        logging.info(f"Found {len(userIDs)} unique users in {csvPath}.")

        timestampCol = "Date" if "Date" in df.columns else "timestamp"
        df[timestampCol] = pd.to_datetime(df[timestampCol]).astype(int) / 10**9 # convert to the unix epoch

        groupedMsgs = df.groupby(usernameCol)[msgCol].apply(list)
        groupedTimestamps = df.groupby(usernameCol)[timestampCol].apply(list)
        msgsPerUser = []
        timestampsPerUser = []
        for userID in userIDs:
            userMsgs = groupedMsgs.get(userID, [])
            userTimestamps = groupedTimestamps.get(userID, [])
            cleanedMsgs = []
            cleanedTimestamps = []
            for msg, timestamp in zip(userMsgs, userTimestamps):
                if isinstance(msg, str):
                    cleanedMsg = self.clean_msg(msg)
                    if cleanedMsg.strip():
                        cleanedMsgs.append(cleanedMsg)
                        cleanedTimestamps.append(timestamp)
            msgsPerUser.append(cleanedMsgs)
            timestampsPerUser.append(cleanedTimestamps)
        logging.info(f"Loaded messages from {csvPath} for users {userIDs}.")
        originalMsgCount = sum(len(userMsgs) for userMsgs in msgsPerUser)
        filteredMsgCount = sum(len(cleanedMsgs) for cleanedMsgs in msgsPerUser)
        logging.info(f"Filtered out {originalMsgCount - filteredMsgCount} messages!")
        return msgsPerUser, userIDs, timestampsPerUser

    def load_msgs_from_dat(self, datPath: str, limitToUsers: List[str] = None) -> Tuple[List[List[str]], List[str], List[str]]:
        userIDs = set()
        msgsPerUser = defaultdict(list)
        timestampsPerUser = defaultdict(list)
        with open(datPath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                msgs = line.split("\n")
                msgs = line.split("\\n")
                for msg in msgs:
                    if "::" in msg:
                        parts = msg.split("::")
                        if len(parts) >= 2:
                            userMsg = parts[1]
                            timestampStr = parts[0]
                            try:
                                timestamp = datetime.strptime(timestampStr.strip(), "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                print(parts)
                                breakpoint()
                            timestamp = int(timestamp.timestamp())
                            if ":" in userMsg:
                                userID, msg = userMsg.split(":", 1)
                                userID = userID.strip()
                                timestamp = pd.to_datetime(timestampStr).timestamp()
                                if userID != "SYS" and (limitToUsers is None or userID in limitToUsers or userID.rstrip("#")[0] in limitToUsers):
                                    userIDs.add(userID)
                                    cleanedMsg = self.clean_msg(msg.strip())
                                    if cleanedMsg:
                                        msgsPerUser[userID].append(cleanedMsg)
                                        timestampsPerUser[userID].append(timestamp)
                                    else:
                                        logging.debug(f"Skipping message '{msg}' from user {userID} because it is empty after cleaning.")
                                else:
                                    logging.debug(f"Skipping message '{msg}' from user {userID} because user is not in {limitToUsers}.")
                            else:
                                logging.debug(f"Skipping message '{msg}' from user {userID} because it does not contain a colon.")
                        else:
                            logging.debug(f"Skipping message '{msg}' from user {userID} because it is empty after splitting on a colon.")
        for userID in list(userIDs):
            if userID.endswith("#0"):
                base = userID.rstrip("#0")
                msgsPerUser[base].extend(msgsPerUser[userID])
                timestampsPerUser[base].extend(timestampsPerUser[userID])
                del msgsPerUser[userID]
                del timestampsPerUser[userID]
                userIDs.remove(userID)
                userIDs.add(base)
        userIDs = list(userIDs)
        logging.info(f"Found {len(userIDs)} unique users in {datPath}.")
        msgsPerUser = [msgsPerUser[userID] for userID in userIDs]
        timestampsPerUser = [timestampsPerUser[userID] for userID in userIDs]
        logging.info(f"Loaded messages from {datPath} for users {userIDs}.")
        originalMsgCount = sum(len(userMsgs) for userMsgs in msgsPerUser)
        filteredMsgCount = sum(len(cleanedMsgs) for cleanedMsgs in msgsPerUser)
        logging.info(f"Filtered out {originalMsgCount - filteredMsgCount} messages!")
        return msgsPerUser, userIDs, timestampsPerUser

    def load_direct_msgs_from_copied_discord_txt(self, txtPath: str) -> Tuple[List[List[str]], List[str], List[str]]:
        with open(txtPath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        pattern = re.compile(r"^(.+) — (\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M)$")

        currentUser = None
        currentUserMsgs = []
        currentUserTimestamps = []
        msgsPerUser = {}
        timestampsPerUser = {}
        users = set()

        for line in lines:
            match = pattern.match(line.strip())
            if match:
                if currentUser:
                    cleaned = [self.clean_msg(msg).strip() for msg in currentUserMsgs]
                    cleaned = [msg for msg in cleaned if msg]
                    msgsPerUser.setdefault(currentUser, []).extend(cleaned)
                    timestampsPerUser.setdefault(currentUser, []).extend(currentUserTimestamps)
                    currentUserMsgs = []
                    currentUserTimestamps = []
                currentUser = match.group(1)
                timestamp = pd.to_datetime(match.group(2)).timestamp()
                users.add(currentUser)
            elif currentUser and not line.strip().startswith("Image"):
                if not line.strip().startswith("Image"):
                    cleaned = self.clean_msg(line.strip())
                    if cleaned.strip():
                        currentUserMsgs.append(cleaned)
                        currentUserTimestamps.append(timestamp)

        if currentUser:
            cleaned = [self.clean_msg(msg).strip() for msg in currentUserMsgs]
            cleaned = [msg for msg in cleaned if msg]
            msgsPerUser.setdefault(currentUser, []).extend(cleaned)
            timestampsPerUser.setdefault(currentUser, []).extend(currentUserTimestamps)

        sort = sorted(list(users))
        msgs = [msgsPerUser[user] for user in sort]
        timestamps = [timestampsPerUser[user] for user in sort]

        logging.info(f"Found {len(sort)} unique users in {txtPath}.")
        logging.info(f"Loaded {len(msgs)} messages from {txtPath} for users {sort}.")
        originalMsgCount = sum(len(userMsgs) for userMsgs in msgs)
        filteredMsgCount = sum(len(cleanedMsgs) for cleanedMsgs in msgs)
        logging.info(f"Filtered out {originalMsgCount - filteredMsgCount} messages!")

        return msgs, sort, [list(map(int, userTimestamps)) for userTimestamps in timestamps]