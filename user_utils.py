from typing import Dict, Tuple, List
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial import ConvexHull
import umap.umap_ as umap
from hdbscan import HDBSCAN
import pandas as pd
import mplcyberpunk
from shapely.geometry import Polygon
import logging
from termcolor import colored

""" to be used once we have user embs """



class UserUtils:
    def __init__(self) -> None:
        pass

    def get_user_embs_mean(self, embs1: Tensor, embs2: Tensor, bDimReduce: bool = True, nComps: int = 2) -> Tensor:
        if bDimReduce:
            embs = torch.cat((embs1, embs2))
            _umap = umap.UMAP(n_neighbors=100, n_components=nComps, metric="cosine", random_state=69)
            embs = _umap.fit_transform(embs.cpu().numpy())
            embs = torch.tensor(embs, dtype=torch.float32)
            embs1 = embs[:len(embs1)]
            embs2 = embs[len(embs1):]
        else:
            embs1 = embs1.cpu()
            embs2 = embs2.cpu()
        embs1Mean = torch.mean(embs1, dim=0)
        embs2Mean = torch.mean(embs2, dim=0)
        return embs1Mean, embs2Mean

    def compare_two_users(self, embs1: Tensor, embs2: Tensor) -> Dict:
        # TODO: make sure that the embs are centered and normalized
        jaccards = {"UMAP": None}
        dimReducers = {"UMAP": umap.UMAP(n_neighbors=100, n_components=2, metric="cosine", random_state=69)}
        for name, reducer in dimReducers.items():
            embsReduced = reducer.fit_transform(torch.cat((embs1, embs2)).cpu().numpy())

            hull1 = ConvexHull(embsReduced[:len(embs1)])
            hull2 = ConvexHull(embsReduced[len(embs1):])

            poly1 = Polygon(embsReduced[:len(embs1)][hull1.vertices]).buffer(0)
            poly2 = Polygon(embsReduced[len(embs1):][hull2.vertices]).buffer(0)

            if not (poly1.is_valid and poly2.is_valid):
                raise ValueError("Polygons are not valid!")

            inter = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            jaccards[name] = inter / union if union != 0 else 0

        emb1Mean, emb2Mean = self.get_user_embs_mean(embs1, embs2, True, 2)
        emb1MeanTip = (emb1Mean[0], emb1Mean[1])
        emb2MeanTip = (emb2Mean[0], emb2Mean[1])
        triArea = 0.5 * np.linalg.norm(np.cross(emb1MeanTip, emb2MeanTip))
        cosim = torch.nn.functional.cosine_similarity(emb1Mean.unsqueeze(0), emb2Mean.unsqueeze(0), dim=1).item()
        dot = torch.dot(emb1Mean, emb2Mean).item()
        euclidean = torch.norm(emb1Mean - emb2Mean).item()
        magnitude1, magnitude2 = self.calc_magnitude(emb1Mean).item(), self.calc_magnitude(emb2Mean).item()
        return {"cosim": cosim, "dot": dot, "euclidean": euclidean, "jaccards": jaccards, "magnitude1": magnitude1, "magnitude2": magnitude2, "triArea": triArea}

    def normalize(self, x: Tensor) -> Tensor:
        return x / torch.norm(x)

    def calc_magnitude(self, x: Tensor) -> Tensor:
        return torch.norm(x)

    def calc_spread(self, embsReduced: Tensor) -> Tuple[Tensor, Tensor, float]:
        ranges = np.ptp(embsReduced, axis=0)
        stds = np.std(embsReduced, axis=0)
        hull = ConvexHull(embsReduced)
        hullVol = hull.volume
        return ranges, stds, hullVol, hull

    def train_isolation_forest_for_user(self, embs: Tensor) -> IsolationForest:
        forest = IsolationForest(n_estimators=10000, n_jobs=-1, max_features=768).fit(embs.cpu().numpy())
        logging.info(f"Trained IsolationForest for user.")
        logging.debug(f"Trained IsolationForest for user with {len(embs)} embeddings.\nmaxSamples: {forest.max_samples_}\ncontamination: {forest.contamination}\nnEstimators: {forest.n_estimators}\nmaxFeatures: {forest.max_features}\nnJobs: {forest.n_jobs}\nn_features_in: {forest.n_features_in_}")
        return forest

    def plot_and_measure_spread(self, embs1: Tuple[Tensor, str], embs2: Tuple[Tensor, str], reduceFunc, title: str) -> None:
        embs = torch.cat((embs1[0], embs2[0]))
        embsReduced = reduceFunc(embs.cpu().numpy())

        spreadEmbs1 = self.calc_spread(embsReduced[:len(embs1[0])])
        spreadEmbs2 = self.calc_spread(embsReduced[len(embs1[0]):])

        plt.figure(figsize=(16, 16))
        plt.scatter(embsReduced[:len(embs1[0]), 0], embsReduced[:len(embs1[0]), 1], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")
        plt.scatter(embsReduced[len(embs1[0]):, 0], embsReduced[len(embs1[0]):, 1], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")
        plt.legend()
        plt.title(title)
        plt.xlabel(f"{title}0")
        plt.ylabel(f"{title}1")
        mplcyberpunk.add_glow_effects()

        plt.show()

        print(f"Spread of {embs1[1]}: {spreadEmbs1}\nSpread of {embs2[1]}: {spreadEmbs2}")
        return spreadEmbs1, spreadEmbs2

    def plot_embs(self, embs1: Tuple[Tensor, str, List[str]], embs2: Tuple[Tensor, str, List[str]], kmeansClusters: int = 30) -> None:
        plt.style.use("cyberpunk")

        pca = PCA(n_components=2)
        embs = torch.cat((embs1[0], embs2[0]))
        embsReduced = pca.fit_transform(embs.cpu())

        plt.figure(figsize=(16, 16))

        kmeans = KMeans(n_clusters=kmeansClusters, random_state=69, n_init="auto").fit(embsReduced)
        centroids = kmeans.cluster_centers_

        def find_closest_point_idx(points, center):
            return np.linalg.norm(points - center, axis=1).argmin()

        plt.scatter(embsReduced[:len(embs1[0]), 0], embsReduced[:len(embs1[0]), 1], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")

        for center in centroids:
            idx = find_closest_point_idx(embsReduced[:len(embs1[0])], center)
            plt.annotate(embs1[2][idx], (embsReduced[idx, 0], embsReduced[idx, 1]), textcoords="offset points", xytext=(5, -5), ha="right", color="#ff00ff")

        hullPts1 = embsReduced[:len(embs1[0])]
        hull1 = ConvexHull(hullPts1)
        poly1 = Polygon(hullPts1[hull1.vertices]).buffer(0)
        first = True
        for simplex in hull1.simplices:
            if first:
                plt.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], c="#ffffff", linewidth=3, label=embs1[1])
                first = False
            else:
                plt.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], c="#ffffff", linewidth=3)

        plt.scatter(embsReduced[len(embs1[0]):, 0], embsReduced[len(embs1[0]):, 1], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")

        offset = len(embs1[0])
        for center in centroids:
            idx = find_closest_point_idx(embsReduced[offset:], center)
            plt.annotate(embs2[2][idx], (embsReduced[offset + idx, 0], embsReduced[offset + idx, 1]), textcoords="offset points", xytext=(5, -5), ha="right", color="#bfff00")

        hullPts2 = embsReduced[len(embs1[0]):]
        hull2 = ConvexHull(hullPts2)
        poly2 = Polygon(hullPts2[hull2.vertices]).buffer(0)
        first = True
        for simplex in hull2.simplices:
            if first:
                plt.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], "k-", linewidth=3, label=embs2[1])
                first = False
            else:
                plt.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], "k-", linewidth=3)

        inter = poly1.intersection(poly2)

        plt.fill(*poly1.exterior.xy, alpha=0.15, fc='r', label=f"{embs1[1]}'s hull")
        plt.fill(*poly2.exterior.xy, alpha=0.15, fc='b', label=f"{embs2[1]}'s hull")

        plt.legend()
        plt.title("PCA")
        plt.xlabel("PC0")
        plt.ylabel("PC1")

        mplcyberpunk.make_lines_glow()

        tsne = TSNE(n_components=2, random_state=69)
        embsReduced = tsne.fit_transform(embs.cpu().numpy())

        plt.figure(figsize=(16, 16))

        kmeans = KMeans(n_clusters=kmeansClusters, random_state=69, n_init="auto").fit(embsReduced)
        centroids = kmeans.cluster_centers_

        plt.scatter(embsReduced[:len(embs1[0]), 0], embsReduced[:len(embs1[0]), 1], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")

        for center in centroids:
            idx = find_closest_point_idx(embsReduced[:len(embs1[0])], center)
            plt.annotate(embs1[2][idx], (embsReduced[idx, 0], embsReduced[idx, 1]), textcoords="offset points", xytext=(5, -5), ha="right", color="#ff00ff")

        hullPts1 = embsReduced[:len(embs1[0])]
        hull1 = ConvexHull(hullPts1)
        poly1 = Polygon(hullPts1[hull1.vertices]).buffer(0)
        first = True
        for simplex in hull1.simplices:
            if first:
                plt.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], c="#ffffff", linewidth=3, label=embs1[1])
                first = False
            else:
                plt.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], c="#ffffff", linewidth=3)

        plt.scatter(embsReduced[len(embs1[0]):, 0], embsReduced[len(embs1[0]):, 1], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")

        offset = len(embs1[0])
        for center in centroids:
            idx = find_closest_point_idx(embsReduced[offset:], center)
            plt.annotate(embs2[2][idx], (embsReduced[offset + idx, 0], embsReduced[offset + idx, 1]), textcoords="offset points", xytext=(5, -5), ha="right", color="#bfff00")

        hullPts2 = embsReduced[len(embs1[0]):]
        hull2 = ConvexHull(hullPts2)
        poly2 = Polygon(hullPts2[hull2.vertices]).buffer(0)
        first = True
        for simplex in hull2.simplices:
            if first:
                plt.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], "k-", linewidth=3, label=embs2[1])
                first = False
            else:
                plt.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], "k-", linewidth=3)

        inter = poly1.intersection(poly2)

        plt.fill(*poly1.exterior.xy, alpha=0.15, fc='r', label=f"{embs1[1]}'s hull")
        plt.fill(*poly2.exterior.xy, alpha=0.15, fc='b', label=f"{embs2[1]}'s hull")

        plt.legend()
        plt.title("t-SNE")
        plt.xlabel("t-SNE0")
        plt.ylabel("t-SNE1")

        mplcyberpunk.make_lines_glow()


        ump = umap.UMAP(n_neighbors=100, n_components=2, metric="cosine").fit_transform(embs.cpu())

        plt.figure(figsize=(16, 16))

        kmeans = KMeans(n_clusters=kmeansClusters, random_state=69, n_init="auto").fit(ump)
        centroids = kmeans.cluster_centers_

        plt.scatter(ump[:len(embs1[0]), 0], ump[:len(embs1[0]), 1], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")

        for center in centroids:
            idx = find_closest_point_idx(ump[:len(embs1[0])], center)
            plt.annotate(embs1[2][idx], (ump[idx, 0], ump[idx, 1]), textcoords="offset points", xytext=(5, -5), ha="right", color="#ff00ff")

        hullPts1 = ump[:len(embs1[0])]
        hull1 = ConvexHull(hullPts1)
        poly1 = Polygon(hullPts1[hull1.vertices]).buffer(0)
        first = True
        for simplex in hull1.simplices:
            if first:
                plt.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], c="#ffffff", linewidth=3, label=embs1[1])
                first = False
            else:
                plt.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], c="#ffffff", linewidth=3)

        plt.scatter(ump[len(embs1[0]):, 0], ump[len(embs1[0]):, 1], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")

        offset = len(embs1[0])
        for center in centroids:
            idx = find_closest_point_idx(ump[offset:], center)
            plt.annotate(embs2[2][idx], (ump[offset + idx, 0], ump[offset + idx, 1]), textcoords="offset points", xytext=(5, -5), ha="right", color="#bfff00")

        hullPts2 = ump[len(embs1[0]):]
        hull2 = ConvexHull(hullPts2)
        poly2 = Polygon(hullPts2[hull2.vertices]).buffer(0)
        first = True
        for simplex in hull2.simplices:
            if first:
                plt.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], "k-", linewidth=3, label=embs2[1])
                first = False
            else:
                plt.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], "k-", linewidth=3)

        inter = poly1.intersection(poly2)

        plt.fill(*poly1.exterior.xy, alpha=0.15, fc='r', label=f"{embs1[1]}'s hull")
        plt.fill(*poly2.exterior.xy, alpha=0.15, fc='b', label=f"{embs2[1]}'s hull")

        plt.legend()
        plt.title("UMAP")
        plt.xlabel("UMAP0")
        plt.ylabel("UMAP1")

        mplcyberpunk.make_lines_glow()


        plt.show()

        # spreadPCAEmbs1, spreadPCAEmbs2 = self.plot_and_measure_spread(embs1, embs2, lambda x: PCA(n_components=2).fit_transform(x), "PCA")

    def plot_embs_3D(self, embs1: Tuple[Tensor, str], embs2: Tuple[Tensor, str]) -> None:
        pca = PCA(n_components=3)
        embs = torch.cat((embs1[0], embs2[0]))
        embsReduced = pca.fit_transform(embs.cpu())

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embsReduced[:len(embs1[0]), 0], embsReduced[:len(embs1[0]), 1], embsReduced[:len(embs1[0]), 2], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")
        hullPts1 = embsReduced[:len(embs1[0])]
        hull1 = ConvexHull(hullPts1)
        first = True
        for simplex in hull1.simplices:
            if first:
                ax.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], hullPts1[simplex, 2], c="#ffffff", linewidth=3, label=embs1[1])
                first = False
            else:
                ax.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], hullPts1[simplex, 2], c="#ffffff", linewidth=3)

        ax.scatter(embsReduced[len(embs1[0]):, 0], embsReduced[len(embs1[0]):, 1], embsReduced[len(embs1[0]):, 2], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")
        hullPts2 = embsReduced[len(embs1[0]):]
        hull2 = ConvexHull(hullPts2)
        first = True
        for simplex in hull2.simplices:
            if first:
                ax.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], hullPts2[simplex, 2], "k-", linewidth=3, label=embs2[1])
                first = False
            else:
                ax.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], hullPts2[simplex, 2], "k-", linewidth=3)

        ax.legend()
        ax.set_title("PCA")
        ax.set_xlabel("PC0")
        ax.set_ylabel("PC1")
        ax.set_zlabel("PC2")


        tsne = TSNE(n_components=3, random_state=69)
        embsReduced = tsne.fit_transform(embs.cpu().numpy())
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embsReduced[:len(embs1[0]), 0], embsReduced[:len(embs1[0]), 1], embsReduced[:len(embs1[0]), 2], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")
        hullPts1 = embsReduced[:len(embs1[0])]
        hull1 = ConvexHull(hullPts1)
        first = True
        for simplex in hull1.simplices:
            if first:
                ax.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], hullPts1[simplex, 2], c="#ffffff", linewidth=3, label=embs1[1])
                first = False
            else:
                ax.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], hullPts1[simplex, 2], c="#ffffff", linewidth=3)

        ax.scatter(embsReduced[len(embs1[0]):, 0], embsReduced[len(embs1[0]):, 1], embsReduced[len(embs1[0]):, 2], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")
        hullPts2 = embsReduced[len(embs1[0]):]
        hull2 = ConvexHull(hullPts2)
        first = True
        for simplex in hull2.simplices:
            if first:
                ax.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], hullPts2[simplex, 2], "k-", linewidth=3, label=embs2[1])
                first = False
            else:
                ax.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], hullPts2[simplex, 2], "k-", linewidth=3)

        ax.legend()
        ax.set_title("t-SNE")
        ax.set_xlabel("t-SNE0")
        ax.set_ylabel("t-SNE1")
        ax.set_zlabel("t-SNE2")


        ump = umap.UMAP(n_neighbors=100, n_components=3, metric="cosine").fit_transform(embs.cpu())
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ump[:len(embs1[0]), 0], ump[:len(embs1[0]), 1], ump[:len(embs1[0]), 2], label=embs1[1], c="#ff00ff", alpha=0.7, marker="o")
        hullPts1 = ump[:len(embs1[0])]
        hull1 = ConvexHull(hullPts1)
        first = True
        for simplex in hull1.simplices:
            if first:
                ax.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], hullPts1[simplex, 2], c="#ffffff", linewidth=3, label=embs1[1])
                first = False
            else:
                ax.plot(hullPts1[simplex, 0], hullPts1[simplex, 1], hullPts1[simplex, 2], c="#ffffff", linewidth=3)

        ax.scatter(ump[len(embs1[0]):, 0], ump[len(embs1[0]):, 1], ump[len(embs1[0]):, 2], label=embs2[1], c="#bfff00", alpha=0.7, marker="^")
        hullPts2 = ump[len(embs1[0]):]
        hull2 = ConvexHull(hullPts2)
        first = True
        for simplex in hull2.simplices:
            if first:
                ax.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], hullPts2[simplex, 2], "k-", linewidth=3, label=embs2[1])
                first = False
            else:
                ax.plot(hullPts2[simplex, 0], hullPts2[simplex, 1], hullPts2[simplex, 2], "k-", linewidth=3)

        ax.legend()
        ax.set_title("UMAP")
        ax.set_xlabel("UMAP0")
        ax.set_ylabel("UMAP1")
        ax.set_zlabel("UMAP2")

        plt.show()

    def plot_mean_embs(self, emb1Mean: Tensor, emb2Mean: Tensor, userIDs: Tuple[str, str]) -> None:
        plt.style.use("cyberpunk")

        fig, ax = plt.subplots(figsize=(16, 16))

        emb1MeanTip = (emb1Mean[0], emb1Mean[1])
        emb2MeanTip = (emb2Mean[0], emb2Mean[1])
        opposite = np.linalg.norm(np.array(emb1MeanTip) - np.array(emb2MeanTip))

        # ax.plot(*zip((0, 0), emb1MeanTip), "k--", linewidth=3)

        pts = np.array([emb1MeanTip, emb2MeanTip, (0, 0)])
        tri = patches.Polygon(pts, closed=True, fill=False, edgecolor="k", linewidth=1)
        ax.add_patch(tri)
        triArea = 0.5 * np.linalg.norm(np.cross(emb1MeanTip, emb2MeanTip))
        areaUnderEmb1 = 0.5 * abs(emb1Mean[0]) * abs(emb1Mean[1])
        areaUnderEmb2 = 0.5 * abs(emb2Mean[0]) * abs(emb2Mean[1])
        logging.debug(f"areaUnderEmb1: {areaUnderEmb1}\nareaUnderEmb2: {areaUnderEmb2}\ntriArea: {triArea}")
        logging.debug(f"ratio of areaUnderEmb1 to areaUnderEmb2: {areaUnderEmb1 / areaUnderEmb2}")
        logging.debug(f"ratio of len of emb1 to len of emb2: {np.linalg.norm(emb1MeanTip) / np.linalg.norm(emb2MeanTip)}")
        logging.debug(f"len of opposite: {opposite}")

        offset = 0.00075
        ax.text(emb1Mean[0] / 2, (emb1Mean[1] / 2) + (offset * -1), userIDs[0], fontsize=12, ha="left", color="b")
        ax.text(emb2Mean[0] / 2, (emb2Mean[1] / 2) + (offset * -1), userIDs[1], fontsize=12, ha="left", color="r")
        ax.text((emb1MeanTip[0] + emb2MeanTip[0]) / 2, (emb1MeanTip[1] + emb2MeanTip[1]) / 2, "L2", fontsize=12, ha="left", color="k")
        ax.text(0, -0.0005, "Origin", fontsize=12, ha="center")

        ax.fill_between([0, emb1MeanTip[0]], [0, emb1MeanTip[1]], color="g", alpha=0.2, hatch="\\", label=f"area under {userIDs[0]}: {areaUnderEmb1:.4f}")
        ax.fill_between([0, emb2MeanTip[0]], [0, emb2MeanTip[1]], color="r", alpha=0.2, hatch="/" , label=f"area under {userIDs[1]}: {areaUnderEmb2:.4f}")
        ax.fill(*zip((0, 0), emb1MeanTip, emb2MeanTip), color="violet", alpha=0.75, label=f"area of triangle: {triArea:.4f}")

        ax.quiver(0, 0, emb1Mean[0], emb1Mean[1], angles='xy', scale_units='xy', scale=1, color="b", label=userIDs[0])
        ax.quiver(0, 0, emb2Mean[0], emb2Mean[1], angles='xy', scale_units='xy', scale=1, color="r", label=userIDs[1])

        side1Mag = np.linalg.norm(np.array(emb1MeanTip))
        side2Mag = np.linalg.norm(np.array(emb2MeanTip))
        side3Mag = np.linalg.norm(np.array(emb1MeanTip) - np.array(emb2MeanTip))
        print(f"side1Mag: {side1Mag}\nside2Mag: {side2Mag}\nside3Mag: {side3Mag}")

        angle1 = np.arccos((side1Mag**2 + side3Mag**2 - side2Mag**2) / (2 * side1Mag * side3Mag)) * 180 / np.pi
        angle2 = np.arccos((side2Mag**2 + side3Mag**2 - side1Mag**2) / (2 * side2Mag * side3Mag)) * 180 / np.pi
        angle3 = np.arccos((side1Mag**2 + side2Mag**2 - side3Mag**2) / (2 * side1Mag * side2Mag)) * 180 / np.pi

        logging.debug(f"angle1: {angle1}\nangle2: {angle2}\nangle3: {angle3}")
        
        cosAngle3 = np.cos(np.radians(angle3))
        check = side1Mag**2 + side2Mag**2 - 2 * side1Mag * side2Mag * cosAngle3
        if not np.isclose(check, side3Mag**2):
            raise ValueError("side3Mag is not equal to check!")
        else:
            logging.info(colored("law of cosines check passed!", "green"))


        # arcR = 0.001
        # emb1MeanAngle = np.arctan2(emb1Mean[1], emb1Mean[0]) * (180 / np.pi)
        # emb1MeanAngle = (emb1MeanAngle + 360) % 360
        # emb2MeanAngle = np.arctan2(emb2Mean[1], emb2Mean[0]) * (180 / np.pi)
        # emb2MeanAngle = (emb2MeanAngle + 360) % 360
        # arc1 = patches.Arc((0, 0), arcR, arcR, angle=0, theta1=0, theta2=emb1MeanAngle, edgecolor="k", linewidth=3)
        # arc2 = patches.Arc(emb1MeanTip, arcR, arcR, angle=emb1MeanAngle - 180, theta1=0, theta2=180 - (angle1 + angle2), edgecolor="k", linewidth=3)
        # arc3 = patches.Arc(emb2MeanTip, arcR, arcR, angle=emb2MeanAngle - 180, theta1=0, theta2=angle2, edgecolor="k", linewidth=3)
        # ax.add_patch(arc1)
        # ax.add_patch(arc2)
        # ax.add_patch(arc3)

        ax.text((emb1MeanTip[0] / 2), (emb1MeanTip[1] / 2) + offset, f"{side1Mag:.4f}", fontsize=10, ha="right", alpha=0.33)
        ax.text((emb2MeanTip[0] / 2), (emb2MeanTip[1] / 2) + offset, f"{side2Mag:.4f}", fontsize=10, ha="right", alpha=0.33)
        ax.text((emb1MeanTip[0] + emb2MeanTip[0]) / 2, (emb1MeanTip[1] + emb2MeanTip[1]) / 2, f"{side3Mag:.4f}", fontsize=10, ha="right", alpha=0.5)

        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.title(f"Mean Embeddings of {userIDs[0]} and {userIDs[1]} in dim-reduced (UMAP) space\n(random state == 69, n_neighbors == 100, n_components == 2, metric == cosine)")
        plt.grid()
        plt.axis("equal")

        mplcyberpunk.make_lines_glow()

        plt.show()

    def plot_mean_embs_3D(self, emb1Mean: Tensor, emb2Mean: Tensor, userIDs: Tuple[str, str]) -> None:
        _umap = umap.UMAP(n_neighbors=15, n_components=3, metric="cosine")
        emb1Mean = _umap.fit_transform(emb1Mean.cpu().numpy())
        emb2Mean = _umap.fit_transform(emb2Mean.cpu().numpy())

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')

        ax.quiver(0, 0, 0, emb1Mean[0], emb1Mean[1], emb1Mean[2], color="b", label=userIDs[0])
        ax.quiver(0, 0, 0, emb2Mean[0], emb2Mean[1], emb2Mean[2], color="r", label=userIDs[1])

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()
        ax.set_title("Mean Embeddings")
        ax.grid()
        ax.axis("equal")

        plt.show()

    def plot_vector_field_3D(self, embs: Tuple[Tensor, str], embs2: Tuple[Tensor, str]) -> None:
        tsne = TSNE(n_components=3, random_state=69)
        embsCombined = torch.cat((embs[0], embs2[0]))
        embsReduced = tsne.fit_transform(embsCombined.cpu())

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')

        for i, emb in enumerate(embsReduced):
            color = "b" if i < len(embs[0]) else "r"
            ax.quiver(0, 0, 0, emb[0], emb[1], emb[2], length=np.linalg.norm(emb), normalize=True, color=color)

        ax.set_xlabel("t-SNE0")
        ax.set_ylabel("t-SNE1")
        ax.set_zlabel("t-SNE2")
        ax.legend([embs[1], embs2[1]])
        ax.set_title("Vector Field of Embeddings")

        plt.show()

    def plot_vector_field_3D_mean(self, embs):
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')

        for emb in embs:
            if isinstance(emb, Tensor):
                emb = emb.cpu().numpy()
            norm = np.linalg.norm(emb)
            if norm != 0:
                emb = emb / norm
            ax.quiver(0, 0, 0, emb[0], emb[1], emb[2], length=norm, normalize=True)

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.set_title("Vector Field of Mean Embeddings")

        plt.show()

    def plot_topics(self, embs1: Tuple[Tensor, str], embs2: Tuple[Tensor, str]) -> None:
        _umap = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine").fit_transform(torch.cat((embs1[0], embs2[0])).cpu().numpy())

        cluster = HDBSCAN(min_cluster_size=15, min_samples=1, metric="euclidean", cluster_selection_method="eom").fit(umap)

        res = pd.DataFrame(_umap, columns=["x", "y"])
        res["labels"] = cluster.labels_

        fig, ax = plt.subplots(figsize=(16, 16))
        outliers = res.loc[res.labels == -1, :]
        clustered = res.loc[res.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, cmap='hsv', s=10, alpha=0.5)
        plt.colorbar()
        plt.title("UMAP")
        plt.xlabel("UMAP0")
        plt.ylabel("UMAP1")

        plt.show()
