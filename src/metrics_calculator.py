from typing import Any

import numpy as np


class MetricsCalculator:
    def __init__(self):
        pass

    def precision_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        if len(pred_items) == 0:
            return 0
        precision_at_k = (len(set(true_items) & set(pred_items))) / len(pred_items)
        return precision_at_k

    def recall_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        if len(true_items) == 0:
            return 0
        recall_at_k = (len(set(true_items) & set(pred_items))) / len(true_items)
        return recall_at_k

    def f1_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        precision = self.precision_at_k(true_items, pred_items)
        recall = self.recall_at_k(true_items, pred_items)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    def rr_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        rr = 0
        for rank, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                rr = 1 / (rank + 1)
        return rr

    def get_user_relevance(
        self, true_items: list[Any], pred_items: list[Any]
    ) -> list[int]:
        user_relevances = [0 for _ in range(len(pred_items))]
        for rank, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                user_relevances[rank] = 1
        return user_relevances

    def ap_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        user_relevances = self.get_user_relevance(true_items, pred_items)
        if sum(user_relevances) == 0:
            return 0
        nonzero_indices = np.asarray(user_relevances).nonzero()[0]
        return sum(
            [sum(user_relevances[: idx + 1]) / (idx + 1) for idx in nonzero_indices]
        ) / sum(user_relevances)

    def dcg_at_k(self, user_relevances: list[int]) -> float:
        return user_relevances[0] + np.sum(
            user_relevances[1:] / np.log2(np.arange(2, len(user_relevances) + 1))
        )

    def ndcg_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        user_relevances = self.get_user_relevance(true_items, pred_items)
        if sum(user_relevances) == 0:
            return 0
        dcg_max = self.dcg_at_k(sorted(user_relevances, reverse=True))
        if not dcg_max:
            return 0
        return self.dcg_at_k(user_relevances) / dcg_max
