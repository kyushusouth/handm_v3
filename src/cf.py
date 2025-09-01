from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix


def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir.parent.joinpath("handm_v2", "data")
    result_dir = root_dir.joinpath("result", datetime.now().strftime("%Y%m%d-%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)
    past_start_date = pd.to_datetime("2020-01-01")
    past_end_date = pd.to_datetime("2020-07-31")
    val_start_date = pd.to_datetime("2020-08-18")
    val_end_date = pd.to_datetime("2020-08-24")
    test_start_date = pd.to_datetime("2020-08-25")
    test_end_date = pd.to_datetime("2020-08-31")
    chunksize = 100000
    num_use_customers = 10000
    random_state = 42

    filt_chunks = []
    for chunk in pd.read_csv(
        data_dir.joinpath("transactions_train.csv"), chunksize=chunksize
    ):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        filtered_chunk = chunk.loc[
            (past_start_date <= chunk["t_dat"]) & (chunk["t_dat"] <= test_end_date)
        ]
        if not filtered_chunk.empty:
            filt_chunks.append(filtered_chunk)
    trans_df = pd.concat(filt_chunks)
    # use_customers = np.random.choice(
    #     trans_df["customer_id"].unique(), size=num_use_customers, replace=False
    # )
    # trans_df = trans_df.loc[trans_df["customer_id"].isin(use_customers)]
    trans_df["is_purchased"] = 1

    past_trans_df = trans_df.loc[
        (past_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= past_end_date),
        ["customer_id", "article_id", "is_purchased"],
    ].copy()
    val_trans_df = trans_df.loc[
        (val_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= val_end_date),
        ["customer_id", "article_id", "is_purchased"],
    ].copy()
    test_trans_df = trans_df.loc[
        (test_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= test_end_date),
        ["customer_id", "article_id", "is_purchased"],
    ].copy()

    cf_customer_ids = past_trans_df["customer_id"].unique()
    customer_index_map = {
        customer_id: index for index, customer_id in enumerate(cf_customer_ids)
    }

    cf_article_ids = past_trans_df["article_id"].unique()
    article_index_map = {
        article_id: index for index, article_id in enumerate(cf_article_ids)
    }
    index_article_map = {
        index: article_id for index, article_id in enumerate(cf_article_ids)
    }

    past_trans_df["customer_index"] = past_trans_df["customer_id"].map(
        customer_index_map
    )
    past_trans_df["article_index"] = past_trans_df["article_id"].map(article_index_map)

    customer_indices = past_trans_df["customer_index"].values
    article_indices = past_trans_df["article_index"].values
    is_purchased = past_trans_df["is_purchased"].values

    cf_matrix = csr_matrix(
        (is_purchased, (customer_indices, article_indices)),
        shape=(cf_customer_ids.shape[0], cf_article_ids.shape[0]),
    )

    np.savez(
        result_dir.joinpath("cf_data"),
        cf_customer_ids=cf_customer_ids,
        cf_article_ids=cf_article_ids,
        cf_matrix=cf_matrix,
    )

    als = AlternatingLeastSquares(factors=64, iterations=15, random_state=random_state)
    als.fit(cf_matrix)
    als.save(result_dir.joinpath("als.npz"))

    bpr = BayesianPersonalizedRanking(
        factors=64, iterations=100, random_state=random_state
    )
    bpr.fit(cf_matrix)
    bpr.save(result_dir.joinpath("bpr.npz"))

    val_customer_ids = val_trans_df["customer_id"].unique()
    valid_val_customer_ids = np.intersect1d(cf_customer_ids, val_customer_ids)
    valid_val_customer_indices = np.array(
        [customer_index_map[customer_id] for customer_id in valid_val_customer_ids]
    )
    test_customer_ids = test_trans_df["customer_id"].unique()
    valid_test_customer_ids = np.intersect1d(cf_customer_ids, test_customer_ids)
    valid_test_customer_indices = np.array(
        [customer_index_map[customer_id] for customer_id in valid_test_customer_ids]
    )
    print(f"val: {valid_val_customer_ids.shape[0] / val_customer_ids.shape[0]}")
    print(f"test: {valid_test_customer_ids.shape[0] / test_customer_ids.shape[0]}")

    als_val_pred_indices, _ = als.recommend(
        valid_val_customer_indices,
        cf_matrix[valid_val_customer_indices],
        N=10,
    )
    als_test_pred_indices, _ = als.recommend(
        valid_test_customer_indices,
        cf_matrix[valid_test_customer_indices],
        N=10,
    )
    bpr_val_pred_indices, _ = bpr.recommend(
        valid_val_customer_indices,
        cf_matrix[valid_val_customer_indices],
        N=10,
    )
    bpr_test_pred_indices, _ = bpr.recommend(
        valid_test_customer_indices,
        cf_matrix[valid_test_customer_indices],
        N=10,
    )

    val_customer_purchase_map = (
        val_trans_df.groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    test_customer_purchase_map = (
        test_trans_df.groupby("customer_id")["article_id"].apply(list).to_dict()
    )

    metrics_records = []
    for i, customer_id in enumerate(valid_val_customer_ids):
        true_items_cand = val_customer_purchase_map[customer_id]
        true_items = set()
        for true_item in true_items_cand:
            if true_item in cf_article_ids:
                true_items.add(true_item)

        if len(true_items) == 0:
            continue

        pred_indices = als_val_pred_indices[i]
        pred_items = [index_article_map[index] for index in pred_indices]
        for k in [1, 3, 5, 10]:
            recall = len(true_items & set(pred_items[:k])) / len(true_items)
            metrics_records.append(
                {"data": "val", "model": "als", "k": k, "recall": recall}
            )

        pred_indices = bpr_val_pred_indices[i]
        pred_items = [index_article_map[index] for index in pred_indices]
        for k in [1, 3, 5, 10]:
            recall = len(true_items & set(pred_items[:k])) / len(true_items)
            metrics_records.append(
                {"data": "val", "model": "bpr", "k": k, "recall": recall}
            )

    for i, customer_id in enumerate(valid_test_customer_ids):
        true_items_cand = test_customer_purchase_map[customer_id]
        true_items = set()
        for true_item in true_items_cand:
            if true_item in cf_article_ids:
                true_items.add(true_item)

        if len(true_items) == 0:
            continue

        pred_indices = als_test_pred_indices[i]
        pred_items = [index_article_map[index] for index in pred_indices]
        for k in [1, 3, 5, 10]:
            recall = len(true_items & set(pred_items[:k])) / len(true_items)
            metrics_records.append(
                {"data": "test", "model": "als", "k": k, "recall": recall}
            )

        pred_indices = bpr_test_pred_indices[i]
        pred_items = [index_article_map[index] for index in pred_indices]
        for k in [1, 3, 5, 10]:
            recall = len(true_items & set(pred_items[:k])) / len(true_items)
            metrics_records.append(
                {"data": "test", "model": "bpr", "k": k, "recall": recall}
            )

    metrics_df = (
        pd.DataFrame(metrics_records)
        .groupby(["data", "model", "k"])
        .agg(recall=("recall", "mean"))
        .reset_index()
    )
    print(metrics_df)


if __name__ == "__main__":
    main()
