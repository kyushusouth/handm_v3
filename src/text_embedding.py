from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sns.set_style("whitegrid")


def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir.parent.joinpath("handm_v2", "data")
    result_dir = root_dir.joinpath("result", datetime.now().strftime("%Y%m%d-%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)
    chunksize = 100000
    past_start_date = pd.to_datetime("2020-06-01")
    past_end_date = pd.to_datetime("2020-06-30")
    test_start_date = pd.to_datetime("2020-08-01")
    test_end_date = pd.to_datetime("2020-08-31")
    num_use_customers = 1000
    customer_usecols = [
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
    ]
    article_usecols = [
        "article_id",
        "product_code",
        "colour_group_name",
        "department_name",
        "department_no",
        "garment_group_name",
        "graphical_appearance_name",
        "index_group_name",
        "index_name",
        "perceived_colour_master_name",
        "perceived_colour_value_name",
        "product_group_name",
        "product_type_name",
        "section_name",
        "detail_desc",
    ]
    customer_df = pd.read_csv(
        data_dir.joinpath("customers.csv"), usecols=customer_usecols
    )
    customer_df.loc[~customer_df["age"].isna(), "age_bin"] = (
        customer_df.loc[~customer_df["age"].isna(), "age"] // 10 * 10
    )
    article_df = pd.read_csv(data_dir.joinpath("articles.csv"), usecols=article_usecols)

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
    use_customers = np.random.choice(
        trans_df["customer_id"].unique(), size=num_use_customers, replace=False
    )
    trans_df = trans_df.loc[trans_df["customer_id"].isin(use_customers)]
    trans_df["is_purchased"] = 1

    customer_age_bin_map = (
        customer_df.loc[~customer_df["age_bin"].isna()]
        .set_index("customer_id")["age_bin"]
        .to_dict()
    )

    past_trans_df = trans_df.loc[
        (past_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= past_end_date),
        ["t_dat", "customer_id", "article_id", "is_purchased"],
    ].copy()
    test_trans_df = trans_df.loc[
        (test_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= test_end_date),
        ["t_dat", "customer_id", "article_id", "is_purchased"],
    ].copy()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    article_ids = article_df["article_id"].values
    id_to_index = {article_id: index for index, article_id in enumerate(article_ids)}
    sentences = article_df["detail_desc"].values.tolist()
    batchsize = 512
    embedding_batches = []
    for i in tqdm(range(0, len(sentences), batchsize)):
        batch = sentences[i : i + batchsize]
        batch_embs = model.encode(batch)
        embedding_batches.append(batch_embs)
    embeddings = np.concatenate(embedding_batches, axis=0)  # (n_sample, n_dim)

    user_cv_count = (
        past_trans_df.groupby("customer_id")
        .agg(user_cv_count=("article_id", "count"))
        .reset_index()
    )
    warm_customers = set(
        user_cv_count.loc[
            user_cv_count["user_cv_count"] > 0, "customer_id"
        ].values.tolist()
    )
    cold_customers = set(use_customers) - warm_customers
    customer_topk_items = {}
    customer_rec_status = {}
    num_rec = 10

    popular_items = (
        past_trans_df.groupby("article_id")
        .agg(item_cv_count=("customer_id", "count"))
        .reset_index()
        .sort_values("item_cv_count", ascending=False)
        .head(num_rec)
    )

    last_purchase_item_df = (
        past_trans_df.loc[past_trans_df["customer_id"].isin(warm_customers)]
        .sort_values("t_dat", ascending=False)
        .groupby("customer_id")
        .head(1)
    )
    last_purchase_item_map = last_purchase_item_df.set_index("customer_id")[
        "article_id"
    ]
    for customer_id in tqdm(warm_customers):
        last_purchase_item = last_purchase_item_map[customer_id]
        last_purchase_item_index = id_to_index[last_purchase_item]
        query_emb = embeddings[last_purchase_item_index]
        scores = (query_emb[None, :] @ embeddings.T)[0]  # (n_sample,)
        sorted_indices = np.argsort(scores)
        topk_indices = [
            index
            for index in sorted_indices[-(num_rec + 1) :]
            if index != last_purchase_item_index
        ]  # 直近購入したアイテムと同じアイテムは選択しない
        topk_items = article_ids[topk_indices]
        customer_topk_items[customer_id] = topk_items
        customer_rec_status[customer_id] = "personalized"

    age_bin_to_customers_map = (
        customer_df.loc[customer_df["customer_id"].isin(warm_customers)]
        .groupby("age_bin")
        .agg(customer_ids=("customer_id", list))["customer_ids"]
        .to_dict()
    )
    for customer_id in tqdm(cold_customers):
        customer_age_bin = customer_age_bin_map.get(customer_id)
        if not customer_age_bin:
            customer_topk_items[customer_id] = popular_items
            customer_rec_status[customer_id] = "popular"
            continue
        related_warm_customers = age_bin_to_customers_map[customer_age_bin]
        last_purchase_items = [
            last_purchase_item_map[cid] for cid in related_warm_customers
        ]
        last_purchase_item_indices = [id_to_index[item] for item in last_purchase_items]
        query_emb = np.mean(embeddings[last_purchase_item_indices], axis=0)
        scores = (query_emb[None, :] @ embeddings.T)[0]  # (n_sample,)
        sorted_indices = np.argsort(scores)
        topk_indices = [index for index in sorted_indices[-num_rec:]]
        topk_items = article_ids[topk_indices]
        customer_topk_items[customer_id] = topk_items
        customer_rec_status[customer_id] = "related"

    metrics_records = []
    for customer_id, group_df in tqdm(test_trans_df.groupby("customer_id")):
        pred_items = customer_topk_items[customer_id]
        true_items = group_df["article_id"].unique().tolist()
        for k in [1, 3, 5, 10]:
            recall = len(set(pred_items[:k]) & set(true_items)) / len(true_items)
            metrics_records.append(
                {
                    "customer_id": customer_id,
                    "rec_status": customer_rec_status[customer_id],
                    "k": k,
                    "recall": recall,
                }
            )
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df = (
        metrics_df.groupby(["rec_status", "k"])
        .agg(n_sample=("customer_id", "count"), recall=("recall", "mean"))
        .reset_index()
    )

    print(metrics_df)

    plt.figure(figsize=(12, 8))
    sns.barplot(metrics_df, x="k", y="recall", hue="rec_status")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("recall.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.barplot(metrics_df, x="k", y="n_sample", hue="rec_status")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("n_sample.png"))
    plt.close()


if __name__ == "__main__":
    main()
