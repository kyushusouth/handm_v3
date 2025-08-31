from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir.parent.joinpath("handm_v2", "data")
    result_dir = root_dir.joinpath("result", datetime.now().strftime("%Y%m%d-%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)
    chunksize = 100000
    past_start_date = pd.to_datetime("2020-07-01")
    past_end_date = pd.to_datetime("2020-07-31")
    train_start_date = pd.to_datetime("2020-08-01")
    train_end_date = pd.to_datetime("2020-08-17")
    val_start_date = pd.to_datetime("2020-08-18")
    val_end_date = pd.to_datetime("2020-08-24")
    test_start_date = pd.to_datetime("2020-08-25")
    test_end_date = pd.to_datetime("2020-08-31")
    num_use_customers = 10000
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

    past_trans_df = trans_df.loc[
        (past_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= past_end_date),
        ["customer_id", "article_id", "is_purchased"],
    ].copy()
    future_trans_df = trans_df.loc[
        train_start_date <= trans_df["t_dat"],
        ["customer_id", "article_id", "is_purchased"],
    ].copy()

    past_trans_df = past_trans_df.merge(
        customer_df, on="customer_id", how="left"
    ).merge(article_df, on="article_id", how="left")

    num_popular_items = 10
    num_age_popular_items = 10

    popular_items = (
        past_trans_df.groupby("article_id")
        .agg(item_cv_count=("customer_id", "count"))
        .sort_values("item_cv_count", ascending=False)
        .head(num_popular_items)
        .index.values.tolist()
    )
    age_popular_items = (
        past_trans_df.groupby(["age_bin", "article_id"])
        .agg(item_cv_count=("customer_id", "count"))
        .reset_index()
        .sort_values("item_cv_count", ascending=False)
        .groupby("age_bin")
        .head(num_age_popular_items)
        .groupby("age_bin")["article_id"]
        .apply(list)
        .to_dict()
    )

    customer_age_bin_map = customer_df.set_index("customer_id")["age_bin"].to_dict()
    recalls = []
    for customer_id, group_df in future_trans_df.groupby("customer_id"):
        age_bin = customer_age_bin_map[customer_id]
        true_items = group_df["article_id"].unique().tolist()
        pred_items = popular_items + age_popular_items.get(age_bin, [])
        recall = len(set(true_items) & set(pred_items)) / len(set(true_items))
        recalls.append(recall)

    print(np.mean(recalls))


if __name__ == "__main__":
    main()
