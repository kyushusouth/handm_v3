from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from lightgbm import LGBMRanker, early_stopping, log_evaluation, plot_importance
from sklearn.preprocessing import OrdinalEncoder

from metrics_calculator import MetricsCalculator

sns.set_style("whitegrid")


def transform_cat(
    df: pd.DataFrame, enc: OrdinalEncoder | None = None
) -> OrdinalEncoder:
    cat_cols = [
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
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
    if enc:
        df[cat_cols] = enc.transform(df[cat_cols]).astype(int)
    else:
        enc = OrdinalEncoder(
            dtype=np.float64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        df[cat_cols] = enc.fit_transform(df[cat_cols]).astype(int)
    return enc


def main():
    with open(Path(__file__).parent.joinpath("conf", "config.yaml")) as f:
        conf = yaml.safe_load(f)

    root_dir = Path(__file__).parent.parent
    data_dir = root_dir.parent.joinpath("handm_v2", "data")
    result_dir = root_dir.joinpath("result", datetime.now().strftime("%Y%m%d-%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)
    chunksize = 100000
    past_start_date = pd.to_datetime("2019-08-01")
    past_end_date = pd.to_datetime("2019-08-31")
    train_start_date = pd.to_datetime("2020-07-01")
    train_end_date = pd.to_datetime("2020-07-21")
    val_start_date = pd.to_datetime("2020-07-22")
    val_end_date = pd.to_datetime("2020-07-30")
    test_start_date = pd.to_datetime("2020-08-01")
    test_end_date = pd.to_datetime("2020-08-31")
    num_use_customers = 100
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
    num_cols = [
        "age",
        "user_item_cv_count",
        "user_cv_count",
        "item_cv_count",
        "user_item_cv_ratio",
        "item_user_cv_ratio",
    ]
    cat_cols = [
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
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
        "is_freq_product_code_match",
        "is_freq_colour_group_name_match",
        "is_freq_department_name_match",
        "is_freq_department_no_match",
        "is_freq_garment_group_name_match",
        "is_freq_graphical_appearance_name_match",
        "is_freq_index_group_name_match",
        "is_freq_index_name_match",
        "is_freq_perceived_colour_master_name_match",
        "is_freq_perceived_colour_value_name_match",
        "is_freq_product_group_name_match",
        "is_freq_product_type_name_match",
        "is_freq_section_name_match",
    ]
    num_neg_samples = 5000
    seed = 42

    customer_df = pd.read_csv(
        data_dir.joinpath("customers.csv"), usecols=customer_usecols
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
    train_trans_df = trans_df.loc[
        (train_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= train_end_date),
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

    past_trans_df = past_trans_df.merge(
        customer_df, on="customer_id", how="left"
    ).merge(article_df, on="article_id", how="left")

    print("negative sampling ...")

    def merge_negative_samples(df: pd.DataFrame, seed: int) -> pd.DataFrame:
        neg_records = []
        for customer_id in df["customer_id"].unique():
            purchased_items = set(
                trans_df.loc[
                    trans_df["customer_id"] == customer_id, "article_id"
                ].values
            )
            neg_cand_items = article_df.loc[
                ~article_df["article_id"].isin(purchased_items)
            ]
            size = min(num_neg_samples, len(neg_cand_items))
            neg_items = neg_cand_items.sample(n=size, replace=False, random_state=seed)
            neg_items["customer_id"] = customer_id
            neg_records.append(neg_items[["customer_id", "article_id"]])
        neg_samples_df = pd.concat(neg_records)
        neg_samples_df["is_purchased"] = 0
        df = pd.concat([df, neg_samples_df])
        return df

    train_trans_df = (
        merge_negative_samples(train_trans_df, seed=seed)
        .merge(customer_df, on="customer_id", how="left")
        .merge(article_df, on="article_id", how="left")
    )
    val_trans_df = (
        merge_negative_samples(val_trans_df, seed=seed + 1)
        .merge(customer_df, on="customer_id", how="left")
        .merge(article_df, on="article_id", how="left")
    )
    test_trans_df = (
        merge_negative_samples(test_trans_df, seed=seed + 2)
        .merge(customer_df, on="customer_id", how="left")
        .merge(article_df, on="article_id", how="left")
    )

    print("feature engineering ...")
    cat_enc = transform_cat(past_trans_df)
    _ = transform_cat(train_trans_df, cat_enc)
    _ = transform_cat(val_trans_df, cat_enc)
    _ = transform_cat(test_trans_df, cat_enc)

    user_item_cv_count = past_trans_df.groupby(["customer_id", "article_id"]).agg(
        user_item_cv_count=("customer_id", "count")
    )
    user_cv_count = past_trans_df.groupby("customer_id").agg(
        user_cv_count=("customer_id", "count")
    )
    item_cv_count = past_trans_df.groupby("article_id").agg(
        item_cv_count=("article_id", "count")
    )

    def merge_cv_count_df(df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df.merge(user_item_cv_count, on=["customer_id", "article_id"], how="left")
            .merge(user_cv_count, on="customer_id", how="left")
            .merge(item_cv_count, on="article_id", how="left")
            .assign(
                user_item_cv_ratio=lambda x: x["user_item_cv_count"]
                / x["user_cv_count"],
                item_user_cv_ratio=lambda x: x["user_item_cv_count"]
                / x["item_cv_count"],
            )
            .fillna(0)
        )
        return df

    train_trans_df = merge_cv_count_df(train_trans_df)
    val_trans_df = merge_cv_count_df(val_trans_df)
    test_trans_df = merge_cv_count_df(test_trans_df)

    for col in [
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
    ]:
        user_freq_cat_df = (
            past_trans_df.groupby(["customer_id", col])
            .agg(count=("customer_id", "count"))
            .reset_index()
            .sort_values("count", ascending=False)
            .groupby(["customer_id"])
            .head(1)[["customer_id", col]]
            .rename(columns={col: f"user_freq_{col}"})
        )

        def merge_freq_cat_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.merge(user_freq_cat_df, on="customer_id", how="left")
            df[f"is_freq_{col}_match"] = (df[col] == df[f"user_freq_{col}"]).astype(int)
            df = df.drop(columns=f"user_freq_{col}")
            return df

        train_trans_df = merge_freq_cat_df(train_trans_df)
        val_trans_df = merge_freq_cat_df(val_trans_df)
        test_trans_df = merge_freq_cat_df(test_trans_df)

    train_trans_df = train_trans_df.sort_values("customer_id")
    val_trans_df = val_trans_df.sort_values("customer_id")
    test_trans_df = test_trans_df.sort_values("customer_id")

    ranker = LGBMRanker(
        objective="lambdarank",
        metric=["ndcg"],
        learning_rate=0.01,
        n_estimators=1000,
        importance_type="gain",
        random_state=seed,
        verbosity=-1,
    )
    ranker.fit(
        X=train_trans_df[cat_cols + num_cols],
        y=train_trans_df["is_purchased"],
        group=train_trans_df.groupby("customer_id").size().values,
        eval_set=[(val_trans_df[cat_cols + num_cols], val_trans_df["is_purchased"])],
        eval_group=[val_trans_df.groupby("customer_id").size().values],
        eval_metric=["ndcg"],
        eval_at=[1, 3, 5, 10],
        categorical_feature=cat_cols,
        callbacks=[log_evaluation(), early_stopping(stopping_rounds=50)],
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    metric_colormaps = {"ndcg": "Blues"}
    eval_at_list = [1, 3, 5, 10]
    num_k_values = len(eval_at_list)
    for metric_name, base_cmap_name in metric_colormaps.items():
        cmap = plt.get_cmap(base_cmap_name)
        colors = [cmap(i) for i in np.linspace(0.3, 0.9, num_k_values)]
        for i, k in enumerate(eval_at_list):
            label = f"{metric_name}@{k}"
            metric_values = ranker.evals_result_["valid_0"][label]
            ax.plot(metric_values, color=colors[i], label=label)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Score")
    ax.legend(title="Metrics", bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("learning_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plot_importance(ranker, ax=ax, max_num_features=20, importance_type="gain")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("feature_importance.png"))
    plt.close(fig)

    explainer = shap.TreeExplainer(ranker)
    shap_samples = val_trans_df.sample(1000, random_state=seed)[cat_cols + num_cols]
    shap_values = explainer.shap_values(shap_samples)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("shap_summary_plot.png"))
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    shap.summary_plot(shap_values, shap_samples, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("shap_summary_plot_dot.png"))
    plt.close(fig)

    metrics_calc = MetricsCalculator()
    metrics_records = []
    for _, group_df in test_trans_df.groupby("customer_id"):
        score = ranker.predict(group_df[cat_cols + num_cols])
        group_df["score"] = score

        true_items = group_df.loc[
            group_df["is_purchased"] == 1,
            "article_id",
        ].values.tolist()
        true_items_new = group_df.loc[
            (group_df["is_purchased"] == 1) & (group_df["user_item_cv_count"] == 0),
            "article_id",
        ].values.tolist()
        true_items_repeat = group_df.loc[
            (group_df["is_purchased"] == 1) & (group_df["user_item_cv_count"] > 0),
            "article_id",
        ].values.tolist()

        pred_items = (
            group_df.sort_values("score", ascending=False)
            .head(10)["article_id"]
            .values.tolist()
        )

        for k in [1, 3, 5, 10]:
            recall = metrics_calc.recall_at_k(true_items, pred_items[:k])
            ndcg = metrics_calc.ndcg_at_k(true_items, pred_items[:k])
            metrics_records.append(
                {
                    "model": "lgbm",
                    "eval_type": "all",
                    "k": k,
                    "recall": recall,
                    "ndcg": ndcg,
                }
            )

        if true_items_new:
            for k in [1, 3, 5, 10]:
                recall = metrics_calc.recall_at_k(true_items_new, pred_items[:k])
                ndcg = metrics_calc.ndcg_at_k(true_items_new, pred_items[:k])
                metrics_records.append(
                    {
                        "model": "lgbm",
                        "eval_type": "new",
                        "k": k,
                        "recall": recall,
                        "ndcg": ndcg,
                    }
                )

        if true_items_repeat:
            for k in [1, 3, 5, 10]:
                recall = metrics_calc.recall_at_k(true_items_repeat, pred_items[:k])
                ndcg = metrics_calc.ndcg_at_k(true_items_repeat, pred_items[:k])
                metrics_records.append(
                    {
                        "model": "lgbm",
                        "eval_type": "repeat",
                        "k": k,
                        "recall": recall,
                        "ndcg": ndcg,
                    }
                )

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df = metrics_df.groupby(["model", "eval_type", "k"]).agg(
        num_sample=("recall", "count"), recall=("recall", "mean"), ndcg=("ndcg", "mean")
    )

    for metric_name in ["recall", "ndcg"]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        sns.lineplot(
            metrics_df, x="k", y=metric_name, hue="model", style="eval_type", ax=ax
        )
        fig.tight_layout()
        fig.savefig(result_dir.joinpath(f"{metric_name}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
