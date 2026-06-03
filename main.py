import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from skope_rules import SkopeRules


def main():
    # Breast cancer: malignant (0) -> đổi thành 1 (anomaly), benign (1) -> 0 (normal)
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = SkopeRules(
        feature_names=feature_names,
        precision_min=0.60,
        recall_min=0.10,
        n_estimators=15,
        max_depth=6,
        bootstrap=True,
        random_state=42,
    )

    print("Đang train SkopeRules trên tập dữ liệu Breast Cancer...")
    clf.fit(X_train, y_train)

    n_rules = len(clf.rules_)
    n_clusters = len(clf.clustered_rules_)
    print(f"  -> Tổng số rule: {n_rules}")
    print(f"  -> Tổng số cluster: {n_clusters}")

    # Build lookup: rule_str (BASE_FEATURE_NAME form) -> (precision, recall, count)
    perf_by_idx = {i: clf.rules_[i][1] for i in range(n_rules)}

    print("\n" + "=" * 70)
    for cluster_idx, (rule_indices, readable_rules) in enumerate(
        zip(clf.clusters_, clf.clustered_rules_)
    ):
        n = len(readable_rules)
        print(f"\nCluster {cluster_idx + 1}  [{n} rule{'s' if n > 1 else ''}]")
        print("-" * 70)
        for rank, (rule_idx, rule_str) in enumerate(
            zip(rule_indices, readable_rules)
        ):
            precision, recall, count = perf_by_idx[rule_idx]
            support = int(round(recall * y_train.sum()))
            print(f"  ({rank + 1}) {rule_str}")
            print(
                f"       precision={precision:.2f}  "
                f"recall={recall:.2f}  "
                f"support≈{support} mẫu  "
                f"xuất hiện trong {count} cây"
            )

    print("\n" + "=" * 70)
    print(f"\nKết thúc. {n_rules} rule được nhóm thành {n_clusters} cluster.")


if __name__ == "__main__":
    main()
