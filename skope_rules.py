import numpy as np
from collections import Counter
from collections.abc import Iterable
import pandas
import numbers
from warnings import warn

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import _tree

from rule import Rule, replace_feature_name

from itertools import combinations

INTEGER_TYPES = (numbers.Integral, np.integer)
BASE_FEATURE_NAME = "__C__"


class SkopeRules(BaseEstimator):
    """An easy-interpretable classifier optimizing simple logical rules.

    Parameters
    ----------

    feature_names : list of str, optional
        The names of each feature to be used for returning rules in string
        format.

    precision_min : float, optional (default=0.5)
        The minimal precision of a rule to be selected.

    recall_min : float, optional (default=0.01)
        The minimal recall of a rule to be selected.

    n_estimators : int, optional (default=10)
        The number of base estimators (rules) to use for prediction. More are
        built before selection. All are available in the estimators_ attribute.

    max_samples : int or float, optional (default=.8)
        The number of samples to draw from X to train each decision tree, from
        which rules are generated and selected.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_samples_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each decision tree, from
        which rules are generated and selected.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    max_depth : integer or List or None, optional (default=3)
        The maximum depth of the decision trees. If None, then nodes are
        expanded until all leaves are pure or until all leaves contain less
        than min_samples_split samples.
        If an iterable is passed, you will train n_estimators
        for each tree depth. It allows you to create and compare
        rules of different length.

    max_depth_duplication : integer, optional (default=None)
        The maximum depth of the decision tree for rule deduplication,
        if None then no deduplication occurs.

    max_features : int, float, string or None, optional (default="auto")
        The number of features considered (by each decision tree) when looking
        for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node for
        each decision tree.
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a percentage and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    rules_ : dict of tuples (rule, precision, recall, nb).
        The collection of `n_estimators` rules used in the ``predict`` method.
        The rules are generated by fitted sub-estimators (decision trees). Each
        rule satisfies recall_min and precision_min conditions. The selection
        is done according to OOB precisions.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators used to generate candidate
        rules.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    max_samples_ : integer
        The actual number of samples

    n_features_ : integer
        The number of features when ``fit`` is performed.

    classes_ : array, shape (n_classes,)
        The classes labels.
    """

    def __init__(self,
                 feature_names=None,
                 precision_min=0.5,
                 recall_min=0.01,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=1.,
                 bootstrap=False,
                 bootstrap_features=False,
                 max_depth=6,
                 max_depth_duplication=None,
                 max_features=1.,
                 min_samples_split=2,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        self.precision_min = precision_min
        self.recall_min = recall_min
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_samples_features = max_samples_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_depth = max_depth
        self.max_depth_duplication = max_depth_duplication
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X. Has to follow the convention 0 for
            normal data, 1 for anomalies.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples, typically
            the amount in case of transactions data. Used to grow regression
            trees producing further rules to be tested.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.n_features_ = X.shape[1]

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("This method needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])

        if not isinstance(self.max_depth_duplication, int) \
                and self.max_depth_duplication is not None:
            raise ValueError("max_depth_duplication should be an integer"
                             )
        if not set(self.classes_) == set([0, 1]):
            warn("Found labels %s. This method assumes target class to be"
                 " labeled as 1 and normal data to be labeled as 0. Any label"
                 " different from 0 will be considered as being from the"
                 " target class."
                 % set(self.classes_))
            y = (y > 0)

        # ensure that max_samples is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, str):
            raise ValueError('max_samples (%s) is not supported.'
                             'Valid choices are: "auto", int or'
                             'float' % self.max_samples)

        elif isinstance(self.max_samples, INTEGER_TYPES):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1], got %r"
                                 % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples

        self.rules_ = {}
        self.estimators_ = []
        self.estimators_samples_ = []
        self.estimators_features_ = []

        # default columns names :
        feature_names_ = [BASE_FEATURE_NAME + x for x in
                          np.arange(X.shape[1]).astype(str)]
        if self.feature_names is not None:
            self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                  for i, feat in enumerate(self.feature_names)}
        else:
            self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                  for i, feat in enumerate(feature_names_)}
        self.feature_names_ = feature_names_

        clfs = []

        self._max_depths = self.max_depth \
            if isinstance(self.max_depth, Iterable) else [self.max_depth]

        for max_depth in self._max_depths:
            bagging_clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=max_depth,
                    max_features=self.max_features,
                    min_samples_split=self.min_samples_split,
                ),
                n_estimators=self.n_estimators,
                max_samples=max_samples,
                max_features=self.max_samples_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

            clfs.append(bagging_clf)

        # define regression target:
        if sample_weight is not None:
            if sample_weight is not None:
                sample_weight = check_array(sample_weight, ensure_2d=False)
            weights = sample_weight - sample_weight.min()
            contamination = float(sum(y)) / len(y)
            y_reg = (
                pow(weights, 0.5) * 0.5 / contamination * (y > 0) -
                pow((weights).mean(), 0.5) * (y == 0))
            y_reg = 1. / (1 + np.exp(-y_reg))  # sigmoid
        else:
            y_reg = y  # same as an other classification bagging

        for clf in clfs:
            clf.fit(X, y_reg)
            self.estimators_ += clf.estimators_
            self.estimators_samples_ += clf.estimators_samples_
            self.estimators_features_ += clf.estimators_features_

        rules_ = []
        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):

            # Create mask for OOB samples
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[samples] = True
            mask = ~in_bag
                        
            if sum(mask) == 0:
                warn("OOB evaluation not possible: doing it in-bag."
                     " Performance evaluation is likely to be wrong"
                     " (overfitting) and selected rules are likely to"
                     " not perform well! Please use max_samples < 1.")
                mask = samples
            rules_from_tree = self._tree_to_rules(
                estimator, np.array(self.feature_names_)[features])

            # XXX todo: idem without dataframe
            X_oob = pandas.DataFrame((X[mask, :])[:, features],
                                     columns=np.array(
                                         self.feature_names_)[features])

            if X_oob.shape[1] > 1:  # otherwise pandas bug (cf. issue #16363)
                y_oob = y[mask]
                y_oob = np.array((y_oob != 0))

                # Add OOB performances to rules:
                rules_from_tree = [(r, self._eval_rule_perf(r, X_oob, y_oob))
                                   for r in set(rules_from_tree)]
                rules_ += rules_from_tree

        # Factorize rules before semantic tree filtering
        rules_ = [
            tuple(rule)
            for rule in
            [Rule(r, args=args) for r, args in rules_]]

        # keep only rules verifying precision_min and recall_min:
        for rule, score in rules_:
            if score[0] >= self.precision_min and score[1] >= self.recall_min:
                if rule in self.rules_:
                    # update the score to the new mean
                    c = self.rules_[rule][2] + 1
                    b = self.rules_[rule][1] + 1. / c * (
                        score[1] - self.rules_[rule][1])
                    a = self.rules_[rule][0] + 1. / c * (
                        score[0] - self.rules_[rule][0])

                    self.rules_[rule] = (a, b, c)
                else:
                    self.rules_[rule] = (score[0], score[1], 1)

        self.rules_ = sorted(self.rules_.items(),
                             key=lambda x: (x[1][0], x[1][1]), reverse=True)

        # ========================
        # ✅ NEW LOGIC (ADD HERE)
        # ========================

        # ⚠️ rules_ ở đây phải là list Rule (NOT dict)

        rules_list = [Rule(rule) for rule, perf in self.rules_]

        # ✅ STEP 1: activation sets
        activation_sets = self._build_activation_sets(
            X,
            rules_list,
            self.feature_names_  # dùng "C0, C1" trước khi replace
        )

        # ✅ STEP 2: clustering
        clusters = self._cluster_rules(
            rules_list,
            activation_sets,
            threshold=0.6
        )

        # ✅ save cluster
        self.clusters_ = clusters

        # ✅ OPTIONAL: convert lại rules theo cluster
        self.clustered_rules_ = [
            [str(rules_list[i]) for i in cluster]
            for cluster in clusters
        ]

        # ✅ raw rules (C0, C1)
        self.clustered_rules_raw_ = [
            [str(rules_list[i]) for i in cluster]
            for cluster in clusters
        ]

        # ✅ readable rules
        self.clustered_rules_ = [
            [
                replace_feature_name(rule, self.feature_dict_)
                for rule in cluster
            ]
            for cluster in self.clustered_rules_raw_
        ]

        return self


    def _tree_to_rules(self, tree, feature_names):
        """
        Return a list of rules from a tree

        Parameters
        ----------
            tree : Decision Tree Classifier/Regressor
            feature_names: list of variable names

        Returns
        -------
        rules : list of rules.
        """
        # XXX todo: check the case where tree is build on subset of features,
        # ie max_features != None

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules = []

        def recurse(node, base_name):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                symbol = '<='
                symbol2 = '>'
                threshold = tree_.threshold[node]
                text = base_name + ["{} {} {}".format(name, symbol, threshold)]
                recurse(tree_.children_left[node], text)

                text = base_name + ["{} {} {}".format(name, symbol2,
                                                      threshold)]
                recurse(tree_.children_right[node], text)
            else:
                rule = str.join(' and ', base_name)
                rule = (rule if rule != ''
                        else ' == '.join([feature_names[0]] * 2))
                # a rule selecting all is set to "c0==c0"
                rules.append(rule)

        recurse(0, [])

        return rules if len(rules) > 0 else 'True'

    def _eval_rule_perf(self, rule, X, y):
        detected_index = list(X.query(rule).index)
        if len(detected_index) <= 1:
            return (0, 0)
        y_detected = y[detected_index]
        true_pos = y_detected[y_detected > 0].sum()
        if true_pos == 0:
            return (0, 0)
        pos = y[y > 0].sum()
        return y_detected.mean(), float(true_pos) / pos

    def _build_activation_sets(self, X, rules, feature_names):
        """Convert all rules to binary activation masks over X."""
        return [r.evaluate(X, list(feature_names)) for r in rules]

    # ==========================
    # NEW: similarity functions
    # ==========================

    def _jaccard(self, m1, m2):
        inter = np.sum(m1 & m2)
        union = np.sum(m1 | m2)
        return inter / union if union > 0 else 0.0


    def _adjusted_similarity(self, m1, m2):
        """
        Fix big-rule domination
        """
        j = self._jaccard(m1, m2)
        cov1 = np.mean(m1)
        cov2 = np.mean(m2)

        return j * (1 - max(cov1, cov2))


    def _asymmetric_similarity(self, m1, m2):
        """
        Detect small rule inside big rule
        """
        inter = np.sum(m1 & m2)
        denom = min(np.sum(m1), np.sum(m2))

        return inter / denom if denom > 0 else 0.0
    
    # ==========================
    # NEW: clustering rules
    # ==========================

    # def _cluster_rules(
    #     self,
    #     rules,
    #     activation_sets,
    #     threshold=0.6,
    #     max_cluster_size=10,
    #     rule_scores=None,
    #     ):
    #     """
    #     Graph-based clustering of rules by activation-set similarity.
 
    #     Parameters
    #     ----------
    #     rules : list of Rule
    #     activation_sets : list of boolean np.ndarray
    #     threshold : float
    #         Minimum similarity to draw an edge between two rules.
    #     max_cluster_size : int
    #         Maximum number of rules kept per cluster. When a cluster exceeds
    #         this limit the rules with the highest precision are kept, so that
    #         truncation is deterministic and quality-preserving.
    #     rule_scores : list of (precision, recall) tuples, optional
    #         Used to rank rules inside an oversized cluster. When None every
    #         rule is treated as having equal precision (arbitrary order kept).
    #     """
 
    #     n = len(rules)
    #     adj = {i: set() for i in range(n)}
 
    #     for i, j in combinations(range(n), 2):
    #         sim = self._adjusted_similarity(
    #             activation_sets[i],
    #             activation_sets[j]
    #         )
    #         if sim > threshold:
    #             adj[i].add(j)
    #             adj[j].add(i)
 
    #     visited = set()
    #     clusters = []
 
    #     for i in range(n):
    #         if i in visited:
    #             continue
 
    #         # DFS to collect the connected component
    #         stack = [i]
    #         comp = []
 
    #         while stack:
    #             node = stack.pop()
    #             if node in visited:
    #                 continue
    #             visited.add(node)
    #             comp.append(node)
    #             stack.extend(adj[node])
 
    #         # Truncate oversized clusters: keep highest-precision rules first
    #         if len(comp) > max_cluster_size:
    #             if rule_scores is not None:
    #                 comp = sorted(
    #                     comp,
    #                     key=lambda idx: rule_scores[idx][0],  # precision
    #                     reverse=True
    #                 )
    #             comp = comp[:max_cluster_size]
 
    #         clusters.append(comp)
 
    #     return clusters

    def _cluster_rules(
        self,
        rules,
        activation_sets,
        threshold=0.6,
        max_cluster_size=10,
        rule_scores=None,
        use_clique=True,          # ← tham số mới
    ):
        n = len(rules)

        # Bước 1: build similarity matrix (dùng chung cho cả 2 mode)
        sim = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            s = self._adjusted_similarity(activation_sets[i], activation_sets[j])
            sim[i, j] = sim[j, i] = s

        adj = {i: {j for j in range(n) if i != j and sim[i, j] > threshold}
            for i in range(n)}

        return self._clique_based_clustering(
            n, adj, sim, rule_scores, max_cluster_size
        )
    
        # return self._component_based_clustering(
        #     n, adj, rule_scores, max_cluster_size
        # )


    def _clique_based_clustering(self, n, adj, sim, rule_scores, max_cluster_size):
        """
        Greedy maximal clique clustering.
        Mỗi vòng: tìm clique lớn nhất có thể từ node chưa được assign,
        gom vào cluster, tiếp tục cho đến khi hết node.
        """
        assigned = set()
        clusters = []

        # Sắp xếp node theo precision giảm dần làm seed
        if rule_scores is not None:
            seed_order = sorted(range(n), key=lambda i: rule_scores[i][0], reverse=True)
        else:
            seed_order = list(range(n))

        for seed in seed_order:
            if seed in assigned:
                continue

            # Grow clique từ seed: chỉ thêm node kết nối với TẤT CẢ node hiện tại
            clique = [seed]
            candidates = adj[seed] - assigned

            for node in sorted(candidates,
                            key=lambda i: rule_scores[i][0] if rule_scores else 0,
                            reverse=True):
                if node in assigned:
                    continue
                # Kiểm tra node có edge với mọi thành viên clique hiện tại không
                if all(node in adj[m] for m in clique):
                    clique.append(node)
                    if len(clique) >= max_cluster_size:
                        break

            # Truncate nếu cần (hiếm xảy ra với clique nhưng giữ cho an toàn)
            if len(clique) > max_cluster_size:
                if rule_scores is not None:
                    clique = sorted(clique, key=lambda i: rule_scores[i][0], reverse=True)
                clique = clique[:max_cluster_size]

            for node in clique:
                assigned.add(node)
            clusters.append(clique)

        return clusters


    # def _component_based_clustering(self, n, adj, rule_scores, max_cluster_size):
    #     """
    #     Logic DFS cũ — giữ lại để so sánh.
    #     FIX BUG: nodes bị truncate được đưa vào singleton thay vì biến mất.
    #     """
    #     visited = set()
    #     clusters = []

    #     for i in range(n):
    #         if i in visited:
    #             continue

    #         stack, comp = [i], []
    #         while stack:
    #             node = stack.pop()
    #             if node in visited:
    #                 continue
    #             visited.add(node)
    #             comp.append(node)
    #             stack.extend(adj[node])

    #         if len(comp) > max_cluster_size:
    #             if rule_scores is not None:
    #                 comp_sorted = sorted(comp, key=lambda idx: rule_scores[idx][0], reverse=True)
    #             else:
    #                 comp_sorted = comp

    #             kept    = comp_sorted[:max_cluster_size]
    #             dropped = comp_sorted[max_cluster_size:]   # ← FIX: không bỏ mất

    #             clusters.append(kept)
    #             for node in dropped:
    #                 clusters.append([node])                # → mỗi node thành singleton
    #         else:
    #             clusters.append(comp)

    #     return clusters
    
    # ==========================
    # NEW: ensemble scoring
    # ==========================

    # def _ensemble_score(self, activation_sets, weights=None):
    #     """
    #     Replace max rule with ensemble voting
    #     """

    #     n_samples = len(activation_sets[0])
    #     score = np.zeros(n_samples)

    #     if weights is None:
    #         weights = np.ones(len(activation_sets))

    #     for w, mask in zip(weights, activation_sets):
    #         score += w * mask.astype(float)

    #     return score / np.sum(weights)