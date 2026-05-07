from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - fallback for minimal environments
    KMeans = None

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover - fallback for minimal environments
    leaves_list = None
    linkage = None
    squareform = None


class AdvancedPortfolioPipeline:
    """
    Portfolio constructor for MOEX-style 3D tensors: [days, tickers, features].

    The first feature is treated as a price proxy for return estimation. Other
    features are still useful indirectly through clustering if they are present
    in the input tensor.
    """

    METHODS = {
        "ensemble": "ML ensemble: HRP + Black-Litterman views + CVaR policy search + graph attention",
        "hrp": "Hierarchical Risk Parity with shrinkage covariance",
        "black_litterman": "Black-Litterman with ML-style momentum/volatility views",
        "cvar": "Distribution-aware CVaR optimization via cross-entropy policy search",
        "graph_attention": "Graph neural network inspired risk propagation and attention weighting",
    }

    def __init__(
        self,
        decay_factor: float = 0.97,
        n_clusters: int = 8,
        n_iterations: int = 10,
        method: str = "ensemble",
        risk_free_rate: float = 0.0,
        target_assets: Optional[int] = None,
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        random_state: int = 42,
    ):
        self.decay_factor = decay_factor
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.method = method
        self.risk_free_rate = risk_free_rate
        self.target_assets = target_assets
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.random_state = random_state
        self.is_trained = False

    def fit(self, tensor_3d: np.ndarray, tickers: np.ndarray):
        self.tickers_ = np.array(tickers)
        self.tensor_ = np.array(tensor_3d, dtype=np.float64)
        n_days, n_tickers, _ = self.tensor_.shape

        if n_days < 3:
            raise ValueError("Для продвинутых методов нужно минимум 3 дня наблюдений")
        if n_tickers < 2:
            raise ValueError("Для портфеля нужно минимум 2 инструмента")

        self.n_clusters = max(1, min(self.n_clusters, n_tickers))
        self.max_weight = max(float(self.max_weight), 1.0 / n_tickers)
        self.target_assets_ = min(self.target_assets or self.n_clusters, n_tickers)

        self.prices_ = self._extract_price_matrix(self.tensor_)
        self.returns_ = self._calculate_returns(self.prices_)
        self.expected_returns_ = self._ewma_mean(self.returns_)
        self.covariance_ = self._shrinkage_covariance(self.returns_)
        self.correlation_ = self._correlation_from_covariance(self.covariance_)
        self.labels_ = self._cluster_assets()

        method = self.method.lower()
        if method not in self.METHODS:
            raise ValueError(f"Неизвестный метод '{self.method}'. Доступно: {list(self.METHODS)}")

        weights_by_method = {
            "hrp": self._hrp_weights(),
            "black_litterman": self._black_litterman_weights(),
            "cvar": self._cvar_policy_search_weights(),
            "graph_attention": self._graph_attention_weights(),
        }

        if method == "ensemble":
            weights = (
                0.35 * weights_by_method["hrp"]
                + 0.25 * weights_by_method["black_litterman"]
                + 0.25 * weights_by_method["cvar"]
                + 0.15 * weights_by_method["graph_attention"]
            )
        else:
            weights = weights_by_method[method]

        self.method_weights_ = {
            name: self._weights_to_dict(value) for name, value in weights_by_method.items()
        }
        self.weights_ = self._project_to_bounds(weights)
        self.portfolio_indices_ = self._select_portfolio_indices(self.weights_)
        self.diagnostics_ = self._build_diagnostics()
        self.is_trained = True
        return self

    def predict(self) -> List[str]:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        return [str(self.tickers_[i]) for i in self.portfolio_indices_]

    def get_weights(self) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        selected = {str(self.tickers_[i]): self.weights_[i] for i in self.portfolio_indices_}
        total = sum(selected.values())
        if total > 0:
            selected = {ticker: value / total for ticker, value in selected.items()}
        return {ticker: round(float(value), 6) for ticker, value in selected.items()}

    def get_clusters(self) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        clusters = {}
        sharpe = self._asset_scores()
        for cluster_id in range(self.n_clusters):
            indices = np.where(self.labels_ == cluster_id)[0]
            if len(indices) == 0:
                continue
            center_idx = indices[np.argmax(sharpe[indices])]
            sub_corr = self.correlation_[np.ix_(indices, indices)]
            avg_corr = 0.0 if len(indices) == 1 else float(np.mean(sub_corr[np.triu_indices(len(indices), 1)]))
            clusters[f"cluster_{cluster_id}"] = {
                "tickers": [str(self.tickers_[i]) for i in indices],
                "size": int(len(indices)),
                "center": str(self.tickers_[center_idx]),
                "avg_correlation": round(avg_corr, 4),
            }
        return clusters

    def get_diagnostics(self) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        return self.diagnostics_

    def _extract_price_matrix(self, tensor_3d: np.ndarray) -> np.ndarray:
        prices = tensor_3d[:, :, 0].astype(np.float64)
        prices[~np.isfinite(prices)] = np.nan
        column_medians = np.nanmedian(prices, axis=0)
        column_medians[~np.isfinite(column_medians)] = 1.0
        prices = np.where(np.isfinite(prices), prices, column_medians)
        return np.maximum(prices, 1e-12)

    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        returns = prices[1:] / prices[:-1] - 1.0
        returns[~np.isfinite(returns)] = 0.0
        if returns.shape[0] > 4:
            lower = np.nanpercentile(returns, 1, axis=0)
            upper = np.nanpercentile(returns, 99, axis=0)
            returns = np.clip(returns, lower, upper)
        return returns

    def _ewma_mean(self, returns: np.ndarray) -> np.ndarray:
        weights = self.decay_factor ** np.arange(len(returns) - 1, -1, -1)
        weights = weights / np.sum(weights)
        return weights @ returns

    def _shrinkage_covariance(self, returns: np.ndarray) -> np.ndarray:
        centered = returns - self._ewma_mean(returns)
        weights = self.decay_factor ** np.arange(len(returns) - 1, -1, -1)
        weights = weights / np.sum(weights)
        sample_cov = (centered * weights[:, None]).T @ centered
        avg_variance = float(np.mean(np.diag(sample_cov)))
        target = np.eye(sample_cov.shape[0]) * max(avg_variance, 1e-8)
        shrinkage = 0.25
        cov = (1.0 - shrinkage) * sample_cov + shrinkage * target
        cov += np.eye(cov.shape[0]) * 1e-8
        return cov

    def _correlation_from_covariance(self, covariance: np.ndarray) -> np.ndarray:
        vol = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        corr = covariance / np.outer(vol, vol)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _cluster_assets(self) -> np.ndarray:
        features = self._asset_feature_matrix()
        if KMeans is not None and len(features) >= self.n_clusters:
            best_labels = None
            best_inertia = float("inf")
            for i in range(self.n_iterations):
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state + i, n_init=1)
                labels = kmeans.fit_predict(features)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_labels = labels
            return best_labels

        order = np.argsort(self._asset_scores())
        labels = np.zeros(len(features), dtype=int)
        for position, asset_idx in enumerate(order):
            labels[asset_idx] = position % self.n_clusters
        return labels

    def _asset_feature_matrix(self) -> np.ndarray:
        volatility = np.std(self.returns_, axis=0) + 1e-12
        downside = np.sqrt(np.mean(np.minimum(self.returns_, 0.0) ** 2, axis=0))
        momentum = np.prod(1.0 + self.returns_[-min(20, len(self.returns_)):], axis=0) - 1.0
        corr_embedding = self._correlation_embedding()
        base = np.column_stack([self.expected_returns_, volatility, downside, momentum, corr_embedding])
        return self._zscore(base)

    def _correlation_embedding(self) -> np.ndarray:
        try:
            values, vectors = np.linalg.eigh(self.correlation_)
            order = np.argsort(values)[::-1][: min(3, len(values))]
            embedding = vectors[:, order] * values[order]
            if embedding.shape[1] < 3:
                padding = np.zeros((embedding.shape[0], 3 - embedding.shape[1]))
                embedding = np.column_stack([embedding, padding])
            return embedding
        except np.linalg.LinAlgError:
            return np.zeros((self.correlation_.shape[0], 3))

    def _asset_scores(self) -> np.ndarray:
        volatility = np.sqrt(np.maximum(np.diag(self.covariance_), 1e-12))
        return (self.expected_returns_ - self.risk_free_rate / 252.0) / volatility

    def _hrp_weights(self) -> np.ndarray:
        order = self._hierarchical_order()
        weights = np.ones(len(order))
        clusters = [list(range(len(order)))]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            left_var = self._cluster_variance(order[left])
            right_var = self._cluster_variance(order[right])
            alpha = 1.0 - left_var / (left_var + right_var + 1e-12)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            clusters.extend([left, right])

        result = np.zeros(len(order))
        result[order] = weights
        return self._project_to_bounds(result)

    def _hierarchical_order(self) -> np.ndarray:
        if linkage is not None and leaves_list is not None and squareform is not None:
            distance = np.sqrt(np.maximum((1.0 - self.correlation_) / 2.0, 0.0))
            np.fill_diagonal(distance, 0.0)
            try:
                return leaves_list(linkage(squareform(distance, checks=False), method="single"))
            except Exception:
                pass
        return np.lexsort((self._asset_scores(), self.labels_))

    def _cluster_variance(self, indices: np.ndarray) -> float:
        sub_cov = self.covariance_[np.ix_(indices, indices)]
        inverse_variance = 1.0 / np.maximum(np.diag(sub_cov), 1e-12)
        weights = inverse_variance / np.sum(inverse_variance)
        return float(weights.T @ sub_cov @ weights)

    def _black_litterman_weights(self) -> np.ndarray:
        n_assets = len(self.tickers_)
        market_weights = np.ones(n_assets) / n_assets
        risk_aversion = 2.5
        prior = risk_aversion * self.covariance_ @ market_weights
        views = self.expected_returns_
        volatility = np.sqrt(np.maximum(np.diag(self.covariance_), 1e-12))
        confidence = 1.0 / (1.0 + volatility / np.median(volatility))
        omega = np.diag(np.maximum((1.0 - confidence) * volatility ** 2, 1e-8))
        tau_cov = 0.05 * self.covariance_

        try:
            precision = np.linalg.pinv(tau_cov) + np.linalg.pinv(omega)
            posterior = np.linalg.pinv(precision) @ (np.linalg.pinv(tau_cov) @ prior + np.linalg.pinv(omega) @ views)
            raw = np.linalg.pinv(self.covariance_) @ posterior
        except np.linalg.LinAlgError:
            raw = views / np.maximum(volatility, 1e-12)

        return self._long_only_from_scores(raw)

    def _cvar_policy_search_weights(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_assets = len(self.tickers_)
        mean = np.ones(n_assets) / n_assets
        concentration = 40.0

        for _ in range(5):
            samples = rng.dirichlet(np.maximum(mean * concentration, 0.2), size=600)
            samples = np.array([self._project_to_bounds(sample) for sample in samples])
            scores = np.array([self._cvar_objective(sample) for sample in samples])
            elite_count = max(10, len(samples) // 10)
            elite = samples[np.argsort(scores)[-elite_count:]]
            mean = elite.mean(axis=0)
            concentration *= 1.25

        return self._project_to_bounds(mean)

    def _cvar_objective(self, weights: np.ndarray) -> float:
        portfolio_returns = self.returns_ @ weights
        losses = -portfolio_returns
        threshold = np.quantile(losses, 0.95)
        tail_losses = losses[losses >= threshold]
        cvar = float(np.mean(tail_losses)) if len(tail_losses) else float(threshold)
        expected = float(np.mean(portfolio_returns))
        diversification = 1.0 - float(np.sum(weights ** 2))
        return expected - 2.5 * cvar + 0.02 * diversification

    def _graph_attention_weights(self) -> np.ndarray:
        adjacency = np.abs(self.correlation_)
        np.fill_diagonal(adjacency, 0.0)
        threshold = np.quantile(adjacency[adjacency > 0], 0.6) if np.any(adjacency > 0) else 1.0
        adjacency = np.where(adjacency >= threshold, adjacency, 0.0)
        row_sum = adjacency.sum(axis=1, keepdims=True)
        transition = np.divide(adjacency, row_sum, out=np.zeros_like(adjacency), where=row_sum > 0)

        centrality = np.ones(len(self.tickers_)) / len(self.tickers_)
        teleport = centrality.copy()
        for _ in range(40):
            centrality = 0.85 * transition.T @ centrality + 0.15 * teleport

        volatility = np.sqrt(np.maximum(np.diag(self.covariance_), 1e-12))
        momentum = np.prod(1.0 + self.returns_[-min(20, len(self.returns_)):], axis=0) - 1.0
        attention_logits = self._zscore_vector(self.expected_returns_ / volatility)
        attention_logits += 0.25 * self._zscore_vector(momentum)
        attention_logits -= 0.50 * self._zscore_vector(centrality)
        attention = self._softmax(attention_logits)
        return self._project_to_bounds(0.65 * attention + 0.35 * self._hrp_weights())

    def _select_portfolio_indices(self, weights: np.ndarray) -> np.ndarray:
        order = np.argsort(weights)[::-1]
        selected = order[: self.target_assets_]
        return selected[np.argsort(weights[selected])[::-1]]

    def _build_diagnostics(self) -> Dict[str, Any]:
        selected_weights = np.zeros_like(self.weights_)
        selected_weights[self.portfolio_indices_] = self.weights_[self.portfolio_indices_]
        selected_weights = self._normalize(selected_weights)
        portfolio_returns = self.returns_ @ selected_weights
        volatility = float(np.std(portfolio_returns))
        expected = float(np.mean(portfolio_returns))
        losses = -portfolio_returns
        cvar = float(np.mean(losses[losses >= np.quantile(losses, 0.95)])) if len(losses) else 0.0

        return {
            "method": self.method,
            "method_description": self.METHODS[self.method.lower()],
            "expected_return_daily": round(expected, 6),
            "volatility_daily": round(volatility, 6),
            "sharpe_daily": round(expected / (volatility + 1e-12), 4),
            "cvar_95_daily": round(cvar, 6),
            "effective_assets": round(1.0 / max(float(np.sum(selected_weights ** 2)), 1e-12), 3),
            "available_methods": self.METHODS,
            "scientific_novelty": [
                "робастная shrinkage-ковариация вместо нестабильной выборочной матрицы",
                "иерархическая диверсификация HRP без обращения плохо обусловленной ковариации",
                "CVaR-оптимизация хвостового риска через cross-entropy policy search",
                "графовая attention-поправка: штраф за системно центральные и сильно связанные акции",
                "ансамбль риск-моделей, снижающий зависимость от одного предположения о рынке",
            ],
        }

    def _long_only_from_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.array(scores, dtype=np.float64)
        scores[~np.isfinite(scores)] = 0.0
        scores = scores - np.min(scores)
        if np.sum(scores) <= 1e-12:
            scores = self._asset_scores() - np.min(self._asset_scores())
        if np.sum(scores) <= 1e-12:
            scores = np.ones_like(scores)
        return self._project_to_bounds(scores / np.sum(scores))

    def _project_to_bounds(self, weights: np.ndarray) -> np.ndarray:
        weights = np.array(weights, dtype=np.float64)
        weights[~np.isfinite(weights)] = 0.0
        weights = np.maximum(weights, 0.0)
        weights = self._normalize(weights)
        lower = max(0.0, float(self.min_weight))
        upper = min(1.0, max(float(self.max_weight), 1.0 / len(weights)))
        weights = np.clip(weights, lower, upper)

        for _ in range(30):
            total = float(np.sum(weights))
            if abs(total - 1.0) < 1e-10:
                break
            if total > 1.0:
                candidates = weights > lower + 1e-12
                excess = total - 1.0
                capacity = np.sum(weights[candidates] - lower)
                if capacity <= 1e-12:
                    break
                weights[candidates] -= excess * (weights[candidates] - lower) / capacity
            else:
                candidates = weights < upper - 1e-12
                deficit = 1.0 - total
                capacity = np.sum(upper - weights[candidates])
                if capacity <= 1e-12:
                    break
                weights[candidates] += deficit * (upper - weights[candidates]) / capacity
            weights = np.clip(weights, lower, upper)
        return self._normalize(weights)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = float(np.sum(weights))
        if total <= 1e-12:
            return np.ones_like(weights) / len(weights)
        return weights / total

    def _weights_to_dict(self, weights: np.ndarray) -> Dict[str, float]:
        return {str(ticker): round(float(weight), 6) for ticker, weight in zip(self.tickers_, weights)}

    def _zscore(self, values: np.ndarray) -> np.ndarray:
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0) + 1e-12
        result = (values - mean) / std
        result[~np.isfinite(result)] = 0.0
        return result

    def _zscore_vector(self, values: np.ndarray) -> np.ndarray:
        values = np.array(values, dtype=np.float64)
        std = np.nanstd(values) + 1e-12
        result = (values - np.nanmean(values)) / std
        result[~np.isfinite(result)] = 0.0
        return result

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        values = values - np.max(values)
        exp_values = np.exp(np.clip(values, -50, 50))
        return exp_values / np.sum(exp_values)
