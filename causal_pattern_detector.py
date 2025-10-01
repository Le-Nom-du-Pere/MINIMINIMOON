# coding=utf-8
import warnings

import dcor
import networkx as nx
import numpy as np
import pandas as pd
import pygam
import torch
import torch.nn as nn
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class IndustrialCausalPatternDetector:
    """
    工业级因果模式检测器，集成多种现代因果推断方法
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.causal_graph = nx.DiGraph()
        self.models = {}
        self.scalers = {}
        self.results = {}

    def preprocess_data(self, data):
        """
        数据预处理模块
        """
        # 处理缺失值
        data = data.fillna(data.mean())

        # 标准化数值特征
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        self.scalers["main"] = scaler

        return data

    @staticmethod
    def detect_correlation_patterns(data):
        """
        基于多种相关性度量的模式检测
        """
        results = {}
        numeric_data = data.select_dtypes(include=[np.number])

        # Pearson相关系数
        pearson_corr = numeric_data.corr(method="pearson")
        results["pearson"] = pearson_corr

        # Spearman秩相关
        spearman_corr = numeric_data.corr(method="spearman")
        results["spearman"] = spearman_corr

        # 距离相关性 (Distance Correlation)
        cols = numeric_data.columns
        n_cols = len(cols)
        dist_corr_matrix = np.zeros((n_cols, n_cols))

        for i in range(n_cols):
            for j in range(i, n_cols):
                if i == j:
                    dist_corr_matrix[i, j] = 1.0
                else:
                    dcor_val = dcor.distance_correlation(
                        numeric_data.iloc[:, i], numeric_data.iloc[:, j]
                    )
                    dist_corr_matrix[i, j] = dcor_val
                    dist_corr_matrix[j, i] = dcor_val

        results["distance_correlation"] = pd.DataFrame(
            dist_corr_matrix, columns=cols, index=cols
        )

        return results

    def build_causal_graph(self, data, method="pc"):
        """
        构建因果图
        """
        if method == "pc":
            return self._pc_algorithm(data)
        elif method == "ges":
            return self._ges_algorithm(data)
        else:
            raise ValueError("Unsupported method")

    @staticmethod
    def _pc_algorithm(data):
        """
        PC算法实现因果发现
        """
        # 初始化完全无向图
        g = nx.Graph()
        variables = data.columns.tolist()
        g.add_nodes_from(variables)

        # 添加所有可能的边
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                g.add_edge(variables[i], variables[j])

        # 逐步移除边 based on conditional independence tests
        # 这里简化实现，实际应用中需要更复杂的条件独立性测试
        removed_edges = []
        for node in variables:
            neighbors = list(g.neighbors(node))
            if len(neighbors) > 1:
                # 简化的条件独立性判断
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        var1, var2 = neighbors[i], neighbors[j]
                        if g.has_edge(var1, var2):
                            corr = np.abs(data[var1].corr(data[var2]))
                            # 如果两个邻居之间的相关性很低，则移除边
                            if corr < 0.1:
                                g.remove_edge(var1, var2)
                                removed_edges.append((var1, var2))

        # 转换为有向图（简化版）
        dag = nx.DiGraph()
        dag.add_nodes_from(g.nodes())
        for edge in g.edges():
            # 简单的因果方向判断（实际中需要更复杂的方法）
            if data[edge[0]].corr(data[edge[1]]) > 0:
                dag.add_edge(edge[0], edge[1])
            else:
                dag.add_edge(edge[1], edge[0])

        return dag

    @staticmethod
    def causal_inference_dml(data, treatment, outcome, confounders):
        """
        使用Double Machine Learning进行因果推断
        """
        # 准备数据
        X = data[confounders]
        T = data[treatment]
        Y = data[outcome]

        # 初始化Causal Forest DML模型
        est = CausalForestDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            discrete_treatment=False,
            cv=3,
        )

        # 拟合模型
        est.fit(Y, T, X=X, W=None)

        # 估计因果效应
        causal_effect = est.effect(X)
        confidence_intervals = est.effect_interval(X, alpha=0.05)

        return {
            "causal_effect": causal_effect,
            "confidence_intervals": confidence_intervals,
            "model": est,
        }

    def deep_causal_inference(self, data, treatment, outcome, confounders):
        """
        使用深度学习方法进行因果推断
        """
        # 数据准备
        X = torch.tensor(data[confounders].values, dtype=torch.float32)
        T = torch.tensor(data[treatment].values,
                         dtype=torch.float32).unsqueeze(1)
        Y = torch.tensor(data[outcome].values,
                         dtype=torch.float32).unsqueeze(1)

        # 构建神经网络模型
        class CausalNet(nn.Module):
            def __init__(self, input_dim):
                super(CausalNet, self).__init__()
                self.shared_layers = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.ReLU(
                    ), nn.Linear(128, 64), nn.ReLU()
                )
                self.treatment_layer = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
                )
                self.outcome_layer = nn.Sequential(
                    nn.Linear(65, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),  # 64 + 1 (treatment)
                )

            def forward(self, x, t):
                shared = self.shared_layers(x)
                t_pred = self.treatment_layer(shared)
                y_input = torch.cat([shared, t], dim=1)
                y_pred = self.outcome_layer(y_input)
                return y_pred, t_pred

        model = CausalNet(len(confounders))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练模型
        dataset = TensorDataset(X, T, Y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(100):
            for x_batch, t_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred, t_pred = model(x_batch, t_batch)
                loss = criterion(y_pred, y_batch) + criterion(t_pred, t_batch)
                loss.backward()
                optimizer.step()

        # 估计因果效应
        with torch.no_grad():
            # 在治疗条件下
            y_t1, _ = model(X, torch.ones_like(T))
            # 在对照条件下
            y_t0, _ = model(X, torch.zeros_like(T))
            causal_effect = (y_t1 - y_t0).numpy().flatten()

        return {"causal_effect": causal_effect, "model": model}

    @staticmethod
    def gam_causal_analysis(data, treatment, outcome, confounders):
        """
        使用广义可加模型(GAM)进行因果分析
        """
        # 准备数据
        X = data[confounders + [treatment]]
        y = data[outcome]

        # 构建GAM模型
        # 为每个特征创建spline项
        formula = " + ".join([f"s({col})" for col in confounders + [treatment]])

        # 由于pygam的API限制，我们手动构建模型
        gam = pygam.GAM()
        gam.fit(X, y)

        # 提取治疗变量的影响
        treatment_idx = X.columns.get_loc(treatment)
        treatment_effect = gam.coef_[treatment_idx]

        return {"treatment_effect": treatment_effect, "model": gam}

    def detect_anomalies_in_causal_patterns(self, data, reference_patterns):
        """
        检测因果模式中的异常
        """
        current_patterns = self.detect_correlation_patterns(data)

        anomalies = {}
        for pattern_type, current_matrix in current_patterns.items():
            if pattern_type in reference_patterns:
                reference_matrix = reference_patterns[pattern_type]

                # 计算差异
                diff = np.abs(current_matrix.values - reference_matrix.values)
                anomaly_score = np.mean(diff)

                # 识别具体异常位置
                threshold = np.percentile(diff, 95)
                anomaly_positions = np.where(diff > threshold)

                anomalies[pattern_type] = {
                    "score": anomaly_score,
                    "positions": list(zip(anomaly_positions[0], anomaly_positions[1])),
                    "details": diff,
                }

        return anomalies

    def generate_causal_report(self, data):
        """
        生成完整的因果分析报告
        """
        # 数据预处理
        processed_data = self.preprocess_data(data)

        # 相关性模式检测
        correlation_patterns = self.detect_correlation_patterns(processed_data)

        # 构建因果图
        causal_graph = self.build_causal_graph(processed_data)

        # 存储结果
        self.results = {
            "correlation_patterns": correlation_patterns,
            "causal_graph": causal_graph,
            "processed_data": processed_data,
        }

        return self.results


# 使用示例
def main():
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000

    # 生成具有已知因果关系的数据
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 0.5, n_samples)
    X4 = 0.7 * X3 + np.random.normal(0, 0.5, n_samples)
    Y = 1.2 * X3 + 0.8 * X4 + np.random.normal(0, 0.5, n_samples)

    data = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4, "Y": Y})

    # 初始化因果模式检测器
    detector = IndustrialCausalPatternDetector()

    # 生成报告
    results = detector.generate_causal_report(data)

    # 执行特定因果推断
    if len(data.columns) >= 3:
        confounders = ["X1", "X2", "X3"]
        treatment = "X4"
        outcome = "Y"

        # 确保列存在于数据中
        available_cols = [col for col in confounders if col in data.columns]
        if (
            treatment in data.columns
            and outcome in data.columns
            and len(available_cols) > 0
        ):
            try:
                dml_result = detector.causal_inference_dml(
                    data, treatment, outcome, available_cols
                )
                print("DML因果效应估计完成")
            except Exception as e:
                print(f"DML分析出错: {e}")

    print("因果模式检测完成")
    print(f"检测到的因果图节点: {list(results['causal_graph'].nodes())}")
    print(f"检测到的因果图边: {list(results['causal_graph'].edges())}")


if __name__ == "__main__":
    main()
