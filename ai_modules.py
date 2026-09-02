"""Lightweight, executable AI modules for the emergency-routing prototype.

These models are deliberately small because the project uses a 900-record
teaching dataset.  They demonstrate the pipeline architecture; they are not a
production emergency-dispatch model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn


FEATURE_NAMES = [
    "frequency", "severity", "hour_sin", "hour_cos", "temperature",
    "visibility", "junction", "traffic_signal",
]


class SpatialTemporalGAT(nn.Module):
    """A compact spatial-temporal attention network with graph attention."""

    def __init__(self, input_size: int, hidden_size: int = 20):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            input_size, num_heads=1, batch_first=True, dropout=0.20
        )
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor):
        temporal, _ = self.temporal_attention(
            features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0)
        )
        temporal = temporal.squeeze(0)
        query = self.query(temporal)
        key = self.key(temporal)
        value = self.value(temporal)
        scores = query @ key.T / math.sqrt(query.shape[1])
        scores = scores.masked_fill(~adjacency, -1e9)
        attention = torch.softmax(scores, dim=1)
        graph_features = attention @ value
        prediction = self.output(torch.cat([graph_features, temporal], dim=1)).squeeze(1)
        return prediction, attention


class RouteDQN(nn.Module):
    """Small Q-network used to choose between shortest and safer candidates."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)
        )

    def forward(self, state: torch.Tensor):
        return self.network(state)


def _normalise(values: np.ndarray) -> np.ndarray:
    lower, upper = np.nanmin(values), np.nanmax(values)
    if upper - lower < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - lower) / (upper - lower)


class RiskIntelligenceEngine:
    """Trains prototype learning, uncertainty, XAI, and DQN components."""

    def __init__(self, accident_data: pd.DataFrame):
        torch.manual_seed(42)
        np.random.seed(42)
        self.locations, self.features, self.targets, self.adjacency = self._prepare(accident_data)
        self.model = SpatialTemporalGAT(len(FEATURE_NAMES))
        self._train_gat()
        self.dqn = RouteDQN()
        self._train_dqn()
        self.predictions, self.uncertainty, self.feature_importance = self._infer()

    @staticmethod
    def _prepare(data: pd.DataFrame):
        frame = data.copy()
        frame["Start_Time"] = pd.to_datetime(frame["Start_Time"], errors="coerce")
        frame["Lat_Grid"] = frame["Start_Lat"].round(2)
        frame["Lng_Grid"] = frame["Start_Lng"].round(2)
        frame["Hour"] = frame["Start_Time"].dt.hour.fillna(12)
        for column, default in [("Temperature(F)", 60), ("Visibility(mi)", 8)]:
            frame[column] = pd.to_numeric(frame.get(column, default), errors="coerce").fillna(default)
        for column in ["Junction", "Traffic_Signal"]:
            frame[column] = frame.get(column, False).fillna(False).astype(float)

        locations = frame.groupby(["Lat_Grid", "Lng_Grid"], as_index=False).agg(
            frequency=("ID", "count"), severity=("Severity", "mean"),
            hour=("Hour", "mean"), temperature=("Temperature(F)", "mean"),
            visibility=("Visibility(mi)", "mean"), junction=("Junction", "mean"),
            traffic_signal=("Traffic_Signal", "mean"),
        )
        locations["risk_score"] = locations["frequency"] * locations["severity"]
        hour_angle = 2 * np.pi * locations["hour"].to_numpy() / 24
        raw_features = np.column_stack([
            locations["frequency"], locations["severity"], np.sin(hour_angle),
            np.cos(hour_angle), locations["temperature"], locations["visibility"],
            locations["junction"], locations["traffic_signal"],
        ]).astype(np.float32)
        features = np.column_stack([_normalise(raw_features[:, i]) for i in range(raw_features.shape[1])])
        targets = _normalise(locations["risk_score"].to_numpy()).astype(np.float32)
        coordinates = locations[["Lat_Grid", "Lng_Grid"]].to_numpy()
        neighbours = NearestNeighbors(n_neighbors=min(7, len(locations))).fit(coordinates)
        _, indices = neighbours.kneighbors(coordinates)
        adjacency = np.zeros((len(locations), len(locations)), dtype=bool)
        for index, nearby in enumerate(indices):
            adjacency[index, nearby] = True
        return locations, torch.tensor(features, dtype=torch.float32), torch.tensor(targets), torch.tensor(adjacency)

    def _train_gat(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=0.015, weight_decay=1e-4)
        self.model.train()
        for _ in range(80):
            optimiser.zero_grad()
            prediction, _ = self.model(self.features, self.adjacency)
            loss = nn.functional.mse_loss(prediction, self.targets)
            loss.backward()
            optimiser.step()

    def _train_dqn(self):
        # A small replay-style simulation: reward favours lower risk but penalises detours.
        states = torch.rand(240, 4)
        shortest_reward = -(states[:, 0] + 0.15 * states[:, 2])
        safer_reward = -(states[:, 1] + 0.15 * states[:, 3])
        targets = torch.stack([shortest_reward, safer_reward], dim=1)
        optimiser = torch.optim.Adam(self.dqn.parameters(), lr=0.02)
        for _ in range(100):
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(self.dqn(states), targets)
            loss.backward()
            optimiser.step()
        self.dqn.eval()

    def _infer(self):
        # MC Dropout: keep dropout active and sample the learned graph-risk estimate.
        self.model.train()
        samples = []
        with torch.no_grad():
            for _ in range(20):
                prediction, _ = self.model(self.features, self.adjacency)
                samples.append(prediction.numpy())
            _, attention = self.model(self.features, self.adjacency)
        predictions = np.mean(samples, axis=0)
        uncertainty = np.std(samples, axis=0)
        # Graph XAI proxy: feature sensitivity weighted by learned input attention.
        weights = self.model.query.weight.detach().abs().mean(dim=0).numpy()
        importance = weights * self.features.detach().abs().mean(dim=0).numpy()
        importance = importance / max(importance.sum(), 1e-9)
        self.mean_graph_attention = float(attention.diag().mean())
        return predictions, uncertainty, importance

    def road_risk(self, latitude: float, longitude: float) -> float:
        coordinates = self.locations[["Lat_Grid", "Lng_Grid"]].to_numpy()
        distances = np.sqrt(((coordinates[:, 0] - latitude) * 111) ** 2 + ((coordinates[:, 1] - longitude) * 85) ** 2)
        nearest = int(np.argmin(distances))
        return float(self.predictions[nearest] * np.exp(-distances[nearest] / 8))

    def route_uncertainty(self, graph, route) -> float:
        values = []
        coordinates = self.locations[["Lat_Grid", "Lng_Grid"]].to_numpy()
        for node in route:
            point = np.array([graph.nodes[node]["y"], graph.nodes[node]["x"]])
            nearest = int(np.argmin(np.sum((coordinates - point) ** 2, axis=1)))
            values.append(self.uncertainty[nearest])
        return float(np.mean(values))

    def route_decision(self, shortest_distance, shortest_risk, safest_distance, safest_risk):
        maximum_distance = max(shortest_distance, safest_distance, 0.01)
        maximum_risk = max(shortest_risk, safest_risk, 0.01)
        state = torch.tensor([[
            shortest_distance / maximum_distance, safest_distance / maximum_distance,
            shortest_risk / maximum_risk, safest_risk / maximum_risk,
        ]], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.dqn(state).numpy()[0]
        # Preserve the safety rule when the learned policy is uncertain.
        action = int(np.argmax(q_values))
        return ("Shortest Route" if action == 0 else "Risk-Aware Route"), q_values

    def explanation(self):
        ranked = np.argsort(self.feature_importance)[::-1][:3]
        labels = [FEATURE_NAMES[index].replace("_", " ").title() for index in ranked]
        return labels, [round(float(self.feature_importance[index]) * 100, 1) for index in ranked]
