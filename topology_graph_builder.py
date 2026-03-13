"""Topology graph construction for weighted local connectivity."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Tuple

import networkx as nx


class TopologyGraphBuilder:
    """Build weighted graph from inlier counts and extract optimal-root SPST."""

    def __init__(self, inlier_counts: Dict[Tuple[Hashable, Hashable], float]):
        self.inlier_counts = inlier_counts
        self.graph = nx.Graph()

    def build_graph(self) -> nx.Graph:
        sum_inliers = defaultdict(float)
        for (i, j), nij in self.inlier_counts.items():
            sum_inliers[i] += nij
            sum_inliers[j] += nij

        for (i, j), nij in self.inlier_counts.items():
            w_ij = 0.5 * (nij / (sum_inliers[i] + 1e-12) + nij / (sum_inliers[j] + 1e-12))
            distance = 1.0 / (w_ij + 1e-6)
            self.graph.add_edge(i, j, inliers=nij, weight=w_ij, distance=distance)
        return self.graph

    def find_optimal_reference_node(self):
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Graph is empty. Build graph first.")

        all_pairs = dict(nx.floyd_warshall(self.graph, weight="distance"))
        total_distance = {
            node: sum(dist for target, dist in dists.items() if target != node)
            for node, dists in all_pairs.items()
        }
        return min(total_distance, key=total_distance.get)

    def extract_spanning_tree(self, root) -> nx.DiGraph:
        if root not in self.graph:
            raise ValueError(f"Root {root} not in graph")

        paths = nx.single_source_dijkstra_path(self.graph, source=root, weight="distance")
        tree = nx.DiGraph()
        tree.add_node(root)

        for node, path in paths.items():
            if len(path) < 2:
                continue
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.graph.get_edge_data(u, v)
                tree.add_edge(u, v, **edge_data)
        return tree
