# path_finder.py

import heapq
import math
import os
from typing import Optional

from .graph_loader import GraphLoader
from .graph_model import Graph, Node, NodeRef

from ..geo_utils import euclidean_dist_m, haversine_dist_m

# 그래프 경계 횡단 비용 (meters 단위 거리와 동일한 스케일)
# 실내/실외 전환, 계단 등에 부여할 패널티를 타입별로 정의
_TRANSITION_COSTS: dict[str, float] = {
    "enter_building": 0.0,
    "exit_building": 0.0,
    "stairs_up": 5.0,
    "stairs_down": 5.0,
}
_DEFAULT_TRANSITION_COST = 0.0


def _edge_weight(graph: Graph, a: Node, b: Node) -> float:
    if graph.coordinate_frame == "wgs84":
        assert a.lat is not None and a.lon is not None
        assert b.lat is not None and b.lon is not None
        return haversine_dist_m(a.lat, a.lon, b.lat, b.lon)
    else:
        assert a.x is not None and a.y is not None
        assert b.x is not None and b.y is not None
        return euclidean_dist_m(a.x, a.y, b.x, b.y)


def _build_transition_index(
    loader: GraphLoader,
) -> dict[NodeRef, list[tuple[str, int, float]]]:
    """
    connections.json을 (from_graph_id, from_node_id) → [(to_graph_id, to_node_id, cost)]
    인덱스로 변환한다.
    """
    index: dict[NodeRef, list[tuple[str, int, float]]] = {}
    for t in loader.transitions:
        key: NodeRef = (t.from_graph_id, t.from_node_id)
        cost = _TRANSITION_COSTS.get(t.transition_type, _DEFAULT_TRANSITION_COST)
        index.setdefault(key, []).append((t.to_graph_id, t.to_node_id, cost))
    return index


def _reconstruct(prev: dict[NodeRef, Optional[NodeRef]], end: NodeRef) -> list[NodeRef]:
    """
    Dijkstra 탐색 중 기록한 prev 딕셔너리를 역추적하여 경로를 복원한다.
    end에서 출발점(prev가 None인 노드)까지 거슬러 올라간 뒤 reverse로 순서를 바로잡는다.
    """
    path: list[NodeRef] = []
    cur: Optional[NodeRef] = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


class PathFinder:
    """
    graphs를 넘나드는 dijkstra path finder.

    - 같은 그래프 내 이동: node.transitions edge, 좌표계에 맞는 거리 penality weight
    - 그래프 간 이동: connections.json의 전환 비용(기본 0m, 계단 5m 패널티)
    - 그래프 파일은 경로 탐색 중 필요할 때만 lazy load된다.
    """

    def __init__(self) -> None:
        manifest_path = os.path.join(os.path.dirname(__file__), "graphs", "manifest.json")
        self._loader = GraphLoader(manifest_path, max_cache_size=5)
        self._conn_index = _build_transition_index(self._loader)

    def find_path(self, start: NodeRef, end: NodeRef) -> Optional[list[list[NodeRef]]]:
        """
        start에서 end까지 최단 경로를 coordinate_frame 별로 분리해 반환한다.

        Args:
            start: (graph_id, node_id) 출발 노드
            end:   (graph_id, node_id) 도착 노드

        Returns:
            coordinate_frame이 같은 NodeRef 묶음의 리스트 (segment 단위).
            예: [[outdoor_1, outdoor_2], [indoor_1, indoor_2]]
            경로 없으면 None
        """
        if start == end:
            return [[start]]

        dist: dict[NodeRef, float] = {start: 0.0}
        prev: dict[NodeRef, Optional[NodeRef]] = {start: None}
        # (누적 거리, graph_id, node_id)
        priority_queue: list[tuple[float, str, int]] = [(0.0, start[0], start[1])]

        while priority_queue:
            d, graph_id, node_id = heapq.heappop(priority_queue)
            cur: NodeRef = (graph_id, node_id)

            if d > dist.get(cur, math.inf):
                continue

            if cur == end:
                return _split_by_frame(_reconstruct(prev, end), self._loader)

            graph = self._loader.get_graph(graph_id)
            node = graph.nodes.get(node_id)
            if node is None:
                continue

            # ── 같은 그래프 내 이웃 ──────────────────────────────────
            for neighbor_id in node.transitions:
                neighbor = graph.nodes.get(neighbor_id)
                if neighbor is None:
                    continue
                new_dist = d + _edge_weight(graph, node, neighbor)
                neighbor_ref: NodeRef = (graph_id, neighbor_id)
                if new_dist < dist.get(neighbor_ref, math.inf):
                    dist[neighbor_ref] = new_dist
                    prev[neighbor_ref] = cur
                    heapq.heappush(priority_queue, (new_dist, graph_id, neighbor_id))

            # ── 그래프 간 이웃 (lazy load 발생 가능) ────────────────
            for to_gid, to_nid, cost in self._conn_index.get(cur, []):
                target_ref: NodeRef = (to_gid, to_nid)
                new_dist = d + cost
                if new_dist < dist.get(target_ref, math.inf):
                    dist[target_ref] = new_dist
                    prev[target_ref] = cur
                    heapq.heappush(priority_queue, (new_dist, to_gid, to_nid))

        return None


def _split_by_frame(path: list[NodeRef], loader: GraphLoader) -> list[list[NodeRef]]:
    """
    NodeRef 리스트를 coordinate_frame이 바뀌는 경계에서 분리한다.

    예: [outdoor_1, outdoor_2, indoor_1, indoor_2]
        → [[outdoor_1, outdoor_2], [indoor_1, indoor_2]]
    """
    if not path:
        return []

    segments: list[list[NodeRef]] = []
    current_segment: list[NodeRef] = [path[0]]
    current_frame = loader.get_graph(path[0][0]).coordinate_frame

    for ref in path[1:]:
        frame = loader.get_graph(ref[0]).coordinate_frame
        if frame != current_frame:
            segments.append(current_segment)
            current_segment = [ref]
            current_frame = frame
        else:
            current_segment.append(ref)

    segments.append(current_segment)
    return segments

    def nearest_wgs_node(self, lat: float, lon: float, graph_id: str = "kist_outdoor") -> NodeRef:
        """wgs84 그래프에서 (lat, lon)에 가장 가까운 노드를 반환한다."""
        graph = self._loader.get_graph(graph_id)
        best = min(
            graph.nodes.values(),
            key=lambda n: haversine_dist_m(lat, lon, n.lat, n.lon),
        )
        return (graph_id, best.id)

    def nearest_uwb_node(self, x: float, y: float, graph_id: str) -> NodeRef:
        """UWB 그래프에서 (x, y)에 가장 가까운 노드를 반환한다."""
        graph = self._loader.get_graph(graph_id)
        best = min(
            graph.nodes.values(),
            key=lambda n: euclidean_dist_m(x, y, n.x, n.y),
        )
        return (graph_id, best.id)

    def resolve_place_to_node(self, place_id: str) -> NodeRef:
        """등록된 모든 그래프에서 place_id를 검색하여 NodeRef를 반환한다."""
        for graph_id in self._loader.graph_metas:
            graph = self._loader.get_graph(graph_id)
            if place_id in graph.places:
                place = graph.places[place_id]
                return (graph_id, place.node_id)
        raise KeyError(f"Place '{place_id}' not found in any registered graph")