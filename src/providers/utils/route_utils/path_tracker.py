# path_tracker.py

import logging

from dataclasses import dataclass
from typing import Optional

from .graph_loader import GraphLoader
from .graph_model import Graph, Node, NodeRef

from ..geo_utils import haversine_dist_m, euclidean_dist_m


@dataclass(frozen=True)
class TrackerNode:
    """PathTracker가 현재/이전 노드 정보를 반환할 때 사용하는 컨테이너."""
    ref: NodeRef
    node: Node
    graph: Graph


_REACH_TOL_WGS84_M: float = 5.0   # RTK 정확도 기준 실외 도달 판정
_REACH_TOL_UWB_M: float = 3.0     # UWB 정확도 기준 실내 도달 판정


class PathTracker:
    """
    PathFinder가 반환한 NodeRef 경로를 순서대로 추적한다.

    update()로 현재 위치를 전달하면 목표 노드 도달 여부를 판단하고
    자동으로 다음 노드로 전진한다. 한 번의 호출에 여러 노드를 연속 통과할 수 있다.

    coordinate_frame 별 도달 허용 반경:
        wgs84  : _REACH_TOL_WGS84_M (RTK 정확도 기준)
        uwb_*  : _REACH_TOL_UWB_M   (UWB 정확도 기준)
    """

    def __init__(self, path: list[NodeRef], loader: GraphLoader) -> None:
        self._path = list(path)
        self._loader = loader
        self._idx: int = 0
        self._current: Optional[TrackerNode] = None
        self._prev: Optional[TrackerNode] = None
        self._rebuild_cache()
        logging.info("PathTracker: %d nodes", len(self._path))

    def update(self, lat: Optional[float], lon: Optional[float], x: Optional[float] = None, y: Optional[float] = None) -> None:
        """
        현재 위치를 기준으로 목표 노드 도달 여부를 확인하고, 도달 시 다음 노드로 전진한다.

        outdoor 노드 (wgs84) : (lat, lon) 사용 — None 이면 해당 노드 판정 스킵
        indoor 노드  (uwb_*) : (x, y) 사용    — None 이면 해당 노드 판정 스킵
        """
        while not self.is_done:
            current = self._current
            if current is None:
                break

            node = current.node
            if current.graph.coordinate_frame == "wgs84":
                if lat is None or lon is None or node.lat is None or node.lon is None:
                    break
                dist = haversine_dist_m(lat, lon, node.lat, node.lon)
                reach_tol = _REACH_TOL_WGS84_M
            else:
                if x is None or y is None or node.x is None or node.y is None:
                    break
                dist = euclidean_dist_m(x, y, node.x, node.y)
                reach_tol = _REACH_TOL_UWB_M

            if dist < reach_tol:
                logging.info(
                    "PathTracker: reached %s (dist=%.2fm, tol=%.2fm) [%d/%d]",
                    current.ref, dist, reach_tol, self._idx + 1, len(self._path),
                )
                self._idx += 1
                self._rebuild_cache()
            else:
                break
    
    @property
    def current_node(self) -> Optional[TrackerNode]:
        """현재 목표 노드. 경로 완료 시 None."""
        return self._current

    @property
    def prev_node(self) -> Optional[TrackerNode]:
        """직전에 통과한 노드. 첫 번째 노드 도달 전에는 None."""
        return self._prev

    @property
    def is_done(self) -> bool:
        """경로의 모든 노드를 통과했으면 True."""
        return self._idx >= len(self._path)

    @property
    def path(self) -> list[NodeRef]:
        """경로 전체 NodeRef 리스트."""
        return self._path

    @property
    def progress(self) -> tuple[int, int]:
        """(현재 인덱스, 전체 노드 수) 반환."""
        return self._idx, len(self._path)

    def _rebuild_cache(self) -> None:
        self._current = self._resolve(self._idx)
        self._prev = self._resolve(self._idx - 1) if self._idx > 0 else None

    def _resolve(self, idx: int) -> Optional[TrackerNode]:
        if idx < 0 or idx >= len(self._path):
            return None
        ref = self._path[idx]
        graph = self._loader.get_graph(ref[0])
        node = graph.nodes.get(ref[1])
        if node is None:
            return None
        return TrackerNode(ref=ref, node=node, graph=graph)
