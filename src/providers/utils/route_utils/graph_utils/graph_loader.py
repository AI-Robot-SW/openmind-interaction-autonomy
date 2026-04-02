# graph_loader.py

import os
import json
import logging
from collections import OrderedDict

from .graph_model import Graph, GraphMeta, Node, Place, Transition




class GraphLoader:
    """
    manifest.json과 connections.json은 즉시 load하고,
    개별 그래프 파일은 요청 시점에 lazy load한다.
    로드된 그래프는 LRU(Least Recently Used) 캐시에 보관하며, 캐시가 가득 차면
    가장 오래 사용하지 않은 그래프를 evict한다.

    단일 스레드에서만 호출한다고 가정하므로 thread-safe하지 않다.
    멀티스레드 환경에서 사용할 경우 get_graph()에 Lock을 추가해야 한다.
    """

    def __init__(self, manifest_path: str, max_cache_size: int = 5) -> None:
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
        self._max_cache_size = max_cache_size

        self._graph_metas: dict[str, GraphMeta] = {}
        self._transitions: list[Transition] = []
        self._cache: OrderedDict[str, Graph] = OrderedDict()

        self._load_manifest(manifest_path)

    # ------------------------------------------------------------------
    # initial load (manifest + connections load)
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: str) -> None:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        for g in data["graphs"]:
            meta = GraphMeta(
                graph_id=g["graph_id"],
                path=g["path"],
                graph_type=g["graph_type"],
                site=g["site"],
                building=g.get("building"),
                floor=g.get("floor"),
            )
            self._graph_metas[meta.graph_id] = meta

        connections_path = os.path.join(self._base_dir, data["connections_path"])
        with open(connections_path, encoding="utf-8") as f:
            conn_data = json.load(f)

        for c in conn_data["connections"]:
            self._transitions.append(
                Transition(
                    from_graph_id=c["from_graph_id"],
                    from_node_id=c["from_node_id"],
                    to_graph_id=c["to_graph_id"],
                    to_node_id=c["to_node_id"],
                    transition_type=c["transition_type"],
                )
            )

        logging.info(
            "GraphLoader: %d graph(s) registered, %d transition(s) loaded",
            len(self._graph_metas),
            len(self._transitions),
        )

    # ------------------------------------------------------------------
    # lazy load + LRU cache (graphs load)
    # ------------------------------------------------------------------

    def get_graph(self, graph_id: str) -> Graph:
        """그래프를 반환한다. 캐시에 없으면 디스크에서 로드한다."""
        if graph_id in self._cache:
            self._cache.move_to_end(graph_id)
            return self._cache[graph_id]

        graph = self._load_graph(graph_id)
        self._cache[graph_id] = graph
        self._cache.move_to_end(graph_id)

        if len(self._cache) > self._max_cache_size:
            evicted, _ = self._cache.popitem(last=False)
            logging.debug("GraphLoader: evicted '%s' from cache", evicted)

        logging.debug(
            "GraphLoader: loaded '%s' (%d nodes), cache size=%d/%d",
            graph_id,
            len(graph.nodes),
            len(self._cache),
            self._max_cache_size,
        )
        return graph

    def _load_graph(self, graph_id: str) -> Graph:
        if graph_id not in self._graph_metas:
            raise KeyError(f"Unknown graph_id: '{graph_id}'")

        meta = self._graph_metas[graph_id]
        path = os.path.join(self._base_dir, meta.path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found: '{path}' (graph_id='{graph_id}')")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        nodes: dict[int, Node] = {}
        for n in data["nodes"]:
            node = Node(
                id=n["id"],
                lat=n.get("lat"),
                lon=n.get("lon"),
                x=n.get("x"),
                y=n.get("y"),
                transitions=tuple(n.get("transitions", [])),
            )
            nodes[node.id] = node

        places: dict[str, Place] = {}
        for p in data.get("places", []):
            places[p["id"]] = Place(id=p["id"], node_id=p["node_id"])

        return Graph(
            graph_id=data["graph_id"],
            coordinate_frame=data["coordinate_frame"],
            nodes=nodes,
            places=places,
        )

    # ------------------------------------------------------------------
    # Read only property
    # ------------------------------------------------------------------

    @property
    def transitions(self) -> list[Transition]:
        return self._transitions

    @property
    def graph_metas(self) -> dict[str, GraphMeta]:
        return self._graph_metas

    @property
    def cache_size(self) -> int:
        return len(self._cache)

