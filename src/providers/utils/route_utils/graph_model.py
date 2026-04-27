# graph_model.py

from dataclasses import dataclass, field
from typing import Optional


# (graph_id, node_id) 쌍으로 그래프 전체에서 노드를 식별
NodeRef = tuple[str, int]


@dataclass(frozen=True)
class Node:
    id: int
    # outdoor (wgs84)
    lat: Optional[float]
    lon: Optional[float]
    # indoor (uwb)
    x: Optional[float]
    y: Optional[float]
    transitions: tuple[int, ...]


@dataclass(frozen=True)
class Place:
    id: str
    node_id: int


@dataclass
class Graph:
    graph_id: str
    coordinate_frame: str
    nodes: dict[int, Node] = field(default_factory=dict)
    places: dict[str, Place] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    from_graph_id: str
    from_node_id: int
    to_graph_id: str
    to_node_id: int
    transition_type: str


@dataclass(frozen=True)
class GraphMeta:
    graph_id: str
    path: str
    graph_type: str
    site: str
    building: Optional[str]
    floor: Optional[int]
