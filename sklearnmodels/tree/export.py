from pathlib import Path
from .tree import Condition
from . import Tree
import pygraphviz


def dot_template(body: str, title: str):
    title_dot = f"""0 [label="{title}", shape=plaintext];
0:s -> 1:n [style=invis];
"""
    return (
        """digraph Tree {
splines=false;
graph [pad=".25", ranksep="0.5", nodesep="1"];
node [shape=rect, style="filled", color="black", fontname="helvetica",fillcolor="white"] ;
edge [fontname="helvetica"] ;
"""
        + title_dot
        + body
        + "\n}"
    )


class TreeInfo:
    def __init__(
        self, id: int, parent_id: int, tree: Tree, condition: Condition, height: int
    ):
        self.id = id
        self.parent_id = parent_id
        self.tree = tree
        self.condition = condition
        self.height = height


def make_color(height, max_height):
    hue = height / (max_height)
    max_hue = 0.6
    hue = hue * max_hue
    value = 0.9 if height % 2 else 0.8
    hsv = f"{hue:.3f} 0.7 {value:.3f}"
    return hsv


def make_label(info: TreeInfo, class_names: list[str]):
    prediction = ", ".join([f"{p:.2f}" for p in info.tree.prediction])
    prediction = f"p: ({prediction})"
    params = f"n={info.tree.samples}"
    class_info = ""
    if class_names is not None:
        class_name = class_names[info.tree.prediction.argmax()]
        class_info = f"<b> Class={class_name} </b> <br/>"
    column = "" if info.tree.leaf else f"<br/><b>{info.tree.column}</b>"
    error = f"error: {info.tree.error:.3f}, n={info.tree.samples}"
    label = f"<{class_info} {error} <br/> {prediction} {column}>"
    return label


def make_node(info: TreeInfo, max_height: int, class_names: list[str]):
    color = make_color(info.height, max_height)
    shape = "oval" if info.tree.leaf else "rect"
    label = make_label(info, class_names)
    node = f'{info.id} [label={label}, fillcolor="{color}", shape="{shape}"];\n'
    return node


def make_edge(info: TreeInfo):
    condition = info.condition.short_description()
    return f'{info.parent_id}:s -> {info.id}:n [label="{condition}"] ;\n'


def export_dot(tree: Tree, class_names: list[str], title=""):
    nodes: list[TreeInfo] = []
    global id
    id = 0

    def collect(tree: Tree, parent: int, height: int, condition=None):
        global id
        id += 1
        info = TreeInfo(id, parent, tree, condition, height)
        nodes.append(info)
        for c, t in tree.branches.items():
            collect(t, info.id, height + 1, c)

    collect(tree, 0, 0, 1)
    max_height = max([i.height for i in nodes])
    max_height = max(1, max_height)
    body = ""
    for info in nodes:
        body += make_node(info, max_height, class_names)
        if info.parent_id > 0:
            body += make_edge(info)

    return dot_template(body, title)


def export_dot_file(
    tree: Tree, filepath: Path, title="", class_names: list[str] = None
):
    dot = export_dot(tree, class_names, title=title)
    with open(filepath, "w") as f:
        f.write(dot)


def export_image(
    tree: Tree, filepath: Path, title="", class_names: list[str] = None, prog="dot"
):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(tree.prediction))]
    dot = export_dot(tree, class_names, title=title)
    graph = pygraphviz.AGraph(string=dot)
    graph.draw(path=str(filepath), prog=prog)
