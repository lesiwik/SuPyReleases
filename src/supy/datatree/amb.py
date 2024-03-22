"""Building DataTree objects form non-deterministic computations."""
from collections.abc import AsyncIterable

from supy.amb import Aborted, Done, Event, Failed, NodeClosed, NodeOpened, Path
from supy.datatree.tree import DataTree, Node
from supy.utils import spawn_task


def _indices(path: Path) -> tuple[int, ...]:
    return tuple(choice.idx for choice in path)


class _TreeBuilder:
    def __init__(self, tree: DataTree):
        self.tree = tree
        # node is identified by the choice index sequence
        self.nodes: dict[tuple[int, ...], Node] = {(): tree.root}

    def handle(self, event: Event):
        path = _indices(event.path)

        match event:
            case NodeOpened() if path:  # no need to create root
                parent_path = path[:-1]
                final_choice = event.path[-1]
                parent = self.nodes[parent_path]
                node = parent.add(final_choice.name)
                node["value"] = final_choice.value
                node["index"] = final_choice.idx
                self.nodes[path] = node

            case NodeClosed():
                node = self.nodes[path]
                node.close()

            case Done():
                node = self.nodes[path]
                node.attr["status"] = "done"
                node.attr["result"] = event.result

            case Failed():
                node = self.nodes[path]
                node.attr["status"] = "failed"
                node.attr["exception"] = event.exc

            case Aborted():
                node = self.nodes[path]
                node.attr["status"] = "aborted"
                node.attr["reason"] = event.reason


def execution_tree(events: AsyncIterable[Event]) -> DataTree:
    """
    Build a data tree from non-deterministic execution events.

    Nodes of the created data tree correspond 1-1 to the nodes of the decision tree as
    described in `supy.amb.nondeterministic`. Name of the node is the name of the
    variable whose value is chosen in the given node. Each node has the following
    attributes:

    - 'value': Value chosen at this branching point.
    - 'index': Index of this choice in the list of options.

    In addition, depending on how the execution path ended, each leaf has additional
    attributes:

    - path was successfully completed:
        - 'status': ``"done"``
        - 'result': Value returned by the execution path.
    - path raised an exception:
        - 'status': ``"failed"``
        - 'exception': Exception object that terminated the execution.
    - path was aborted:
        - 'status': ``"aborted"``
        - 'reason': Reason for termination, may be ``None``.

    Parameters
    ----------
    events : async iterable
        Stream of events occurring during a non-deterministic function execution.

    Returns
    -------
    DataTree
        Data tree representing the results of all the execution paths.
    """
    tree = DataTree()
    builder = _TreeBuilder(tree)

    async def build():
        async for event in events:
            builder.handle(event)

    spawn_task(build(), "Building execution tree")
    return tree
