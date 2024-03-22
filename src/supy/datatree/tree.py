from __future__ import annotations

from collections import deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
)
from typing import Any, TypeAlias

from supy.asynciter import AsyncBuffer, AsyncDict
from supy.utils import spawn_task


class Node:
    """
    Node of the data tree.

    Apart from connectivity data (children, parent), each node has a name, a map of
    attributes, and a closed/open status. A node being closed means that the subtree
    rooted at it is structurally complete, i.e. no new nodes can be added to it nor any
    of its descendants.

    The attributes can be awaited asynchronously, suspending execution until they are
    set.

    Parameters
    ----------
    name : str
        Name of the node.
    parent : Node, optional
        Parent node.
    tree : DataTree
        `DataTree` object to which this node belongs.

    Attributes
    ----------
    name : str
        Name of the node.
    parent : Node, optional
        Parent node, ``None`` for root.
    children : iterable of Node
        Children nodes.
    descendants : iterator of Node
        Descendant nodes in post-order.
    ancestors : iterator of Node
        Ancestors of this node, listed in the order from direct parent to root.
    attr : AsyncDict
        Node attributes.
    closed : bool
        ``True`` if the `close` method has been called, ``False`` otherwise.
    """

    def __init__(self, name: str, parent: Node | None, tree: DataTree):
        self.name = name
        self.parent = parent
        self._children: list[Node] = []
        self._closed = False
        self._tree = tree
        self._attr = AsyncDict[str, object]()

    def add(self, name: str) -> Node:
        """
        Create a new child node.

        Parameters
        ----------
        name : str
            Name of the child node.

        Returns
        -------
        Node
            Newly created child node.

        Raises
        ------
        ValueError
            If the node has already been closed.
        """
        self._ensure_open()
        node = Node(name, self, self._tree)
        self._children.append(node)
        return node

    @property
    def children(self) -> Iterable[Node]:
        return tuple(self._children)

    @property
    def descendants(self) -> Iterator[Node]:
        for node in _post_order(self):
            if node is not self:
                yield node

    @property
    def ancestors(self) -> Iterator[Node]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    @property
    def attr(self) -> AsyncDict[str, object]:
        return self._attr

    def __repr__(self):
        return f"<Node({self.name})>"

    def close(self) -> None:
        """
        Close the node, marking its completion.

        New children cannot be added to a closed node.

        Raises
        ------
        ValueError
            If the node has already been closed.
        """
        self._ensure_open()
        self._ensure_children_closed()
        self._closed = True
        self._tree._node_completed(self)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("Node is closed")

    def _ensure_children_closed(self) -> None:
        for child in self._children:
            if not child.closed:
                raise ValueError(f"Child not closed: {child}")

    def __setitem__(self, name: str, value: object) -> None:
        """
        Set node attribute.

        All the tasks that have been asynchronously waiting for it are awakened.

        Parameters
        ----------
        name : str
            Attribute name.
        value : object
            Attribute value.
        """
        self._attr[name] = value

    def __getitem__(self, name: str) -> Awaitable[object]:
        """
        Get node attribute, waiting until it is available if needed.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        awaitable
            Coroutine that produces an attribute value.
        """
        return self._attr[name]


def _post_order(root: Node) -> Iterator[Node]:
    # Iterative post-order tree traversal
    stack = [deque([root])]

    while stack:
        entry = stack.pop()
        match entry:
            case deque() as nodes if nodes:
                first = nodes.popleft()
                stack.append(nodes)
                stack.append(first)
                stack.append(deque(first._children))
            case Node() as node:
                yield node


TreeDecorator: TypeAlias = Callable[[Node], Awaitable[None]]
"""
Function decorating `DataTree` nodes.
"""


class DataTree:
    """
    Data structure for incrementally storing tree-structured computation results.

    `DataTree` is comprised of `Node` objects, which have a name, (possibly empty) list
    of children, and a dictionary of attributes. There is no hard distinction between
    internal nodes and leaves, any node can hold arbitrary data in its attributes.

    The intended purpose of the `DataTree` is to facilitate simultaneous production and
    consumption of tree-structured data using Python's asynchronous execution framework
    (``async``/``await`` and the `asyncio` module). The tree can be built by appending
    new nodes. Once all the nodes in a given subtree have been added, its root can be
    marked as complete using `Node.close` method. While the tree is being built, one can
    simultaneously iterate over its completed nodes, as `DataTree` is ``AsyncIterable``.

    Closed nodes of the tree can be decorated by adding attributes. Since complex tree
    computations involving multiple attributes can result in intractable data
    dependencies and be difficult to correctly schedule manually (especially with nodes
    being added "on line"), `DataTree` offers a flexible decoration mechanism backed by
    the `asyncio` task scheduler in the form of `decorate` function.

    Attributes
    ----------
    root : Node
        Root of the data tree.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Attribute_grammar
    """

    def __init__(self):
        self._root = Node("", None, self)
        self._nodes = AsyncBuffer[Node]()

    def __aiter__(self) -> AsyncIterator[Node]:
        return self._nodes.__aiter__()

    def _node_completed(self, node: Node) -> None:
        self._nodes.add(node)
        if node is self._root:
            self._nodes.close()

    @property
    def root(self) -> Node:
        return self._root

    def decorate(self, fun: TreeDecorator | None = None):
        """
        Decorate tree by applying function to all the nodes.

        The function passed as an argument is called on every closed node of the tree,
        including those added and/or closed in the future. It should decorate nodes of
        the tree by setting their attribute values. Since the passed function is a
        coroutine, if other attributes are required to compute a new one for this node,
        one can ``await`` for them to be available. Any attribute of any node can be
        safely awaited (as long as dependency graph not contain cycles), since each
        invocation of the decorating coroutine on a node is scheduled as a separate
        asynchronous task, allowing arbitrarily complex data dependencies between node
        attributes.

        This method can also be used as a decorator.

        Parameters
        ----------
        fun : callable, optional
            Function used to add node attributes. If not present, `decorate` returns a
            function decorator.

        Returns
        -------
        function decorator
            If called without `fun`. The returned decorator, when applied to a function,
            uses it to decorate nodes of the tree, as if this function were passed to
            `decorate` directly.
        None
            If called with `fun`.

        Examples
        --------
        >>> import asyncio
        >>> async def fun():
        ...     tree = DataTree()
        ...     @tree.decorate()
        ...     async def add_attr(node):
        ...         node["val"] = 3
        >>> asyncio.run(fun())
        """
        if fun is not None:
            self._decorate(self, fun)
            return fun
        else:
            return self.decorate

    def _decorate(self, nodes: AsyncIterable[Node], fun: TreeDecorator) -> None:
        async def decorate_nodes():
            async for node in nodes:
                self._schedule(fun(node))

        self._schedule(decorate_nodes())

    def _schedule(self, coro: Coroutine[Any, Any, Any]) -> None:
        spawn_task(coro, "Decorating tree {self!r}")

    async def wait_until_complete(self) -> None:
        """
        Asynchronously wait until the tree has been completed.

        Tree is complete after all its nodes have been closed. After that point, no new
        nodes can appear, but new node attributes can be added.
        """
        # This is not a great way to do it, since the task will
        # be repeatedly awakened before it leaves this loop.
        async for _ in self:
            pass
