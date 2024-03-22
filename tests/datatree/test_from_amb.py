from collections.abc import AsyncIterable, Iterable
from typing import TypeVar

import pytest

from supy.amb import Aborted, Choice, Done, Failed, NodeClosed, NodeOpened, Path
from supy.datatree import Node
from supy.datatree.amb import execution_tree

T = TypeVar("T")


async def as_async(items: Iterable[T]) -> AsyncIterable[T]:
    for item in items:
        yield item


async def collect_nodes(nodes: AsyncIterable[Node]) -> list[Node]:
    return [node async for node in nodes]


async def path_to(node: Node) -> Path:
    items = []

    while node.parent is not None:
        value = await node["value"]
        index: int = await node["index"]  # type: ignore[assignment]
        items.append(Choice(node.name, index, value))
        node = node.parent

    return Path(items)


async def assert_status_matches(node: Node, expected: str) -> None:
    if (status := await node["status"]) != expected:
        pytest.fail(f"Node status is '{status}', not '{expected}'")


async def assert_path_matches(node: Node, path: Path) -> None:
    __tracebackhide__ = True

    node_path = await path_to(node)
    assert node_path == path, "Node path does not match"


async def assert_success(node: Node, *, result: object, path: Path) -> None:
    __tracebackhide__ = True

    await assert_status_matches(node, "done")
    actual_result = await node["result"]
    assert actual_result == result, "Node 'result' does not match"
    await assert_path_matches(node, path)


async def assert_failed(node: Node, exc_type: type, *, match: str, path: Path) -> None:
    __tracebackhide__ = True

    await assert_status_matches(node, "failed")
    exc: Exception = await node["exception"]  # type: ignore[assignment]
    with pytest.raises(exc_type, match=match):
        raise exc
    await assert_path_matches(node, path)


async def assert_aborted(node: Node, *, reason: str | None, path: Path) -> None:
    __tracebackhide__ = True

    await assert_status_matches(node, "aborted")
    actual_reason = await node["reason"]
    assert actual_reason == reason, "Aborted node 'reason' does not match"
    await assert_path_matches(node, path)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_no_choice():
    events = [
        NodeOpened(Path()),
        Done(Path(), result=2),
        NodeClosed(Path()),
    ]

    tree = execution_tree(as_async(events))

    nodes = await collect_nodes(tree)

    assert nodes == [tree.root]
    await assert_success(tree.root, result=2, path=())


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_no_choice_failed():
    events = [
        NodeOpened(Path()),
        Failed(Path(), exc=ValueError("boom")),
        NodeClosed(Path()),
    ]

    tree = execution_tree(as_async(events))

    nodes = await collect_nodes(tree)

    assert nodes == [tree.root]
    await assert_failed(tree.root, ValueError, match="boom", path=())


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_no_choice_aborted():
    events = [
        NodeOpened(Path()),
        Aborted(Path(), reason="some reason"),
        NodeClosed(Path()),
    ]

    tree = execution_tree(as_async(events))

    nodes = await collect_nodes(tree)

    assert nodes == [tree.root]
    await assert_aborted(tree.root, reason="some reason", path=())


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_single_choice():
    events = [
        NodeOpened(Path()),
        #
        NodeOpened(Path([Choice("a", 0, 3.14)])),
        Done(Path([Choice("a", 0, 1)]), result=1),
        NodeClosed(Path([Choice("a", 0, 3.14)])),
        #
        NodeOpened(Path([Choice("a", 1, 2.71)])),
        Done(Path([Choice("a", 1, 2)]), result=2),
        NodeClosed(Path([Choice("a", 1, 2.71)])),
        #
        NodeClosed(Path()),
    ]

    tree = execution_tree(as_async(events))

    nodes = await collect_nodes(tree)

    assert len(nodes) == 3
    assert len(tree.root.children) == 2
    left, right = tree.root.children

    await assert_success(left, result=1, path=Path([Choice("a", 0, 3.14)]))
    await assert_success(right, result=2, path=Path([Choice("a", 1, 2.71)]))
