import asyncio
import unittest
from collections.abc import AsyncIterable

import pytest
import pytest_asyncio

from supy.datatree import DataTree, Node


@pytest.fixture()
def tree():
    """An empty DataTree."""
    return DataTree()


@pytest_asyncio.fixture()
async def complete_tree(tree):
    root = tree.root
    node_1 = root.add("problem-1")
    node_2 = root.add("problem-2")

    node_1_1 = node_1.add("method-1")
    node_1_2 = node_1.add("method-2")
    node_2_1 = node_2.add("method-1")
    node_2_2 = node_2.add("method-2")

    node_1_1.close()
    node_1_2.close()
    node_2_1.close()
    node_2_2.close()

    node_2.close()
    node_1.close()
    root.close()

    return tree


async def collect_nodes(nodes: AsyncIterable[Node]) -> list[Node]:
    return [node async for node in nodes]


def assert_list_elements_equal(list1, list2) -> None:
    case = unittest.TestCase()
    case.assertCountEqual(list1, list2)  # noqa: PT009


def assert_post_order(nodes: list[Node]) -> None:
    __tracebackhide__ = True

    for idx, node in enumerate(nodes):
        for child in node.children:
            child_idx = nodes.index(child)
            if child_idx > idx:
                msg = (
                    f"Post-order violated: child node\n\n"
                    f"    {child}\n\n"
                    f"at index {child_idx} occurs after its parent\n\n"
                    f"    {node}\n\n at index {idx}"
                )
                pytest.fail(msg)


@pytest.mark.asyncio()
async def test_empty_tree_has_only_root(tree):
    nodes = collect_nodes(tree)
    tree.root.close()
    assert await nodes == [tree.root]


@pytest.mark.asyncio()
async def test_can_add_top_level_node(tree):
    root = tree.root
    node = root.add("problem")

    node.close()
    root.close()

    assert node.parent is root

    nodes = await collect_nodes(tree)
    assert_list_elements_equal(nodes, [node, root])


@pytest.mark.asyncio()
async def test_can_add_subnodes(tree):
    root = tree.root
    node = root.add("problem")
    child = node.add("method")
    child.close()
    node.close()
    root.close()

    nodes = await collect_nodes(tree)
    assert_list_elements_equal(nodes, [node, root, child])


@pytest.mark.asyncio()
async def test_iteration_is_postorder(complete_tree):
    nodes = await collect_nodes(complete_tree)
    assert_post_order(nodes)


@pytest.mark.asyncio()
async def test_cannot_add_child_to_closed_node(tree):
    root = tree.root
    node = root.add("problem")
    node.close()

    with pytest.raises(ValueError, match="closed"):
        node.add("method")


@pytest.mark.asyncio()
async def test_cannot_close_node_twice(tree):
    root = tree.root
    node = root.add("problem")
    node.close()

    with pytest.raises(ValueError, match="closed"):
        node.close()


@pytest.mark.asyncio()
async def test_cannot_close_node_with_open_child(tree):
    root = tree.root
    root.add("problem")

    with pytest.raises(ValueError, match="not closed"):
        root.close()


@pytest.mark.asyncio()
async def test_can_set_attributes(tree):
    root = tree.root
    root["error"] = 3
    root["foo"] = "bar"
    assert dict(root.attr.items()) == {"error": 3, "foo": "bar"}


@pytest.mark.asyncio()
async def test_can_read_attribute(tree):
    root = tree.root
    root.attr.update(error=3, foo="bar")
    assert await root["error"] == 3
    assert await root["foo"] == "bar"


@pytest.mark.asyncio()
async def test_can_read_attributes_set_later(tree):
    root = tree.root
    event = asyncio.Event()

    async def read_foo():
        event.set()
        return await root["foo"]

    val = asyncio.create_task(read_foo())

    await event.wait()
    root["foo"] = "bar"

    assert await val == "bar"


@pytest_asyncio.fixture()
async def error_tree(tree):
    root = tree.root
    problem_a = root.add("problem")
    problem_a["value"] = "probA"

    for i in range(3):
        node = problem_a.add("iter")
        node["value"] = i
        node["error"] = 3 * i**2
        node.close()

    problem_a.close()

    problem_b = root.add("problem")
    problem_a["value"] = "probB"

    for i in range(3):
        node = problem_b.add("iter")
        node["value"] = i
        node["error"] = i**3 + 1
        node.close()

    problem_b.close()
    root.close()

    return tree


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_decorate_synthesized_attributes(error_tree):
    @error_tree.decorate
    async def compute_avg(node):
        if node.name == "problem":
            errors = [await c["error"] for c in node.children]
            node["avg"] = sum(errors) / len(errors)

    problem_a, problem_b = error_tree.root.children
    assert await problem_a["avg"] == 5
    assert await problem_b["avg"] == 4


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_decorate_inherited_attributes(error_tree):
    @error_tree.decorate()
    async def compute_avg(node):
        if node.name == "problem":
            errors = [await c["error"] for c in node.children]
            node["avg"] = sum(errors) / len(errors)

    @error_tree.decorate()
    async def mark_better_than_avg(node):
        if node.name == "iter":
            avg = await node.parent["avg"]
            node["good"] = await node["error"] < avg

    good_nodes = [
        node async for node in error_tree if node.name == "iter" and await node["good"]
    ]
    assert {await node["error"] for node in good_nodes} == {0, 3, 1, 2}


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_get_descendants_post_order(complete_tree):
    node_1, node_2 = complete_tree.root.children
    node_1_1, node_1_2 = node_1.children
    node_2_1, node_2_2 = node_2.children

    nodes = complete_tree.root.descendants
    assert_list_elements_equal(
        nodes, [node_1, node_2, node_1_1, node_1_2, node_2_1, node_2_2]
    )
    assert_post_order(nodes)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_get_ancestors(complete_tree):
    root = complete_tree.root
    _, node_2 = complete_tree.root.children
    node_2_1, _ = node_2.children

    nodes = node_2_1.ancestors
    assert_list_elements_equal(nodes, [node_2, root])


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_wait_until_complete(tree, mocker):
    event = asyncio.Event()
    mock = mocker.MagicMock()

    async def wait_for_tree():
        await tree.wait_until_complete()
        mock()
        event.set()

    _ = asyncio.create_task(wait_for_tree())

    # cannot be called before tree is completed
    mock.assert_not_called()

    tree.root.close()

    await event.wait()
    mock.assert_called_once()
