from collections.abc import AsyncIterable

import pytest

from supy.asynciter import AsyncPushIterator


async def collect(items: AsyncIterable) -> list[object]:
    return [x async for x in items]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_iter_added_items_are_produced():
    it = AsyncPushIterator()

    it.add(1)
    it.add(3)
    it.close()

    items = collect(it)

    assert await items == [1, 3]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_iter_can_add_items_later():
    it = AsyncPushIterator()

    items = collect(it)

    it.add(1)
    it.add(3)
    it.close()

    assert await items == [1, 3]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_iter_cannot_add_items_after_close():
    it = AsyncPushIterator()
    it.close()

    with pytest.raises(ValueError, match="closed"):
        it.add(1)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_iter_cannot_close_twice():
    it = AsyncPushIterator()
    it.close()

    with pytest.raises(ValueError, match="closed"):
        it.close()
