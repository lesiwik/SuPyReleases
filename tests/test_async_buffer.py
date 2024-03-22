import asyncio
import pickle
from collections.abc import AsyncIterable

import pytest

from supy.asynciter import AsyncBuffer


async def collect(items: AsyncIterable) -> list[object]:
    return [x async for x in items]


@pytest.fixture()
def buff():
    """Empty AsyncBuffer instance."""
    return AsyncBuffer()


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_old_items_appear_in_iteration(buff):
    buff.add(1)
    buff.add(3)
    buff.close()

    items = await collect(buff)

    assert items == [1, 3]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_new_items_appear_in_iteration(buff):
    buff.add(1)

    # This is to force the iterators to already exist when buff.close() is called

    started = asyncio.Event()

    async def collect_and_notify():
        collected = []

        async for item in buff:
            started.set()
            collected.append(item)

        return collected

    items = asyncio.create_task(collect_and_notify())

    await started.wait()

    buff.add(3)
    buff.close()

    assert await items == [1, 3]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_can_have_multiple_iterators(buff):
    buff.add(1)

    items_1 = asyncio.create_task(collect(buff))

    buff.add(2)
    buff.add(3)

    items_2 = asyncio.create_task(collect(buff))

    buff.add(4)
    buff.close()

    assert await items_1 == await items_2


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_is_context_manager(buff):
    with buff:
        buff.add(1)

        items = asyncio.create_task(collect(buff))

        buff.add(2)
        buff.add(3)

    assert await items == [1, 2, 3]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_cannot_add_items_after_close(buff):
    buff.close()

    with pytest.raises(ValueError, match="closed"):
        buff.add(1)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_cannot_close_twice(buff):
    buff.close()

    with pytest.raises(ValueError, match="closed"):
        buff.close()


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_buffer_from_iterable():
    event = asyncio.Event()

    async def generator():
        yield 1
        await event.wait()
        yield 2

    buff = AsyncBuffer.from_async_iterable(generator())
    event.set()

    assert await collect(buff) == [1, 2]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_open_buffer_can_be_pickled(buff):
    buff.add(1)
    buff.add(2)

    data = pickle.dumps(buff)
    new_buff = pickle.loads(data)

    buff.add(3)
    new_buff.close()

    assert await collect(new_buff) == [1, 2]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_closed_buffer_can_be_pickled(buff):
    buff.add(1)
    buff.add(2)
    buff.close()

    data = pickle.dumps(buff)
    new_buff = pickle.loads(data)

    assert await collect(new_buff) == [1, 2]
