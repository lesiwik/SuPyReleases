import asyncio
import pickle

import pytest
import pytest_asyncio

from supy.asynciter import AsyncDict


@pytest.fixture()
def adict():
    return AsyncDict()


@pytest_asyncio.fixture()
async def foobar_adict():
    return AsyncDict(foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_retrieve_value(adict):
    adict["foo"] = "bar"
    val = adict["foo"]

    assert await val == "bar"


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_retrieve_value_set_later(adict):
    val = adict["foo"]
    adict["foo"] = "bar"

    assert await val == "bar"


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_get_existing_with_no_default(adict):
    adict["foo"] = "bar"
    assert adict.get("foo") == "bar"


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_get_missing_with_default(adict):
    assert adict.get("foo", default=3) == 3


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_get_missing_with_no_default_gives_none(adict):
    assert adict.get("foo") is None


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_check_if_key_exists(adict):
    adict["foo"] = "bar"
    val = adict["bar"]

    assert "foo" in adict
    assert "bar" not in adict

    # to avoid "was never awaited" runtime warning
    adict["bar"] = 1
    await val


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_delete_key(adict):
    adict["foo"] = 3

    del adict["foo"]

    assert "foo" not in adict


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_pop_key(adict):
    adict["foo"] = 3

    val = await adict.pop("foo")

    assert "foo" not in adict
    assert val == 3


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_popitem(adict):
    adict["foo"] = 3

    assert adict.popitem() == ("foo", 3)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_create_from_dict():
    adict = AsyncDict(dict(foo=3, bar=5))

    assert dict(adict.items()) == dict(foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_create_from_kv_pairs():
    adict = AsyncDict([("foo", 3), ("bar", 5)])

    assert dict(adict.items()) == dict(foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_create_from_kwargs():
    adict = AsyncDict(foo=3, bar=5)

    assert dict(adict.items()) == dict(foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_create_from_mix():
    adict = AsyncDict(dict(foo=3), bar=5)

    assert dict(adict.items()) == dict(foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_update_from_dict(adict):
    adict["a"] = 1
    adict.update(dict(foo=3, bar=5))

    assert dict(adict.items()) == dict(a=1, foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_update_from_pairs(adict):
    adict["a"] = 1
    adict.update([("foo", 3), ("bar", 5)])

    assert dict(adict.items()) == dict(a=1, foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_update_from_kwargs(adict):
    adict["a"] = 1
    adict.update(foo=3, bar=5)

    assert dict(adict.items()) == dict(a=1, foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_update_from_mix(adict):
    adict["a"] = 1
    adict.update(dict(foo=3), bar=5)

    assert dict(adict.items()) == dict(a=1, foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_update_wakes_up_getters(adict):
    event = asyncio.Event()

    async def get():
        event.set()
        return await adict["foo"]

    val = asyncio.create_task(get())

    await event.wait()
    adict.update(foo=3)

    assert await val == 3


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_iterate_over_keys(foobar_adict):
    assert list(foobar_adict) == ["foo", "bar"]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_iterate_over_values(foobar_adict):
    assert list(foobar_adict.values()) == [3, 5]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_iterate_over_items(foobar_adict):
    assert list(foobar_adict.items()) == [("foo", 3), ("bar", 5)]


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_await_gets_fresh_value(adict):
    val = adict["foo"]
    adict["foo"] = 3
    adict["foo"] = 5

    assert await val == 5


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_length_includes_only_already_set_values(adict):
    val = adict["foo"]
    adict["bar"] = 3

    assert len(adict) == 1

    # to avoid "was never awaited" runtime warning
    adict["foo"] = 1
    await val


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_update_inplace_using_operator(adict):
    adict["foo"] = 3
    adict |= dict(bar=5)

    assert dict(adict.items()) == dict(foo=3, bar=5)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_equal_to_same_async_dict():
    assert AsyncDict(foo=5, bar=3) == AsyncDict(foo=5, bar=3)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_equal_to_same_dict():
    assert AsyncDict(foo=5, bar=3) == dict(foo=5, bar=3)


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_be_pickled(foobar_adict):
    data = pickle.dumps(foobar_adict)
    new_adict = pickle.loads(data)
    assert new_adict == foobar_adict


@pytest.mark.timeout(1)
@pytest.mark.asyncio()
async def test_can_be_pickled_with_pending(foobar_adict):
    event = asyncio.Event()

    async def get():
        event.set()
        return await foobar_adict["asdf"]

    val = asyncio.create_task(get())

    # to ensure get() reached its await
    await event.wait()

    data = pickle.dumps(foobar_adict)
    new_adict = pickle.loads(data)

    assert new_adict == foobar_adict

    # to avoid "destroyed while pending" runtime error
    foobar_adict["asdf"] = 8
    await val
