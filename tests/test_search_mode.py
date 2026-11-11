"""
M-Team 标题模式透传、分类覆盖和搜索上下文的离线回归。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.endpoints import search as endpoint
from app.application.search.state import SearchStateService, normalize_search_params
from app.modules.indexer import IndexerModule
from app.modules.indexer.spider import mtorrent
from app.runtime.extensions.module.contracts import compatible_module_kwargs
from app.schemas.token import TokenPayload
from app.schemas.types import MediaType


@pytest.mark.parametrize("mode", [None, "", "normal", "adult", "custom-mode"])
@pytest.mark.parametrize("cat", [None, "401,402"])
@pytest.mark.asyncio
async def test_mteam_mode_overrides_categories_in_both_transports(monkeypatch, mode, cat):
    """
    同步与异步实际发送相同模式和分类，未知非空模式保留原值。
    """
    monkeypatch.setattr(mtorrent, "get_configured_system_config", lambda: None)
    spider = mtorrent.MTorrentSpider({"id": 1, "domain": "https://mteam.example/", "apikey": "fake"})
    calls = []

    def post(**params):
        """
        捕获请求体并回放空结果，不访问真实站点。
        """
        calls.append(params["json"])
        return SimpleNamespace(status_code=200, json=lambda: {"code": 0, "data": {"data": []}})

    monkeypatch.setattr(mtorrent, "RequestUtils", lambda **_kwargs: SimpleNamespace(post_res=post))
    monkeypatch.setattr(mtorrent, "AsyncRequestUtils", lambda **_kwargs: SimpleNamespace(post_res=AsyncMock(side_effect=post)))
    spider.search("Show", mtype=MediaType.TV, cat=cat, mode=mode)
    await spider.async_search("Show", mtype=MediaType.TV, cat=cat, mode=mode)
    assert len(calls) == 2 and calls[0] == calls[1]
    expected = [] if mode else cat.split(",") if cat else spider._tv_category
    assert calls[0]["categories"] == expected
    assert calls[0].get("mode") == (mode or None)
    assert calls[0]["pageNumber"] == 1


@pytest.mark.parametrize("parser", ["mTorrent", "SunnyPT", "HDDolby", ""])
def test_indexer_only_passes_mode_to_mteam(monkeypatch, parser):
    """
    混合站点搜索不会向其他专站或通用解析器泄漏新增参数。
    """
    monkeypatch.setattr(IndexerModule, "_IndexerModule__search_check", lambda *_args: True)
    request = IndexerModule._IndexerModule__create_search_request(
        {"parser": parser}, "Show", mode="adult",
    )
    assert ("mode" in request.arguments) == (parser == "mTorrent")


@pytest.mark.parametrize("mode", [None, "", "adult"])
@pytest.mark.asyncio
async def test_title_api_passes_mode_and_persists_context(monkeypatch, mode):
    """
    普通标题接口透传模式并保存归属用户的搜索条件。
    """
    search, save = AsyncMock(return_value=[]), AsyncMock()
    monkeypatch.setattr(endpoint, "SearchChain", lambda: SimpleNamespace(
        async_search_by_title=search, async_save_search_context=save,
    ))
    await endpoint.search_by_title(keyword="Show", mode=mode, _=TokenPayload(sub=51))
    assert search.await_args.kwargs["mode"] == mode
    assert save.await_args.args[1].params.get("mode") == (mode or None)


@pytest.mark.parametrize("keyword", ["Show", ""])
def test_mode_context_roundtrip(keyword):
    """
    包含模式的普通搜索和无关键词浏览都保留可重放的条件。
    """
    params = SearchStateService.build_params(keyword=keyword, mode="adult")
    assert normalize_search_params(params)["mode"] == "adult"
    assert "mode" not in SearchStateService.build_params(keyword="Show")


def test_legacy_module_signature_ignores_optional_mode():
    """
    旧模块无需新增 mode 参数即可继续接收既有输入。
    """
    def legacy(site, keyword, mtype, page):
        """
        表示未升级的模块调用签名。
        """
        return site, keyword, mtype, page

    params = {"site": {}, "keyword": "Show", "mtype": None, "page": 0, "mode": "adult"}
    assert "mode" not in compatible_module_kwargs("search_torrents", legacy, params)


@pytest.mark.parametrize("entry", ["sync", "async", "stream"])
@pytest.mark.asyncio
async def test_title_chain_carries_mode_to_provider_and_cache(monkeypatch, entry):
    """
    三种标题入口共用模式参数，流式完成不会重复展开缓存字段。
    """
    from app.chain.search.facade import SearchChain

    chain = object.__new__(SearchChain)
    provider_calls, cache_calls = [], []

    def search(**params):
        """
        捕获站点搜索参数并返回空结果。
        """
        provider_calls.append(params)
        return []

    async def stream(**params):
        """
        捕获渐进式搜索参数并完成空结果流。
        """
        provider_calls.append(params)
        yield {"type": "done", "stage": "done", "items": []}

    monkeypatch.setattr(chain, "_SearchChain__search_all_sites", search)
    monkeypatch.setattr(chain, "_SearchChain__async_search_all_sites", AsyncMock(side_effect=search))
    monkeypatch.setattr(chain, "_SearchChain__async_search_all_sites_stream", stream)
    monkeypatch.setattr(chain, "cancel_ai_recommend", lambda: None)
    monkeypatch.setattr(chain, "save_last_search_params", lambda **params: cache_calls.append(params))
    monkeypatch.setattr(chain, "async_save_last_search_params", AsyncMock(side_effect=lambda **params: cache_calls.append(params)))
    monkeypatch.setattr(chain, "_save_results", lambda _items: None)
    monkeypatch.setattr(chain, "_async_save_results", AsyncMock())
    params = {"title": "Show", "mode": "adult", "cache_local": True, "rule_groups": []}
    if entry == "sync":
        chain.search_by_title(**params)
    elif entry == "async":
        await chain.async_search_by_title(**params)
    else:
        assert [event async for event in chain.async_search_by_title_stream(**params)]
    assert provider_calls[0]["mode"] == "adult"
    assert cache_calls[0]["mode"] == "adult"


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["sync", "async", "stream"])
async def test_provider_carries_mode_to_site_without_changing_plugin_input(monkeypatch, entry):
    """
    真实 provider 调度链把模式送到站点，独立插件搜索源不接收该参数。
    """
    from concurrent.futures import Future
    from unittest.mock import MagicMock

    from app.chain.search import provider
    from app.chain.search.facade import SearchChain

    chain = object.__new__(SearchChain)
    chain.runtime_config = SimpleNamespace(search_threadpool_size=1)
    site = {"id": 1, "name": "M-Team"}
    calls, plugin_calls = [], []

    def submit(function, *args, **kwargs):
        """
        就地执行共享线程任务以隔离并发资源。
        """
        result = Future()
        result.set_result(function(*args, **kwargs))
        return result

    def search(**params):
        """
        记录单站点调用并以空页结束翻页。
        """
        calls.append(params)
        return []

    def plugin(**params):
        """
        记录独立插件资源源的原有参数。
        """
        plugin_calls.append(params)
        return []

    monkeypatch.setattr(provider, "ThreadHelper", lambda: SimpleNamespace(submit=submit))
    monkeypatch.setattr(provider, "ProgressHelper", MagicMock())
    monkeypatch.setattr(provider, "AsyncProgressHelper", lambda *_args: SimpleNamespace(
        start=AsyncMock(), update=AsyncMock(), end=AsyncMock(),
    ))
    monkeypatch.setattr(provider, "_wait_for_site_request", lambda _site: None)
    monkeypatch.setattr(provider, "_async_wait_for_site_request", AsyncMock(return_value=None))
    monkeypatch.setattr(chain, "_sync_indexers", lambda _sites: [site])
    monkeypatch.setattr(chain, "_async_indexers", AsyncMock(return_value=[site]))
    monkeypatch.setattr(chain, "search_site_torrents", search)
    monkeypatch.setattr(chain, "async_search_site_torrents", AsyncMock(side_effect=search))
    monkeypatch.setattr(chain, "search_plugin_torrents", plugin)
    monkeypatch.setattr(chain, "async_search_plugin_torrents", AsyncMock(side_effect=plugin))
    monkeypatch.setattr(chain, "_should_continue_search_pages", lambda **_kwargs: False)
    params = {"keyword": "Show", "mode": "adult", "page": 0}
    if entry == "sync":
        chain._SearchChain__search_all_sites(**params)
    elif entry == "async":
        await chain._SearchChain__async_search_all_sites(**params)
    else:
        assert [item async for item in chain._SearchChain__async_search_all_sites_stream(**params)]
    assert calls and all(call["mode"] == "adult" for call in calls)
    assert plugin_calls and all("mode" not in call for call in plugin_calls)
