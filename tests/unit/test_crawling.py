from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from llm_engineering.application.crawlers import custom_article
from llm_engineering.application.crawlers.custom_article import CustomArticleCrawler
from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.application.crawlers.github import GithubCrawler
from llm_engineering.application.crawlers.linkedin import LinkedInCrawler
from llm_engineering.application.crawlers.medium import MediumCrawler
from llm_engineering.domain.documents import ArticleDocument
from steps.etl.crawl_links import _crawl_link

USER = SimpleNamespace(id="00000000-0000-4000-8000-000000000000", full_name="Test User")


class _FakeLoader:
    """Stands in for AsyncHtmlLoader, so the test does not use the network."""

    html = ""

    def __init__(self, links: list[str]) -> None:
        pass

    def load(self) -> list[Document]:
        return [Document(page_content=self.html, metadata={"title": "Some Substack"})]


@pytest.fixture
def saved(monkeypatch) -> list[ArticleDocument]:
    saved_articles = []
    monkeypatch.setattr(custom_article, "AsyncHtmlLoader", _FakeLoader)
    monkeypatch.setattr(ArticleDocument, "find", classmethod(lambda cls, **kwargs: None))
    monkeypatch.setattr(ArticleDocument, "save", lambda self, **kwargs: saved_articles.append(self))

    return saved_articles


def test_error_page_is_rejected(saved) -> None:  # 3.8
    _FakeLoader.html = "<html><body><h1>Page not found</h1><p>No posts</p></body></html>"

    with pytest.raises(ValueError, match="Not a real article"):
        CustomArticleCrawler().extract(link="https://example.substack.com/p/gone", user=USER)

    assert saved == []


def test_real_article_is_saved(saved) -> None:  # 3.8
    _FakeLoader.html = "<html><body>" + "<p>A real paragraph about RAG pipelines.</p>" * 60 + "</body></html>"

    CustomArticleCrawler().extract(link="https://example.substack.com/p/real", user=USER)

    assert len(saved) == 1


def test_failing_crawler_counts_as_one_failed_link() -> None:  # 3.2, 3.7
    class _BrokenDispatcher:
        def get_crawler(self, link: str):
            raise DeprecationWarning("switched off")

    successful, domain = _crawl_link(_BrokenDispatcher(), "https://linkedin.com/in/someone", USER)

    assert (successful, domain) == (False, "linkedin.com")


def test_linkedin_crawler_fails_before_starting_chrome() -> None:  # 3.2
    with pytest.raises(DeprecationWarning):
        LinkedInCrawler()


def test_default_dispatcher_registers_every_crawler_with_a_domain() -> None:  # 3.4
    dispatcher = CrawlerDispatcher.build_default()

    assert set(dispatcher._crawlers.values()) == {MediumCrawler, LinkedInCrawler, GithubCrawler}
    assert isinstance(dispatcher.get_crawler("https://github.com/user/repo"), GithubCrawler)
    assert isinstance(dispatcher.get_crawler("https://some-blog.com/post"), CustomArticleCrawler)
