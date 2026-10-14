from urllib.parse import urlparse

from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.domain.documents import UserDocument


@step
def crawl_links(user: UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    dispatcher = CrawlerDispatcher.build_default()

    logger.info(f"Starting to crawl {len(links)} link(s).")

    metadata = {}
    successfull_crawls = 0
    for link in tqdm(links):
        successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        successfull_crawls += successfull_crawl
 
        metadata = _add_to_metadata(metadata, crawled_domain, successfull_crawl)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} links.")

    success_rate = successfull_crawls / len(links) if links else 1.0
    if success_rate < 0.8:
        raise RuntimeError(
            f"Only {successfull_crawls}/{len(links)} links crawled ({success_rate:.0%}); expected at least 80%."
        )

    return links


def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> tuple[bool, str]:
    crawler_domain = urlparse(link).netloc

    try:
        # Inside the try: creating a crawler can fail too (Chrome not starting, LinkedIn switched off),
        # and that should count as one failed link, not stop the whole step.
        crawler = dispatcher.get_crawler(link)
        crawler.extract(link=link, user=user)

        return (True, crawler_domain)
    except Exception as e:
        logger.error(f"An error occurred while crowling: {e!s}")

        return (False, crawler_domain)


def _add_to_metadata(metadata: dict, domain: str, successfull_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = metadata[domain].get("successful", 0) + successfull_crawl
    metadata[domain]["total"] = metadata[domain].get("total", 0) + 1

    return metadata
