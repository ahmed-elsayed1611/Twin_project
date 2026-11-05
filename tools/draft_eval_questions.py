"""Draft retrieval-evaluation questions with an LLM: two per article, for a human to check.

The questions are labelled with the article's document id, not a chunk id, so the set still works after re-chunking.
Output: data/eval/retrieval_questions_draft.json. Review it, then save the kept questions as retrieval_questions.json.

Usage: poetry run python -m tools.draft_eval_questions --n-articles 25
"""

import json
import random
from pathlib import Path

import click
from loguru import logger
from openai import OpenAI

from llm_engineering.domain.documents import ArticleDocument
from llm_engineering.settings import settings

OUTPUT_PATH = Path("data/eval/retrieval_questions_draft.json")
MAX_ARTICLE_CHARS = 8000

PROMPT = """You write test questions for a search system over technical blog articles.

Read the article and write exactly 2 questions that this article answers:
1. "simple": about the article's main topic.
2. "specific": about one concrete detail (a number, a setting, a tool, a step) found in the article.

Rules:
- A reader must understand each question without seeing the article: never write "this article" or "the author".
- Do not copy phrases from the article; ask the way a curious engineer would.
- Each question must be answerable from this article alone.

Reply with JSON only: {{"simple": "...", "specific": "..."}}

Title: {title}

Article:
{content}"""


def article_text(article: ArticleDocument) -> tuple[str, str]:
    content = article.content
    title = content.get("Title") or ""
    body = content.get("Content") or ""

    return title, body[:MAX_ARTICLE_CHARS]


@click.command()
@click.option("--n-articles", default=25, help="How many articles to draft questions for.")
@click.option("--seed", default=42, help="Random seed for choosing the articles.")
def main(n_articles: int, seed: int) -> None:
    articles = ArticleDocument.bulk_find()
    random.Random(seed).shuffle(articles)
    articles = articles[:n_articles]

    client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    drafts = []
    for i, article in enumerate(articles, start=1):
        title, body = article_text(article)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL_ID,
            messages=[{"role": "user", "content": PROMPT.format(title=title, content=body)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            questions = json.loads(response.choices[0].message.content or "")
        except json.JSONDecodeError:
            logger.warning(f"Skipping '{title}': the reply was not JSON.")
            continue

        for question_type in ("simple", "specific"):
            if question := (questions.get(question_type) or "").strip():
                drafts.append(
                    {
                        "question": question,
                        "type": question_type,
                        "document_id": str(article.id),
                        "title": title,
                        "author": article.author_full_name,
                    }
                )
        logger.info(f"{i}/{len(articles)} {title}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote {len(drafts)} draft questions to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
