# Job Collector — full documentation

Written 7 October 2026. Everything here describes the tool as it stands in `~/job-collector`.

## 1. What it is

Job Collector is a small Python program that does three things:

1. **Collects** job postings from the careers boards of 82 companies.
2. **Scores** every posting against my CV and writes down the reasons for the score.
3. **Tracks** what I have seen and applied to, in a local database.

The result is one short ranked list I can read each morning, instead of opening eighty careers pages.

### What it is not

- It does **not** log in to any website.
- It does **not** apply to jobs. Applying stays manual, so every application can be tailored.
- It does **not** read LinkedIn or Wuzzuf pages. Both block programs. Section 9 explains how their jobs still reach the tool.
- It does **not** understand a posting. It matches words and rules. It ranks; I still read before applying.

## 2. Quick start

```bash
cd ~/job-collector
pip install -r requirements.txt
```

```bash
python -m job_collector collect
```

```bash
python -m job_collector list --min-score 16
```

```bash
python -m job_collector show <job_id>
```

```bash
python -m job_collector mark <job_id> applied
```

A full collection takes about 15 minutes, because the program reads each board politely, one request at a time.

## 3. How it works

```
companies.yaml ──▶ Dispatcher ──▶ Source ──▶ JobPosting ──▶ Scoring ──▶ SQLite ──▶ list / show / CSV
 82 boards          "which         reads one   one shape      against      jobs.db
                     system?"      board       for all        profile.yaml

alert emails (LinkedIn, Wuzzuf) ──▶ alerts.py ──▶ JobPosting ──▶ Scoring ──▶ SQLite
```

One run, step by step:

1. **Read the list.** `companies.yaml` holds one link per company board.
2. **Check the fetch log.** A board read less than 6 hours ago is skipped.
3. **Dispatch.** The dispatcher looks at the website part of the link and returns the source class that can read that careers system. An unknown website is an error; the program never guesses.
4. **Fetch.** The source asks the board for its jobs and turns each one into a `JobPosting`, the same shape whatever the system.
5. **Score.** Each posting is scored against `profile.yaml`. The score comes with a list of reasons.
6. **Save.** The posting is inserted into SQLite. The link is unique, so a job already stored is not stored again.
7. **Report.** The run prints how many jobs each board had, how many were new, and the best new ones.

If one board fails (site down, changed format), the error is recorded for that board and the run continues with the rest.

## 4. The files

```
job-collector/
├── companies.yaml            the 82 boards I watch
├── profile.yaml              my profile, from my CV: roles, skills, experience, location
├── requirements.txt          requests, pydantic, pyyaml, pytest
├── jobs.db                   the database (created on first run; not shared)
├── jobs.csv                  the last export (not shared)
├── README.md                 short version of this document
├── DOCUMENTATION.md          this document
├── job_collector/
│   ├── __main__.py           the commands: collect, list, show, mark, rescore, export, import-alerts, fetch-alerts
│   ├── collect.py            the loop over boards
│   ├── dispatcher.py         website → source
│   ├── models.py             JobPosting
│   ├── scoring.py            the scoring rules and the per-job skills view
│   ├── storage.py            SQLite: save, query, status, fetch log
│   ├── alerts.py             reads LinkedIn and Wuzzuf alert emails
│   └── sources/              one file per careers system (12 sources)
├── tools/
│   ├── discover.py           visits company websites and detects their careers system
│   ├── probe.py              tests company names against the public feeds
│   ├── companies.txt         140 company websites used for discovery
│   └── more_names.txt        about 400 company names used for probing
└── tests/
    └── test_job_collector.py 30 tests, none of which touch the network
```

## 5. The core pieces

### 5.1 JobPosting (`models.py`)

Every source must produce this one shape. Pydantic checks the types when the object is built.

| Field | Meaning |
|---|---|
| `source` | Which system it came from (`workday`, `ashby`, `linkedin-alert`…) |
| `company` | Company name |
| `title` | Job title |
| `location` | Place, as the company wrote it |
| `remote` | `True`, `False`, or unknown |
| `url` | The link to the posting. This is what makes a job unique |
| `description` | Plain text of the posting, when the board provides it |
| `published` | When it was posted, when the board says |

Each posting also has a short `id`: the first 10 characters of a hash of its link. The same link always gives the same ID, which is what I type in `show` and `mark`.

### 5.2 The dispatcher (`dispatcher.py`)

It holds a table of "website ending → source class". `get_source(link)` returns the source whose website matches the end of the link's host name, so `acme.recruitee.com` matches `recruitee.com`.

Three design choices, made on purpose and different from the book's crawler dispatcher that this is modelled on:

- **No fallback.** An unknown website raises `UnknownSourceError`. Guessing how to read an unknown job board would give wrong data silently.
- **The database refuses duplicates.** The link column is unique and the insert is a single call, not "check, then insert".
- **A fetch log.** A board is not read again within 6 hours, so the program cannot hammer a site by accident.

### 5.3 The sources (`sources/`)

Every source inherits from `BaseSource` and implements one method, `fetch(link)`, which returns a list of `JobPosting`.

| Source | Boards | What it reads | Example link |
|---|---|---|---|
| `workday` | 24 | The board's own search, with a search word | `https://valeo.wd3.myworkdayjobs.com/valeo_jobs?q=Egypt` |
| `ashby` | 15 | The company's public feed | `https://jobs.ashbyhq.com/thndr` |
| `greenhouse` | 13 | The company's public feed | `https://job-boards.greenhouse.io/canonical` |
| `smartrecruiters` | 10 | The company's public feed, filtered to a country | `https://jobs.smartrecruiters.com/BoschGroup?country=eg` |
| `recruitee` | 5 | The company's public feed | `https://synapseanalytics.recruitee.com` |
| `workable` | 4 | The company's public feed | `https://apply.workable.com/nawy-real-estate` |
| `oraclecloud` | 3 | The board's own search | `https://…oraclecloud.com/hcmUI/CandidateExperience/en/sites/RayaCareers` |
| `rss` | 3 | The careers site's own RSS feed (Siemens, Vodafone / _VOIS) | `https://jobs.siemens.com/en_US/externaljobs/SearchJobs/feed/?search=Egypt` |
| `lever` | 2 | The company's public feed | `https://jobs.lever.co/klivvr` |
| `bamboohr` | 1 | The company's job list (titles and places only) | `https://instabug.bamboohr.com/careers` |
| `amazon` | 1 | amazon.jobs search results for a country | `https://www.amazon.jobs/en/search?loc_query=Egypt&country=EGY` |
| `remotive` | 1 | A remote-jobs board | `https://remotive.com/api/remote-jobs` |

**Descriptions on large boards.** Workday and SmartRecruiters return a list of titles without descriptions, and each description needs its own request. To stay polite, the program reads at most 25 (Workday) or 30 (SmartRecruiters) descriptions per board. It picks the most promising titles first, by scoring the titles against my profile. The rest are scored on title and place only. A job stored without a description gets one filled in on a later run.

**Limits per board.** Workday reads up to 100 search results; SmartRecruiters up to 100; the Siemens feed up to 3 pages of 20.

### 5.4 The companies (`companies.yaml`)

82 boards, each tested and answering on 7 October 2026.

| Group | Count | Examples |
|---|---|---|
| `eg` — Egyptian companies | 17 | Thndr, Instabug, Nawy, Robusta, Integrant, Synapse Analytics, Si-Ware, Klivvr, Raya, Homzmart, Brimore |
| `global` — global and regional companies hiring in Egypt | 40 | Valeo, Siemens, Vodafone / _VOIS, Amazon, PwC ETIC, DXC, Cisco, Oracle, HPE, Mastercard, Visa, Bosch, Delivery Hero (Talabat), Sutherland |
| `remote` — companies that hire remotely | 25 | Canonical, GitLab, Elastic, Cohere, Mistral AI, LangChain, Supabase, Cursor, ClickHouse, Remotive |

For global companies the link carries the search (`q=Egypt`, `country=eg`), so only Egypt roles are collected.

An entry looks like this:

```yaml
- {name: Valeo, group: global, link: "https://valeo.wd3.myworkdayjobs.com/valeo_jobs?q=Egypt"}
```

`default_location` is optional. It is used when a feed is already filtered to Egypt and does not repeat the place on each job (Siemens, Raya).

### 5.5 My profile and the scoring (`profile.yaml`, `scoring.py`)

`profile.yaml` was written from my CV (AI Engineer version, October 2026):

- **Years of experience:** 0 years of paid work in this field.
- **Target roles:** words in a job title and the points each adds (LLM, machine learning, AI, NLP, RAG, MLOps, data scientist, Python, backend, data engineer, software engineer…).
- **Off-target titles:** other fields (sales, finance, hardware design, operations…).
- **Level words:** too senior (senior, lead, manager…) and junior (junior, graduate, intern, trainee…).
- **Skills:** strong (Python, LLM, fine-tuning, LoRA, RAG, NLP, PyTorch, Hugging Face, LangChain, FastAPI, Docker, SQL, Qdrant, MongoDB…) and familiar (ZenML, MLOps, GitHub Actions, pandas, scikit-learn, AWS, Linux…).
- **Location:** home (Egypt, Cairo, Giza…) and remote regions I can work in (worldwide, EMEA, MENA, Africa, Middle East).
- **Languages I do not speak:** German, French, and others.

The rules, in the order they run:

| # | Rule | Points |
|---|---|---|
| 1 | The title contains a target role | the role's points, total capped at 14 |
| 1 | No target role, and the title is in another field | −8 |
| 2 | The title is too senior | −10 |
| 2 | The title is junior level | +6 |
| 3 | Skills from my CV appear in the text: strong +2 each, familiar +1 each | capped at 14 |
| 4 | The text asks for N years (N at least 2 more than I have) | −4 for each missing year, down to −16 |
| 4 | The text is open to new graduates, or asks for at most 1 year | +4 |
| 5 | The job is in Egypt | +8 |
| 5 | Remote, and open to my region or with no country named | +5 |
| 5 | Remote, but tied to another country | −8 |
| 5 | On-site in another country | −12 |
| 6 | Requires a language I do not speak | −8 |
| 7 | The title has no target role | the total is capped at 14 |

Words are matched as whole words, so "ai" does not match inside "maintain". "Years asked" is the largest "N years" in the text, ignoring numbers above 15.

How to read a score:

| Score | Label |
|---|---|
| 24 or more | strong |
| 16 to 23 | good |
| 8 to 15 | maybe |
| under 8 | weak |

**A worked example.** Valeo, "Graduate In Training Program (GIT), AI Engineer", Cairo:

| Reason | Points |
|---|---|
| `role:ai` | +7 |
| `junior level` | +6 |
| `skills: python, pytorch, langchain, fastapi, deep learning, rest api, aws` | +13 |
| `open to new graduates` | +4 |
| `in Egypt` | +8 |
| **Total** | **38 (strong)** |

After changing `profile.yaml`, run `python -m job_collector rescore` to apply the new rules to every stored job without fetching anything.

### 5.6 The database (`storage.py`, `jobs.db`)

A single SQLite file with two tables.

`jobs` — one row per posting:

| Column | Meaning |
|---|---|
| `id` | The 10-character ID |
| `url` | The link. Unique: the database refuses a second row with the same link |
| `source`, `company`, `title`, `location`, `remote`, `description`, `published` | The posting |
| `first_seen` | When the program first stored it |
| `score`, `reasons` | The score and why |
| `status` | `new`, `applied` or `skipped` |

`fetch_log` — one row per board link, with the time it was last read. This is what enforces the 6-hour wait.

## 6. The commands

All commands start with `python -m job_collector`, run from `~/job-collector`.

| Command | What it does |
|---|---|
| `collect` | Reads every board in `companies.yaml` and saves new jobs |
| `collect --only valeo` | Only companies whose name contains "valeo" |
| `collect --force` | Ignores the 6-hour wait |
| `list` | Shows stored jobs with status `new`, best score first (default: score 8 or more, 30 jobs) |
| `list --min-score 16` | Only good and strong matches |
| `list --days 1` | Only jobs first seen in the last day |
| `list --status applied` | Jobs I applied to (`new`, `applied`, `skipped`, or `all`) |
| `list --limit 100` | Show more rows |
| `show <job_id>` | One job in full: why it scored, which of my skills to lead with, what it asks for that my CV lacks, the years asked, and the start of the description |
| `mark <job_id> applied` | Sets the status (`new`, `applied`, `skipped`) |
| `rescore` | Recomputes every score after I edit `profile.yaml` |
| `export jobs.csv --min-score 16` | Writes a CSV for Excel or Notion |
| `import-alerts [folder]` | Reads saved alert emails (`.eml`) from a folder (default `inbox/`) |
| `fetch-alerts [folder] --days 7` | Downloads alert emails from my mailbox, then imports them |

Options that apply to every command, placed before the command name: `--companies`, `--profile`, `--db`, to point at different files.

### The `show` view

```
Graduate In Training Program (GIT), AI Engineer  |  Valeo  |  Cairo, Egypt
https://valeo.wd3.myworkdayjobs.com/valeo_jobs/job/Cairo/…
Score 38 (strong)   status: new
Why: role:ai +7; junior level +6; skills:… +13; open to new graduates +4; in Egypt +8
Experience asked: no number of years stated

On my CV and in this posting, in the order the posting mentions them (lead with these):
  python, rest api, fastapi, deep learning, pytorch, langchain, aws, arabic
In this posting but not on my CV (be ready to be asked about these):
  flask, django, javascript, typescript, pytest, tensorflow, llamaindex, gcp
```

The first list is the order to put my skills in for that application. The second is what to prepare for.

## 7. A daily routine

1. `python -m job_collector collect` (about 15 minutes; start it and do something else).
2. `python -m job_collector list --min-score 16 --days 1` to see what is new.
3. `python -m job_collector show <job_id>` for each one worth applying to; reorder the CV skills to match.
4. Apply on the company's site.
5. `python -m job_collector mark <job_id> applied`, or `skipped` for ones I rule out.
6. Once a week: `python -m job_collector export jobs.csv --min-score 16`, then Notion → Import → CSV into the Job Applications Tracker.

## 8. Adding companies and systems

### Add a company

1. Find its board. `tools/discover.py` visits company websites, follows the careers link, and reports the careers system and the company's identifier. `tools/probe.py` tests company names directly against six public feeds and keeps boards with jobs in Egypt or open remote jobs.

```bash
python tools/discover.py tools/companies.txt
```

```bash
python tools/probe.py tools/more_names.txt
```

2. Add one line to `companies.yaml`.
3. Test it: `python -m job_collector collect --only "<name>" --force`.

`discover.py` obeys each site's `robots.txt` and reads only a few pages per company. Many careers pages are built by JavaScript, so it found the system for 38 of 140 websites; `probe.py` found more.

### Add a new careers system

1. Write a class in `job_collector/sources/` that inherits from `BaseSource` and implements `fetch(link)`.
2. Export it in `job_collector/sources/__init__.py`.
3. Add one `.register("<website>", YourSource)` line in `SourceDispatcher.build()`.
4. Add a test with a small made-up response in the system's real shape.

## 9. LinkedIn and Wuzzuf

Both sites block programs from reading their pages. Wuzzuf answered `403` when tested, and LinkedIn forbids automated access in its terms. Disguising the program as a browser would risk the accounts I need for the job search, so the tool does not do it. The same jobs arrive two other ways.

**From the source.** Most postings on those sites originate on a company's own careers board. The tool reads those boards directly, usually with the full description.

**From my alert emails.** Both sites email me new matching jobs, and those emails are mine to read.

1. Set up job alerts on LinkedIn and Wuzzuf for my target roles.
2. Then either:
   - save alert emails as `.eml` files into `inbox/` (Gmail: three dots → *Download message*) and run `python -m job_collector import-alerts`; or
   - let the program read them from my mailbox. It connects read-only and changes nothing. It needs two environment variables, so no password is written in any file:

```bash
export JOB_ALERT_EMAIL="me@gmail.com"
export JOB_ALERT_APP_PASSWORD="an app password, not my real password"
python -m job_collector fetch-alerts
```

An alert gives the title, the link, and usually the company and place, but no description. These jobs are scored on title and place only. Tracking codes are removed from the links, so the same job arriving in two emails is stored once.

**This part is not yet proven.** It was built from the known link formats and tested on made-up emails. It needs checking against real LinkedIn and Wuzzuf alerts.

## 10. Rules the tool follows

- It identifies itself honestly in every request (`job-collector/… (personal job search; one user)`).
- It waits between requests to the same site (0.5 to 1 second).
- It does not read a board twice within 6 hours.
- It limits how many descriptions it reads per board.
- `discover.py` obeys `robots.txt`.
- Remotive's terms are respected: link back to the job on Remotive, name Remotive as the source, at most about 4 requests a day.
- It uses only feeds and search pages that the careers sites publish. It does not bypass a block.

## 11. Tests

```bash
python -m pytest tests
```

30 tests, none of which touch the network. They cover:

- each source, given a small made-up response in the real shape of that system's feed;
- the dispatcher: correct routing, and refusal of unknown websites;
- the scoring rules one by one, including whole-word matching, the experience gap, the location rules, and the cap;
- a check on my real `profile.yaml`: an off-field graduate programme stays out of the top;
- storage: one row per link, description backfill, status changes, the recent-jobs filter;
- the collect loop: a broken board does not stop the run, and a board is not re-read too soon;
- the alert reader, on made-up emails.

## 12. Results of the first full run (7 October 2026)

| Measure | Value |
|---|---|
| Boards read | 82 (one more, Thales, failed and was removed) |
| Postings collected | 3,302 |
| Strong matches | 5 |
| Good matches | 27 |
| Maybe | 211 |

The top five: Valeo Graduate In Training Program, AI Engineer (38); Valeo Graduate In Training Program, AI Development (35); Canonical Python and AI/ML software engineer, remote (30); Valeo Fullstack AI Engineer (25); Valeo Junior Software Engineer (24).

## 13. Limits

- **82 boards, not every employer.** Companies with a custom careers page, or that post only on LinkedIn, are covered only through alert emails. Several large employers block or hide their feeds (Microsoft, IBM, Dell, Ericsson, Orange) and are not included.
- **The score is rules and keywords.** A posting can score well and still be wrong for me. A good posting with an unusual title can be capped at "maybe".
- **Some jobs are scored on title and place only:** the less promising titles on large boards, the Instabug list, the Siemens feed, and everything from alert emails.
- **"Remote" is whatever the company wrote.** It often means remote inside one country.
- **Posting dates can be old.** A board may still list a job posted months ago.
- **Boards change.** A company can move to another careers system, and its link will start failing. The run reports which boards failed.
- **It is not a git repository yet** and is not published.

## 14. Troubleshooting

| What I see | Cause | What to do |
|---|---|---|
| `skipped: fetched less than 6 h ago` | The board was read recently | Wait, or add `--force` |
| `FAILED: HTTPError 404` or `422` for one board | The company changed or closed its board | Find the new link with `tools/discover.py` and update `companies.yaml` |
| `No source registered for '…'` | The link's website is not a supported system | Check the link, or add a source (section 8) |
| `Nothing to show.` | No job passes the filters | Lower `--min-score`, drop `--days`, or try `--status all` |
| A job I applied to is still listed | Its status is still `new` | `mark <job_id> applied` |
| Scores look wrong after editing the profile | Stored scores are old | `rescore` |
| `Set JOB_ALERT_EMAIL and JOB_ALERT_APP_PASSWORD first.` | The mailbox variables are not set | Set both (section 9), or use `import-alerts` with saved emails |
| `ModuleNotFoundError` | A library is missing | `pip install -r requirements.txt` |
