# Study guide 1 — Data preparation

Code: [`twin_dataset_builder.py`](../whatsapp-twin/data-preparation/src/twin_dataset_builder.py) (655 lines, standard library only).
Tests: [`test_twin_dataset_builder.py`](../whatsapp-twin/data-preparation/tests/test_twin_dataset_builder.py) (11 tests).

## The one-sentence version

A raw WhatsApp `.txt` export goes in; `train.jsonl` and `validation.jsonl` come out, where every record is "a few turns of context, then one real reply of mine".

## The whole flow is four calls

Everything happens in `run()` (line 575):

```
parse_messages  →  sessionize  →  build_samples  →  export
   text lines       messages        sessions         samples
   → messages       → sessions      → samples        → files
```

If you remember these four names and what each one receives and returns, you can rebuild the rest from memory.

## What happened to my real data at each step

These numbers are from `audit_report.json` of the real build. They are the best way to explain the pipeline, because each drop has a reason.

| Step | Count | Why it changed |
|---|---:|---|
| Lines in the file | 135,289 | |
| Lines that start a message (header matched) | 119,589 | The other 15,700 are continuation lines of multi-line messages |
| After removing WhatsApp system notices | 113,816 | 5,773 notices: media omitted, deleted message, encryption banner |
| After cleaning | 111,829 | 1,917 too short (under 2 characters), 70 empty after redaction |
| After merging bursts | 62,852 turns | 48,977 merges: back-to-back messages from one person become one turn |
| Sessions | 874 | A new session starts after 4 hours of silence |
| Samples | 26,989 | One per "Contact spoke, then I replied". 911 dropped for no Contact message before, 3,274 dropped by the duplicate cap |
| Train / validation | 24,915 / 2,074 | Whole sessions go to one side |

## Step 1 — `parse_messages` (line 276)

**Problem it solves:** the file is lines, not messages. One message can cover several lines, and the header is in Arabic locale.

- **`HEADER_PATTERN` (line 33)** decides whether a line starts a new message. It captures date, time, AM/PM (`ص` / `م`), sender and text.
- **`_normalize_header_text` (line 188)** runs first. It removes invisible direction marks (WhatsApp inserts them in Arabic text) and converts Arabic digits `٠١٢` to `012`. Without it the regex matches nothing.
- **Continuation lines.** A line with no header is appended to the message being built (`current`). `flush_current()` finishes that message when the next header arrives.
- **`_is_system_event` (line 238).** A header with no sender, or text containing a known marker, is a WhatsApp notice and is skipped.
- **Senders.** My name stays; every other sender becomes `Contact` (line 334). The real name never reaches the output.
- **`_redact_content` (line 242).** Strips the "edited" tag, then replaces emails, URLs, phone numbers and credential-like tokens with `[EMAIL]`, `[URL]`, `[PHONE]`, `[CREDENTIAL]`.
- **`_merge_bursts` (line 347).** People send five short messages in a row. Same sender within 300 seconds becomes one turn, joined with newlines.

One detail worth knowing: a merged turn keeps the timestamp of its *first* message (line 359), so the 300 seconds are measured from the start of the burst, not from the previous message.

## Step 2 — `sessionize` (line 369)

A gap of more than 4 hours starts a new session. The reason: context from yesterday's conversation should not be used to explain today's reply.

## Step 3 — `build_samples` (line 395)

For every turn of mine, it checks in order:

1. Is the turn before it from `Contact`? If not, there is nothing I was replying to → rejected.
2. Has this exact reply text already been kept 3 times? → rejected (duplicate cap).
3. Otherwise take up to 5 previous turns as context and build the record: `system` message, context as `user` / `assistant`, and my reply as the final `assistant` message.

The sample id is a SHA-256 hash of the record's content, so the same input always gives the same id.

## Step 4 — `export` (line 510)

- **Split by session, not by sample (`_is_validation_session`, line 458).** The session id is hashed to a number between 0 and 1; under 0.10 goes to validation. No random seed is needed, and the split is identical on every run.
- **Self-validation (`_validate_outputs`, line 496).** It re-reads the files it just wrote and checks every record: roles are valid, last message is `assistant`, no email / URL / phone / credential pattern is left, no edit tag is left. If anything fails, the build raises an error.
- **Atomic write.** Files are written to a temporary folder and moved into place only after validation passes. A failed build never leaves half-written files.

## Bug log

Each row is recorded in the code comments, the tests, or the previous session.

| # | Symptom | Root cause | Fix | Test that guards it |
|---|---|---|---|---|
| 1 | The text `<تم تعديل هذه الرسالة>` appeared inside 3,142 of 27,877 training records (about 11%) | WhatsApp glues an "edited" tag onto the message text. It is not a separate system line, so the system-notice filter did not see it | `EDIT_TAG_PATTERN` strips it in `_redact_content`, and the validator fails the build if one survives | `test_edited_message_tag_is_stripped` |
| 2 | A login and PIN written as one token (a word followed by digits) was not redacted | `PHONE_PATTERN` deliberately refuses digits glued to letters, so that things like `python3.10` are not redacted | A separate, narrower `CREDENTIAL_LIKE_PATTERN`: 3+ letters followed by 6+ digits | `test_credential_like_token_is_redacted` |
| 3 | A few one-word replies ("اه", "لا", "تمام") appeared hundreds of times as targets | The sample-level dedup hashes context plus reply, so the same reply in a different context is not a duplicate | A cap of 3 samples per exact reply text. It removed 3,274 samples | `test_duplicate_responses_are_capped` |
| 4 | A reply typed in parts with a pause in the middle was split, and the later parts were lost as samples | The burst window was too short, so the second part became a turn of mine with no Contact message before it | Burst window widened to 300 seconds | `test_wider_burst_gap_merges_paused_multipart_reply` |
| 5 | (Prevented, not observed) validation loss would look better than it is | Context windows overlap: sample N's reply is inside sample N+1's context. A random split puts nearly the same text on both sides | Whole sessions are assigned to one side | `test_long_time_gap_breaks_context` covers the session boundary |
| 6 | The same real login from bug 2 was hardcoded in a test and a comment | It was copied from the chat as a test example | Replaced with a synthetic value before the first push | — |

Still open, found while writing this guide: the `--burst-gap-seconds` help text (line 602) mentions `samples_dropped_due_to_burst_gap` in the audit report, but no such field exists. It is a leftover from an earlier version.

## What this stage does not do

Say these yourself in an interview before you are asked.

- Redaction is pattern-based. The build the model was trained on had no name redaction (`explicit_redaction_terms_count: 0`): measured on 15 Oct, **24.1% of training samples contained a private name, 1,511 of them (6.1%) in my own reply**. The v1 SFT run computed its loss on every token (system, Contact and me), so the model learned to write the names in all 24.1%, not only in my replies. That explains the evaluation result: SFT mentioned a private name in 23 of 150 replies, against 10 in my real replies.
- **Fixed 15 Oct:** `--redact-names-file` (whole-word matching from `tools/private_names_lib.py`, و/ب/ل prefixes kept) and a `[NUMBER]` rule for 6–7 digit numbers. Rebuilt into `outputs/redacted/`: 4,057 name mentions replaced, **0 names left** (my scan and `check_private_names.py`), 26,969 samples (20 fewer: replies that became identical after redaction hit the duplicate cap). The current SFT and DPO models were trained on the old files; they only benefit after retraining on the new ones.
- Still not removed: places, plans, health or money described in words, names missing from the list, and everything the other person wrote. That is why the data never leaves my machine and Drive.
- It is one conversation with one person, so the model learns how I talk to that person.
- Dates are assumed day-first. A US-locale export would be parsed wrongly without an error.

## Explain-back questions

Answer in your own words, without opening the code first.

1. Why does one WhatsApp message sometimes take several lines in the file, and how does the parser know a line belongs to the previous message?
2. What would happen to the header regex if `_normalize_header_text` were removed?
3. Why merge bursts at all? What would the training samples look like without merging?
4. Why is the 4-hour session gap needed? Give an example of a bad sample without it.
5. I asked for a 10% validation split, but validation is 2,074 of 26,989 samples, which is 7.7%. Why?
6. Explain to a non-expert why splitting by sample would leak, using the 5-turn context window.
7. Why did the normal dedup not catch the repeated "تمام" replies? Why cap them at 3 and not remove them all?
8. Why is the split done with a hash and not `random.shuffle` with a seed?
9. The pipeline validates its own output and writes atomically. What failure does each of those two prevent?
10. A context window can start with one of my own turns. What role does the first message after `system` have then, and why might that matter for the chat template in fine-tuning?
