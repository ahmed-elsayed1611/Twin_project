# WhatsApp Twin Data Contract

## Purpose

This project converts a raw WhatsApp text export into a **local, privacy-reduced, human-persona fine-tuning dataset**. It models Ahmed as a complete person whose technical interests may appear in conversation, rather than defining him by a job title.

## Input Contract

The input is a UTF-8 WhatsApp `.txt` export. The parser is configured for day-month-year Arabic-locale headers such as:

```text
١‏/١‏/٢٠٢٤، ١٠:٣٠ ص - Ahmed: صباح الخير
```

The implementation removes invisible bidirectional marks before parsing, translates Arabic-Indic digits, supports Arabic and English AM/PM markers, merges continuation lines, and ignores WhatsApp system notices.

## Processing Rules

| Stage | Rule | Reason |
|---|---|---|
| Sender handling | Retain the configured self sender as `Ahmed`; map every other sender to `Contact`. | The model learns Ahmed’s responses without exposing contact identities. |
| Content cleaning | Remove system/media/deleted-message notices and normalize whitespace. | Removes non-conversational training noise. |
| PII reduction | Replace phone numbers, email addresses, URLs, credential-like tokens (a word glued to 6+ digits, e.g. a shared login/PIN), 6–7 digit numbers (codes, account numbers, amounts), and explicitly supplied sensitive terms with placeholders. | Reduces accidental exposure in training files. |
| Name redaction | With `--redact-names-file`, replace every name in the private names list with `[NAME]`, as a whole word, keeping an attached و/ب/ل prefix. The build fails if a listed name survives. | Sender labels alone left names inside messages: before this rule, 24.1% of training samples (6.1% of the replies the model learns to write) contained a private name. |
| Turn construction | Merge same-sender bursts that are no more than 300 seconds apart. | Preserves the meaning of short WhatsApp message bursts, including a multi-part reply typed with a short thinking pause in the middle. |
| Session boundary | Start a new session after four hours with no messages. | Prevents unrelated conversations from sharing context. |
| Sample creation | Keep a sample only when `Contact` is followed by `Ahmed`. | Makes Ahmed’s actual message the supervised target. |
| Response repetition cap | Keep at most 3 samples with the exact same response text. | Prevents a handful of generic short acknowledgements (e.g. "اه", "تمام") from dominating the response distribution. |
| Context | Include up to five previous turns. | Teaches conversational flow without excessive overlap. |
| Split | Assign whole sessions deterministically to train or validation. | Prevents overlapping conversation windows from leaking across splits. |

## Output Contract

The pipeline creates the following files under `outputs/`.

| File | Format | Intended use |
|---|---|---|
| `train.jsonl` | Chat `messages` records | Primary fine-tuning input. |
| `validation.jsonl` | Chat `messages` records | Evaluation during fine-tuning. |
| `all_samples.json` | Instruction/response records | Easy inspection and compatibility with older instruction-format loaders. |
| `audit_report.json` | JSON statistics | Evidence of parser behavior, cleaning, filtering, and split counts. |
| `rejections.jsonl` | Redacted rejection metadata | Debugging rejected sample reasons without retaining rejected message text. |

Each chat-format record has a neutral system message, the sanitized conversation history as role-based messages, and Ahmed’s response as the final assistant message. Raw timestamps and raw contact names are not written to training files.

## Privacy Boundary

This is **privacy reduction, not a guarantee of anonymization**. The dataset still contains personal conversation text. Keep the raw export and generated dataset local, do not upload them to a public repository, and always build with `--redact-names-file` (the private names list) before sharing data with any training provider. Use `--redact-term` only for other literal terms: it matches inside longer words, so a short name passed there would also damage everyday words.

Name redaction removes the listed names only. Places, plans, health, money and relationships described in words, a name missing from the list, and everything the other person wrote are still in the text.

Automated PII redaction (email, URL, phone, credential-like tokens) is pattern-based and will not catch everything — for example, a phone number or reference code that doesn't match the expected shape, or a sensitive word with no digits attached at all. Treat `audit_report.json`'s `pii_redactions` counts as a lower bound, not a guarantee, and spot-check `all_samples.json` directly before treating the dataset as safe to use.

## Expected Command

```bash
./scripts/run_from_raw.sh /absolute/path/to/whatsapp_chat.txt.txt
```

The command first runs the test suite and then produces a validated dataset locally.

## Acceptance Criteria

A successful run exits with status code `0`, writes all five output artifacts, reports a non-zero `samples_total` in `audit_report.json`, and produces no contact names in sender fields.

> The exact sample count is intentionally not treated as a success criterion. It depends on configuration choices such as burst merging, minimum response length, time-gap boundaries, and explicit redaction terms.

## References

[1] [WhatsApp Help Center — Export your chat history](https://faq.whatsapp.com/1180414079177245/)
[2] [Hugging Face TRL — Dataset formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
[3] [NIST — De-identification of Personal Information](https://csrc.nist.gov/publications/detail/sp/800-188/final)
