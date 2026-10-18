# Chunking

Chunking cuts each document section into passages small enough to embed and retrieve precisely, yet large enough to carry a complete piece of information. The chunk is the unit that is embedded, stored, retrieved and cited.

## Where chunking sits

```
Document ─► sections (page / sheet / heading) ─► chunker, per section ─► ChunkRecord(text, source, start, end, metadata)
```

`SearchIndex` chunks **each section separately**, so a chunk never spans two pages or two sheets and inherits that section's metadata (`page`, `sheet`, `heading`). Offsets `start`/`end` are relative to the section's normalized text.

## Chunkers

| | `FixedSizeChunker` | `RecursiveChunker` (default) |
|---|---|---|
| Splits at | Every `chunk_size` characters | The last paragraph, line, sentence or word break that fits |
| Respects sentences | No; cuts mid-word | Yes, when a separator exists in the window |
| Overlap | `overlap` characters, step = `chunk_size - overlap` | `overlap` characters, moved forward to a word boundary |
| Defaults | 200 / 0 | 300 / 50 (used by `SearchIndex`) |
| Use | Baseline and comparison | Everything else |

Both reject `overlap >= chunk_size`: with a step of zero, `FixedSizeChunker` would never advance.

## How `RecursiveChunker` works

1. **Normalize.** Runs of spaces and tabs become one space and spaces around newlines are trimmed, but **line and paragraph breaks are kept**; three or more newlines become two.
2. **Take a window** of `chunk_size` characters from `start`.
3. **Back off to a natural boundary.** If the window doesn't reach the end of the text, find the **last** occurrence of the first separator that appears in it, in priority order `"\n\n"` → `"\n"` → `". "` → `" "`, and cut **after** it, so the chunk keeps its full stop.
4. **Advance.** The next chunk starts at `end - overlap`, moved forward to the next whitespace so it never begins mid-word.

```
text:     [ paragraph A ........................ ][ paragraph B ......... ]
window:   [------------- chunk_size -------------]
cut:      [ paragraph A ...................... ]↑ last "\n\n" in the window
next:                                   [overlap → paragraph B ...........]
```

Guarantees, all covered by tests in `tests/test_chunking.py`:

- **No gaps:** consecutive chunks touch or overlap, the first starts at 0 and the last ends at the text's end.
- **Paragraph-aware:** a paragraph break wins over a sentence break.
- **Word boundaries:** no chunk starts in the middle of a word.
- **Punctuation kept:** a chunk cut at a sentence ends with its full stop.

The name follows LangChain's `RecursiveCharacterTextSplitter`. This implementation does not recurse into oversized pieces; it backs off through the separators in priority order within one window.

### Bugs fixed in this chunker
Earlier versions had three silent defects, found through the evaluation benchmark rather than unit tests:

| Bug | Effect |
|---|---|
| The next chunk started at `start + chunk_size - overlap` even after an early cut | 269 characters fell between chunks and were never indexed; one benchmark answer was unreachable |
| All whitespace, including newlines, was collapsed **before** searching for `"\n\n"` | Paragraph splitting never happened |
| Overlap started at an arbitrary character | Chunks began mid-word ("…ased alloy"), cutting keywords and evidence phrases |

## Choosing size and overlap

Sweep on the 30-question benchmark (default hashing embedder, hybrid retrieval). This is **analysis only**: the benchmark is also the test set, so defaults are not picked from it.

| chunk_size / overlap | Chunks | hit@1 | recall@3 | MRR@3 |
|---|---|---|---|---|
| 150 / 30 | 94 | 0.467 | 0.811 | 0.639 |
| 200 / 40 | 67 | 0.600 | 0.922 | 0.756 |
| 300 / 0 | 30 | 0.967 | 0.989 | 0.983 |
| **300 / 50 (default)** | 31 | 0.933 | 0.989 | 0.967 |
| 500 / 50 | 17 | 0.933 | 0.989 | 0.967 |
| 800 / 50 | 11 | 0.933 | 0.967 | 0.944 |

How to read it:

- **Too small hurts the most.** At 150 characters a chunk holds part of a sentence: it loses context, and evidence phrases get split across chunks. Some of this drop is how the benchmark scores (a split evidence phrase counts as a miss), but a fragment is also genuinely weaker evidence for an LLM.
- **Around one paragraph works best here.** The sample notes have paragraphs of 130–289 characters (median 193), so a 300-character window holds a whole paragraph and the cut lands on the paragraph break: one chunk ≈ one paragraph ≈ one idea.
- **Too large dilutes.** At 800 characters several topics share one chunk, so its embedding is an average of topics and recall@3 starts to drop. Large chunks also cost more prompt tokens at generation time.
- **Overlap is a trade-off.** 300/0 scored one question higher than 300/50. When chunks already end at paragraph breaks, overlap mostly creates near-duplicate fragments. Overlap helps when an answer straddles a boundary, which is more common with long paragraphs or fixed-size cuts. The default keeps a modest overlap for that case; the difference here is one question (0.033).

Practical guidance:

| Content | Starting point |
|---|---|
| Short technical notes, FAQs, procedures | 200–400 characters, ~10–20 % overlap |
| Long narrative documents, manuals | 500–1,000 characters, more overlap |
| Tables (labelled rows) | One or a few rows per chunk; rows are self-describing, so little overlap is needed |

Whatever the setting, keep chunks within the embedding model's input limit (all-MiniLM-L6-v2 truncates beyond 256 word pieces, roughly 1,000 characters of English) and re-run `scripts/evaluate.py` after any change.

## Chunk statistics on the sample corpus

With the default 300/50 over the non-PDF samples in `data/raw/` (11 documents): **40 chunks, median 226 characters, range 45–289.** The single chunk under 80 characters is an overlap tail ("work rather than from the annealed condition."), the kind of fragment that can outrank the full passage.

## Limitations

| Limitation | Effect | Possible fix |
|---|---|---|
| No minimum chunk length | A separator early in the window, or an overlap tail, yields a very short chunk | Merge chunks below a threshold into their neighbour |
| Character-based size | Size in characters only approximates model tokens | Count tokens with the embedding model's tokenizer |
| No structure awareness beyond sections | Lists and tables can be cut between related lines | Chunk tables by rows; keep list items together |
| Overlap fragments | Near-duplicates compete in ranking | Reranking, or de-duplicating overlapping hits at query time |
| Offsets refer to normalized section text | Not byte positions in the original file | Map back through the normalization if exact highlighting is needed |
