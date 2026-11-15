# Financial annotation profiles and qualified-symbol source mentions

The optional search/answer parameter `nlp_profile` accepts exactly `v1` or `v2`.
The default remains `v1`. `v2` requires `extract_facts: true`; unknown profiles,
wrong types and incompatible settings fail before any persistence. The feature
adds no dependency, model download, network request, external service or identity
lookup. These are original finite mention rules, not pretrained NLP or an issuer
or security master.

`nlp_v1.py` is the published parser copied without changes. Its numeric/currency,
event, entity and citation behavior is frozen, including its earlier limited
qualified-symbol matching. `nlp.py` uses explicit allowlisted dispatch, never a
module name supplied by the caller or a "latest" alias. `nlp_v2.py` invokes the
frozen numeric/currency/tenor/event rules and replaces only entity extraction.
New parser behavior must receive a new explicit profile; changing v1/v2 in place
would break the replay contract.

Default and explicit v1 snapshot bodies omit `nlp_profile` entirely. This keeps
old serialized annotation output and saved identities byte-for-byte identical,
including queries without annotation and existing hybrid queries. v2 bodies save
`nlp_profile: "v2"` and `gig.financial-mention-rules.v2` in the annotation result.
Replay explicitly selects the stored profile; an absent historical profile means
v1 regardless of future default choices. Unknown stored profiles fail closed.
Saved latent models continue replaying without refitting. No schema migration or
relabeling of old snapshots is performed.

## Exact syntax and boundaries

v2 recognizes literal uppercase `NYSE:` or `NASDAQ:`, optional ASCII spaces/tabs,
then one to five ASCII uppercase letters, optionally a dot and one ASCII
uppercase share-class letter. For example, `NYSE:BRK.B` and `NASDAQ:ZZZZZ.Z` are
syntactically supported source mentions, whether or not such an instrument
exists. Hyphen classes, multi-letter suffixes, warrants, preferred/rights syntax,
other namespaces, bare symbols and permanent identity resolution are outside
this profile.

Extraction first consumes the complete non-delimited candidate, then validates
its whole syntax. Quotes (including curly quotes and guillemets), brackets,
whitespace and `,;!?` delimit candidates. One terminal full stop is treated as
sentence punctuation; periods inside candidates participate in validation.
A prefix must start at a delimiter or the passage start, so an embedded namespace
cannot produce a ticker subfragment. Unicode letters, homoglyphs, combining marks,
zero-width suffixes, `/WS`, `.BOGUS`, repeated dots and other unsupported suffixes
cannot produce a shortened ticker mention. Entire candidates longer than
32 characters (including namespace/spacing) are unsupported. The parser scans at
most the existing 20 retrieved passages of at most 4,096 characters each; it never
truncates a long candidate into a valid symbol.

Supported qualified entities retain `mention` and `provenance`, and add:

| Field | Meaning |
| --- | --- |
| `mention_kind` | `EXCHANGE_QUALIFIED_SYMBOL_SOURCE_MENTION` |
| `namespace_mention` | Literal source `NYSE` or `NASDAQ`; no authority assertion |
| `symbol_mention` | Complete validated source token, e.g. `BRK.B` |
| `share_class_mention` | Literal one-letter suffix or null |
| `link_state` | `UNKNOWN_NO_ISSUER_AUTHORITY` |
| `instrument_identity_state` | `UNKNOWN` |
| `issuer_linking_state` | `UNKNOWN` |
| `exchange_membership_state` | `UNKNOWN` |
| `existence_state` | `UNKNOWN` |

Unsupported candidates add `{reason, provenance}` to `unsupported_mentions` with
`UNSUPPORTED_QUALIFIED_SYMBOL_SYNTAX` or `QUALIFIED_SYMBOL_BUDGET_EXCEEDED`.
For exact authored UTF-8 lines, source offsets are original byte ranges and quote
boundaries exclude enclosing punctuation. Normalized passages retain the existing
explicit raw-passage citation semantics; they do not claim direct source bytes
for rendered text. General Treasury/Federal Reserve mentions keep their prior
unknown-link schema.

All event occurrences remain `UNKNOWN`; sentiment remains `UNAVAILABLE`; bare
`$` remains `UNKNOWN`/`UNSPECIFIED_DOLLAR`. A known namespace, share-class syntax,
nearby number or event keyword establishes no issuer link, exchange membership,
permanent instrument identity, financial truth or tradable observation.

## Acceptance

`tests/test_financial_nlp_profiles.py` pins pre-change serialized v1 annotation
hashes and lexical/annotated/hybrid saved IDs, explicit-v1/default equivalence,
legacy/v2 replay under changed dispatcher defaults, saved-model replay without
refitting, strict API/profile/resource rejection, Unicode byte quote boundaries,
whole-candidate rejections, unknown authority and the original numeric attacks.
The existing financial intelligence and latent/NLP tests remain unchanged.
