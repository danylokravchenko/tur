# Speculative decoding

Speculation proposes several next tokens for a decoding sequence and verifies them all in
one pass. Each accepted draft is a token that needed no pass of its own. It is opt-in:
it raises throughput when drafts are often right, but tokens then arrive in bursts.

## Proposals

- **Prompt lookup** (`--speculative-draft N`). The longest suffix n-gram (1–4 tokens) of
  the request's own prompt and output that occurred earlier proposes the tokens that
  followed it. A continuation that reaches the end of the history repeats with the period
  between the two occurrences, so a short cycle fills the draft limit.
- **The native MTP head** (`--mtp`). Qwen3.5 checkpoints ship a one-layer
  multi-token-prediction head. It joins the normalized embedding of the token at
  `p + 1` with the final hidden state at `p`, runs one attention layer, and predicts the
  token at `p + 2`. Tur runs the head after each pass over the newly committed positions,
  writing its own keys and values, and keeps its draft for the next pass when prompt
  lookup has no proposal or a less accepted one.
- **The Gemma 4 assistant** (`--draft-model`). Google publishes a four-layer drafter for
  each Gemma 4 size. It joins the target's scaled embedding of the token at `p + 1` with
  the target's final hidden state at `p`, and attends at position `p + 1` over the
  target's own K/V: sliding layers read the last sliding owner's last `window + 1` keys,
  full layers the last full owner's. It writes no state. Its logits cover only the
  tokens of its 32 best-scoring vocabulary clusters (4,096 of 262,144 on E2B), and the
  pipeline uses it like the MTP head. Its drafts chain: the next step joins the
  embedding of its greedy draft with its own state through `post_projection`, at the
  same position and over the same keys, as Transformers' assisted generation does.

## Verification is exact

The engine requests logits at every drafted position, and uses **target matching**:

1. Draw from the target's full distribution, after penalties, temperature and
   truncation, at the first drafted position.
2. If the draw equals the draft, accept it and continue with the next position.
3. On the first mismatch, publish the target's draw and discard the remaining drafts.
4. If every draft matches, also publish the last row's draw as a bonus token.

For a deterministic proposal `d`, acceptance happens with probability `p(d)`, and a
mismatching draw already follows `p(x | x ≠ d)`. The published tokens therefore follow
the target distribution exactly: same seed, same tokens as ordinary decoding. There is no
stochastic `min(1, p/q)` correction and there are no draft trees.

Rolling back is selecting a boundary. Every recurrent layer keeps a slot at each
candidate boundary, and attention pages past the accepted one are released. RNG,
penalty and grammar bookkeeping advance only for published tokens.

## Adapting to cost

Drafting pays only while expected accepted tokens outpace the extra verification cost:

- Each request keeps an online estimate of its per-draft acceptance, separately for each
  draft source.
- `DraftCosts` fits measured device time against drafted tokens, separately for each
  count of decoding sequences.
- Planning drafts the most likely tokens first and stops where the pass's expected
  tokens per second would fall. Batching that makes verification slower therefore
  disables speculation by itself.

For the MTP head:

- Each draft carries the head's own probability, calibrated online against observed
  acceptance.
- The head's logits are computed only while its acceptance is at least 0.3, for at most
  four decoding sequences, and only while predicted to pay. That acceptance estimate
  counts only each verification's first head draft.
- The assistant chains drafts only when one sequence decodes, and only while the
  probability of reaching the next draft (the acceptance rate to the power of its depth)
  stays at least 0.3. Verification then keeps the chain's leading drafts while their
  calibrated acceptances multiply to at least 0.3.
- A head skipped as unprofitable is retried after 16 passes, and the interval doubles up
  to 256 while it keeps losing.

## Constraints

- Requires F32 recurrent state. With F16 or BF16 state the pipeline logs a warning and
  decodes normally, because a multi-token verification keeps F32 intermediates between
  drafts.
- MTP disables prefix reuse (see [Prefix reuse](prefix-reuse.md)).
- Grammar-constrained requests never draft, and EOS is never proposed.
- Drafting uses extra state memory (a recurrent slot per candidate boundary and an
  output row per draft), which the memory plan reserves up front.

## Decisions

Measured on Qwen3.5-0.8B on an M1:

- **Exact target matching over draft-model sampling.** Outputs are identical to ordinary
  decoding; 1,278 same-seed request comparisons produced identical token IDs.
- **Opt-in throughput mode.** Prompt lookup reached 3.25× on repetitive single-request
  output, but delivers tokens in bursts with longer pauses, which raises p95 inter-token
  gaps.
- **MTP drafts from a reduced vocabulary.** The head projects onto the 32,768 lowest token
  IDs, through a copy of the LM head's leading rows. Byte-level BPE numbers tokens in
  merge order, so these are the most frequent. This cuts a head pass from about 6.5 ms
  to 2.4–3.1 ms at unchanged acceptance; 16K lost prose drafts and 64K added cost.
- **Calibrated confidence enables sampled MTP.** Drafts with confidence ≥ 0.8 were 99%
  accepted, and those below 0.4 only 20%.
- **Lookup plus MTP gives 1.12–1.38× ordinary decoding** on novel prose and code at one
  to four sequences, greedy and sampled (1.09–1.36× over the best lookup-only policy).
- **The head is gated off above four decoding sequences.** At eight, a draft costs about
  2.4 ms against 3.25 ms per decode token, and the head's upkeep costs 3–5%. For the same
  reason a finer cost model was not built: it could not gain more than about 10%.
- **No learned drafter.** No smaller Qwen3.5 checkpoint exists to serve as a draft model,
  and no EAGLE-3, DFlash or DSpark drafter targets these checkpoints.
- **Gemma's assistant chains drafts at one decoding sequence.** Against one draft per
  pass on E2B (Q8_0, F16 K/V, greedy, 96 tokens, lookup plus the assistant, two
  alternating rounds): novel prose 31.6–33.8 against 26.3–29.9 tokens/s (1.14×),
  repetitive text 50.0–50.5 against 33.2–33.6 (1.50×), code 27.1–29.8 against
  28.7–29.9 (within noise). At four sequences chains lost 6% on prose and 13% on code,
  so above one sequence the assistant drafts one token per pass. Every run reproduced
  the token IDs of single drafts.
- **The assistant against ordinary decoding,** one draft per pass. On E2B (Q8_0, F16 K/V, greedy,
  96 tokens, two repeats) lookup plus the assistant against ordinary decoding: novel
  prose 28.5–28.9 against 20.0–20.3 tokens/s at one sequence (1.42×) and 58.9–61.1
  against 48.6–48.8 at four (1.23×); code 23.3–26.1 against 19.9–20.1 (1.17–1.30×) and
  64.1–65.4 against 47.7–48.1 (1.35×). On repetitive prompts, where lookup alone reaches
  3.2×, it changes lookup by −7% to +5%. 63–66% of its drafts were accepted, a draft
  costs about 2.9 ms against a 47 ms decode, and all 36 runs reproduced ordinary
  decoding's token IDs.
- **Next candidate:** chained MTP drafts, estimated at about 10% at one sequence (see
  [Open work](../design/open.md)).
