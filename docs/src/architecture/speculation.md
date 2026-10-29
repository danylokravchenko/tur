# Speculative decoding

Decoding one token per pass leaves the GPU underused, because the weights are read for a single row of work. Speculative decoding guesses several next tokens cheaply and checks them all in one pass. Every guess that turns out right is a token that didn't need a pass of its own. It's opt-in: it raises throughput when guesses are often right, but tokens then arrive in small bursts.

## Where drafts come from

Prompt lookup (`--speculative-draft N`) looks for the longest recent run of 1 to 4 tokens that also appeared earlier in the request's prompt or output, and proposes whatever followed it last time. It costs almost nothing and works well when output repeats its input, as in code edits, extraction or JSON with repeated keys. If the continuation reaches the end of the history, it repeats with the period between the two occurrences, so a short cycle can fill the whole draft.

The MTP head (`--mtp`) ships inside Qwen3.5 checkpoints. It is a single extra layer that combines the embedding of the token at `p + 1` with the model's final hidden state at `p` and predicts the token at `p + 2`. Tur runs the head after each pass over the newly committed positions, writing its own keys and values, and uses its draft when prompt lookup has nothing better.

The Gemma 4 assistant (`--draft-model`) is a four-layer drafter Google publishes for each Gemma 4 size. It combines the target's embedding of the token at `p + 1` with the target's final hidden state at `p`, and attends at position `p + 1` over the target's own keys and values, so it needs no state of its own. Its output covers only the tokens in its 32 best-scoring vocabulary clusters (4,096 of 262,144 on E2B), which keeps it cheap. With one request decoding, it chains drafts: each next step feeds its own greedy draft back in at the same position, as Transformers' assisted generation does.

## Verification is exact

The engine asks for logits at every drafted position and checks drafts by target matching:

1. Draw a token from the model's full distribution at the first drafted position, after penalties, temperature and truncation.
2. If it equals the draft, accept it and move to the next position.
3. At the first mismatch, publish the model's own draw and discard the remaining drafts.
4. If every draft matches, also publish the draw at the last row as a bonus token.

A draft `d` is accepted with exactly the probability the model gives it, and a mismatching draw already follows the model's distribution over everything else. So the published tokens follow the model's distribution exactly: the same seed gives the same tokens as normal decoding. Unlike classic speculative sampling, there is no `min(1, p/q)` correction and no draft tree.

Undoing rejected drafts means picking a boundary. Every recurrent layer keeps a slot at each candidate position, and attention pages past the accepted one are released. The random stream, penalties and grammar state advance only for published tokens.

## Drafting only when it pays

Verifying drafts costs GPU time, and a wrong draft wastes it. Tur keeps track of both sides:

- Each request estimates how often its drafts are accepted, separately for each draft source.
- `DraftCosts` fits measured GPU time against the number of drafted tokens, separately for each number of decoding requests.
- Planning drafts the most likely tokens first and stops where expected tokens per second would start to fall. When a larger batch makes verification more expensive, speculation turns itself off.

The MTP head and the Gemma assistant have extra gates:

- Each draft carries the head's own probability, calibrated online against how often such drafts were accepted.
- The head runs only while its acceptance rate is at least 0.3, with at most four requests decoding, and only while it is predicted to pay. Its acceptance estimate counts only the first head draft of each verification.
- The assistant chains drafts only when one request is decoding, and only while the chance of reaching the next draft stays at least 0.3. Verification keeps the leading drafts of a chain while their calibrated acceptance rates multiply to at least 0.3.
- A head turned off as unprofitable is retried after 16 passes, and the interval doubles up to 256 while it keeps losing.

## Constraints

- Speculation needs F32 recurrent state. With F16 or BF16 state, the pipeline logs a warning and decodes normally, because verifying several tokens keeps F32 intermediate state between drafts.
- MTP turns off prefix reuse (see [Prefix reuse](prefix-reuse.md)).
- Grammar-constrained requests never draft, and the end-of-sequence token is never proposed.
- Drafting needs extra state memory, a recurrent slot per candidate boundary and an output row per draft, which the memory plan reserves up front.

## Decisions

Measured on Qwen3.5-0.8B on an M1 unless noted:

- Target matching over draft-model sampling. Outputs are identical to normal decoding; 1,278 same-seed comparisons produced identical tokens.
- Speculation is an opt-in throughput mode. Prompt lookup ran 3.25× faster on repetitive single-request output, but tokens arrive in bursts with longer pauses between them, which hurts the slowest token gaps.
- The MTP head drafts from a reduced vocabulary: the 32,768 lowest token IDs, through a copy of the output layer's first rows. Byte-level BPE numbers tokens in merge order, so these are the most frequent ones. This cut a head pass from about 6.5 ms to under 3 ms with unchanged acceptance; 16K tokens lost prose drafts and 64K added cost.
- Calibrated confidence makes MTP useful with sampling. Drafts with confidence of at least 0.8 were 99% accepted, and those below 0.4 only 20%.
- Lookup plus MTP gives 1.12 to 1.38× normal decoding on new prose and code with one to four requests, greedy and sampled.
- The head turns off above four decoding requests. At eight, its upkeep costs 3 to 5% and a draft costs nearly as much as decoding the token. For the same reason, a finer cost model wasn't built: it could gain at most about 10%.
- There is no learned drafter for Qwen. No smaller Qwen3.5 checkpoint exists to draft with, and no EAGLE-3, DFlash or DSpark drafter targets these checkpoints.
- The Gemma assistant drafts one token per pass for several requests. On E2B (Q8_0, F16 K/V, greedy), lookup plus the assistant beat normal decoding by 1.42× on prose and 1.17 to 1.30× on code with one request, and 1.23× and 1.35× with four. On repetitive prompts, where lookup alone gives 3.2×, it changes little. About two thirds of its drafts were accepted, at about 2.9 ms each against a 47 ms decode, and all 36 runs reproduced normal decoding's tokens.
- The Gemma assistant chains drafts with one request decoding. Against one draft per pass, chains were 1.14× faster on prose and 1.50× on repetitive text, and level on code. With four requests they lost 6% on prose and 13% on code, so chaining is limited to one request. Every run reproduced the single-draft tokens.
- Chained MTP drafts are the next candidate, estimated at about 10% with one request (see [Open work](../design/open.md)).
