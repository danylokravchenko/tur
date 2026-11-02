# Prefix reuse

Many requests start with the same tokens: a shared system prompt, a long document several users ask about, or an agent conversation that grows by one step at a time. Prefix reuse computes that shared start once and lets later requests resume from it, so their first token comes sooner and the GPU spends its time on new work.

In a plain transformer you can cache keys and values and cut them back to any length. A hybrid model's recurrent state doesn't work that way: the state after token `n` exists only if a pass ended exactly at `n`. So Tur stores snapshots at exact boundaries. Each `PrefixCache` entry is a committed `ModelState` (a page table plus a recurrent slot) after some request's first `n` tokens.

## Where snapshots are taken

- At every multiple of the spacing, 256 tokens by default. The scheduler never lets a prompt chunk cross one of these, so the boundaries always exist.
- At the end of each prompt, together with the logits of its last token. An identical prompt can then sample its first output without running the model at all.
- At the point where the chat template will rewrite the conversation, for prompts that already contain assistant replies (explained below).
- At the last token of each finished request, so a conversation's next turn can resume right after the reply.

Chat templates often render the previous assistant turn differently once a new user turn follows it, for example by dropping its thinking section. The next request then matches the old one only up to that point. To find it, the pipeline renders a probe with an empty assistant reply and a new empty user turn, and takes the longest prefix it shares with the actual prompt as an extra chunk boundary. First turns, prompts without assistant history, and templates that reject the probe keep the ordinary boundaries, so they pay nothing extra. The prediction only decides where to cut; a restore still checks every token of the next request.

A snapshot shares its pages and slot with the request that produced it, so taking one copies nothing. It keeps one recurrent slot and its pages alive. If a snapshot ends partway through a page, the first request to append there copies that one page.

## Lookup

Keys are a chained hash over the tokens, seeded by a namespace: the tenant, or the request's `cache_salt`. Equal prefixes in different namespaces never meet, and a hash match is used only after comparing every token. The pool records the model's identity and state layout, so entries can't cross models, quantizations or positional settings.

A lookup probes only the lengths where entries exist, so its cost depends on the number of entries rather than the prompt's length.

## Waiting for a producer

When several requests reach the same uncached prefix together, only the oldest computes it. The others wait until its snapshot exists and then restore it, instead of each computing their own copy. The set of waiters is rebuilt every pass, so if the producer is cancelled or finishes early, nobody stays stuck behind it. The scheduler's queue-age limit bounds the wait.

## Retention and eviction

By default the cache keeps one snapshot per running request (`--prefix-snapshots`), using at most 1/8 of the state pool. Under memory pressure, snapshots go before any running request is preempted.

Eviction removes the entry that saves the least compute per byte it holds. An entry's value is the number of tokens it covers beyond the deepest retained entry it extends, times how often its prefix was recently computed or restored. Its cost is its recurrent slot plus the pages only it holds. Counts are halved periodically, so prefixes that stop recurring lose their protection. A new entry is always kept at first. Pages and slots return to the pool only when no request or entry holds them.

A reply that ends in tool calls is a strong sign that the conversation will return, and its next prompt arrives only after the tools have run, while other traffic may be pushing its snapshots out. So those snapshots go on hold for 60 seconds, and eviction takes every other entry first. How far the next prompt matches depends on how the template renders the reply, so the hold covers the reply's end, then the prompt's end, then the point where history gets rewritten or, on a first turn, the last aligned snapshot. Holds take at most half the snapshots, and a newer hold displaces the oldest. A hold reserves nothing: when memory runs short, held entries still go before any running request is preempted.

## Decisions

Measured on Qwen3.5-0.8B on an M1:

- Snapshots are zero-copy and verified. Requests that restore a prefix generate exactly the text of uncached runs. Eight users behind a cached 1.5K-token system prompt got their first token about 17× sooner than without reuse (0.56 s against 9.5 s at the median).
- Spacing is 256 tokens, chosen from a sweep of 64 to 512. A spacing of 128 cached 12% more tokens in multi-turn chats but didn't improve time to first token.
- Prompt-end snapshots keep their logits, so an exact repeat needs no model pass.
- Requests wait for an in-flight producer. Eight users sending the same new 1.5K-token prompt got their first token 5.6× sooner and computed 5.7× fewer prompt tokens than if each had computed it.
- Eviction ranks by compute saved per byte. Under churn, a recurring prefix survived 3 of 5 waves where least-recently-used eviction kept it in none, and an agent loop cached 83% of its prompt tokens instead of 72%, finishing 1.25× sooner. Low-reuse, decode-only and mixed traffic were unchanged.
- Letting cached requests jump the queue is opt-in. With `engine.prefix_restorable_first`, cached requests got their first token 5.4× sooner under contention (16 requests, 4 places), but uncached ones waited 15% longer and throughput fell 10%. It stays off by default because it slows unrelated requests.
- Tool-calling replies hold their snapshots for 60 seconds, a judgment of how long tools take rather than a measured value. In a tool loop with three users behind a shared system prompt between each call, follow-ups restored all but the new tool result, against at most 1,536 of 2,346 tokens without a hold, and their first token came 3× sooner (206 ms against 641 ms). The other users kept all their cached tokens and their time (4.30 s against 4.32 s).
- Holds are capped at half the snapshots. With no cap, three of the four snapshots were held in that loop, and the other users cached 24% fewer tokens and took 1.7× as long.
- Lookup uses an indexed hash, not a radix tree. A hit at 32K tokens costs 6.6 µs and a miss under 0.1 µs.
- Prefix reuse and the MTP drafter exclude each other. The head's state for boundary B depends on the token sampled after B, which a snapshot at B can't hold, so `--mtp` turns reuse off.
- The template-rewrite boundary is taken once a prompt has assistant history. Over four six-turn conversations with 16 snapshots, it raised the share of cached prompt tokens from 52% to 68%, cut last-turn time to first token by 30%, and finished 11% sooner. With 8 snapshots, eviction limited the gain to 2% on the last turn and nothing on total time. Taking it on first turns too made a single uncached request 15% slower to its first token, so first turns skip it.
