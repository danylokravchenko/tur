# Grammars and structured output

A grammar describes the text a model may write. Before each token, Tur works out which tokens can still lead to a valid result and masks the rest on the GPU, so the output matches the format whatever the temperature. Your code can parse the answer without retries or repair.

A grammar controls the format and not the facts. Asked for the Apollo 11 landing date under a date pattern, Qwen3.5-0.8B once answered `"1969-11-20"`: a well-formed date, and the wrong one.

## What you can do

| Goal | Over HTTP | In the Rust library |
| --- | --- | --- |
| Pick one label from a list | `grammar` with regex, or `json_schema` with `enum` | Regex or Lark |
| Get a value in a set pattern, such as a date | `grammar` with regex, or `json_schema` with `pattern` | Regex |
| Extract records as JSON | `json_schema` | `structured`, or a JSON schema grammar |
| Get any JSON object | `json_object` | `structured` |
| Force a tool call | `tool_choice` | `structured` |
| Keep tool calls valid while the model may also answer in text | `tool_choice: auto` | `tools` |
| Generate a small language, such as restricted SQL | `grammar` with Lark | Lark |
| Mix prose with JSON | `grammar` with Lark | Lark |

## Over HTTP

Every example here goes in the body of `POST /v1/chat/completions`. The [HTTP API](http-api.md#structured-output) page covers each field.

### Labels

A schema doesn't have to be an object. This one classifies a review, and `content` is the JSON string `"negative"`, quotes included:

```json
{
  "messages": [{"role": "user", "content": "Is this review positive or negative? 'The battery died in a day.'"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "label", "schema": {
    "type": "string", "enum": ["positive", "negative", "neutral"]
  }}}
}
```

### Patterns

`pattern` constrains a string with a regular expression:

```json
{"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
```

### Records and alternatives

Schemas can nest, limit their sizes and offer alternatives. This one accepts either shape, and the model chose `{"kind":"circle","radius":2}` when told about a circle:

```json
{"anyOf": [
  {"type": "object", "required": ["kind", "radius"], "additionalProperties": false,
   "properties": {"kind": {"const": "circle"}, "radius": {"type": "number"}}},
  {"type": "object", "required": ["kind", "side"], "additionalProperties": false,
   "properties": {"kind": {"const": "square"}, "side": {"type": "number"}}}
]}
```

A keyword the compiler doesn't support answers 400, so a schema that's accepted is enforced.

### Regex and Lark

`response_format` also takes a grammar directly, in either syntax. This is a Tur extension, shaped like OpenAI's grammar format for custom tools. The text comes back bare, so this label has no quotes around it, unlike a JSON schema's:

```json
{
  "messages": [{"role": "user", "content": "Is this review positive or negative? 'The battery died in a day.'"}],
  "response_format": {"type": "grammar", "syntax": "regex", "definition": "positive|negative|neutral"}
}
```

With `"syntax": "lark"`, `definition` holds a grammar like those [below](#lark), with its own `start` rule. An invalid grammar answers 400. Only the answer has to match: thinking and gpt-oss's channels work as they do for a JSON schema.

### Tool calls

`"tool_choice": "required"` makes the model call one of your tools, and `{"type": "function", "function": {"name": "get_weather"}}` makes it call that one. Under `auto` the model decides, and on Qwen and Gemma any call it starts is held to its schema. Text around the call runs free and can still be drafted.

## In the Rust library

Set `grammar` on a `GenerationRequest`. A `TopLevelGrammar` comes from a regex, a Lark grammar or a JSON schema:

```rust,ignore
use tur::backend::guidance::TopLevelGrammar;

let mut request = GenerationRequest::new("When did Apollo 11 land? Date only.".into(), 32);
request.render.enable_thinking = Some(false);
request.grammar = Some(TopLevelGrammar::from_regex(r"\d{4}-\d{2}-\d{2}"));
let stats = pipeline.run(&request, |_| {})?;
```

The `grammar` field covers everything the model generates, so turn thinking off, or the reasoning has to match it too. On gpt-oss it must also spell out Harmony's channel tokens. To avoid both, set `structured` instead, which fits the grammar to the model's format and returns reasoning apart from the answer. The HTTP API uses it:

```rust,ignore
use tur::backend::structured::StructuredOutput;

request.structured = Some(StructuredOutput::JsonSchema(schema));
request.structured = Some(StructuredOutput::Lark(definition));
```

### Lark

[Lark grammars](https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md) describe languages that JSON schemas can't. Lowercase names are rules and uppercase names are terminals. This one limits a model to simple `SELECT` statements, and Qwen3.5-0.8B answered `SELECT name FROM users WHERE age > 30;`:

```text
start: "SELECT " columns " FROM " NAME (" WHERE " condition)? ";"
columns: "*" | NAME (", " NAME)*
condition: NAME " " OP " " VALUE
OP: "=" | "<" | ">"
VALUE: /[0-9]+/ | /'[a-zA-Z ]*'/
NAME: /[a-z_]+/
```

`%json` embeds a schema in a larger grammar, which lets you mix prose with data:

```text
start: note "\n" data
note: "Note: " /[^\n]{1,200}/
data: %json {"type": "object", "required": ["city"], "additionalProperties": false, "properties": {"city": {"type": "string"}}}
```

Special tokens have no text the grammar can match, so write them by id: `<[248069]>` is that one token, and `<[^248069]>*` is any run of tokens without it. Look ids up with the tokenizer, since they differ between models.

## Writing grammars that work

A mask only removes choices, and the model still picks among what's left. Most bad results come from a schema and a prompt that disagree.

- Say what you want in a message as well. The model isn't told about the schema, so asking for the fields by name gives it a reason to write them.
- Bound what can grow. Told only to "describe Ada Lovelace", Qwen3.5-0.8B wrote `"born":18151010…` under a plain integer until `max_tokens`, because it wanted to continue and digits were still legal. With `"maximum": 2100` it wrote `1815` and stopped. Use `maximum`, `maxLength` and `maxItems`, or `{1,200}` in a regex.
- Leave room in `max_tokens`. Running out cuts the output off mid-value with `finish_reason` set to `length`, so check for that before parsing. Reasoning counts too.
- Check the content. An empty list satisfies a list schema: `{"people":[]}` came back for a sentence naming three people.

## What it costs

Masks run on the GPU in the same pass as sampling, so a constrained token costs about what an unconstrained one does. Tokens the grammar alone decides, such as JSON keys, need no sampling and run together in one pass. On JSON with long keys that made Qwen3.5-0.8B on an M1 3× faster with one request and 1.8× with eight, with tokens arriving in bursts.

Constrained tokens are never drafted, so `--speculative-draft` doesn't speed them up, and `logprobs` turns the shortcut for decided tokens off. The first grammar on a server takes about 0.6 s longer, while Tur indexes the tokenizer, and every other request waits for that. [Sampling](../architecture/sampling.md) explains how this works and has the measurements.

## Limits

- Parser limits bound what a grammar can cost. One that compiles can still exceed them partway through generation, which answers 400 with the code `invalid_grammar`, or an error event if the response is streaming.
- Special-token ids differ between models, so a grammar that writes them by id works only on the model it was written for.
- `response_format` can't be combined with `tools` unless `tool_choice` is `none`.
- On Gemma, forced tool calls accept only `type`, `enum`, `properties`, `required`, `items` and annotations. Its call syntax isn't JSON, so Tur translates schemas itself.
- On Qwen, string parameters of forced calls are free text unless they have an `enum`, so `pattern` and `maxLength` don't apply to them.
- Under `auto`, gpt-oss calls aren't constrained, because they have no opening token to wake a grammar on.
