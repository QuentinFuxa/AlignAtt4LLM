# MT server protocol (v1)

`alignatt-mt-server` serves the AlignAtt MT policy to external ASR frontends
over WebSocket. The client owns transcription; the server owns translation.
This document is the normative protocol spec. WhisperLiveKit
(`--translation-backend alignatt`) is the reference client.

```bash
alignatt-mt-server --preset gemma_low_latency --port 8765
```

All frames are JSON text. One WebSocket connection = one translation session.

## Session lifecycle

1. Client connects and sends `init`.
2. Server replies `init_ok` (or `error`, then closes).
3. Client streams `update` messages; the server answers each serviced update
   with a `translation` message.
4. Plain WebSocket close ends the session. The client should send its last
   update with `is_final: true` before closing if it wants an end-of-stream
   quality pass.

## Client to server

### `init`

```json
{
  "type": "init",
  "protocol_version": 1,
  "source_lang": "en",
  "target_lang": "de",
  "preset": "gemma_low_latency",
  "overrides": {"translation_alignatt_inaccessible_ms": 0},
  "context_text": "Talk title, glossary, or domain hints (optional)",
  "history": [["Previous sentence.", "Vorheriger Satz."]],
  "accepted_target_prefix": ""
}
```

- `protocol_version` (required): must be `1`.
- `source_lang`, `target_lang`: language codes. The server validates the
  direction against the calibrated alignment-heads files it ships and
  returns `unsupported_direction` with the `supported` list otherwise.
- `preset` (optional): server-side runtime preset name.
- `overrides` (optional): per-session knobs; only whitelisted keys apply
  (see `CLIENT_OVERRIDE_WHITELIST` in `mt_server.py`), unknown keys are
  ignored silently.
- `context_text` (optional): free-text domain context injected into the MT
  prompt outside the source-marked region (it is never translated).
- `history` (optional): `[source, target]` pairs restored after a reconnect
  so translation-history windows keep working.
- `accepted_target_prefix` (optional): the partial target text already shown
  to the user for the open utterance before a reconnect. The server continues
  from it and never re-emits or contradicts it.

### `update`

```json
{
  "type": "update",
  "seq": 42,
  "utterance_id": 3,
  "words": [["Hello", 120.0, 400.0], ["world,", 520.0, 900.0]],
  "tail": {"words": [["how", null, null], ["are", null, null]]},
  "clock_ms": 2350.0,
  "is_final": false
}
```

- Updates carry the FULL state of the open utterance, not deltas: `words` is
  every committed word of the utterance, in order, as
  `[text, start_ms, end_ms]` (timestamps may be `null`). A newer update
  supersedes an older unserviced one losslessly, which is what makes
  server-side coalescing and reconnects trivial.
- `tail.words` (optional): the ASR's unstable hypothesis tail. The MT model
  drafts over it, but AlignAtt only commits target tokens whose attention
  lands on committed words. Feed it when the upstream ASR is append-only
  (e.g. the qwen3 causal backend); omit it for hypothesis-churning backends.
- `clock_ms`: monotone stream clock (the audio head position). Must be at
  least the largest committed `end_ms`.
- `is_final: true` closes the utterance: the server runs a full-quality
  translation of the whole utterance (reusing the streamed partial as a
  prefill) and the next update must use `utterance_id + 1`.
- `seq`: client-chosen monotone integer, echoed back. Because partial
  updates coalesce (latest wins), some `seq` values never get a response.
  `is_final` updates are sticky and always processed, in order.

## Server to client

### `init_ok`

```json
{"type": "init_ok", "protocol_version": 1, "direction": "en-de",
 "preset": "gemma_low_latency", "model": "gemma_vllm_alignatt"}
```

### `translation`

```json
{
  "type": "translation",
  "seq": 42,
  "utterance_id": 3,
  "final": false,
  "committed_text": "Hallo Welt,",
  "committed_delta": " Welt,",
  "buffer_text": "wie",
  "covered_source_units": 2,
  "stop_reason": "alignatt:commit_fast_path"
}
```

- `committed_text`: cumulative accepted target for the open utterance.
  Append-only across the utterance: each value extends the previous one
  (`previous + committed_delta == committed_text`, delta verbatim including
  leading whitespace).
- `buffer_text`: the model's draft beyond the accepted prefix (display-only,
  may be rewritten at any time).
- `covered_source_units`: 1 + the highest source word index the accepted
  target attends to (from the calibrated alignment heads); `null` when
  unavailable. Clients use it to timestamp translation segments.
- When `final: true`, `committed_text` is the full-quality translation of
  the closed utterance and replaces the streamed partial at the line level;
  `forced_final: true` additionally signals a server-side rollover
  (`--max-utterance-words` exceeded), after which the client must bump its
  `utterance_id` exactly as for its own finalization.

### `error`

```json
{"type": "error", "code": "unsupported_direction",
 "message": "...", "supported": ["en-de", "en-it", "en-zh"]}
```

Codes: `bad_protocol`, `unsupported_direction`, `busy` (server at
`--max-sessions`), `processing_failed` (that update was dropped; the session
stays usable).

## Scheduling and backpressure

The server serializes all MT work on one engine (the vLLM attention observer
supports a single in-flight request). Per connection it keeps a mailbox of
size one for partial updates (latest wins) plus a sticky ordered queue for
finals. Clients should self-pace: coalesce commits, avoid calling per single
word, and treat `translation` responses as the natural pacing signal.

## Frontier semantics (why this protocol is shaped like this)

`build_source_accessibility_frontier` marks committed words accessible
(they carry real timestamps) and tail words inaccessible (they carry none).
The prompt text itself does not encode accessibility, so when the ASR
commits words that were already in the prompt as tail, the prompt bytes do
not change; with `translation_alignatt_commit_fast_path` (default on for
this server) the previously held target tokens are released by re-cutting
the cached draft, without a new MT engine call. Feeding the tail therefore
converts the ASR commit latency into MT speculation time.
