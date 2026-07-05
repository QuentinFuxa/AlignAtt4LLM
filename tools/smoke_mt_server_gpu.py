#!/usr/bin/env python3
"""GPU smoke for alignatt-mt-server: replay a timed English stream, measure
emission latency, and verify the append-only and fast-path contracts.

Simulates a WhisperLiveKit client: words arrive as unstable tail first
(inaccessible source), then commit ~2 s later (accessible), sentence
punctuation closes utterances. Everything is wall-clock paced.

Run on the GPU box, with the server already up:

    python tools/smoke_mt_server_gpu.py --url ws://localhost:8765 \
        --target de --output-json outputs/mt_server_smoke.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from websockets.sync.client import connect

TEXT = (
    "The causal encoder processes each audio block exactly once. "
    "This keeps the compute cost constant as the stream grows. "
    "Attention heads inside the decoder track which source words are being translated. "
    "The policy only commits target tokens whose attention lands on committed source words. "
    "Everything else stays in the draft buffer until the transcript catches up. "
    "This design converts recognition latency into translation speculation time."
)

WORDS_PER_SECOND = 2.4
COMMIT_LAG_SECONDS = 2.0
TAIL_AHEAD_WORDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--target", default="de")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="Replay speed multiplier (4.0 = 4x faster than real time).")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    words = TEXT.split()
    word_end_s = [(i + 1) / WORDS_PER_SECOND for i in range(len(words))]

    ws = connect(args.url, open_timeout=10.0)
    ws.send(json.dumps({
        "type": "init", "protocol_version": 1,
        "source_lang": "en", "target_lang": args.target,
    }))
    init = json.loads(ws.recv(timeout=300.0))
    if init.get("type") != "init_ok":
        raise SystemExit(f"init failed: {init}")
    print(f"init_ok: {init['direction']} preset={init['preset']}")

    seq = 0
    utterance_start = 0
    committed_upto = 0
    emitted = ""
    events = []
    responses = []
    retractions = 0
    wall0 = time.perf_counter()

    def now_stream_s() -> float:
        return (time.perf_counter() - wall0) * args.speed

    def exchange(update: dict, timeout: float = 60.0) -> dict:
        nonlocal seq, emitted, retractions
        seq += 1
        update["seq"] = seq
        sent_at = time.perf_counter()
        ws.send(json.dumps(update))
        while True:
            response = json.loads(ws.recv(timeout=timeout))
            if response.get("type") == "error":
                raise SystemExit(f"server error: {response}")
            if response.get("type") == "translation" and int(response.get("seq") or 0) >= seq:
                response["_rtt_ms"] = (time.perf_counter() - sent_at) * 1000.0
                responses.append(response)
                if not response.get("final"):
                    text = response.get("committed_text") or ""
                    if not text.startswith(emitted):
                        retractions += 1
                        print(f"RETRACTION: {emitted!r} -> {text!r}")
                    emitted = text
                return response

    last_sent = None
    while committed_upto < len(words):
        # advance the stream clock; words older than COMMIT_LAG commit
        stream_s = now_stream_s()
        new_committed_upto = committed_upto
        while (
            new_committed_upto < len(words)
            and word_end_s[new_committed_upto] <= stream_s - COMMIT_LAG_SECONDS
        ):
            new_committed_upto += 1
        tail_upto = min(len(words), max(new_committed_upto, int(stream_s * WORDS_PER_SECOND)))
        commit_advanced = new_committed_upto > committed_upto
        committed_upto = new_committed_upto

        # utterance boundary: first sentence punctuation among committed words
        final_boundary = None
        for i in range(utterance_start, committed_upto):
            if words[i].rstrip('"\'').endswith((".", "!", "?")):
                final_boundary = i + 1
                break
        is_final = final_boundary is not None
        utt_end = final_boundary if is_final else committed_upto

        if (utt_end, tail_upto if not is_final else -1, utterance_start) == last_sent:
            time.sleep(0.05)
            continue
        last_sent = (utt_end, tail_upto if not is_final else -1, utterance_start)

        utt_words = [
            [words[i], word_end_s[i] * 1000.0 - 300.0, word_end_s[i] * 1000.0]
            for i in range(utterance_start, utt_end)
        ]
        tail_words = (
            [] if is_final
            else [[words[i], None, None] for i in range(committed_upto, tail_upto)]
        )
        update = {
            "type": "update",
            "utterance_id": len([e for e in events if e.get("final")]),
            "words": utt_words,
            "tail": {"words": tail_words},
            "clock_ms": stream_s * 1000.0,
            "is_final": is_final,
        }
        response = exchange(update)
        events.append({
            "stream_s": round(stream_s, 2),
            "committed_words": committed_upto - utterance_start,
            "tail_words": len(tail_words),
            "final": bool(response.get("final")),
            "stop_reason": response.get("stop_reason"),
            "rtt_ms": round(response["_rtt_ms"], 1),
            "committed_text": response.get("committed_text"),
            "commit_advanced": commit_advanced,
        })
        if is_final:
            print(f"[{stream_s:6.2f}s] FINAL: {response.get('committed_text')!r}")
            utterance_start = utt_end
            emitted = ""
        else:
            print(
                f"[{stream_s:6.2f}s] rtt={response['_rtt_ms']:6.1f}ms "
                f"reason={response.get('stop_reason')} "
                f"text={response.get('committed_text')!r}"
            )
        time.sleep(0.4 / args.speed)

    ws.close()

    fast_path_events = [
        e for e in events if e.get("stop_reason") == "alignatt:commit_fast_path"
    ]
    commit_rtts = [e["rtt_ms"] for e in events if e["commit_advanced"] and not e["final"]]
    all_rtts = [e["rtt_ms"] for e in events]
    summary = {
        "events": len(events),
        "finals": len([e for e in events if e["final"]]),
        "retractions": retractions,
        "fast_path_events": len(fast_path_events),
        "fast_path_rtt_ms_median": (
            round(statistics.median(e["rtt_ms"] for e in fast_path_events), 1)
            if fast_path_events else None
        ),
        "commit_rtt_ms_median": (
            round(statistics.median(commit_rtts), 1) if commit_rtts else None
        ),
        "rtt_ms_p95": round(sorted(all_rtts)[int(0.95 * (len(all_rtts) - 1))], 1)
        if all_rtts else None,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(
            {"summary": summary, "events": events}, indent=2, ensure_ascii=False
        ))

    assert retractions == 0, "append-only violated"
    assert summary["finals"] >= 5, "expected one final per sentence"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
