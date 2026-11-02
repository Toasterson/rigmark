#!/usr/bin/env python3
"""Opt-in deterministic tool/history service profile, separate from the core suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import bench
from receipt import check_stream_row


PROFILE = "rigmark-tools-v1"
TOOL = {"type": "function", "function": {
    "name": "read_config", "description": "Read the public benchmark fixture config.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "enum": ["config.json"]}},
        "required": ["path"], "additionalProperties": False}}}
CONFIG = {"timeout_ms": 250, "retries": 2}
SYSTEM = "Follow the user's instructions exactly. Do not add markdown fences or commentary."


def arrivals(sessions, rate, seed):
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(sessions - 1):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


def turn_valid(index, row):
    if row.get("stream_done_marker") is not True:
        return False
    if index == 0:
        calls = row.get("tool_calls", [])
        if row.get("finish_reason") != "tool_calls" or len(calls) != 1 or not calls[0].get("id"):
            return False
        try:
            return (calls[0]["function"]["name"] == "read_config" and
                    json.loads(calls[0]["function"]["arguments"]) == {"path": "config.json"})
        except (KeyError, TypeError, ValueError):
            return False
    if row.get("finish_reason") != "stop":
        return False
    if index == 1:
        try:
            return json.loads(row["output"]) == CONFIG
        except (KeyError, TypeError, ValueError):
            return False
    return row.get("output", "").strip() == "READY"


def initial_messages(index):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
             f"Session {index}. Read config.json with read_config. Do not answer yet."}]


def advance_messages(messages, turn, row):
    if turn == 0:
        call = row["tool_calls"][0]
        messages.extend([{"role": "assistant", "content": row["output"], "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(CONFIG)},
                         {"role": "user", "content": "Return exactly the config JSON object."}])
    elif turn == 1:
        messages.extend([{"role": "assistant", "content": row["output"]},
                         {"role": "user", "content": "Now reply exactly READY."}])


def run_session(client, model, index, offset, origin, settings):
    due = origin + offset
    time.sleep(max(due - time.monotonic(), 0))
    started = time.monotonic()
    messages = initial_messages(index)
    rows = []
    for turn in range(3):
        turn_started = time.monotonic()
        if turn == 1:
            time.sleep(settings["tool_latency_seconds"])
        payload = bench.build_chat_payload(model, SYSTEM, "", settings["max_tokens"],
                                           settings["seed"], {})
        payload["messages"] = messages
        if turn == 0:
            payload.update(tools=[TOOL], tool_choice={"type": "function", "function": {"name": "read_config"}})
        try:
            row = client.stream("/v1/chat/completions", payload)
        except (OSError, RuntimeError, ValueError) as error:
            return {"session": index, "arrival_seconds": offset,
                    "start_lag_seconds": max(started - due, 0), "turns": rows,
                    "completed": False, "error": type(error).__name__,
                    "attempted_turns": turn + 1, "failed_request": payload,
                    "wall_seconds": time.monotonic() - started}
        row.update(turn=turn + 1, request=json.loads(json.dumps(payload)),
                   turn_seconds=time.monotonic() - turn_started,
                   simulated_tool_seconds=settings["tool_latency_seconds"] if turn == 1 else 0)
        row["completed"] = turn_valid(turn, row)
        rows.append(row)
        if not row["completed"]:
            break
        advance_messages(messages, turn, row)
    return {"session": index, "arrival_seconds": offset,
            "start_lag_seconds": max(started - due, 0), "turns": rows,
            "attempted_turns": len(rows),
            "completed": len(rows) == 3 and all(row["completed"] for row in rows),
            "wall_seconds": time.monotonic() - started}


def summarise(sessions):
    rows = [row for session in sessions for row in session["turns"] if row["completed"]]
    times = sorted(row["turn_seconds"] for row in rows)
    return {"completed_sessions": sum(session["completed"] for session in sessions),
            "completed_turns": len(rows), "attempted_turns": sum(s["attempted_turns"] for s in sessions),
            "time_per_completed_turn_seconds": sum(s["wall_seconds"] for s in sessions) / len(times) if times else None,
            "mean_completed_turn_seconds": sum(times) / len(times) if times else None,
            "p95_turn_seconds": times[math.ceil(.95 * len(times)) - 1] if times else None,
            "p99_turn_seconds": times[math.ceil(.99 * len(times)) - 1] if times else None}


def validate_result(result):
    errors = []
    if not isinstance(result, dict):
        return ["receipt is not an object"]
    if result.get("schema") != 1 or result.get("profile") != PROFILE:
        return ["unsupported agentic receipt"]
    try:
        settings, sessions = result["settings"], result["agentic"]["sessions"]
        if any(type(settings[key]) is not int or settings[key] < 1
               for key in ("sessions", "workers", "max_tokens")):
            return ["invalid profile counts"]
        if type(settings["seed"]) is not int or any(
            type(settings[key]) not in (int, float) or not math.isfinite(settings[key])
            for key in ("arrival_rate", "tool_latency_seconds")
        ) or settings["arrival_rate"] <= 0 or settings["tool_latency_seconds"] < 0:
            return ["invalid profile scheduling settings"]
        offsets = arrivals(settings["sessions"], settings["arrival_rate"], settings["seed"])
        if len(sessions) != settings["sessions"]:
            errors.append("session count mismatch")
        for index, session in enumerate(sessions):
            if session["session"] != index + 1 or session["arrival_seconds"] != offsets[index]:
                errors.append("session arrival/identity mismatch")
            rows = session["turns"]
            if session.get("attempted_turns") != len(rows) + int("error" in session):
                errors.append("attempted turn count mismatch")
            messages = initial_messages(index + 1)
            for field in ("wall_seconds", "start_lag_seconds"):
                if not math.isfinite(session[field]) or session[field] < 0:
                    errors.append("invalid session timing")
            for turn, row in enumerate(rows):
                check_stream_row(errors, row, f"session[{index}].turn[{turn}]", True)
                if row["turn"] != turn + 1 or row["completed"] != turn_valid(turn, row):
                    errors.append("turn completion mismatch")
                expected_payload = bench.build_chat_payload(result["run"]["model"], SYSTEM, "",
                                                              settings["max_tokens"], settings["seed"], {})
                expected_payload["messages"] = messages
                if turn == 0:
                    expected_payload.update(tools=[TOOL], tool_choice={"type": "function", "function": {"name": "read_config"}})
                if row.get("request") != expected_payload:
                    errors.append("request differs from deterministic profile/history")
                if not math.isfinite(row["turn_seconds"]) or row["turn_seconds"] < 0:
                    errors.append("invalid turn timing")
                if row["completed"]:
                    advance_messages(messages, turn, row)
                elif turn != len(rows) - 1:
                    errors.append("session continued after failed turn")
                expected_tool = settings["tool_latency_seconds"] if turn == 1 else 0
                if row["simulated_tool_seconds"] != expected_tool or row["turn_seconds"] < expected_tool:
                    errors.append("simulated tool timing mismatch")
                if row["turn_seconds"] + 0.001 < row["wall_seconds"] + expected_tool:
                    errors.append("turn duration excludes request/tool time")
            if len(rows) > 3 or session["completed"] != (len(rows) == 3 and all(r["completed"] for r in rows)):
                errors.append("session completion mismatch")
        if result["agentic"]["summary"] != summarise(sessions):
            errors.append("agentic summary differs from raw turns")
    except (KeyError, TypeError, ValueError, IndexError):
        errors.append("malformed agentic receipt")
    return errors


def render(result, fingerprint):
    errors = validate_result(result)
    if errors:
        raise ValueError("invalid receipt: " + "; ".join(errors))
    from report import line, rule, wrapped_lines

    summary = result["agentic"]["summary"]
    lines = [rule("╭", "─", "╮"), line("RIGMARK // OPT-IN TOOL/HISTORY PROFILE"),
             line("Separate synthetic serving profile; not a coding quality score.")]
    for key in sorted(summary):
        lines.extend(wrapped_lines(f"{key}: {summary[key]}"))
    lines.extend(wrapped_lines("Settings: " + json.dumps(result["settings"], sort_keys=True)))
    lines.extend(wrapped_lines("Appliance: " + json.dumps(result["run"]["appliance"], sort_keys=True)))
    lines.extend(wrapped_lines("Source: " + json.dumps(result["protocol"], sort_keys=True)))
    lines.extend([line("Tail: nearest-rank percentiles of completed turns, includes tool delay."),
                  line("Inspect failures and arrival/start lag in JSON; never hide failed sessions."),
                  line("JSON sha256:" + fingerprint), rule("╰", "─", "╯")])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--arrival-rate", type=float, default=0.5)
    parser.add_argument("--tool-latency", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    if args.sessions < 1 or args.workers < 1 or args.max_tokens < 1 or not math.isfinite(args.arrival_rate) or args.arrival_rate <= 0 or not math.isfinite(args.tool_latency) or args.tool_latency < 0:
        parser.error("counts/rate must be positive; tool latency must be finite and nonnegative")
    settings = {"sessions": args.sessions, "arrival_rate": args.arrival_rate,
                "tool_latency_seconds": args.tool_latency, "workers": args.workers,
                "max_tokens": args.max_tokens, "seed": args.seed}
    result = {"schema": 1, "profile": PROFILE,
              "protocol": {"version": bench.PROTOCOL_VERSION, **bench.git_identity(),
                           "profile_sha256": hashlib.sha256(json.dumps(
                               {"tool": TOOL, "config": CONFIG, "system": SYSTEM, "profile": PROFILE},
                               sort_keys=True).encode()).hexdigest()},
              "run": {"model": args.model, "appliance": bench.load_metadata(args.metadata),
                      "started_at": datetime.now(timezone.utc).isoformat()}, "settings": settings}
    client = bench.Client(args.base_url, os.environ.get(args.api_key_env, ""), args.timeout)
    offsets = arrivals(args.sessions, args.arrival_rate, args.seed)
    origin = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_session, client, args.model, i + 1, offset, origin, settings)
                   for i, offset in enumerate(offsets)]
        sessions = [future.result() for future in futures]
    result["agentic"] = {"sessions": sessions, "summary": summarise(sessions)}
    result["run"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    bench.write_result(result, args.output)
    card = render(result, hashlib.sha256(args.output.read_bytes()).hexdigest())
    card_path = args.output.with_suffix(".card.txt")
    card_path.write_text(card + "\n")
    print(card)
    print(f"card: {card_path}\nreceipt: {args.output}")


if __name__ == "__main__":
    main()
