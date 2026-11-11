#!/usr/bin/env python3
"""
Standalone Agent Fight Club agent for Escape Games.

It plays one Escape Games scenario: create a game, start it, then repeat —
ask the model what to do, run the command it picked, give it the result.

It only uses the Python standard library. What it plays and how — the
competition, the scenario, and the four settings from Agent Builder (model,
system prompt, turn limit, token limit) — come from agent.json next to this
file. The keys come from .env.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request


def load_dotenv():
    """Reads KEY=VALUE lines from a .env file next to this script, without
    overriding variables already set in the environment."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

AFC_SERVER_URL = os.environ.get("AFC_SERVER_URL") or "https://agentfightclub.today"
AFC_API_KEY = os.environ.get("AFC_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Where model calls go. OpenRouter itself, or — for an agent on the
# competition's managed keys — Agent Fight Club's OpenRouter-compatible proxy,
# which takes the AFC token as the key and pays with the admin's key.
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"


def load_config():
    """Reads agent.json from next to this script: which game to play, and the
    agent's settings from Agent Builder. Edit it to change either."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.json")
    if not os.path.exists(path):
        print("error: agent.json not found next to agent.py.", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()

# The game this agent plays. Point it at another scenario in agent.json.
COMPETITION_ID = CONFIG["competitionId"]
SCENARIO_ID = CONFIG["scenarioId"]

# The agent itself, as configured in Agent Builder.
AGENT_NAME = CONFIG["name"]
# Leaderboards group games by "<your display name>_<context>", so this keeps
# runs of this script on their own row instead of mixing them in with the games
# you play from Agent Builder. Change it to split your runs up further.
AGENT_CONTEXT = "compiled-agent"
MODEL = CONFIG["model"]
SYSTEM_PROMPT = CONFIG["instructions"]
MAX_TURNS = CONFIG["maxTurns"]  # how many times the agent may call the model in one run
MAX_TOKENS = CONFIG["maxTokens"]  # how long a single model reply may be

# The only tool the agent has: run one command in the game.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "game_command",
            "description": "Execute a command in the current game.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute in the game"},
                },
                "required": ["command"],
            },
        },
    },
]


# ANSI colors, so the log skims the way it does in Agent Builder. Only the
# markers and captions are colored — the model's text and the game's answers
# print in the terminal's own foreground color, which stays readable on a light
# background as well as a dark one. Left off when the output is redirected to a
# file, where the codes would just be noise.
GREY, CYAN, GREEN, RED, BOLD, RESET = "\033[90m", "\033[36m", "\033[32m", "\033[31m", "\033[1m", "\033[0m"
COLORS = {"•": GREY, "→": CYAN, "←": GREEN, "✕": RED}
USE_COLOR = sys.stdout.isatty()

# Every log line is also kept here and sent to the platform when the run ends,
# so this agent's reasoning shows up on the game log page (see submit_agent_log).
EVENTS = []
EVENT_TYPES = {"•": "system", "→": "message", "←": "tool_result", "✕": "error"}

# What the run cost so far, totalled across every model call and reported at the
# end (see submit_stats). OpenRouter only returns these when asked — that is the
# "usage" field in ask_model's request body. Not every provider reports a cost,
# so COST_SEEN tracks whether the dollar figure is real or just zero.
TOTAL_TOKENS = 0
TOTAL_COST_USD = 0.0
COST_SEEN = False

# The agents in this run, by agent_id. "main" plays the game. A model call made
# for anything else — say, compacting the history with a cheaper model — passes
# its own agent_id to ask_model, which adds it here; the leaderboard and the game
# log then show it as a row of its own, with its model, tokens and cost.
AGENTS = {
    "main": {"agentId": "main", "agentName": AGENT_NAME, "model": MODEL, "systemPrompt": SYSTEM_PROMPT},
}
# What each agent's calls cost: {"agentId", "model", "tokenCount"}, plus "costUsd"
# once the provider has reported one.
USAGE_BY_AGENT = {}


def log(marker, text, event_type=None):
    """Prints one "<marker> <text>" line and records it for the agent log.
    Markers mirror the agent log in Agent Builder: "•" status, "✕" error."""
    color = COLORS.get(marker, "") if USE_COLOR else ""
    print(f"{color}{marker} {text}{RESET if color else ''}")
    sys.stdout.flush()
    record(event_type or EVENT_TYPES[marker], text)


def log_caption(marker, caption):
    """Prints the caption heading a block of lines — "→ from LLM" for everything
    one model reply produced, "← to LLM (tool result)" for what goes back to it.
    Mirrors the agent log in Agent Builder. Console only: what the transcript
    records is the lines beneath it."""
    color = f"{COLORS.get(marker, '')}{BOLD}" if USE_COLOR else ""
    print(f"{color}{marker} {caption}{RESET if color else ''}")
    sys.stdout.flush()


def log_line(text, event_type, prefix="", agent_id="main"):
    """Prints one line under a caption and records it for the agent log, under
    the agent that produced it. The prefix is console-only, so the transcript
    keeps the raw content."""
    print(f"{prefix}{text}")
    sys.stdout.flush()
    record(event_type, text, agent_id)


def record(event_type, text, agent_id="main"):
    """Keeps a line for the transcript posted when the run ends (see
    submit_agent_log)."""
    EVENTS.append({
        "timestamp": int(time.time() * 1000),
        "type": event_type,
        "content": text,
        "agentId": agent_id,
    })


class AfcError(RuntimeError):
    """A failed Agent Fight Club API call. status is the HTTP status code."""

    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def call_afc(method, path, body=None):
    """Calls the Agent Fight Club API and returns the parsed JSON response."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {AFC_API_KEY}"}
    if data is not None:
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(f"{AFC_SERVER_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        # Failures come back as {"message": "..."} — use that, it reads better
        # than raw JSON both in the log and to the model.
        error_body = error.read().decode()
        try:
            reason = json.loads(error_body).get("message") or error_body
        except json.JSONDecodeError:
            reason = error_body or str(error)
        raise AfcError(f"{method} {path} -> {error.code}: {reason}", error.code) from error

    return json.loads(text) if text else None


def send_game_command(command_path, command_line):
    """Sends one game command, e.g. "go north", and returns what the game says back."""
    parts = (command_line or "").strip().split()
    command = parts[0] if parts else ""
    args = parts[1:]
    result = call_afc("POST", command_path, {"command": command, "args": args})
    if isinstance(result, dict) and result.get("message"):
        return result["message"]
    return json.dumps(result)


def ask_model(messages, agent_id="main", model=MODEL, tools=TOOLS):
    """Sends the conversation so far to the model and returns its reply, adding
    what the call cost to this run's totals and to agent_id's share of them.

    The game loop calls it as the "main" agent. Anything else that needs a model
    call passes its own agent_id (and, if it differs, its model and tools — None
    for none), so the leaderboard shows what it cost separately."""
    if agent_id not in AGENTS:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        AGENTS[agent_id] = {"agentId": agent_id, "agentName": agent_id, "model": model, "systemPrompt": system}

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        # Ask OpenRouter to report token counts and dollar cost on the response.
        "usage": {"include": True},
    }
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"OpenRouter API error ({error.code}): {error.read().decode()}") from error

    if payload.get("error"):
        raise RuntimeError(f"OpenRouter error: {payload['error'].get('message')}")

    count_usage(agent_id, model, payload.get("usage") or {})

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("No response from model")

    message = choices[0]["message"]
    return {
        "text": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
        # "length" means the reply was cut off because it hit MAX_TOKENS.
        "finish_reason": choices[0].get("finish_reason"),
    }


def is_unfinished_game(error):
    """The stats endpoints answer 400 for a game that is still being played."""
    return isinstance(error, AfcError) and error.status == 400


def count_usage(agent_id, model, usage):
    """Adds one call's usage to the run's totals and to agent_id's own row."""
    global TOTAL_TOKENS, TOTAL_COST_USD, COST_SEEN
    if not usage:
        return
    tokens = usage.get("total_tokens") or (
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    )
    TOTAL_TOKENS += tokens
    row = USAGE_BY_AGENT.setdefault(agent_id, {"agentId": agent_id, "model": model, "tokenCount": 0})
    row["tokenCount"] += tokens
    if usage.get("cost") is not None:
        TOTAL_COST_USD += usage["cost"]
        COST_SEEN = True
        row["costUsd"] = row.get("costUsd", 0.0) + usage["cost"]


def log_game_result(game_id):
    """Prints how the game ended — the same summary Agent Builder shows — and
    returns the game's status ("completed" means won), or None if unknown."""
    try:
        stats = call_afc("GET", f"/escapegames/api/games/{game_id}/stats")
    except Exception as error:
        if is_unfinished_game(error):
            log("•", "Game not finished — the agent stopped before winning or losing.")
        else:
            log("✕", f"Could not read game stats: {error}")
        return
    if not stats:
        return

    minutes, seconds = divmod((stats.get("timeSpentMs") or 0) // 1000, 60)
    time_spent = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    status = stats.get("status")
    outcome = "Won" if status == "completed" else "Failed" if status == "failed" else status
    log("•", f"Game {outcome} — {stats.get('turnCount')} turns, {time_spent}")
    return status


def submit_stats(game_id):
    """Reports what the run cost — tokens, dollars — and the settings it ran with,
    so the game log page shows the same numbers Agent Builder does for a browser
    run. Best-effort: the platform only accepts stats for a finished game, so a
    run that stopped early is turned away here, and that is fine."""
    # One row per agent that made a model call, each with its own model; a cost
    # only where the provider reported one.
    per_agent = list(USAGE_BY_AGENT.values())
    body = {"totalTokenCount": TOTAL_TOKENS, "perAgent": per_agent}
    if COST_SEEN:
        body["totalCostUsd"] = TOTAL_COST_USD

    # A snapshot of what this agent was, kept next to the run it explains.
    body["agentConfig"] = {
        "name": AGENT_NAME,
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "maxTurns": MAX_TURNS,
        "maxTokens": MAX_TOKENS,
    }

    cost = f", ${TOTAL_COST_USD:.4f}" if COST_SEEN else ""
    try:
        call_afc("POST", f"/escapegames/api/competitions/{COMPETITION_ID}/games/{game_id}/stats", body)
        log("•", f"Run cost: {TOTAL_TOKENS} tokens{cost}")
    except Exception as error:
        reason = "the game is not finished" if is_unfinished_game(error) else error
        log("•", f"Stats not submitted ({TOTAL_TOKENS} tokens{cost}): {reason}")


def submit_agent_log(game_id):
    """Sends everything this run logged to the platform, so it shows up under
    "Agent log" on the game log page. Works whether or not the agent won —
    nothing about the game itself depends on it."""
    try:
        call_afc(
            "POST",
            f"/escapegames/api/competitions/{COMPETITION_ID}/games/{game_id}/agent-log",
            {
                "version": 1,
                "agents": list(AGENTS.values()),
                "events": [dict(event, agentName=AGENTS[event["agentId"]]["agentName"]) for event in EVENTS],
            },
        )
        log("•", f"Agent log: {AFC_SERVER_URL}/escapegames/ui/games/{game_id}/log")
    except Exception as error:
        log("•", f"Agent log not submitted: {error}")


# The annotated explanation of this agent's code, written by a coding agent
# (see AGENTS.md, if this folder has one) and uploaded after a won game.
CODE_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-report.html")


def submit_code_report(game_id, status):
    """After a won game, attaches agent-report.html to it, so the competition
    leaderboard links the report next to the result. Skipped when the game was
    not won or there is no report. Best-effort, like the stats."""
    if status != "completed" or not os.path.exists(CODE_REPORT):
        return
    with open(CODE_REPORT, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        call_afc(
            "POST",
            f"/escapegames/api/competitions/{COMPETITION_ID}/games/{game_id}/attachments",
            {"files": [{"name": "agent-report.html", "content": content}]},
        )
        log("•", f"Code report attached: {AFC_SERVER_URL}/escapegames/ui/competitions/{COMPETITION_ID}/leaderboard")
    except Exception as error:
        log("•", f"Code report not attached: {error}")


def run_agent():
    log("•", f'Starting agent "{AGENT_NAME}" with model {MODEL}...')

    game = call_afc(
        "POST",
        f"/escapegames/api/competitions/{COMPETITION_ID}/games",
        {"scenarioId": SCENARIO_ID, "context": AGENT_CONTEXT},
    )
    game_id = game["gameId"]
    log("•", f"Game created: {game_id}")
    log("•", f"Live map: {AFC_SERVER_URL}/escapegames/ui/competitions/{COMPETITION_ID}/games/{game_id}/livemap")

    command_path = f"/escapegames/api/competitions/{COMPETITION_ID}/games/{game_id}/command"
    start_result = call_afc("POST", command_path, {"command": "start"})
    log("•", f"Game started: {(start_result or {}).get('message', 'OK')}")

    # The whole conversation with the model. Every turn appends to it, so the
    # model always sees everything that has happened so far.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Follow the system prompt."},
    ]

    turn = 0
    while turn < MAX_TURNS:
        turn += 1
        log("•", f"Turn {turn}/{MAX_TURNS} — thinking...")
        reply = ask_model(messages)

        # Everything one reply produced — its text and the command it chose —
        # sits under a single caption, the way Agent Builder draws it.
        if reply["text"] or reply["tool_calls"]:
            log_caption("→", "from LLM")
        if reply["text"]:
            log_line(reply["text"], "message")

        # No tool calls means the model just talked instead of playing, so
        # there is nothing left to run and the agent stops here.
        if not reply["tool_calls"]:
            if reply["finish_reason"] == "length":
                log("✕", f"Model hit MAX_TOKENS ({MAX_TOKENS}) before calling a tool. Raise MAX_TOKENS.")
            break

        messages.append({"role": "assistant", "content": reply["text"], "tool_calls": reply["tool_calls"]})

        for tool_call in reply["tool_calls"]:
            name = tool_call["function"]["name"]
            log_line(f"{name}({tool_call['function']['arguments']})", "tool_call", "tool call ")

            if name == "game_command":
                arguments = json.loads(tool_call["function"]["arguments"])
                # A rejected command is not fatal: hand the error to the model as
                # the tool's result and let it try something else.
                try:
                    result = send_game_command(command_path, arguments.get("command"))
                except Exception as error:
                    result = f"Error: {error}"
                log_caption("←", "to LLM (tool result)")
                log_line(result, "tool_result")
            else:
                # The model asked for a tool that does not exist. Say so in the
                # result rather than guessing, so it can correct itself.
                result = f'Error: unknown tool "{name}". The only tool available is game_command.'
                log("✕", result)

            # The result goes back to the model as the answer to its tool call.
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

    if turn >= MAX_TURNS:
        log("✕", f"Turn limit reached ({MAX_TURNS}). Agent stopped.")

    status = log_game_result(game_id)
    log("•", "Agent finished.")
    # Stats first, so the cost line it logs is part of the transcript that
    # submit_agent_log then sends.
    submit_stats(game_id)
    submit_agent_log(game_id)
    submit_code_report(game_id, status)


if __name__ == "__main__":
    if not AFC_API_KEY:
        print("error: AFC_API_KEY env var required. See env.sample.", file=sys.stderr)
        sys.exit(1)
    if not OPENROUTER_API_KEY:
        print("error: OPENROUTER_API_KEY env var required. See env.sample.", file=sys.stderr)
        sys.exit(1)
    # On managed keys the AFC token doubles as the OpenRouter key, and only the
    # competition's proxy accepts it. Never send it to openrouter.ai.
    if OPENROUTER_API_KEY == AFC_API_KEY and not os.environ.get("OPENROUTER_BASE_URL"):
        print("error: OPENROUTER_API_KEY is your AFC token, which only works on the competition's managed keys. "
              "Set OPENROUTER_BASE_URL as in env.sample.", file=sys.stderr)
        sys.exit(1)

    try:
        run_agent()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
