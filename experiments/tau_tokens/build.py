"""Build the τ-banking token-level test set. For every task, take one successful public leaderboard run, remove its
search calls and their results, and cut out one decision per gold agent action that the run makes, matched the way
tau2 matches actions (only `compare_args` when the action lists them). Each decision is the agent's view up to and
including the gold call, rendered in Qwen3-4B-Instruct-2507's chat format with the no-knowledge policy and with nested
argument JSON in json.dumps form, plus the character span of every argument value the gold action specifies and a
label saying whether that value is copyable from the text before it, found only in the task's required documents, or
neither (booleans and nulls always count as neither).

Run inside a tau2 v1.0.1 checkout:
  cd <tau2-bench> && uv run --with transformers python <repo>/experiments/tau_tokens/build.py \
      --trajectories <dir> --out <dir>
"""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage
from tau2.data_model.simulation import Results
from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR
from tau2.utils.llm_utils import to_litellm_messages

TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
RETRIEVAL = "no_knowledge"
URL = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions/{source}/trajectories/banking_knowledge_results.json"
SOURCES = {
    "qwen3-8-max_sierra_2026-08-04": "8c8191c43dfb2d21c1322cc154740e5e6044151837ab817c2cbbcd13ffeb626e",
    "claude-opus-5_sierra_2026-08-04": "80bd60cd0dc19855e685b85ef2c40efcbc2dabae1ffa16988b40b781cbeee2a3",
    "grok-4-5_sierra_2026-08-04": "f6a9327bdcd8df56a71ba9f59d23ce6a0ef844eac7753cad5ce0d04b87e849b5",
    "gpt-5-6-sol_sierra_2026-08-04": "614cdc47ae910fb97936c839e59cf43ec087f9c3c548af395159ea751759c292",
}
END_OF_TURN = "<|im_end|>\n"
CALL_CLOSE = "}\n</tool_call>"


def log(message):
    print(f"[tau_tokens {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source, expected in SOURCES.items():
        path = directory / f"{source}.json"
        if not path.exists():
            log(f"downloading {source}")
            urllib.request.urlretrieve(URL.format(source=source), path)
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"TRAJECTORY_HASH_MISMATCH {source} expected={expected} actual={actual}")
        paths[source] = path
    return paths


def parsed_container(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, (dict, list)) else None


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, str):
        container = parsed_container(value)
        return value if container is None else canonical(container)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def written(value):
    """The same value with every string that holds a JSON object or array rewritten in json.dumps form."""
    if isinstance(value, dict):
        return {key: written(item) for key, item in value.items()}
    if isinstance(value, list):
        return [written(item) for item in value]
    if isinstance(value, str):
        container = parsed_container(value)
        return value if container is None else json.dumps(written(container))
    return value


def compared(action, arguments):
    keys = arguments.keys() if action.compare_args is None else action.compare_args
    return {key: arguments[key] for key in keys if key in arguments}


def same_call(call, action):
    if call.name != action.name:
        return False
    if action.compare_args is None:
        return json.dumps(canonical(call.arguments), sort_keys=True) == \
            json.dumps(canonical(action.arguments), sort_keys=True)
    return json.dumps(canonical(compared(action, call.arguments)), sort_keys=True) == \
        json.dumps(canonical(compared(action, action.arguments)), sort_keys=True)


def target_arguments(call, action):
    """Gold arguments, except that keys outside compare_args keep the run's own values."""
    if action.compare_args is None:
        return written(action.arguments)
    return {key: written(action.arguments[key] if key in action.compare_args else value)
            for key, value in call.arguments.items()}


def scored(action, path):
    return action.compare_args is None or path.split(".")[0].split("[")[0] in action.compare_args


def escape(text):
    return json.dumps(text)[1:-1]


def leaf_text(leaf):
    return leaf if isinstance(leaf, str) else json.dumps(leaf)


def encoded_leaf(leaf, nesting):
    """How a leaf appears in the call text: its JSON literal (without quotes for strings), escaped once more for
    every JSON string it sits inside."""
    text = escape(leaf) if isinstance(leaf, str) else json.dumps(leaf)
    for _ in range(nesting):
        text = escape(text)
    return text


def encode(value, path=""):
    """json.dumps(value) plus (path, start, end, leaf, nesting) for every leaf. A string holding a JSON object or
    array in json.dumps form is encoded through, so the leaves inside it get spans in the escaped outer text."""
    if isinstance(value, dict):
        text, spans = "{", []
        for index, (key, item) in enumerate(value.items()):
            text += (", " if index else "") + json.dumps(key) + ": "
            inner, inner_spans = encode(item, f"{path}.{key}" if path else key)
            spans += [(p, start + len(text), end + len(text), leaf, n) for p, start, end, leaf, n in inner_spans]
            text += inner
        return text + "}", spans
    if isinstance(value, list):
        text, spans = "[", []
        for index, item in enumerate(value):
            text += ", " if index else ""
            inner, inner_spans = encode(item, f"{path}[{index}]")
            spans += [(p, start + len(text), end + len(text), leaf, n) for p, start, end, leaf, n in inner_spans]
            text += inner
        return text + "]", spans
    if isinstance(value, str):
        container = parsed_container(value)
        if container is not None:
            if json.dumps(container) != value:
                raise SystemExit(f"NESTED_JSON_NOT_WRITTEN path={path}")
            inner, inner_spans = encode(container, path)
            text, spans, position = '"', [], 0
            for p, start, end, leaf, n in inner_spans:
                text += escape(inner[position:start])
                leaf_start = len(text)
                text += escape(inner[start:end])
                spans.append((p, leaf_start, len(text), leaf, n + 1))
                position = end
            return text + escape(inner[position:]) + '"', spans
        body = escape(value)
        return '"' + body + '"', [(path, 1, 1 + len(body), value, 0)]
    if value is None or isinstance(value, (bool, int, float)):
        text = json.dumps(value)
        return text, [(path, 0, len(text), value, 0)]
    raise TypeError(f"UNENCODABLE_VALUE path={path} type={type(value).__name__}")


def strip(messages, allowed):
    """Drop tool calls the no-knowledge agent does not have, their results, and assistant turns left empty."""
    kept, removed, removed_ids, kept_ids = [], Counter(), set(), set()
    for message in messages:
        if isinstance(message, AssistantMessage) and message.tool_calls:
            calls = [call for call in message.tool_calls if call.name in allowed]
            for call in message.tool_calls:
                if call.name not in allowed:
                    removed[call.name] += 1
                    removed_ids.add(call.id)
            kept_ids.update(call.id for call in calls)
            if not calls and not message.content:
                continue
            message = message.model_copy(update={"tool_calls": calls or None})
        elif isinstance(message, ToolMessage):
            if message.id in removed_ids:
                continue
            if message.id not in kept_ids:
                raise SystemExit(f"ORPHAN_TOOL_RESULT id={message.id}")
        kept.append(message)
    return kept, removed


def match(actions, history):
    calls = [(m, c, call) for m, message in enumerate(history) if isinstance(message, AssistantMessage)
             for c, call in enumerate(message.tool_calls or [])]
    used, matched, missing = set(), [], []
    for action in actions:
        hit = next(((m, c) for m, c, call in calls if (m, c) not in used and same_call(call, action)), None)
        if hit is None:
            missing.append(action.action_id)
            continue
        used.add(hit)
        matched.append((action, *hit))
    return matched, missing


def task_context(task):
    environment = get_environment(retrieval_variant=RETRIEVAL, task=task)
    system = SYSTEM_PROMPT.format(domain_policy=environment.get_policy(), agent_instruction=AGENT_INSTRUCTION)
    schemas = [tool.openai_schema for tool in environment.get_tools()]
    documents = []
    for document_id in task.required_documents or []:
        path = KNOWLEDGE_DOCUMENTS_DIR / f"{document_id}.json"
        if not path.exists():
            raise SystemExit(f"REQUIRED_DOCUMENT_MISSING task={task.id} document={document_id}")
        document = json.loads(path.read_text())
        documents.append(f"{document['title']}\n{document['content']}")
    return {"system": system, "schemas": schemas, "allowed": {s["function"]["name"] for s in schemas},
            "documents": "\n".join(documents)}


def prepare(task, simulation, context):
    view = [message for message in simulation.messages if is_valid_agent_history_message(message)]
    history, removed = strip(view, context["allowed"])
    actions = [action for action in task.evaluation_criteria.actions if action.requestor == "assistant"]
    matched, missing = match(actions, history)
    return {"history": history, "removed": removed, "matched": matched, "missing": missing}


def successful(simulation):
    return simulation.reward_info is not None and abs(simulation.reward_info.reward - 1.0) <= 1e-6


def select(task, runs, context):
    """The successful run matching the most gold agent actions; ties go to the earlier source, then trial."""
    best = None
    for source, simulations in runs.items():
        candidates = sorted((s for s in simulations if s.task_id == task.id and successful(s)),
                            key=lambda s: (s.trial, s.id))
        for simulation in candidates:
            prepared = prepare(task, simulation, context)
            if best is None or len(prepared["matched"]) > len(best[2]["matched"]):
                best = (source, simulation, prepared)
    return best


def render(tokenizer, context, history, message_index, call_index, action):
    target = history[message_index]
    arguments = target_arguments(target.tool_calls[call_index], action)
    arguments_text, leaves = encode(arguments)
    if arguments_text != json.dumps(arguments):
        raise SystemExit(f"ENCODER_DRIFT action={action.action_id}")
    calls = list(target.tool_calls[:call_index])
    calls.append(target.tool_calls[call_index].model_copy(update={"arguments": arguments}))
    head = history[:message_index] + [target.model_copy(update={"tool_calls": calls})]
    chat = to_litellm_messages([SystemMessage(role="system", content=context["system"])] + head)
    full = tokenizer.apply_chat_template(chat, tools=context["schemas"], tokenize=False)
    call_text = '<tool_call>\n{"name": "' + action.name + '", "arguments": ' + arguments_text + CALL_CLOSE
    if not full.endswith(call_text + END_OF_TURN):
        raise SystemExit(f"TEMPLATE_DRIFT action={action.action_id}")
    text = full[:-len(END_OF_TURN)]
    arguments_start = len(text) - len(CALL_CLOSE) - len(arguments_text)
    return text, arguments_start, [(p, arguments_start + s, arguments_start + e, leaf, n)
                                   for p, s, e, leaf, n in leaves if scored(action, p)]


def label(leaf, prompt, documents):
    if leaf is None or isinstance(leaf, bool):
        return "derived"
    value = leaf_text(leaf)
    if value in prompt:
        return "copy"
    if value in documents:
        return "documents_only"
    return "derived"


def decisions_for(task, source, simulation, prepared, context, tokenizer):
    records, unscored = [], []
    for action, message_index, call_index in prepared["matched"]:
        text, arguments_start, leaves = render(tokenizer, context, prepared["history"], message_index, call_index,
                                               action)
        if not leaves:
            unscored.append(f"{task.id}:{action.action_id}")
            continue
        values = [{"path": p, "value": leaf, "nesting": n, "start": start, "end": end,
                   "label": label(leaf, text[:start], context["documents"])} for p, start, end, leaf, n in leaves]
        records.append({"decision_id": f"{task.id}:{action.action_id}", "task_id": task.id,
                        "action_id": action.action_id, "action_name": action.name, "compare_args": action.compare_args,
                        "source": {"submission": source, "simulation_id": simulation.id, "trial": simulation.trial},
                        "required_documents": list(task.required_documents or []), "message_index": message_index,
                        "text": text, "arguments_start": arguments_start, "values": values})
    return records, unscored


def verify(record, removed_names):
    text = record["text"]
    for value in record["values"]:
        if text[value["start"]:value["end"]] != encoded_leaf(value["value"], value["nesting"]):
            raise SystemExit(f"SPAN_MISMATCH decision={record['decision_id']} path={value['path']}")
        if value["label"] == "documents_only" and leaf_text(value["value"]) in text[:value["start"]]:
            raise SystemExit(f"DOCUMENT_VALUE_IN_PROMPT decision={record['decision_id']} path={value['path']}")
    for name in removed_names:
        if f'{{"name": "{name}"' in text:
            raise SystemExit(f"REMOVED_TOOL_IN_PROMPT decision={record['decision_id']} tool={name}")


def tokenizer_revision():
    path = try_to_load_from_cache(TOKENIZER, "tokenizer_config.json")
    if not isinstance(path, str) or "/snapshots/" not in path:
        raise SystemExit(f"TOKENIZER_REVISION_UNKNOWN path={path}")
    return path.split("/snapshots/")[1].split("/")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=KNOWLEDGE_DOCUMENTS_DIR, capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != TAU2_COMMIT:
        raise SystemExit(f"TAU2_COMMIT_MISMATCH expected={TAU2_COMMIT} actual={commit}")
    paths = fetch(args.trajectories)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tasks = get_tasks()
    runs = {}
    for source, path in paths.items():
        runs[source] = Results.load(path).simulations
        log(f"loaded {source}: {len(runs[source])} runs, {sum(map(successful, runs[source]))} successful")

    records, chosen, removed, missing, unscored, gold, uncovered = [], Counter(), Counter(), [], [], 0, []
    for task in tasks:
        context = task_context(task)
        best = select(task, runs, context)
        if best is None:
            uncovered.append(task.id)
            continue
        source, simulation, prepared = best
        gold += sum(action.requestor == "assistant" for action in task.evaluation_criteria.actions)
        chosen[source] += 1
        removed.update(prepared["removed"])
        missing += [f"{task.id}:{action_id}" for action_id in prepared["missing"]]
        task_records, task_unscored = decisions_for(task, source, simulation, prepared, context, tokenizer)
        unscored += task_unscored
        for record in task_records:
            verify(record, prepared["removed"])
        records += task_records
        log(f"{task.id}: {source} trial {simulation.trial}, {len(task_records)} decisions, "
            f"{len(prepared['missing'])} gold actions unmatched")

    args.out.mkdir(parents=True, exist_ok=True)
    decisions_path = args.out / "decisions.jsonl"
    with decisions_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    labels = Counter(value["label"] for record in records for value in record["values"])
    manifest = {"tau2_commit": commit, "retrieval_variant": RETRIEVAL, "tokenizer": TOKENIZER,
                "tokenizer_revision": tokenizer_revision(), "trajectories": SOURCES, "tasks": len(tasks),
                "tasks_covered": len(tasks) - len(uncovered), "tasks_uncovered": uncovered,
                "runs_chosen_by_source": dict(chosen), "gold_agent_actions_in_covered_tasks": gold,
                "decisions": len(records), "gold_agent_actions_unmatched": missing,
                "matched_without_specified_values": unscored,
                "values_by_label": dict(labels),
                "values_by_label_and_action": {f"{name}/{lab}": count for (name, lab), count in sorted(Counter(
                    (record["action_name"], value["label"]) for record in records for value in record["values"]
                ).items())},
                "removed_tool_calls": dict(removed), "decisions_sha256": sha256(decisions_path)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"wrote {len(records)} decisions over {manifest['tasks_covered']} tasks to {decisions_path}; "
        f"labels {dict(labels)}")


if __name__ == "__main__":
    main()
