import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def bad_number(value):
    raise ValueError(f"non-JSON number {value}")


def parse(text):
    calls = json.loads(text, object_pairs_hook=unique, parse_constant=bad_number)
    need(type(calls) is list and 1 <= len(calls) <= 4, "call count")
    for call in calls:
        need(type(call) is dict and set(call) == {"tool", "args"}, "call fields")
        need(type(call["tool"]) is str and type(call["args"]) is dict, "call types")
    return calls


FIELDS = {
    "records": {"filter": {"field": str, "value": str}, "sort": {"field": str, "descending": bool},
                "take": {"count": int}, "project": {"fields": list}},
    "text": {"lower": {}, "keep": {"contains": str}, "replace": {"old": str, "new": str},
             "join": {"separator": str}},
    "files": {"copy": {"source": str, "destination": str}, "move": {"source": str, "destination": str},
              "write": {"path": str, "text": str}, "delete": {"path": str}},
    "calendar": {"add": {"id": str, "start": int, "title": str}, "shift": {"id": str, "minutes": int},
                 "rename": {"id": str, "title": str}, "cancel": {"id": str}},
}


def simulate(example, calls, conventions):
    family = example["family"]
    names = {alias: operation for operation, alias in conventions[family].items()}
    state, trace = copy.deepcopy(example["state"]), []
    for call in calls:
        operation = names[call["tool"]]
        args, fields = call["args"], FIELDS[family][operation]
        need(set(args) == set(fields), "argument fields")
        need(all(type(args[k]) is t for k, t in fields.items()), "argument types")
        if family == "records":
            need(type(state) is list and all(type(row) is dict for row in state), "record state")
            if operation == "filter":
                state = [row for row in state if row[args["field"]] == args["value"]]
            elif operation == "sort":
                need(len({type(row[args["field"]]) for row in state}) <= 1, "sort comparability")
                state.sort(key=lambda row: row[args["field"]], reverse=args["descending"])
            elif operation == "take":
                need(1 <= args["count"] <= len(state), "take bounds")
                state = state[:args["count"]]
            else:
                selected = args["fields"]
                need(selected and all(type(k) is str for k in selected), "project fields")
                need(len(set(selected)) == len(selected), "duplicate field")
                state = [{k: row[k] for k in selected} for row in state]
        elif family == "text":
            need(type(state) is list and all(type(x) is str for x in state), "text state")
            if operation == "lower":
                state = [x.lower() for x in state]
            elif operation == "keep":
                need(bool(args["contains"]), "empty filter")
                state = [x for x in state if args["contains"] in x]
            elif operation == "replace":
                need(bool(args["old"]), "empty replacement")
                state = [x.replace(args["old"], args["new"]) for x in state]
            else:
                state = args["separator"].join(state)
        elif family == "files":
            if operation in {"copy", "move"}:
                need(args["destination"] not in state, "existing destination")
                state[args["destination"]] = state[args["source"]]
                if operation == "move":
                    del state[args["source"]]
            elif operation == "write":
                need(bool(args["path"]), "empty path")
                state[args["path"]] = args["text"]
            else:
                del state[args["path"]]
        elif operation == "add":
            need(args["id"] and args["id"] not in state, "existing or empty event")
            need(0 <= args["start"] < 1440, "event time")
            state[args["id"]] = {"start": args["start"], "title": args["title"]}
        else:
            need(args["id"] in state, "missing event")
            if operation == "shift":
                state[args["id"]]["start"] += args["minutes"]
                need(0 <= state[args["id"]]["start"] < 1440, "shifted event time")
            elif operation == "rename":
                state[args["id"]]["title"] = args["title"]
            else:
                del state[args["id"]]
        trace.append({"operation": operation, "state": copy.deepcopy(state)})
    return state, trace


def rescore(record, example, conventions, tokenizer, eos, pad):
    tokens = record["raw_generation_ids"]
    ended = eos in tokens
    end = tokens.index(eos) if ended else len(tokens)
    padding = all(x == pad for x in tokens[end + 1:])
    need(record["generated_ids"] == tokens[:end + 1], "trimmed token mismatch")
    need(record["text"] == tokenizer.decode(tokens[:end], skip_special_tokens=False), "text decode mismatch")
    need(record["native_eos"] == ended and record["padding_only"] == padding, "termination mismatch")
    outcome = {"correct": False, "semantic_correct": False, "format_valid": False, "executable": False}
    if ended and padding:
        try:
            calls = parse(record["text"])
        except (ValueError, RecursionError):
            calls = None
        if calls is not None:
            outcome["format_valid"] = True
            try:
                state, trace = simulate(example, calls, conventions)
            except (ValueError, KeyError, TypeError, IndexError):
                state, trace = None, None
            if trace is not None:
                outcome.update(executable=True, semantic_correct=state == example["expected"], correct=state == example["expected"])
                need(record["final_state"] == state and record["trace"] == trace, "execution trace mismatch")
                need(record["canonical_calls_match"] == (calls == example["calls"]), "canonical calls mismatch")
    for key, value in outcome.items():
        need(record[key] == value, f"score mismatch {key}:{record['id']}")
    for key in ("id", "group_id", "family", "kind", "pattern", "split"):
        need(record[key] == example[key], f"row identity {key}")
    return outcome


def audit_run(run, tokenizer, eos, pad):
    final = json.loads((run / "result.json").read_text())
    execution = json.loads((run / "execution.json").read_text())
    need(execution["status"] == final["status"] == "completed" and execution["exit_code"] == 0, "source completion")
    config = json.loads((run / "config.json").read_text())
    spec = json.loads((run / "stream.json").read_text())
    dataset = json.loads((run / "dataset.json").read_text())
    lookup, groups = {}, set()
    patterns = defaultdict(set)
    for kind, families in dataset.items():
        for family, splits in families.items():
            for split, examples in splits.items():
                for example in examples:
                    need(example["id"] not in lookup and example["group_id"] not in groups, "dataset overlap")
                    lookup[example["id"]] = example
                    groups.add(example["group_id"])
                    state, _ = simulate(example, example["calls"], spec["conventions"])
                    need(state == example["expected"] and state != example["state"], "oracle mismatch")
                    patterns[kind, family, split].add(example["pattern"])
    for family in dataset["workflow"]:
        need(patterns['workflow', family, 'train'].isdisjoint(patterns['workflow', family, 'novel_test']), "novel pattern leakage")
    panels, total = {}, 0
    for relative, checksum in final["artifact_manifest"].items():
        if not relative.endswith('.json'):
            continue
        path = run / relative
        need(sha(path) == checksum, f"artifact hash {relative}")
        panel = json.loads(path.read_text())
        if not isinstance(panel, dict) or set(panel) != {"metrics", "records"}:
            continue
        records = panel["records"]
        need(len({r['id'] for r in records}) == len(records), "duplicate prediction")
        for row in records:
            rescore(row, lookup[row['id']], spec['conventions'], tokenizer, eos, pad)
        metrics = panel['metrics']
        n, correct = len(records), sum(r['correct'] for r in records)
        need(metrics['n'] == n and metrics['correct'] == correct and metrics['accuracy'] == correct/n, 'aggregate accuracy')
        for key in ('format_valid', 'executable', 'native_eos'):
            need(metrics[key] == sum(r[key] for r in records)/n, f'aggregate {key}')
        for pattern, value in metrics['per_pattern'].items():
            subset = [r for r in records if r['pattern'] == pattern]
            need(value == sum(r['correct'] for r in subset)/len(subset), 'pattern aggregate')
        panels[relative] = {'correct': correct, 'n': n, 'per_pattern': metrics['per_pattern']}
        total += n
    transfers = {}
    for arm, receipt in final['transfer'].items():
        curve = receipt['validation_curve']
        need([point['updates'] for point in curve] == [0, *config['workflow_checkpoints']], 'checkpoint order')
        for point in curve:
            panel = panels[f'transfer/{arm}/validation-{point["updates"]}.json']
            need((point['correct'], point['n']) == (panel['correct'], panel['n']), 'curve endpoint')
        auc = sum((b['updates']-a['updates'])*(a['accuracy']+b['accuracy'])/2 for a,b in zip(curve, curve[1:]))/curve[-1]['updates']
        need(math.isclose(auc, receipt['normalized_validation_auc'], abs_tol=1e-12), 'curve AUC')
        reached = [point['updates'] for point in curve if point['accuracy'] >= config['workflow_min_accuracy']]
        need(receipt['first_qualified_checkpoint'] == (min(reached) if reached else None), 'earliest qualification')
        for field, name in [('test','test.json'),('novel_composition_test','novel-test.json')]:
            need(receipt[field]['correct'] == panels[f'transfer/{arm}/{name}']['correct'], 'reported final count')
        transfers[arm] = {
            'validation_curve': [[p['updates'],p['correct'],p['n']] for p in curve],
            'first_qualified_checkpoint': receipt['first_qualified_checkpoint'], 'auc': auc,
            'test': panels[f'transfer/{arm}/test.json'], 'novel': panels[f'transfer/{arm}/novel-test.json'],
            'primitive_after_test': {f:{k:v[k] for k in ('n','correct')} for f,v in receipt['primitive_after_test'].items()},
        }
    return {'status':'passed','source_sha256':execution['source_sha256'], 'execution_sha256':sha(run/'execution.json'),
            'result_sha256':sha(run/'result.json'),'dataset_rows':len(lookup),'panels':len(panels),'records':total,
            'data_seed':config['data_seed'],'order':spec['order'],'target_family':config['target_family'],
            'transfers':transfers,'claim_boundary':'Independent saved-token decode, strict JSON/execution rescore, dataset overlap/oracles, and count/curve audit; GPU persistence and causal attribution require separate experiments.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    path = args.wave/'tokenizer35/tokenizer.json'
    manifest = json.loads((args.repo/'experiments/skill_transfer/provenance/qwen35-4b-snapshot.json').read_text())
    matches = [item for item in manifest['files'] if item.get('path', item.get('name')) == 'tokenizer.json']
    need(len(matches) == 1 and sha(path) == matches[0]['sha256'], 'pinned tokenizer identity')
    tokenizer = Tokenizer.from_file(str(path))
    eos, pad = tokenizer.token_to_id('<|im_end|>'), tokenizer.token_to_id('<|endoftext|>')
    need((eos,pad) == (248046,248044), 'native token IDs')
    result = {'tokenizer_sha256':sha(path),'runs':{}}
    for run in sorted((args.wave/'runs').glob('followthrough-20260912-skills-stream*')):
        if not (run/'result.json').exists() or json.loads((run/'execution.json').read_text())['status'] != 'completed':
            continue
        result['runs'][run.name] = audit_run(run,tokenizer,eos,pad)
        print(json.dumps({'audited':run.name,'records':result['runs'][run.name]['records']}),flush=True)
    need(result['runs'], 'no completed streams collected')
    (args.wave/'tool-stream-audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
