import copy
import json
from dataclasses import asdict, dataclass


class FormatError(ValueError):
    pass


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class Example:
    id: str
    group_id: str
    family: str
    split: str
    kind: str
    pattern: str
    state: object
    request: str
    calls: list
    expected: object

    def record(self):
        return asdict(self)


SCHEMAS = {
    "records": {
        "filter": {"field": str, "value": str},
        "sort": {"field": str, "descending": bool},
        "take": {"count": int},
        "project": {"fields": list},
    },
    "text": {
        "lower": {},
        "keep": {"contains": str},
        "replace": {"old": str, "new": str},
        "join": {"separator": str},
    },
    "files": {
        "copy": {"source": str, "destination": str},
        "move": {"source": str, "destination": str},
        "write": {"path": str, "text": str},
        "delete": {"path": str},
    },
    "calendar": {
        "add": {"id": str, "start": int, "title": str},
        "shift": {"id": str, "minutes": int},
        "rename": {"id": str, "title": str},
        "cancel": {"id": str},
    },
}
FAMILIES = tuple(SCHEMAS)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FormatError(f"DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise FormatError(f"NON_JSON_NUMBER: {value}")


def parse_calls(text, max_calls=4):
    try:
        calls = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise FormatError("INVALID_JSON: whole response must be a JSON array") from error
    if type(calls) is not list or not 1 <= len(calls) <= max_calls:
        raise FormatError(f"CALL_COUNT: require 1..{max_calls} calls")
    for call in calls:
        if type(call) is not dict or set(call) != {"tool", "args"}:
            raise FormatError("CALL_SCHEMA: exactly tool and args are required")
        if type(call["tool"]) is not str or type(call["args"]) is not dict:
            raise FormatError("CALL_TYPES: tool must be a string and args an object")
    return calls


def require(condition, message):
    if not condition:
        raise ExecutionError(message)


def apply_operation(state, family, operation, args):
    schema = SCHEMAS[family][operation]
    require(set(args) == set(schema), "ARGUMENT_KEYS")
    require(all(type(args[key]) is expected for key, expected in schema.items()), "ARGUMENT_TYPES")
    if family == "records":
        require(type(state) is list and all(type(row) is dict for row in state), "RECORD_STATE")
        if operation in {"filter", "sort"}:
            require(all(args["field"] in row for row in state), "MISSING_FIELD")
        if operation == "filter":
            return [row for row in state if row[args["field"]] == args["value"]]
        if operation == "sort":
            values = [row[args["field"]] for row in state]
            require(len({type(value) for value in values}) <= 1, "INCOMPARABLE_VALUES")
            return sorted(state, key=lambda row: row[args["field"]], reverse=args["descending"])
        if operation == "take":
            require(1 <= args["count"] <= len(state), "TAKE_BOUNDS")
            return state[: args["count"]]
        fields = args["fields"]
        require(bool(fields) and all(type(field) is str for field in fields), "PROJECT_FIELDS")
        require(len(set(fields)) == len(fields), "DUPLICATE_FIELD")
        require(all(all(field in row for field in fields) for row in state), "MISSING_FIELD")
        return [{field: row[field] for field in fields} for row in state]
    if family == "text":
        require(type(state) is list and all(type(item) is str for item in state), "TEXT_STATE")
        if operation == "lower":
            return [item.lower() for item in state]
        if operation == "keep":
            require(bool(args["contains"]), "EMPTY_NEEDLE")
            return [item for item in state if args["contains"] in item]
        if operation == "replace":
            require(bool(args["old"]), "EMPTY_NEEDLE")
            return [item.replace(args["old"], args["new"]) for item in state]
        return args["separator"].join(state)
    if family == "files":
        if operation in {"copy", "move"}:
            require(args["source"] in state, "MISSING_SOURCE")
            require(args["destination"] not in state, "DESTINATION_EXISTS")
            state[args["destination"]] = state[args["source"]]
            if operation == "move":
                del state[args["source"]]
        elif operation == "write":
            require(bool(args["path"]), "EMPTY_PATH")
            state[args["path"]] = args["text"]
        else:
            require(args["path"] in state, "MISSING_PATH")
            del state[args["path"]]
        return state
    if operation == "add":
        require(args["id"] not in state and bool(args["id"]), "EVENT_EXISTS_OR_EMPTY")
        require(0 <= args["start"] < 1440, "TIME_BOUNDS")
        state[args["id"]] = {"start": args["start"], "title": args["title"]}
    else:
        require(args["id"] in state, "MISSING_EVENT")
        if operation == "shift":
            start = state[args["id"]]["start"] + args["minutes"]
            require(0 <= start < 1440, "TIME_BOUNDS")
            state[args["id"]]["start"] = start
        elif operation == "rename":
            state[args["id"]]["title"] = args["title"]
        else:
            del state[args["id"]]
    return state


def execute(calls, initial, family, conventions):
    reverse = {alias: operation for operation, alias in conventions[family].items()}
    state = copy.deepcopy(initial)
    trace = []
    for call in calls:
        require(call["tool"] in reverse, "UNKNOWN_TOOL")
        operation = reverse[call["tool"]]
        state = apply_operation(state, family, operation, call["args"])
        trace.append({"operation": operation, "state": copy.deepcopy(state)})
    return state, trace


def grade_text(text, example, conventions, native_eos=True, padding_only=True):
    record = {
        "id": example.id,
        "group_id": example.group_id,
        "family": example.family,
        "kind": example.kind,
        "pattern": example.pattern,
        "split": example.split,
        "text": text,
        "native_eos": native_eos,
        "padding_only": padding_only,
        "format_valid": False,
        "executable": False,
        "semantic_correct": False,
        "correct": False,
        "error": None,
    }
    if not native_eos or not padding_only:
        record["error"] = "NATIVE_TERMINATION"
        return record
    try:
        calls = parse_calls(text)
    except FormatError as error:
        record["error"] = str(error)
        return record
    record["format_valid"] = True
    try:
        state, trace = execute(calls, example.state, example.family, conventions)
    except ExecutionError as error:
        record["error"] = str(error)
        return record
    record.update(
        executable=True,
        semantic_correct=state == example.expected,
        correct=state == example.expected,
        final_state=state,
        trace=trace,
        canonical_calls_match=calls == example.calls,
    )
    return record


def render(example, conventions):
    schemas = {
        conventions[example.family][operation]: {key: value.__name__ for key, value in schema.items()}
        for operation, schema in SCHEMAS[example.family].items()
    }
    return (
        "Use the local tool convention learned from examples. Each call changes the sandbox state. "
        "Return only one JSON array of 1 to 4 calls. A call has exactly the keys tool and args. "
        "Execute the request in the stated order; preserve everything else. No explanation or markdown.\n"
        f"Domain: {example.family}\nAvailable tool argument types: {json.dumps(schemas, sort_keys=True)}\n"
        f"Initial state: {json.dumps(example.state, sort_keys=True)}\nRequest: {example.request}"
    )
