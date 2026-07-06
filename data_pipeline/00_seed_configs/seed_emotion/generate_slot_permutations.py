from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import combinations, islice
from math import comb
from pathlib import Path
import unicodedata
from typing import Any, Dict, Iterable, Iterator, List, Tuple


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_path_part(part: str) -> str:
    return part[:-2] if part.endswith("[]") else part


def set_nested_value(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = [normalize_path_part(part) for part in path.split(".")]
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def get_nested_value(payload: Dict[str, Any], path: str) -> Any:
    parts = [normalize_path_part(part) for part in path.split(".")]
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def auto_request_id(config: Dict[str, Any], combo_index: int) -> str:
    request_id = config["request_id"]
    start_index = int(request_id.get("start_index", 1))
    padding = int(request_id.get("padding", 6))
    numeric_id = start_index + combo_index - 1
    return f"{request_id['prefix']}_{numeric_id:0{padding}d}"


def option_values(field: Dict[str, Any]) -> List[str]:
    return [option["value"] for option in field["options"]]


def option_weights(field: Dict[str, Any]) -> List[float]:
    return [float(option.get("weight", 1)) for option in field["options"]]


def multi_select_values(field: Dict[str, Any]) -> Iterator[List[str]]:
    values = option_values(field)
    for size in field["selection_sizes"]:
        for combo in combinations(values, size):
            yield list(combo)


def activation_condition(field: Dict[str, Any]) -> Dict[str, Any] | None:
    active_when = field.get("active_when")
    legacy_condition = field.get("condition") if field["shape"] == "conditional_select" else None

    if active_when and legacy_condition:
        return {"all": [active_when, legacy_condition]}
    return active_when or legacy_condition


def condition_dependencies(condition: Dict[str, Any] | None) -> set[str]:
    if not condition:
        return set()
    if "all" in condition:
        return set().union(*(condition_dependencies(item) for item in condition["all"]))
    if "any" in condition:
        return set().union(*(condition_dependencies(item) for item in condition["any"]))
    if "field" in condition:
        return {condition["field"]}
    raise ValueError(f"Unsupported condition: {condition}")


def condition_is_met(condition: Dict[str, Any] | None, selected: Dict[str, Any]) -> bool:
    if not condition:
        return True
    if "all" in condition:
        return all(condition_is_met(item, selected) for item in condition["all"])
    if "any" in condition:
        return any(condition_is_met(item, selected) for item in condition["any"])

    field_name = condition.get("field")
    if field_name is None:
        raise ValueError(f"Unsupported condition: {condition}")
    if field_name not in selected:
        return False

    value = selected[field_name]
    has_equals = "equals" in condition
    has_not_equals = "not_equals" in condition
    if has_equals == has_not_equals:
        raise ValueError(f"Condition must define exactly one comparator: {condition}")
    if has_equals:
        return value == condition["equals"]
    return value != condition["not_equals"]


def is_field_active(field: Dict[str, Any], selected: Dict[str, Any]) -> bool:
    return condition_is_met(activation_condition(field), selected)


def condition_to_text(condition: Dict[str, Any] | None, key_to_name: Dict[str, str]) -> str:
    if not condition:
        return "始终生效"
    if "all" in condition:
        return " 且 ".join(condition_to_text(item, key_to_name) for item in condition["all"])
    if "any" in condition:
        return " 或 ".join(condition_to_text(item, key_to_name) for item in condition["any"])

    field_name = condition["field"]
    display_name = key_to_name.get(field_name, field_name)
    if "equals" in condition:
        return f"{display_name} = {condition['equals']}"
    if "not_equals" in condition:
        return f"{display_name} != {condition['not_equals']}"
    raise ValueError(f"Unsupported condition: {condition}")


def validate_config(config: Dict[str, Any]) -> None:
    seen_keys: set[str] = set()

    for field in config["fields"]:
        weights = option_weights(field)
        option_count = len(field["options"])
        if not weights:
            raise ValueError(f"Field {field['key']} must define at least one option.")
        if any(weight < 0 for weight in weights):
            raise ValueError(f"Field {field['key']} has negative option weights.")
        if sum(weights) <= 0:
            raise ValueError(f"Field {field['key']} must have at least one positive option weight.")

        selection_sizes = field.get("selection_sizes", [])
        selection_size_weights = field.get("selection_size_weights", [])
        for size in field.get("selection_sizes", []):
            if size < 0 or size > option_count:
                raise ValueError(
                    f"Invalid selection size {size} for field {field['key']}; "
                    f"expected 0 <= size <= {option_count}."
                )
        if selection_size_weights and len(selection_sizes) != len(selection_size_weights):
            raise ValueError(
                f"Field {field['key']} must align selection_sizes and selection_size_weights."
            )
        if any(weight < 0 for weight in selection_size_weights):
            raise ValueError(f"Field {field['key']} has negative selection size weights.")
        if selection_sizes and selection_size_weights and sum(selection_size_weights) <= 0:
            raise ValueError(
                f"Field {field['key']} must have at least one positive selection size weight."
            )

        for dependency in condition_dependencies(activation_condition(field)):
            if dependency not in seen_keys:
                raise ValueError(
                    f"Field {field['key']} depends on {dependency}, "
                    "but dependencies must be defined earlier in config['fields']."
                )

        seen_keys.add(field["key"])


def iter_field_values(field: Dict[str, Any], selected: Dict[str, Any]) -> Iterable[Any]:
    if not is_field_active(field, selected):
        return [None]

    shape = field["shape"]
    if shape in {"single_select", "conditional_select"}:
        return option_values(field)
    if shape == "multi_select":
        return multi_select_values(field)
    raise ValueError(f"Unsupported field shape: {shape}")


def field_domain_size(field: Dict[str, Any]) -> int:
    shape = field["shape"]
    if shape in {"single_select", "conditional_select"}:
        return len(field["options"])
    if shape == "multi_select":
        option_count = len(field["options"])
        return sum(comb(option_count, size) for size in field["selection_sizes"])
    raise ValueError(f"Unsupported field shape: {shape}")


def weighted_index(weights: List[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weighted sampling requires a positive total weight.")
    return rng.choices(range(len(weights)), weights=weights, k=1)[0]


def weighted_sample_without_replacement(
    values: List[str],
    weights: List[float],
    sample_size: int,
    rng: random.Random,
) -> List[str]:
    remaining = list(zip(values, weights))
    sampled: List[Tuple[int, str]] = []

    for _ in range(sample_size):
        remaining_weights = [max(weight, 0.0) for _, weight in remaining]
        if sum(remaining_weights) <= 0:
            chosen_index = rng.randrange(len(remaining))
        else:
            chosen_index = weighted_index(remaining_weights, rng)
        value, _ = remaining.pop(chosen_index)
        sampled.append((len(sampled), value))

    selected_values = {value for _, value in sampled}
    return [value for value in values if value in selected_values]


def random_field_value(field: Dict[str, Any], selected: Dict[str, Any], rng: random.Random) -> Any:
    if not is_field_active(field, selected):
        return None

    shape = field["shape"]
    if shape in {"single_select", "conditional_select"}:
        values = option_values(field)
        return values[weighted_index(option_weights(field), rng)]
    if shape == "multi_select":
        selection_sizes = field["selection_sizes"]
        selection_size_weights = field.get("selection_size_weights") or [1] * len(selection_sizes)
        sample_size = selection_sizes[weighted_index(selection_size_weights, rng)]
        return weighted_sample_without_replacement(
            option_values(field),
            option_weights(field),
            sample_size,
            rng,
        )
    raise ValueError(f"Unsupported field shape: {shape}")


def dependency_keys_from_index(fields: List[Dict[str, Any]]) -> List[set[str]]:
    dependencies: List[set[str]] = [set() for _ in range(len(fields) + 1)]
    needed: set[str] = set()
    dependencies[len(fields)] = set()
    for index in range(len(fields) - 1, -1, -1):
        needed.update(condition_dependencies(activation_condition(fields[index])))
        dependencies[index] = set(needed)
    return dependencies


def freeze_state(state: Dict[str, Any], needed_keys: set[str]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((key, value) for key, value in state.items() if key in needed_keys))


def count_total_combinations(config: Dict[str, Any]) -> int:
    fields = config["fields"]
    dependencies = dependency_keys_from_index(fields)
    state_counts: Dict[Tuple[Tuple[str, Any], ...], int] = {tuple(): 1}

    for index, field in enumerate(fields):
        current_key = field["key"]
        needed_after = dependencies[index + 1]
        keep_current = current_key in needed_after
        next_state_counts: Dict[Tuple[Tuple[str, Any], ...], int] = {}

        if keep_current:
            for frozen_state, count in state_counts.items():
                state = dict(frozen_state)
                for value in iter_field_values(field, state):
                    if value is None:
                        state.pop(current_key, None)
                    else:
                        state[current_key] = value
                    new_state = freeze_state(state, needed_after)
                    next_state_counts[new_state] = next_state_counts.get(new_state, 0) + count
                state.pop(current_key, None)
        else:
            for frozen_state, count in state_counts.items():
                state = dict(frozen_state)
                branch_size = 1 if not is_field_active(field, state) else field_domain_size(field)
                new_state = freeze_state(state, needed_after)
                next_state_counts[new_state] = next_state_counts.get(new_state, 0) + count * branch_size

        state_counts = next_state_counts

    return sum(state_counts.values())


def field_domain_summary(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    summary: List[Tuple[str, str]] = []
    key_to_name = {field["key"]: field["name"] for field in config["fields"]}

    for field in config["fields"]:
        domain_size = field_domain_size(field)
        condition = activation_condition(field)
        if condition:
            summary.append(
                (
                    field["name"],
                    f"{domain_size}（当 {condition_to_text(condition, key_to_name)} 时生效）",
                )
            )
        else:
            summary.append((field["name"], str(domain_size)))
    return summary


def build_payload(config: Dict[str, Any], selected: Dict[str, Any], combo_index: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    set_nested_value(payload, config["request_id"]["path"], auto_request_id(config, combo_index))

    for field in config["fields"]:
        active = is_field_active(field, selected)
        if not active and field.get("omit_when_inactive"):
            continue
        value = selected.get(field["key"]) if active else None
        set_nested_value(payload, field["path"], value)

    return payload


def iter_cartesian_payloads(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    fields = config["fields"]

    def backtrack(index: int, selected: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if index == len(fields):
            yield dict(selected)
            return

        field = fields[index]
        for value in iter_field_values(field, selected):
            if value is None:
                selected.pop(field["key"], None)
            else:
                selected[field["key"]] = value

            yield from backtrack(index + 1, selected)

        selected.pop(field["key"], None)

    for combo_index, selected in enumerate(backtrack(0, {}), start=1):
        yield build_payload(config, selected, combo_index)


def iter_random_payloads(config: Dict[str, Any], rng: random.Random) -> Iterator[Dict[str, Any]]:
    fields = config["fields"]
    combo_index = 1

    while True:
        selected: Dict[str, Any] = {}
        for field in fields:
            value = random_field_value(field, selected, rng)
            if value is None:
                selected.pop(field["key"], None)
            else:
                selected[field["key"]] = value

        yield build_payload(config, selected, combo_index)
        combo_index += 1


def choose_print_records(records: List[Dict[str, Any]], print_limit: int, seed: int) -> List[Dict[str, Any]]:
    if print_limit <= 0 or not records:
        return []
    if print_limit >= len(records):
        return records

    print_rng = random.Random(seed + 1)
    sampled_indices = sorted(print_rng.sample(range(len(records)), k=print_limit))
    return [records[index] for index in sampled_indices]


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total:.1%}"


def summarize_counter(counter: Counter[str], total: int, top_k: int = 5) -> str:
    if total <= 0 or not counter:
        return "-"
    return "; ".join(
        f"{value}={count} ({format_ratio(count, total)})"
        for value, count in counter.most_common(top_k)
    )


def format_combo_label(values: List[str], preview_items: int = 3) -> str:
    if len(values) <= preview_items:
        return " | ".join(values) if values else "(空)"
    return f"{' | '.join(values[:preview_items])} ... 共{len(values)}项"


def build_distribution_summary(field: Dict[str, Any], values: List[Any]) -> Tuple[int, int, str]:
    active_values = [value for value in values if value is not None]
    active_count = len(active_values)
    shape = field["shape"]

    if active_count == 0:
        return 0, 0, "-"

    if shape == "multi_select":
        combo_counter = Counter(format_combo_label(value) for value in active_values)
        option_counter: Counter[str] = Counter()
        size_counter: Counter[str] = Counter()
        for value in active_values:
            size_counter[f"{len(value)}项"] += 1
            option_counter.update(value)

        unique_count = len(combo_counter)
        size_summary = summarize_counter(size_counter, active_count, top_k=3)
        option_summary = summarize_counter(option_counter, active_count, top_k=4)
        summary = f"组合Top: {summarize_counter(combo_counter, active_count, top_k=2)}"
        summary += f" | 选项频次: {option_summary}"
        summary += f" | 选择数: {size_summary}"
        return active_count, unique_count, summary

    value_counter = Counter(str(value) for value in active_values)
    return active_count, len(value_counter), summarize_counter(value_counter, active_count, top_k=5)


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def pad_display(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def build_distribution_rows(config: Dict[str, Any], records: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for field in config["fields"]:
        values = [get_nested_value(record, field["path"]) for record in records]
        active_count, unique_count, summary = build_distribution_summary(field, values)
        rows.append(
            [
                field["name"],
                field["shape"],
                f"{active_count}/{len(records)} ({format_ratio(active_count, len(records))})",
                str(unique_count),
                summary,
            ]
        )
    return rows


def print_distribution_table(config: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    print("distribution_table:")
    if not records:
        print("  (no records)")
        return

    rows = build_distribution_rows(config, records)
    headers = ["参数", "类型", "生效样本", "唯一值数", "分布概览"]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(columns: List[str]) -> str:
        return " | ".join(
            pad_display(column, widths[index]) for index, column in enumerate(columns)
        )

    separator = "-+-".join("-" * width for width in widths)
    print(f"  {format_row(headers)}")
    print(f"  {separator}")
    for row in rows:
        print(f"  {format_row(row)}")


def build_statistics_markdown(
    config: Dict[str, Any],
    records: List[Dict[str, Any]],
    config_path: Path,
    sampling_mode: str,
    seed: int | None,
    sample_limit: int,
    theoretical_total: int,
) -> str:
    lines = [
        "# Statistics",
        "",
        "## Overview",
        "",
        f"- config: `{config_path}`",
        f"- fields: `{len(config['fields'])}`",
        f"- sampling_mode: `{sampling_mode}`",
    ]
    if seed is not None:
        lines.append(f"- seed: `{seed}`")
    lines.extend(
        [
            f"- theoretical_total_combinations: `{theoretical_total:,}`",
            f"- generated_count: `{len(records)}`",
            f"- requested_sample_limit: `{sample_limit}`",
            "",
            "## Distribution Table",
            "",
        ]
    )

    headers = ["参数", "类型", "生效样本", "唯一值数", "分布概览"]
    rows = build_distribution_rows(config, records)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        escaped = [column.replace("|", "\\|") for column in row]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.append("")
    return "\n".join(lines)


def statistics_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.statistic.md")
    return output_path.with_name(f"{output_path.name}.statistic.md")


def write_text_file(content: str, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slot label packages from config.json.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the label config JSON."
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="How many records to generate."
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=5,
        help="How many generated records to print. Use 0 to disable record printing."
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("cartesian", "random"),
        default="cartesian",
        help="Generate records by deterministic cartesian enumeration or weighted random sampling."
    )
    parser.add_argument(
        "--output-jsonl",
        help="Optional JSONL output path for the preview subset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sampling-mode=random."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config)
    total = count_total_combinations(config)
    summary = field_domain_summary(config)

    print(f"config: {config_path}")
    print(f"fields: {len(config['fields'])}")
    print(f"sampling_mode: {args.sampling_mode}")
    if args.sampling_mode == "random":
        print(f"seed: {args.seed}")
    print(f"theoretical_total_combinations: {total:,}")
    print("domain_breakdown:")
    for name, size in summary:
        print(f"  - {name}: {size}")
    print(f"generated_count: {args.sample_limit}")
    print(f"print_limit: {args.print_limit}")

    if args.sampling_mode == "random":
        rng = random.Random(args.seed)
        records = list(islice(iter_random_payloads(config, rng), args.sample_limit))
    else:
        records = list(islice(iter_cartesian_payloads(config), args.sample_limit))

    if args.print_limit > 0:
        print_records = choose_print_records(records, args.print_limit, args.seed)
        print(json.dumps(print_records, ensure_ascii=False, indent=2))
        if len(records) > len(print_records):
            print(
                f"printed_records: {len(print_records)} / {len(records)} "
                f"(remaining records omitted from stdout)"
            )
    else:
        print("printed_records: 0 (stdout record printing disabled)")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        written = write_jsonl(records, output_path)
        print(f"wrote_jsonl: {output_path} ({written} records)")
        statistics_path = statistics_output_path(output_path)
        statistics_markdown = build_statistics_markdown(
            config=config,
            records=records,
            config_path=config_path,
            sampling_mode=args.sampling_mode,
            seed=args.seed if args.sampling_mode == "random" else None,
            sample_limit=args.sample_limit,
            theoretical_total=total,
        )
        write_text_file(statistics_markdown, statistics_path)
        print(f"wrote_statistics: {statistics_path}")

    print_distribution_table(config, records)


if __name__ == "__main__":
    main()
