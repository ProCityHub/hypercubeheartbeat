#!/usr/bin/env python3
"""Deterministic QASM/thesis reviewer for Hypercube Heartbeat.

Author and concept origin: Adrien D. Thomas / ProCityHub.
This is a scientific inspection tool, not a consciousness or AGI claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PHI = (1 + math.sqrt(5)) / 2
REFERENCES = {
    "phi": PHI,
    "inverse_phi": 1 / PHI,
    "inverse_phi_squared": 1 / PHI**2,
    "half_phi": PHI / 2,
    "pi_over_6": math.pi / 6,
    "observer_angle": 1.8091,
    "actor_angle": 1.3325,
}
CONCEPTS = {
    "consciousness": ("conscious",),
    "observer": ("observer", "origin"),
    "actor": ("actor",),
    "bridge": ("bridge",),
    "memory_latency": ("memory", "latency", "delay", "echo"),
    "integration": ("integration", "integrated information"),
    "uncertainty_noise": ("uncertainty", "noise"),
    "heartbeat_recursion": ("heartbeat", "recursive", "recursion"),
    "phi": ("phi", "golden ratio"),
    "double_slit": ("double slit", "double-slit", "slit"),
    "attractor_collapse": ("attractor", "collapse"),
    "mirror_cube_lattice": ("mirror", "wall", "cube", "lattice"),
}


def number(expr: str):
    allowed = {
        ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }

    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi": return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand); return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError

    try: return float(visit(ast.parse(expr, mode="eval")))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError): return None


def split_csv(text: str):
    out, depth, start = [], 0, 0
    for i, char in enumerate(text):
        depth += char == "("; depth -= char == ")"
        if char == "," and depth == 0:
            out.append(text[start:i].strip()); start = i + 1
    if text[start:].strip(): out.append(text[start:].strip())
    return out


def expand(ref: str, registers: dict[str, int]):
    ref = ref.strip()
    match = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", ref)
    if match: return [f"{match.group(1)}[{int(match.group(2))}]"]
    if ref in registers: return [f"{ref}[{i}]" for i in range(registers[ref])]
    return [ref]


def broadcast(groups):
    width = max((len(group) for group in groups), default=1)
    if any(len(group) not in (1, width) for group in groups): raise ValueError("register width mismatch")
    return [tuple(group[0] if len(group) == 1 else group[i] for group in groups) for i in range(width)]


def parse_qasm(text: str, source="<memory>"):
    qregs, cregs, operations, comments, warnings = {}, {}, [], [], []
    reg_re = re.compile(r"^(qreg|creg)\s+(\w+)\[(\d+)\]\s*;$")
    measure_re = re.compile(r"^measure\s+(.+?)\s*->\s*(.+?)\s*;$", re.I)
    op_re = re.compile(r"^(\w+)(?:\((.*?)\))?\s+(.+?)\s*;$", re.I)
    for line_no, original in enumerate(text.splitlines(), 1):
        code, marker, comment = original.partition("//")
        if marker and comment.strip(): comments.append(comment.strip())
        code = code.strip()
        if not code or code.lower().startswith(("openqasm ", "include ")): continue
        match = reg_re.fullmatch(code)
        if match:
            (qregs if match.group(1) == "qreg" else cregs)[match.group(2)] = int(match.group(3)); continue
        match = measure_re.fullmatch(code)
        if match:
            try: rows = broadcast((expand(match.group(1), qregs), expand(match.group(2), cregs)))
            except ValueError as error: warnings.append(f"line {line_no}: {error}"); continue
            for qubit, bit in rows:
                operations.append({"name": "measure", "params": [], "qubits": [qubit], "classical": [bit], "line": line_no})
            continue
        match = op_re.fullmatch(code)
        if not match: warnings.append(f"line {line_no}: unsupported statement: {code}"); continue
        name, params, operands = match.groups()
        try: rows = broadcast([expand(item, qregs) for item in split_csv(operands)])
        except ValueError as error: warnings.append(f"line {line_no}: {error}"); continue
        for row in rows:
            operations.append({
                "name": name.lower(), "params": split_csv(params) if params else [],
                "qubits": [item for item in row if item.split("[")[0] in qregs],
                "classical": [item for item in row if item.split("[")[0] in cregs], "line": line_no,
            })
    return {"source": source, "qregs": qregs, "cregs": cregs, "operations": operations,
            "comments": comments, "parse_warnings": warnings}


def analyze(circuit):
    counts = Counter(op["name"] for op in circuit["operations"])
    qubits = [f"{name}[{i}]" for name, size in circuit["qregs"].items() for i in range(size)]
    edges, degree = set(), {q: 0 for q in qubits}
    parent = {q: q for q in qubits}

    def find(q):
        while parent[q] != q: parent[q] = parent[parent[q]]; q = parent[q]
        return q

    layers = defaultdict(int); depth = 0
    for op in circuit["operations"]:
        qs = op["qubits"]
        if qs:
            layer = max((layers[q] for q in qs), default=0) + 1
            for q in qs: layers[q] = layer
            depth = max(depth, layer)
        if len(qs) >= 2:
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    a, b = sorted((qs[i], qs[j]))
                    if (a, b) not in edges: edges.add((a, b)); degree[a] += 1; degree[b] += 1
                    ra, rb = find(a), find(b)
                    if ra != rb: parent[rb] = ra
    components = len({find(q) for q in qubits}) if qubits else 0
    values, edge_count, connected = list(degree.values()), len(edges), components == 1
    if not edges: topology = "no-multi-qubit-coupling"
    elif connected and len(qubits) > 2 and edge_count == len(qubits) and all(v == 2 for v in values): topology = "ring"
    elif connected and edge_count == len(qubits) - 1 and max(values) <= 2: topology = "chain"
    elif connected and max(values) == len(qubits) - 1: topology = "star"
    else: topology = "connected-mixed" if connected else "disconnected-mixed"

    corpus = "\n".join([circuit["source"], *circuit["comments"]]).lower()
    mentions = {name: any(term in corpus for term in terms) for name, terms in CONCEPTS.items()}
    status = {
        "consciousness": "comment_only_not_observable", "observer": "semantic_node_label",
        "actor": "semantic_node_label", "bridge": "coupling_proxy_not_validation",
        "memory_latency": "sequence_and_phase_proxy", "integration": "connectivity_proxy",
        "uncertainty_noise": "coherent_uncertainty_not_noise_channel",
        "heartbeat_recursion": "ordered_sequence_proxy", "phi": "parameterized_hypothesis",
        "double_slit": "topology_analogy_only", "attractor_collapse": "measurement_present_attractor_unvalidated",
        "mirror_cube_lattice": "connectivity_topology_proxy",
    }
    assessments = [{"concept": name, "status": status[name]} for name in CONCEPTS if mentions[name]]
    angles = []
    for op in circuit["operations"]:
        for expr in op["params"]:
            value = number(expr); nearest = delta = None
            if value is not None:
                nearest, target = min(REFERENCES.items(), key=lambda item: abs(value - item[1])); delta = abs(value - target)
            angles.append({"gate": op["name"], "qubits": op["qubits"], "expression": expr,
                           "value": value, "nearest_reference": nearest, "delta": delta})
    return {
        "qubit_count": len(qubits), "classical_bit_count": sum(circuit["cregs"].values()),
        "operation_count": len(circuit["operations"]), "gate_counts": dict(sorted(counts.items())),
        "measurement_count": counts["measure"], "approximate_source_depth": depth,
        "topology": topology, "coupling_edges": [list(edge) for edge in sorted(edges)],
        "connected_components": components, "angles": angles, "concept_mentions": mentions,
        "semantic_assessment": assessments,
        "required_controls": [
            "Topology/depth-matched circuit with identical gate families and measurement basis.",
            "Fixed backend, layout, routing, optimization, transpiler seed, shots, and calibration window.",
            "Angle-replacement family: 0, 1.0, 1.5, pi/2, 2.0, and seeded random fixed values.",
            "Rotation-axis, coupling-ablation, edge-shuffle, and angle-permutation controls.",
            "Raw counts, bit order, source/transpiled hashes, software versions, job IDs, and calibration preserved.",
        ],
        "falsification_conditions": [
            "Arbitrary replacement angles perform equivalently to or better than the original values.",
            "The signature disappears under repeated topology-matched runs.",
            "Transpilation, readout bias, drift, or ordinary topology explains the result.",
            "The effect fails on a held-out backend, date, seed, or circuit instance.",
        ],
    }


def theory_record(path):
    path = Path(path); text = path.read_text(encoding="utf-8"); normalized = re.sub(r"\s+", " ", text.lower())
    return {"source": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(), "mentions": {
        "exponent_formula": "1/phi" in normalized and ("phi^2" in normalized or "φ²" in text),
        "scalar_phi_formula": bool(re.search(r"o\s*[×*]\s*a\s*[×*]\s*b\s*[×*]\s*(phi|φ)", normalized)),
        "retraction": "retract" in normalized, "consciousness": "conscious" in normalized,
        "preregistration": "preregister" in normalized or "pre-register" in normalized,
    }}


def build_review(qasm_paths, theory_paths=(), question="Review the thesis and QASM."):
    circuits = []
    for path in qasm_paths:
        circuit = parse_qasm(Path(path).read_text(encoding="utf-8"), str(path)); circuit["analysis"] = analyze(circuit); circuits.append(circuit)
    theories = [theory_record(path) for path in theory_paths]
    scalar = any(item["mentions"]["scalar_phi_formula"] for item in theories)
    exponent = any(item["mentions"]["exponent_formula"] for item in theories)
    conflicts = ["Scalar-phi and exponent-weighted O/A/B formulas both appear; they are not equivalent."] if scalar and exponent else []
    return {
        "schema_version": "1.0", "author_and_concept_origin": "Adrien D. Thomas / ProCityHub",
        "instrument_status": "deterministic_structured_reviewer_not_conscious", "question": question,
        "direct_answer": "The strongest coherent interpretation is layered: Artifact Cube and O/A/B organize the model; heartbeat and memory define temporal hypotheses; QASM encodes rotations, phase, coupling, recurrence, and measurement; claims and preregistration control truth status; GARVIS must provide a separate semantic language layer. The gates do not encode consciousness as an observable. The weakest assumptions are the label-to-physics mapping and unique phi effect.",
        "formula_conflicts": conflicts, "circuits": circuits, "theory_files": theories,
        "strongest_case_against": ["Comments may assign post-hoc meanings to ordinary gates.", "Topology, transpilation, drift, and readout error can explain dominant bitstrings.", "Arbitrary fixed angles may perform like phi-associated values."],
        "smallest_next_experiment": {"title": "Topology-matched angle replacement pilot", "failure_condition": "Original angles do not outperform matched replacements on held-out repeated runs.", "claim_boundary": "A positive result supports a circuit-specific parameter effect, not consciousness, AGI, or new physics."},
        "limitations": ["Rule-based review, not an LLM or subjective opinion.", "Source depth is not transpiled hardware depth.", "No simulation, hardware run, statistical test, or claim upgrade occurs."],
    }


def markdown(review):
    lines = ["# Hypercube QASM and Thesis Review", "", f"**Author:** {review['author_and_concept_origin']}", f"**Status:** `{review['instrument_status']}`", "", "## Direct answer", "", review["direct_answer"], "", "## Formula conflicts", ""]
    lines += [f"- {item}" for item in review["formula_conflicts"]] or ["- None detected in supplied theory files."]
    for circuit in review["circuits"]:
        a = circuit["analysis"]; lines += ["", f"## `{circuit['source']}`", "", f"- Qubits: {a['qubit_count']}", f"- Operations: {a['operation_count']}", f"- Source depth: {a['approximate_source_depth']}", f"- Topology: `{a['topology']}`", f"- Gate counts: `{json.dumps(a['gate_counts'], sort_keys=True)}`", "", "### Gate meaning versus intended meaning", ""]
        lines += [f"- **{item['concept']}** — `{item['status']}`" for item in a["semantic_assessment"]]
        lines += ["", "### Required controls", ""] + [f"- {item}" for item in a["required_controls"]]
        lines += ["", "### Falsification conditions", ""] + [f"- {item}" for item in a["falsification_conditions"]]
    lines += ["", "## Strongest case against", ""] + [f"- {item}" for item in review["strongest_case_against"]]
    lines += ["", "## Smallest next experiment", "", f"**{review['smallest_next_experiment']['title']}**", "", f"- Failure: {review['smallest_next_experiment']['failure_condition']}", f"- Boundary: {review['smallest_next_experiment']['claim_boundary']}", "", "## Limitations", ""] + [f"- {item}" for item in review["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qasm", nargs="*"); parser.add_argument("--qasm-root", action="append", default=[])
    parser.add_argument("--theory", action="append", default=[]); parser.add_argument("--question", default="Review the thesis and QASM.")
    parser.add_argument("--output-dir", default="qasm_thesis_review"); parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv); paths = list(args.qasm)
    for root in args.qasm_root: paths += [str(path) for path in sorted(Path(root).rglob("*.qasm"))]
    paths = list(dict.fromkeys(paths))
    if not paths: parser.error("No QASM files found")
    review = build_review(paths, args.theory, args.question); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = markdown(review); (output / "review.md").write_text(report, encoding="utf-8")
    if args.stdout: print(report, end="")
    else: print(f"Wrote {output / 'review.json'}\nWrote {output / 'review.md'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
