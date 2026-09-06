"""Run the checks needed by one owner criterion; resume applicable results."""

import sys

from .policy import gate_order
from .records import gate_results, read_runs
from .runner import run_gate
from .reviews import require_review


def checkpoint(obligation, policy, manifest, snapshot, store, authority=None):
    require_review(obligation, policy, manifest, store, authority)
    selected = policy["obligations"][obligation]["gates"]
    order = gate_order(policy, selected)
    results = []
    ready = "pass" if authority else "completed"
    for name in order:
        runs, _ = read_runs(store, authority)
        state, _ = gate_results(policy, manifest, runs)
        needed = set()

        def visit(gate):
            if gate in needed or state[gate][ready]:
                return
            needed.add(gate)
            for dependency in policy["gates"][gate].get("requires", []):
                visit(dependency)
            precise = policy["gates"][gate].get("declaration_freshness")
            if precise:
                visit(precise["gate"])

        for root in selected:
            visit(root)
        if name not in needed:
            if state[name][ready]:
                results.append({"gate": name, "action": "reused", "record": state[name]["record"]["path"]})
                print(f"{name}: matching result reused.", file=sys.stderr, flush=True)
            continue
        print(f"{name}: running.", file=sys.stderr, flush=True)
        record = run_gate(name, policy, manifest, snapshot, store, authority)
        results.append({"gate": name, "action": "executed", "outcome": record["outcome"]})
        if record["outcome"] != "pass":
            return {"obligation": obligation, "execution": record["outcome"], "checks": results}
    return {"obligation": obligation, "execution": "passed", "checks": results}
