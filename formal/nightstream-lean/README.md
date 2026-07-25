# Nightstream Lean

This is the active assurance-first formalization for Nightstream. The sibling
Lean packages under `formal/` are legacy reference material, not dependencies,
and no theorem is inherited from them.

## Project map

- [`specs/formal-verification.md`](specs/formal-verification.md) is normative for
  the threat model, property matrix, evidence tiers, supported profiles, and
  completion gates.
- [`docs/architecture.md`](docs/architecture.md) defines layer ownership and
  import direction.
- [`docs/generated-files.md`](docs/generated-files.md) records generated-artifact
  ownership and regeneration commands.
- [`assurance/evidence-ledger.jsonl`](assurance/evidence-ledger.jsonl) records
  mutable evidence. Manifests beside it pin exact artifact identities.
- [`AGENTS.md`](AGENTS.md) contains project-local editing and validation rules.

## Milestone status

| Milestone | Current claim |
|---|---|
| M0 / M0.5 | Project specification, threat model, evidence discipline, and concrete relation foundations are complete for their named model profiles. |
| M1 | Concrete CCS/CE relations, global parameters, encoding, SumCheck, and executable checker equivalences are model-proved. |
| M2 | PiCCS/PiRLC/PiDEC folding shapes and their composed reduction expose explicit sampling, binding, and collision events. Probability bounds remain M6 work. |
| M3 | F' base/recursive/terminal semantics and exact trace induction establish `ValidExecution` for the advertised semantic interface. |
| M4 | `CIR-SOUND` and `CIR-COMPLETE` are artifact-checked for the exact plain, stateless, `[1,1]`, one-recursive-step, terminal-fold, direct-terminal-CE, minimal-bit-carrier profile. |
| M5 | The supported Rust-shaped F' and terminal verifier surfaces are proved equivalent to their Lean predicates and pinned by conformance manifests. |
| M6 | Open: compact-decider soundness, final verifier reduction, and probability bounds for recursive/terminal bad-root events. |

These are scoped claims, not an assertion that every production profile is
verified. M4 does not yet cover stateful mode, Nebula, other schedules, multiple
recursive invocations, alternate carriers, or a parameterized circuit family.
The public compact decider remains fail-closed with `Unsupported`.

Consult the property matrix and evidence ledger for theorem names, hashes,
assumptions, and current gates; do not treat this summary as normative status.

## Validation

Every Lean process is capped at 25 minutes. Run the structural checks, build,
fail-closed axiom report, and executable probes with:

```bash
./scripts/validate.sh all
```

Individual phases are `static`, `build`, `axioms`, and `check`. CI runs the same
bounded wrapper.
