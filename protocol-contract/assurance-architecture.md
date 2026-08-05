# Nightstream protocol-contract architecture

Status: **normative maintenance architecture**.

This document defines ownership, data flow, and validation. The assembled
contract defines protocol behavior. The assurance graph states what evidence
exists for that behavior.

## 1. Irreducible constraints

The architecture follows these necessary facts:

1. Paper text, reviewed errata, and Nightstream choices have different
   authority.
2. A protocol rule, a verifier transition, and an assurance claim are
   different objects.
3. Semantic and assurance dependencies are acyclic. Protocol execution has
   bounded repetition.
4. A large reading view is useful, but it must not be a second authored source.
5. Digests provide identity and compression only.
6. Release state must follow from data. A checker must not require an open or
   blocked state.
7. One fact must have one authored owner.

## 2. Data flow

```text
authored modules
   |
   +-- requirements ----------> semantic requirement DAG
   +-- protocol records ------> verifier state machine
   +-- assurance records -----> assurance DAG and gates
   +-- profile ---------------> exact parameter checks
   +-- evidence -------------> rule-to-evidence edges
   |
typed loader + cross-model validator
   |
deterministic generated views + package manifest
```

`check_contract.py` owns the command line. `contract_model.py` owns the loaded
contract model. `contract_protocol.py` checks the state machine.
`contract_assurance.py` derives assurance and freshness. `contract_checks.py`
checks sources, profile arithmetic, schemas, and repository anchors.
`contract_render.py` creates all generated views. `contract_migration.py`
checks the sealed, lossless import.

## 3. Authored ownership

| Fact | One authored owner |
|---|---|
| Atomic normative words | `src/normative/*.md` |
| Literal paper item | `src/paper/*.md` |
| Rule kind, authority, dependency, replacement | `src/requirements/*.jsonl` |
| Nightstream decision | `src/decisions/decisions.jsonl` |
| Reviewed erratum meaning | `reviewed-errata.md` |
| Protocol state, event, challenge, failure, repetition, schedule | `src/protocol/*` |
| Concrete protocol value | `src/profile/*.toml` |
| Security planning value | `src/security/planning.toml` |
| Evidence applicability and anchor | `src/evidence/*.jsonl` |
| Assurance leaf, issue, review, gate | `src/assurance/*` |
| Source identity | `src/sources/lock.toml` |
| Module order and file role | `src/bundle.json` |

The validator limits authored files to 500 lines and package files to 1,500
lines. It also rejects duplicate IDs, duplicate normative clauses, mixed paper
and decision authority, cycles, redundant transitive edges, invalid
replacements, inconsistent dimensions, and stale generated files.

## 4. Generated views

Root-level contract views are generated. Their banner states this rule.

| View | Derived content |
|---|---|
| `superneo-v1.md` | Ordered final contract with replacement notes |
| `literal-paper-model.md` | Ordered paper extraction |
| `deviations.md` | Decision ledger and derived reverse impact |
| `rule-index.json` | Exact blocks, authority, graph, evidence, and hashes |
| `coverage.csv` | Per-rule evidence matrix |
| `requirement-graph.md` | Semantic DAG and reverse closure |
| `protocol-events.json` | States, events, challenges, failures, repetitions, and transcript schedule |
| `obligations.toml` | Assurance leaves and derived states |
| `obligation-graph.md` | Assurance critical path |
| `release.toml` | Gates and release flags |
| `generated/assurance-status.json` | Full derived assurance snapshot |
| `superneo-v1.toml` | Compatibility profile view |
| `MANIFEST.sha256` | Complete package byte census |

Generation is deterministic. A second refresh with no authored change must
produce no change.

## 5. Semantic requirement DAG

Each requirement record contains one stable rule ID, kind, assembly operation,
direct semantic dependencies, authority, decision blockers, and useful review
flags.

Assembly operations are:

- `adopt`: use a reviewed paper rule;
- `add`: add complete Nightstream behavior;
- `replace`: replace one named paper behavior with one complete rule;
- `remove`: remove one named behavior under an approved decision.

Paper rules and Nightstream changes stay separate in authored data. The
generated contract assembles the final reading view. This preserves provenance
without forcing a reader to merge prose manually.

The current model has 104 atomic rules and 170 direct semantic edges. A rule
has at most 25 lines and four normative keywords. These limits keep modules
small enough for review without splitting one inseparable equation into
several artificial facts.

## 6. Protocol state machine

The state machine is the exact fold-verifier order. It has 12 events, four
challenge families, and five bounded repetitions. Its separate recursive
schedule fixes every frame, squeeze, payload count, tag, and loop nesting. The
selected profile fixes:

- 24 PiCCS SumCheck rounds;
- 15 PiRLC sources;
- 54 coefficients per source;
- one to three sampler attempts per coefficient;
- one to 64 fold steps.

An event owns its input state, output state, inputs, outputs, rules, challenge
use, and rejection conditions. Each proof-rejection code has one registry row
and normative owner. A rejected sampler candidate is a local retry; exhaustion
has the proof-rejection code. A loop is a repetition record, not a cycle in the
semantic or assurance DAG.

Circuit correspondence and deployed verification are assurance edges. They
are not appended as events after native fold acceptance.

## 7. Assurance DAG

An assurance leaf owns only authored facts:

```text
applicability
evidence state
direct claim dependencies
evidence locations
stable issue or decision IDs
```

The tool derives dependency state, blocker state, closure state, display state,
freshness, gate state, implementation readiness, and production eligibility.
Maintainers do not type a derived `blocked` value.

High-level assurance arrows are rollups over leaf claims. Their scopes are
derived by selectors. They do not copy hundreds of rule IDs.

## 8. Evidence and freshness

Evidence locations are not proof. Each per-rule evidence row has one
applicability value, one assurance level, and optional exact file and
declaration anchors. A directory path is an owner location, not a theorem
anchor.

A closure review receipt binds:

- the claim ID;
- the semantic contract hash;
- the exact profile hash;
- a digest of the claim's current evidence set;
- reviewer identity, role, method, and time.

The checker recomputes these values. A stale receipt reopens the claim. The
refresh command does not create or update review receipts.

The semantic contract hash covers normative modules and requirement metadata
with framed input boundaries. The profile hash covers profile TOML files,
security planning, and protocol-machine records. A Markdown rendering change
alone cannot silently redefine either identity.

## 9. Gate meaning

| Gate | Meaning |
|---|---|
| G0 | Reviewed source bytes and exact errata derivation |
| G0B | Paper extraction and normalization review |
| G1 | Complete approved Nightstream design |
| G2 | Lean propositions prove the selected semantic rules |
| G3 | Pinned Rust verifier and Rust-origin refinement evidence |
| G4 | Current circuit correspondence in both directions |
| G5 | Concrete deployed-verifier security reduction |

G1 is the implementation-ready boundary. G5 is the production-claim boundary.
Closing G1 must not change G2 through G5 to closed.

## 10. Change procedure

For each authored change:

1. change the one owning record;
2. refresh generated views;
3. inspect the generated difference;
4. run package and relevant repository checks;
5. repeat affected reviews and issue new receipts;
6. rebuild downstream evidence whose contract or profile identity changed.

A schema-valid file, matching digest, passing fixture, or existing declaration
cannot close a stronger assurance edge by itself.
