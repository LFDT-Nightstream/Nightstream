# Nightstream implementation assurance map

The current source is [site/requirements.json](site/requirements.json). The [public site](https://nightstream-requirements.nicarq.chatgpt.site) and its Markdown download are generated from this snapshot. A repository update is not a deployment receipt; the live site's Source and evidence view identifies the published map revision.

The scope is to prove the selected SuperNeo implementation and the connections needed to validate Rust and reduce constraints. Keep cryptographic assumptions explicit, check their applicability, and calculate error for the intended number of uses. This map does not change protected goals, specifications or phase order.

## Read the current map

- **Proof:** Proved, Assumed, Open and N/A. Every in-scope leaf remains visible in the counts.
- **Link:** Connected, Open and N/A. A local connection does not establish full lifecycle closure.
- **Rust:** Scoped tests, Code only, Recorded, Open and N/A. Code existence does not count as a passed test.
- **Assumption ledger:** Shared premises, parameters, approval state and recorded dependent uses.
- **Error budget:** Conditional per-test bounds and accumulation over a supplied use count, with deployment parameters and remaining security terms shown separately.
- **Readiness:** Existing obligations needed for constraint reduction and complete Rust validation. These views add no proof credit.

The full native path, retained logical-assignment mutation failure, required independent approvals and production obligations remain open. Local diagnostics do not grant Compiler-closed, Conformance-closed or Production-closed status. Read the exact leaf scope and evidence before using a result.

Requirement `scope` annotations support filtering by model, interactive protocol, implementation, production or material outside Stage 1. They are map annotations, not automatically checked Lean statements. Parent/child links describe navigation; `depends_on` records uses of other results. Neither is a complete extracted proof graph.

## Build and check

Run from the repository root. The selected paper Markdown must be available in this checkout or in the checkout passed with `--paper-root`. Paper files absent from Git are identified by content hashes in the reference report.

```bash
python3 -B scripts/lean_graph/guard.py --kind static -- \
  python3 -B docs/reviews/nightstream-fprime-requirements/build_map.py \
  --paper-root /path/to/checkout-with-selected-papers

python3 -B scripts/lean_graph/guard.py --kind static \
  --cwd docs/reviews/nightstream-fprime-requirements/site -- \
  python3 -B -m unittest discover -s tests -p 'test_*.py'

python3 -B scripts/lean_graph/guard.py --kind static -- \
  node docs/reviews/nightstream-fprime-requirements/site/tests/test_assurance.cjs
```

The guard applies the repository's 300-second cap to each non-Lean check. No Lean, Rust or proof-backend build is needed for these website checks.

The build writes `site/dist`: HTML, source JSON, the Markdown index and group files, assumption/readiness/error/evidence views, publication metadata, reference report and a ZIP download. Group counts come from leaf statuses. The HTML and static download use the same data.

## Prepare publication

Commit the exact site source, including `site/requirements.json`, before running the publication build:

```bash
python3 -B scripts/lean_graph/guard.py --kind static -- \
  python3 -B docs/reviews/nightstream-fprime-requirements/build_map.py \
  --paper-root /path/to/checkout-with-selected-papers --publish
```

`--publish` rejects site inputs that do not match `HEAD` and reruns the reference check. It prepares files; it does not deploy them. Use the existing Sites project in `site/.openai/hosting.json` through an account with access. After deployment, compare the live `requirements.json` and `publication.json` with the prepared files.

The revisions have separate meanings:

| Field | Meaning |
| --- | --- |
| `provenance.code_commit` | Protocol source used by the map's code citations. |
| `publication.json: map_commit` | Commit that contains the exact site and map inputs. |
| `provenance.evidence[].code_commit` | Code revision checked by the recorded evidence run. |
| Evidence authority and performed-by fields | Who ran a check and whether it grants any independent approval. |

File/line checks and hashes identify source locations. They do not prove that a Lean statement matches the paper or that protocol evidence is current. The publication step does not issue independent approvals or replace conformance gates.

## Retained review snapshots

`MAP.md`, `OPEN_ITEMS.md`, the JSON files outside `site/`, and the earlier notes are historical review artifacts. Do not use them to regenerate current statuses. `build_map.py` now calls the canonical site build and leaves those snapshots unchanged.

The [conformance report](CONFORMANCE_FIXES.md) and [NIFS review](NIFS_TOP_THREE_PROGRESS.md) describe the retained evidence and its limits. The [reading guide](site/reading-guide.md) defines the current status and scope rules.
