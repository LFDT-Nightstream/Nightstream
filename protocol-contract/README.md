# Nightstream SuperNeo protocol contract

Status: **implementation specification selected; production assurance open**.

This package defines the protocol that Nightstream must implement. It does not
state that the current Lean model, Rust verifier, circuit, decider, or deployed
verifier conforms to the contract.

## Assurance path

```text
reviewed SuperNeo paper + reviewed errata + approved Nightstream decisions
                                  |
                       normative protocol contract
                          /                  \
                Lean semantic model       Rust verifier
                          \                  /
                   Rust-origin conformance evidence
                                  |
                    current circuit correspondence
                                  |
                      end-to-end security reduction
```

Each arrow is an evidence boundary. A digest identifies data. It does not prove
the meaning or correctness of the data.

## Selected v1 result

Nightstream v1 uses `PaddedRowIdentity`:

- one 24-variable row cube;
- exact positive logical dimensions from the verifier-key relation artifact;
- a 54-aligned full committed assignment `z=x||w` of at most 16,777,206 fields;
- the implicit `M_0=[I_m;0]` followed by 13 artifact-owned application matrices;
- one reviewed joint PiCCS SumCheck with 24 rounds;
- the norm terminal from `ct(y_(i,0))` for the same committed assignment;
- no FE/NC split, column proof, column carrier, or extra beta challenge;
- 17 PiRLC ring challenges from one bounded Poseidon2 sampler;
- 16 deterministic signed-binary PiDEC children.

The seeded Ajtai key also has one exact ChaCha8 row-and-chunk expansion with
checked stream and chunk-boundary vectors. Its commitment operation is the
fixed left matrix-vector product.

The package contains 104 atomic normative rules. It also fixes the canonical
Structure stream, verifier-key digest, encoding, Poseidon2 parameters, 12-event
fold transcript, exact nested transcript schedule, sampler bounds, section
sizes, nine-field circuit public image, exact statement prehash, threat model,
and implementation evidence boundary.

## What is ready

G0, G0B, and G1 define the implementation-ready boundary:

- G0 proves byte provenance and exact errata application.
- G0B records the paper extraction and normalization review.
- G1 records all approved and integrated Nightstream decisions.

When these gates are closed by current review receipts, implementation can use
this contract as its target. This state does not permit a production security
claim. G2 through G5 require new Lean, Rust-origin, circuit, decider, and
security evidence.

## Editing model

Edit authored files under `src/`. Do not edit generated root views.

| Authored area | Owner |
|---|---|
| `src/normative/` | Atomic normative text |
| `src/paper/` | Literal reviewed-paper extraction |
| `src/requirements/` | Requirement authority and semantic DAG |
| `src/decisions/` | Nightstream design decisions |
| `src/protocol/` | States, events, challenges, failures, bounded repetitions, and exact transcript schedule |
| `src/profile/` | Exact algebra, encoding, transcript, and backend profile |
| `src/evidence/` | Per-rule evidence applicability and exact anchors |
| `src/assurance/` | Claims, issues, reviews, artifacts, gates, and rollups |
| `src/security/` | Planning arithmetic and open reduction terms |
| `src/sources/` | Source lock |
| `paper-sources/` | Immutable reviewed paper bytes and exact errata patch |

Important generated views include:

- `superneo-v1.md`, the assembled normative contract;
- `rule-index.json`, the exact machine-readable rule index;
- `requirement-graph.md`, the semantic requirement DAG;
- `protocol-events.json`, the state-machine and transcript-schedule view;
- `deviations.md`, the derived decision-to-rule view;
- `obligations.toml` and `obligation-graph.md`, the assurance DAG;
- `release.toml`, the derived release state;
- `MANIFEST.sha256`, the complete package census.

`assurance-architecture.md` gives the ownership and maintenance rules.

## Three separate structures

The package has three structures with stable cross-references:

1. The semantic requirement DAG records protocol dependencies.
2. The protocol state machine records ordered verifier behavior and bounded
   repetition.
3. The assurance DAG records the evidence needed for each gate.

The tool derives dependency, blocker, freshness, gate, and release state. A
validator does not require a specific draft state. A closed package can pass
without validator source changes.

## Commands

Refresh generated views:

```bash
python3 protocol-contract/refresh_derived.py
```

Check the portable package:

```bash
python3 protocol-contract/check_contract.py --package-only
```

This command uses only files under `protocol-contract/`. It does not read the
repository copy of the SuperNeo paper.

Audit the sealed one-time migration record:

```bash
python3 protocol-contract/check_contract.py --package-only --verify-import
```

Check repository paths and declaration anchors:

```bash
python3 protocol-contract/check_contract.py --repository
```

Query any stable ID:

```bash
python3 protocol-contract/check_contract.py --package-only \
  --query NS-PICCS-NORM-BINDING
```

Run fault-injection tests:

```bash
python3 -m unittest discover -s protocol-contract/tests
```

Check production release state:

```bash
python3 protocol-contract/check_contract.py --repository --release
```

Check only the implementation-ready boundary:

```bash
python3 protocol-contract/check_contract.py --package-only --implementation-ready
```

The release command must fail until G2 through G5 close. This is expected for
the present package.

## Change rule

After an authored change:

1. refresh generated views;
2. inspect the generated changes;
3. run package checks;
4. run repository checks when repository evidence is in scope;
5. repeat each affected semantic review and create a new bound receipt;
6. rebuild all downstream evidence for the new contract and profile hashes.

An implementation mismatch reopens an assurance edge. It does not change the
contract by implication.
