# Architecture

Nightstream Lean is an assurance package, not a second implementation of the
protocol. Its modules separate mathematical meaning, implementation meaning,
and evidence so a theorem cannot silently cross a trust boundary.

## Dependency graph

```text
SuperNeo model       HyperNova model
          \           /
        Protocol semantics
                |
   Implementation correspondence
                |
        Assurance composition
                |
       Checks and test harnesses
```

The graph is enforced by `scripts/check-layer-imports.sh`. Umbrella modules may
re-export lower layers, but a lower layer may not import a higher one.

## Layer ownership

| Layer | Owns | Does not own |
|---|---|---|
| `SuperNeo` | CCS/CE relations, concrete algebra and parameters, SumCheck, folding reductions | F' state transitions or Rust/R1CS details |
| `HyperNova` | Construction-2 state shape and defaults | SuperNeo folding or production encodings |
| `Protocol` | F' transition semantics, public-output authority, terminal CE semantics | Sparse-row layout or Rust control flow |
| `Implementation` | Encodings, Rust-shaped programs, R1CS semantics, exact artifacts, local row correspondence | End-to-end execution or security claims |
| `Assurance` | Trace induction and composition of local facts into advertised properties | Generated rows or prover-supplied conclusions |
| `Checks` / `tests` | Executable probes, negative witnesses, drift anchors, and fail-closed axiom reports | Definitions consumed by production theorems |

## R1CS ownership

`Nightstream/Implementation/R1CS` is divided by responsibility:

- `Core`: reusable sparse-R1CS semantics and proof infrastructure;
- `Artifacts/<owner>/Generated`: mechanically emitted data shards;
- `Artifacts/*.lean` and `Artifacts/<owner>/*.lean`: small stable facades over
  generated entrypoints;
- `Ownership`: handwritten manifests, row partitions, and assemblies for large
  multi-shard artifacts;
- `Correspondence`: handwritten compiler, soundness, completeness, and local
  decoding theorems.

Assurance imports stable ownership/correspondence modules. It does not import
generated shards directly. Correspondence modules may consume an exact
single-file generated artifact or an `Ownership` assembly. A generated module
contains data, never the semantic conclusion that data is intended to support.

## Evidence boundaries

The project distinguishes four increasingly strong claims:

1. **Model-level:** a theorem about explicitly defined mathematical or protocol
   semantics.
2. **Artifact-checked:** a theorem consumes the exact committed rows or manifest
   for a named fixed profile.
3. **Rust-conformant:** Rust exports or executes the same artifact and a drift
   gate ties both sides to the same authoritative inputs.
4. **Security-reduced:** verifier acceptance is reduced to stated assumptions
   with named bad events and bounds.

Property scope, threat model, and completion criteria are normative in
[`../specs/formal-verification.md`](../specs/formal-verification.md). Mutable
evidence belongs in `../assurance/`, not in theorem comments or README status
inventories.

## Change rule

When a Rust surface, generated artifact, supported profile, or theorem statement
changes, update the owning property specification and evidence record in the
same change. A hash match establishes identity only; the relevant Lean theorem
establishes meaning.
