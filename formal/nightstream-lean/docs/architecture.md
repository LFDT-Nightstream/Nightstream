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

## Nebula V2 implementation ownership

`Nightstream/Implementation/NebulaV2` is organized by the object or protocol
stage that owns each implementation theorem:

```text
NebulaV2
├── Core          reusable field, bit, comparison, and selector rows
├── Commitment    bundle, compact-token, lane, and terminal-opening rows
├── Memory        carry, claim, operation, product, snapshot, and transition rows
├── NIFS          PiCCS, PiRLC, PiDEC, running-value, and terminal correspondence
├── FPrime        claim, manifest, state, recursive, and terminal correspondence
├── Application   WASM codecs and physical-port refinement
└── Production    selected carrier, memory, NIFS, F-prime, and artifact composition
```

Each directory has a facade that imports its immediate children. The public
`NebulaV2` facade imports only these seven ownership facades. Tests and axiom
guards mirror the same ownership paths below `tests/NebulaV2` and
`tests/Axioms/NebulaV2`. New modules must enter through the narrowest owning
facade; do not restore a flat list in the public facade.

## Proof ownership trees

Protocol-critical components mirror the review path in their module tree:

```text
protocol
└── phase
    └── constraint or theorem family
        └── equation/proof leaf
```

This is an ownership tree, not a requirement for one file per equation.
A leaf should contain the smallest coherent family whose equations share the
same mathematical owner and implementation emitter. Parent modules import
their immediate children and document the boundary they compose; they should
not become editable status ledgers or alternate acceptance predicates.

Some constraints have two legitimate ownership axes. For example,
`paper/nifs/circuit/pi_rlc` owns protocol execution and cost phases, while
`paper/reductions/pi_rlc_circuit` owns the reusable arithmetic emitters by
claim type. Keep those trees separate. Their parent tables must map every cost
leaf to exactly one arithmetic owner and Lean theorem; neither tree may copy
the other's equations merely to make the directories look identical.

Every protocol-critical parent states:

| Field | Meaning |
|---|---|
| child path | stable protocol or implementation stage |
| mathematical obligation | property owned by that child |
| excluded boundary | property deliberately owned elsewhere |

Every constraint-emitting or artifact-refining leaf states:

| Field | Meaning |
|---|---|
| stage path | stable protocol → phase → family address |
| equation or obligation | exact mathematical fact represented |
| authority class | checked, computed, direct dataflow, derived, or security boundary |
| physical owner | Rust emitter, generated piece, or row family |
| Lean owner | soundness, completeness, or necessity theorem |
| multiplicity | symbolic formula; fixed-profile counts remain generated or mechanically checked |

Headers also state `Owns`, `Does not own`, and `Emits constraints`. Exact
measured totals stay in generated manifests and reconciliation tests rather
than duplicated handwritten comments.

`scripts/check-proof-ownership-contracts.sh` enforces this header/table
contract for the independent paper-joint and Split-NC models, the active
ConcretePhi81 F-prime/NIFS spine, Phi81 evaluation homomorphisms, delayed NC
authority, Pi_RLC correspondence and sampler, gadget-native lowering, and
arithmetic-owner trees. It also rejects
editable `Status` columns in Lean source headers: completion evidence belongs
in property specifications and assurance records. The scope is intentional:
trivial modules elsewhere do not gain ceremonial headers merely to satisfy a
repository-wide text rule.

### Structural closure gate

An active protocol-critical slice is structurally closed only when:

- one parent module imports its immediate semantic, artifact-schema, and
  refinement children;
- the project-level barrel imports that parent rather than enumerating its
  leaves;
- generated data depends only on artifact-owned types, never on handwritten
  correspondence modules;
- a focused test checks the exported results and a fail-closed axiom test owns
  their allowed assumptions; and
- claimed degree, multiplicity, and row counts are derived from generated data
  or reconciled traces, not restated as handwritten constants.

This gate applies one review path at a time. File-count reduction is not by
itself a goal: merge modules only after a closed bridge shows that they no
longer own distinct definitions or theorems. Conversely, do not keep
tautological mutation tests, editable status inventories, or proof wrappers
that merely rename an unchecked predicate.

For inclusion-minimality, the counterexample carrier must contain every input
field that a removed family may change. A pointwise plan may fix an outer input
while ranging only over a proposed output; that is sufficient for exactness,
but it cannot honestly prove independent necessity of two checks whose truth
is already fixed by that input. Global minimality modules therefore range over
the complete typed input plus the proposed result, while local modules remain
the smaller proof owners for one fixed input.

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

Property scope, threat model, and completion criteria are normative in the
[`protocol-contract`](../../../protocol-contract/README.md). Mutable evidence
belongs in `../assurance/`, not in theorem comments or README status
inventories.

## Change rule

When a Rust surface, generated artifact, supported profile, or theorem statement
changes, update the owning property specification and evidence record in the
same change. A hash match establishes identity only; the relevant Lean theorem
establishes meaning.
