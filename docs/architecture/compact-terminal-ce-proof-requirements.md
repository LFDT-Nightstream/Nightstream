# Compact Terminal CE Proof Requirements

Status: design gate, not implementation.

## Goal

Replace the direct terminal CE rows in `paper::decider_ce_relation` only when
the replacement proves the same relation more compactly.

Current sound path:

```text
decider R1CS allocates terminal Z
  and enforces Commit(Z), X, low-norm, y_ring, ct, and NC sidecars directly.
```

Allowed compact path:

```text
terminal CE prover proves the terminal CE relation off-circuit
decider R1CS verifies that proof against the NIFS-derived terminal children
```

Digest binding alone is not enough.

## Paper Contract

SuperNeo Definition 13 defines CE for a claim/witness pair:

- `c = L(Z)`
- `X = L_in(Z)`
- `||Z||_infty < b`
- `y_j = M_j * Z(r)` for every CCS matrix `j`

This codebase also carries denormalized sidecar fields. A compact proof must
either prove them or reject them before they become accumulator authority:

- `ct[j] == constant_term(y_ring[j])`
- if present, `y_zcol == Z * chi(s_col)`
- unsupported sidecars remain rejected, not silently omitted from the proof

## Public Statement

The compact proof's public statement must be derived from the same
`terminal_children` wires currently passed to
`decider_ce_relation::enforce_final_ce_relations`.

Minimum public binding:

```text
structure_digest
params_digest
terminal_children_digest
claim_count
```

`terminal_children_digest` must be Poseidon2-only and must bind every public
CE field consumed by the verifier: commitment metadata and data, active `X`,
`r`, `s_col`, `y_ring`, `ct`, `y_zcol`, `m_in`, `fold_digest`, and any
supported offset or auxiliary coordinates. If a field is unsupported in the
clean frontend, reject non-empty values before digesting.

The proof verifier must consume public inputs recomputed from
`emit_terminal_fold`'s NIFS output children. Passing a freshly allocated copy of
`final_state.running.claims` is a soundness bug.

## Required In-Circuit Verifier Contract

The compact verifier rows must prove:

```text
exists Z_i for every terminal child i such that:
  Commit_Ajtai(Z_i) == child_i.c
  project_x(Z_i) == child_i.X
  every entry of Z_i is in {-(b-1), ..., +(b-1)}
  child_i.y_ring[j] == multilinear_eval(M_j * Z_i, child_i.r)
  child_i.ct[j] == child_i.y_ring[j][0]
  child_i.y_zcol == Z_i * chi(child_i.s_col), when the NC channel is present
```

This may be a Spartan/FRI-style proof, matching the SuperNeo paper's
compression posture over Goldilocks, or another explicitly approved backend.
The backend choice changes proof material, not the relation above.

## Current Primitive Inventory

The current tree has enough machinery to keep the direct path sound, but not
enough to replace it with a compact terminal proof verifier yet.

Reusable pieces already present:

- `R1csBuilder` field/K arithmetic and Poseidon2 transcript gadgets.
- In-circuit SuperNeo verifier pieces for `NIFS.V`
  (`Π_CCS.V -> Π_RLC.V -> Π_DEC.V`).
- In-circuit sumcheck verifier helpers used by SplitNc `Π_CCS.V`.
- Native Spartan2 compression over `SpartanCircuit` with the
  Goldilocks/Poseidon2/Merkle-MLE backend.
- Prototype native Spartan adapters and direct terminal circuits.

Missing pieces for this replacement:

- An `R1csBuilder` verifier for the terminal proof's polynomial commitment
  openings (for the paper-aligned Spartan/FRI/Merkle-MLE backend, this means
  Poseidon2 Merkle path verification plus the PCS query checks).
- An in-circuit verifier for the full terminal proof transcript, not only the
  native `spartan2::R1CSSNARKTrait::verify` API.
- A wire-level `TerminalCePublic` constructor from the actual
  `emit_terminal_fold` output children.
- Oracle tests proving direct CE rows and compact verifier rows accept and
  reject the same terminal statements.

Native Spartan verification is useful for a future compression layer, but it
is not a substitute for the in-circuit verifier rows described above. Replacing
`enforce_final_ce_relations` with a native verifier call, a digest equality, or
a proof object that the decider circuit does not verify would weaken the final
artifact.

## Prototype Portability Audit

The prototype has useful reference material, but it must not be ported into
`neo-fold-clean` as-is.

Relevant prototype pieces:

- `neo-fold-prototype/src/frontends/direct_ccs/terminal/ce_bundle.rs`
- `neo-fold-prototype/src/frontends/direct_ccs/terminal/gadgets/final_ce.rs`
- `neo-fold-prototype/src/frontends/rv32im/ivc_snark/mod.rs`

What they provide:

- a native Spartan proof wrapper for a final CE bundle;
- Bellpepper gadgets that allocate final CE witnesses and enforce the paper CE
  relation;
- a two-proof final shape: terminal committed F' proof plus final CE proof.

Why they are not a clean drop-in:

- The direct-CCS CE bundle canonicalizes away transport fields:
  `s_col`, `ct`, `aux_openings`, `y_zcol`, `fold_digest`,
  `c_step_coords`, `u_offset`, and `u_len`.
- In `neo-fold-clean`, several of those fields are deliberately part of the
  carried accumulator authority. In particular, `ct` is denormalized but
  consumed, `y_zcol` is an optional NC side channel, and `fold_digest` is part
  of CE continuity/digest binding.
- The prototype verifier is native Spartan verification. That is useful for a
  terminal compression layer, but it is not the requested decider-circuit
  replacement unless the clean decider circuit verifies the proof or the final
  product verifier directly verifies that Spartan proof against the exact
  public terminal statement.

Safe reuse rule:

```text
Reuse relation ideas, not authority assumptions.
```

Any clean port must keep the clean authority set:

```text
Commit_Ajtai(Z), X, low-norm, y_ring, ct, optional y_zcol,
and full public binding to every terminal_children field.
```

Dropping fields because the prototype canonicalized them is a soundness bug.

## First Safe Implementation Slice

The first implementation slice should not replace production decider rows.

Build a standalone terminal-CE proof module behind the existing direct gadget
oracle:

```text
paper::terminal_ce
  public.rs   -- TerminalCePublic from concrete CeClaim values
  proof.rs    -- opaque proof bytes + typed verification errors
  native.rs   -- native prove / verify against TerminalCePublic
  circuit.rs  -- empty/fail-closed until an in-circuit verifier exists
```

The slice is useful only if it satisfies all of these gates:

- `TerminalCePublic` binds the same data that
  `emit_terminal_fold` exposes as terminal NIFS children, not a private copy of
  `final_state.running.claims`.
- Native verification rejects every direct-gadget tamper class:
  `c.data`, active `X`, `r`, `y_ring`, `ct`, `s_col`, `y_zcol`, and
  out-of-alphabet witness digits.
- The production decider still calls `enforce_final_ce_relations`.
- `circuit::enforce_verify` either verifies the real proof in-circuit or
  returns an explicit unsupported error. It must not accept by checking only a
  digest or by trusting native verification.

Only after the circuit verifier exists may the production decider switch from
direct CE rows to compact proof verification.

## Replacement Tests

Before replacing direct CE rows in production synthesis, add tests that compare
the compact verifier against the direct gadget oracle:

- honest terminal children and witnesses: direct and compact both accept;
- tamper `c.data`, active `X`, `r`, `y_ring`, `ct`, `s_col`, `y_zcol`, or
  low-norm witness digits: compact verifier rejects;
- tamper the compact proof while leaving terminal children honest: rejects;
- recompute prover-controlled digests after a terminal-child tamper: rejects;
- production decider synthesis no longer reads full `final_running.witnesses`
  only in the compact mode, and only because the compact proof supplies the
  witness knowledge;
- row-count test shows the compact terminal verifier does not scale like the
  direct `M * Z` scan.

Until these tests exist and pass, `enforce_final_ce_relations` remains the
soundness contract.
