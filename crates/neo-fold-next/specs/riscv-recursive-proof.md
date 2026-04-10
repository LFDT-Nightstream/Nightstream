# RV64IM Recursive Proof End-State Specification

STATE: WORK IN PROGRESS (NOT READY)

## Scope

This document specifies the end-state theorem-facing proof architecture for
RV64IM in `neo-fold-next`.

It fixes the ownership boundary between:

1. the **inner folding accumulator**,
2. the **recursive verifier/IVC layer**, and
3. the **final exported compressed proof**.

The target is the standard compiler direction described by HyperNova and
explicitly anticipated by SuperNeo:

- use a **public-coin non-interactive folding scheme** obtained by applying
  Fiat-Shamir to the folding protocol, and
- use a recursive circuit or outer SNARK to prove that the verifier of that
  non-interactive folding scheme was run correctly, rather than requiring the
  final external verifier to replay the entire folding transcript from bulky
  public sidecar data.

This spec does **not** redefine the underlying CCS folding relation. It only
defines how the RV64IM proof should be exported in the end-state recursive
design.

It is **not** the full SuperNeo backend or parameter specification. In this
repo, the concrete backend contract, Goldilocks parameterization, challenge
domain, and chunk-local role split are owned by `riscv-kernel.md`.

The concrete recursive backend choice, compression backend choice, canonical
encodings of carried and fresh recursive objects, and the final public Rust API
are owned by `riscv-recursive-instantiation.md`.

Every value that is hashed, committed, projected, or publicly bound in the
recursive verifier path shall have a unique canonical encoding fixed by
versioned backend parameters.

## Normative References

The target in this document is constrained by the following local references:

- `HyperNova.pdf.md`
  - public-coin multi-folding made non-interactive with Fiat-Shamir
  - the recursive step circuit runs the verifier of the non-interactive
    folding scheme
  - Construction 9, the black-box IVC construction
- `01_1_Introduction.md`
  - standard compilers from folding schemes to IVC/PCD
  - Goldilocks is SNARK-friendly
  - efficient proof compression using Spartan with a FRI-based PCS is an
    intended path for SuperNeo accumulators
- `02_2_Technical_overview.md`
  - SuperNeo is an interactive-reduction decomposition
    `Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS`
- `07_7_Neo_s_folding_scheme_for_CCS.md`
  - the folding scheme relations and reduction shapes
- `riscv-witness-backed-side-bridge.md`
  - the exact RV64IM side-bridge relation that must be fixed before later
    recursive or succinct compilation
- `riscv-kernel.md`
  - the concrete theorem-facing SuperNeo backend contract used by RV64IM
  - the Goldilocks parameterization and ring/challenge-domain choices
  - the chunk-local `Π_CCS -> Π_RLC -> Π_DEC` role split
- `riscv-recursive-instantiation.md`
  - the concrete repo instantiation of the recursive backend, compression
    backend, canonical encodings, and public Rust API

This document is normative only where its requirements are implied by, or are a
conservative specialization of, those references plus the repo-owned backend
contract.

## 1. End-State Design Goal

The codebase target exported proof format for an RV64IM execution shall be a
**compressed recursive proof**, not a raw folding accumulator.

This is an export-policy requirement for this repo's end-state proof format. It
is not a claim that the cited papers forbid exposing a less-compressed theorem-
facing artifact in the abstract.

The final external verifier shall verify a proof whose public statement binds:

- the RV64IM program,
- the initial public machine state,
- the final public machine state or an explicitly chosen output projection of
  that state,
- and the step count or equivalent execution length metadata.

The final external verifier shall **not** be required to:

- replay all fold transcripts,
- recompute all Fiat-Shamir challenges from raw per-step or per-row sidecar
  data,
- inspect the full execution trace,
- inspect the full stage witness families,
- or inspect the full accumulator witness.

Those obligations belong to the recursive verifier layer or the outer
compression layer.

## 2. Paper Basis (Informative)

### 2.1 HyperNova

HyperNova states that the public-coin multi-folding scheme is made
non-interactive using the Fiat-Shamir transform in the random oracle model.

Let `NIFS` denote the resulting non-interactive folding scheme. HyperNova then
builds IVC by running the verifier of `NIFS` inside the recursive step circuit.

In HyperNova Construction 9, the step circuit has the form

```text
step(vk, U_i, z_i; (u_i, π_{i+1})) -> (vk, U_{i+1}, z_{i+1})
```

with the key recursive check

```text
U_{i+1} <- NIFS.V(vk, U_i, u_i, π_{i+1}).
```

Thus the external verifier does **not** replay the entire NIFS transcript from
the raw step data. The recursive circuit does that work.

### 2.2 SuperNeo

SuperNeo states that:

1. its folding construction is a composition of interactive reductions,

```text
Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS,
```

2. standard compilers from folding schemes to IVC and PCD apply, and
3. because the construction works over a SNARK-friendly small field such as
   Goldilocks, the resulting accumulators admit efficient proof compression.

Therefore this repo's end-state theorem-facing artifact for a SuperNeo-based
RV64IM system is not the raw accumulator with its full witness sidecar. It is a
compressed proof or argument for the recursive verifier relation, with any
knowledge-soundness claim contingent on the concrete backend assumptions named
in `riscv-recursive-instantiation.md`.

## 3. Objects and Notation

Let:

- `P` be the immutable RV64IM program,
- `dig(P)` be the program binding digest or equivalent public program binding,
- `z_i` be the public machine state after `i` semantic steps,
- `ω_i` be the private witness for step `i`,
- `F_P` be the RV64IM step relation for program `P`,
- `A_i` be the carried paper accumulator bundle after a full SuperNeo fold
  cycle, i.e. a bundle of norm-bounded `CE(b, L)^k` claims under the concrete
  backend contract in `riscv-kernel.md`,
- `U_i` be the recursive-layer running instance that encodes or publicly binds
  `A_i`,
- `W_i` be the witness material that opens or justifies `U_i` as a satisfying
  encoding of `A_i`,
- `u_i` be the full fresh paper instance consumed by the inner NIFS verifier,
- and `u_i'` be the application-visible projection of that fresh instance.

The application-level transition relation is:

```text
z_{i+1} = F_P(z_i, ω_i).
```

The end-state proof statement may expose either:

1. the full final public machine state `z_n`, or
2. an application output projection

```text
y = Out(z_n),
```

provided that `Out` is fixed by the verifier specification and is bound by the
same theorem-facing statement.

For simplicity, the canonical statement in this spec is:

```text
stmt_RV64IM := (dig(P), n, z_0, z_n).
```

If an application wants a smaller public output, it may instead use

```text
stmt_RV64IM,out := (dig(P), n, z_0, y)
```

with the additional requirement that the recursive or compression layer proves

```text
y = Out(z_n).
```

## 4. Inner Folding Layer

The inner folding layer is the non-interactive folding scheme obtained from the
public-coin SuperNeo-style folding protocol by Fiat-Shamir.

The folding relation remains the paper relation:

```text
Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS.
```

This is the Fiat-Shamir non-interactive form of the **full** SuperNeo
composition, not an ad hoc fold of committed step witnesses and not a generic
single-relation placeholder.

At this layer, the prover may still maintain:

- a carried accumulator bundle `A_i` whose theorem-level meaning is the paper's
  `CE(b, L)^k` post-decomposition state under the backend contract,
- a running instance `U_i` and witness `W_i` encoding that bundle,
- a fresh instance `u_i`,
- a fresh witness `w_i`,
- and a folding proof `π_{i+1}` witnessing the transition

```text
(U_i, W_i), (u_i, w_i)  --->  (U_{i+1}, W_{i+1}, π_{i+1}).
```

This raw accumulator state is **not** the final exported artifact.

### 4.1 Base Case / Genesis

The carried relation shall have an explicit genesis state.

Let `Init_SN(P, z_0)` denote the backend-defined initialization path owned by
`riscv-kernel.md` that forms the initial carried SuperNeo boundary from the
fixed program `P` and the initial public machine state `z_0`.

A conforming recursive construction shall take the initial carried bundle `A_0`
to be the carried boundary produced by `Init_SN(P, z_0)`, so that:

- `A_0` has theorem-level meaning equal to a valid carried `CE(b, L)^k` bundle
  under the concrete backend contract fixed by `riscv-kernel.md`,
- `A_0` binds the same fixed-program identity and initial public state as the
  theorem-facing statement,
- and `U_0, W_0` are an encoding and witness for that carried bundle.

`A_0` is therefore not an unconstrained dummy accumulator. It is the carried
pre-execution boundary for the same RV64IM execution whose public statement
binds `(dig(P), n, z_0, z_n)` or `(dig(P), n, z_0, y)`.

If an implementation uses a neutral or synthetic seed object internally, then
the first recursive step shall prove equivalence between that seed object and
the backend-defined carried boundary `A_0` before consuming the first real fresh
RV64IM chunk instance.

The first fresh instance `u_0` therefore represents the first real chunk-local
semantic claim starting from `z_0`, not an unconstrained bootstrap transition.

## 5. Recursive Verifier Layer

This repo adopts a HyperNova-style black-box IVC compiler over the
Fiat-Shamir-compiled public-coin form of the composed SuperNeo reductions.

This document intentionally chooses the **fixed-step black-box IVC** path of
HyperNova Construction 9.

It does **not** specify HyperNova's instruction-indexed NIVC or "a la carte"
compiler path with a program counter and a vector of running instances. If the
codebase later adopts that compiler path, it shall be specified separately.

### 5.1 Recursion Granularity

This repo adopts **chunk-step recursion**.

A recursive step consumes one closed RV64IM chunk produced by the kernel
pipeline, not one semantic RV64IM instruction.

Let chunk `i` cover the contiguous semantic interval

```text
[s_i, s_{i+1})
```

with:

- `s_0 = 0`,
- `s_N = n`,
- initial public state `z_{s_i}`,
- final public state `z_{s_{i+1}}`,
- and chunk ordinal `i`.

The chunk schedule is fixed by the backend contract in `riscv-kernel.md` and
the concrete repo instantiation in `riscv-recursive-instantiation.md`.

Let `Sched_SN` denote that canonical chunk schedule metadata, including the
fixed chunking rule and any required schedule/version identifiers.

Let `ChunkCount(Sched_SN, n)` denote the canonical chunk count for an execution
of exact semantic length `n`.

Let `NextBoundary(Sched_SN, n, i, s_i)` denote the canonical next chunk
boundary after `s_i` for chunk index `i`.

This is an implementation-level specialization for this repo. The cited papers
require only that the recursive layer verify the non-interactive folding
verifier over fresh instances and carried accumulators; they do not require
instruction-level recursion.

The theorem-facing statement remains the full-execution statement

```text
stmt_RV64IM := (dig(P), n, z_0, z_n)
```

or its output-projection variant. Chunk boundaries are recursive-internal.

For this repo, `n` is the exact number of legal RV64IM semantic transitions in
the proved execution. Post-halt padding is forbidden: once the machine is
halted, no further semantic step is legal.

### 5.2 Carried and Fresh Recursive Objects

The recursive layer shall preserve the paper relations directly under the
backend contract.

The carried recursive instance `U_i` is a backend-owned encoding of the paper's
carried `CE(b, L)^k` bundle under the concrete backend contract.

Concretely, `U_i` is not a generic accumulator handle. It is an encoding or
public binding of the carried bundle `A_i` whose theorem-level meaning is
exactly the carried `CE(b, L)^k` meaning required by the backend contract.

The carried witness `W_i` is the corresponding backend-owned witness material
that opens or justifies `U_i` as a satisfying encoding of that carried bundle.

The fresh recursive input consumed at chunk `i` is the chunk-local public
instance bundle for the paper `CCS` relation under the concrete backend
contract.

Write this abstractly as:

```text
U_i := Enc_CE_bundle(A_i)
W_i := Wit_CE_bundle(A_i)
u_i := Enc_CCS_chunk_bundle(C_i)
w_i := Wit_CCS_chunk_bundle(C_i)
```

where `A_i` is the carried `CE(b, L)^k` bundle after chunk `i`, and `C_i` is
the fresh `CCS` chunk-local claim bundle consumed at recursive step `i`.

This file does not permit replacing those paper meanings with a generic “proof
object” abstraction. Any concrete encoding used for `U_i`, `W_i`, `u_i`, or
`w_i` shall have a backend-defined interpretation as the backend-grounded paper
relations above.

The exact encodings are owned by `riscv-recursive-instantiation.md`.

### 5.3 Application-Visible Chunk Projection

Write the application-visible projection of that fresh instance as

```text
u_i' = Enc_chunk_inst(
  dig(P),
  chunk_index = i,
  step_lo = s_i,
  step_hi = s_{i+1},
  z_in = z_{s_i},
  z_out = z_{s_{i+1}},
  halted_out
),
```

where the witness-bearing commitment part is omitted from the public decoding.

This projection is **not** required to be the entire public shape of the fresh
SuperNeo paper instance. The full instance `u_i` still carries whatever
commitment and evaluation data `NIFS.V` requires; `u_i'` is only the
application-visible projection parsed by the recursive step circuit.

`halted_out` and any other terminal flags are present in `u_i'` only if they
belong to the chosen public machine-state projection for this repo. They are
not required merely because the current bridge exports them.

### 5.4 Recursive State and Step Function

The recursive step relation is parameterized by:

```text
RecParams := (vk_NIFS, Sched_SN)
RecPub := (dig(P), n)
```

where:

- `vk_NIFS` is fixed by the recursive backend/verifier configuration,
- `Sched_SN` is the canonical chunk schedule metadata,
- and `dig(P), n` are the theorem-facing public execution bindings.

The recursive state carried across chunk steps is:

```text
State_i := (U_i, chunk_index = i, step_count = s_i, z_{s_i}).
```

The recursive step circuit shall implement:

```text
Step(RecParams, RecPub, State_i; (u_i, π_{i+1})) -> State_{i+1}
```

with the following logic:

1. Compute the application-visible projection

```text
u_i' := Proj_app(u_i).
```

2. Parse

```text
(dig(P)_i, idx_i, lo_i, hi_i, in_i, out_i, halted_i) <- Enc_chunk_inst^{-1}(u_i').
```

3. Check

```text
dig(P)_i = dig(P)
```

and

```text
idx_i = i,
lo_i = s_i,
in_i = z_{s_i},
hi_i = NextBoundary(Sched_SN, n, i, s_i).
```

4. Compute

```text
U_{i+1} <- NIFS.V(vk_NIFS, U_i, u_i, π_{i+1}).
```

5. Output

```text
State_{i+1} := (U_{i+1}, i + 1, hi_i, out_i).
```

The `dig(P)` equality check is an intentional redundant binding at the export
boundary. Program binding may already be captured by key generation and the
encoded step relation; this layer repeats it so the theorem-facing statement is
explicitly bound to the same program identity.

The recursive layer therefore binds both state continuity and execution-length
continuity across chunk boundaries. The final recursive state binds `s_N = n`.

### 5.5 Base and Terminal Acceptance Relations

Let

```text
N := ChunkCount(Sched_SN, n).
```

The recursive construction shall satisfy the following acceptance relations.

Base relation:

```text
Base(RecParams, stmt_RV64IM):
  let A_0 := Init_SN(P, z_0)
  let U_0 := Enc_CE_bundle(A_0)
  output State_0 := (U_0, 0, 0, z_0)
```

where `P` is the fixed program bound by `dig(P)` in `stmt_RV64IM`, and `A_0`
has the carried semantics required by section 4.1.

Terminal relation for the full-state statement:

```text
Terminal(stmt_RV64IM, State_N):
  let State_N = (U_N, chunk_index, step_count, z_term)
  check chunk_index = N
  check step_count = n
  check z_term = z_n
```

Terminal relation for the output-projection statement:

```text
Terminal(stmt_RV64IM,out, State_N):
  let State_N = (U_N, chunk_index, step_count, z_term)
  check chunk_index = N
  check step_count = n
  check Out(z_term) = y
```

These terminal checks are part of the recursive verifier relation accepted by
`IVC.V`. They are not left to informal interpretation.

### 5.6 Binding Inventory

Consensus-critical values are bound in the following locations:

- `dig(P)`, `n`, `z_0`, and `z_n` or `y` are theorem-facing public inputs via
  `stmt_RV64IM` or `stmt_RV64IM,out`,
- `vk_NIFS` is fixed by recursive-backend verifier configuration and is not
  theorem-facing by default,
- `vk_IVC` is fixed by the recursive/compression backend verifier
  configuration and is not theorem-facing by default,
- `Sched_SN`, including `root_params_id`, `fold_schedule`, `chunk_count`,
  and any encoding/transcript version identifiers, is recursive-internal public
  or fixed by backend parameters as specified in
  `riscv-recursive-instantiation.md`,
- and `U_N^pub`, if used, is an auxiliary public handle required by the chosen
  backend and is not part of the canonical semantic theorem statement.

The critical ownership rule is:

- **all challenge re-derivation and transcript-consistency checks for the inner
  folding protocol occur inside `NIFS.V` and therefore inside the recursive
  layer**, not in the final external verifier.

This is the mechanism that replaces the direct-export “option 1” design where
the final external verifier replays the transcript from a large public sidecar.

## 6. Compressed Exported Proof

The final exported proof shall be a compressed proof of the recursive verifier
relation. It is not, by default, a second independent proof of raw terminal
accumulator membership alongside a recursive proof.

Let `Π'_N` denote the recursive proof object produced by the outer IVC system
for the terminal chunk index `N := ChunkCount(Sched_SN, n)`.

Let `Backend_SN` denote the concrete SuperNeo backend contract fixed by
`riscv-kernel.md`.

Let `R_acc^SN` denote the terminal accumulator relation induced by the
Fiat-Shamir non-interactive form of

```text
Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS
```

under `Backend_SN`.

Concretely, `R_acc^SN` is **not** an unspecified generic terminal accumulator
relation. It is the relation that holds exactly when terminal recursive state
material encodes and justifies a satisfying carried accumulator bundle `A_N`
whose theorem-level meaning is the paper's `CE(b, L)^k` accumulator bundle
under `Backend_SN`.

Let `stmt_IVC` denote the public statement accepted by the recursive verifier.
It shall bind `stmt_RV64IM` and may additionally bind a minimal public
accumulator handle `U_N^pub` if the chosen recursive proof system requires one.

The terminal relation proved by the compression layer is:

```text
R_comp(stmt_IVC) :=
{
  (Π'_N) :
    IVC.V(vk_IVC, stmt_IVC, Π'_N) = 1
}.
```

Here:

- `IVC.V` is the verifier of the recursive proof system,
- `stmt_IVC` is the public recursive statement,
- `stmt_RV64IM` is the canonical public RV64IM theorem statement,
- and `R_acc^SN` remains the backend-grounded carried accumulator semantics
  that the recursive step circuit proves transitively by running `NIFS.V` over
  the Fiat-Shamir non-interactive form of the full SuperNeo composition.

In the default export path, `R_acc^SN` is therefore a semantic obligation of
the recursive system, not a second independent theorem that the outer
compression layer re-proves alongside `IVC.V`.

If a future recursive system intentionally leaves some terminal accumulator
handle externally unchecked, that design shall specify a different compression
relation explicitly. This file does not adopt that design by default.

The exported compressed proof shall then be:

```text
Proof_final := (stmt_RV64IM, Π_comp)
```

Here `Π_comp` is a compressed proof or argument for a valid recursive proof
object `Π'_N` satisfying `R_comp(stmt_IVC)`, where `stmt_IVC` binds
`stmt_RV64IM`. Any knowledge-soundness claim for `Π_comp` depends on the
concrete backend assumptions fixed in `riscv-recursive-instantiation.md`.

If the recursive verifier requires a minimal public accumulator handle, then
the exported theorem-facing proof may instead take the form

```text
Proof_final := (stmt_RV64IM, U_N^pub, Π_comp),
```

with

```text
stmt_IVC := (stmt_RV64IM, U_N^pub).
```

The concrete choice of `U_N^pub` is owned by
`riscv-recursive-instantiation.md`. It may be:

- the terminal running instance `U_N`, or
- a canonical backend binding digest of `U_N`.

The following rules are normative:

- prefer a binding public digest of `U_N` over exporting full `U_N` when the
  chosen backend permits it,
- the exported proof shall not expose `W_N`, the sequence of fresh witnesses
  `(w_i)`, or the sequence of fold proofs `(π_i)`.

## 7. Theorem-Facing Statement

The theorem-facing statement shall bind:

1. the fixed program `P`,
2. the initial public state `z_0`,
3. the final public state `z_n` or declared output projection `y`,
4. and the execution length `n` or an equivalent canonical execution-length
   binding.

The intended theorem is:

```text
∃ (ω_0, …, ω_{n-1}), (z_1, …, z_{n-1}) such that
for all i ∈ {0, …, n-1},
z_{i+1} = F_P(z_i, ω_i),
```

together with the public bindings:

```text
dig(P), z_0, z_n, n.
```

This is an exact-length execution theorem. A conforming proof may not insert
post-halt padding steps or stutter transitions after the machine has halted.

If an application-specific output projection `y = Out(z_n)` is used instead of
publishing `z_n`, then the theorem-facing compression relation must also bind:

```text
y = Out(z_n).
```

### 7.1 Current Implementation Auxiliary Obligation Surface

The current `neo-fold-next` RV64IM public proof API exports a stronger, lower-
level **auxiliary obligation surface** in addition to the canonical semantic
statement above.

This auxiliary surface is verifier-required in the current implementation. It is
therefore not accidental bookkeeping, but it is also not the final normalized
semantic theorem statement.

Write the current auxiliary digest-and-metadata bundle as:

```text
aux_RV64IM,current := (
  root_params_id,
  fold_schedule,
  chunk_count,
  public_step_count,
  final_pc,
  halted,
  stage_claims_digest,
  stage_packages_digest,
  kernel_opening_digest,
  prepared_step_bindings_digest,
  execution_digest,
  final_state_digest,
  transcript_final_digest,
  main_lane_surface_digest,
  root_lane_columns_digest
).
```

These fields have the following ownership roles:

- `stage_claims_digest` binds the carried Stage-1 row surface, Stage-2 family
  surface, Stage-3 continuity surface, and transcript claim surface generated by
  the current simple-kernel pipeline,
- `stage_packages_digest` binds the packaged claim proofs currently used for
  those stage-local claim surfaces,
- `kernel_opening_digest` binds the current kernel-opening package bundle,
  including the binding-side opening package and prepared-step opening package,
- `prepared_step_bindings_digest` binds the per-step summary linking exported
  semantic rows to the current root-lane selected-opening surface,
- `execution_digest`, `final_state_digest`, and `transcript_final_digest` bind
  the current terminal execution/transcript summaries carried by the kernel
  claim bundle,
- and `main_lane_surface_digest` plus `root_lane_columns_digest` bind the
  current root main-lane public surface used by the packaged main-lane proof.

The current public proof API also derives repo-local claim bundles over these
digests, including accepted-proof, main-lane, kernel-opening, joint-opening,
and root0 claim objects.

Those claim names are **repo-local binding layers**, not additional paper-level
cryptographic primitives. Their role is to make the current bridge fail closed
on mismatched side-obligation bundles.

The current auxiliary surface may be viewed as a transitional statement

```text
stmt_RV64IM,current := (aux_RV64IM,current)
```

which is lower-level than the canonical semantic statement. A future recursive
or compression layer may internalize, compress, or eliminate parts of this
auxiliary surface, provided it still proves the same semantic theorem and the
same bridge-consistency obligations.

A change in frontend memory-checking or lookup backend may therefore replace
large parts of `aux_RV64IM,current` without changing the semantic theorem,
provided the replacement still proves the same bridge-consistency obligations.

### 7.2 Normalized Public API Boundary

The normalized theorem-facing public API of the end-state recursive system binds
only:

- `dig(P)`,
- `n`,
- `z_0`,
- and `z_n` or `y = Out(z_n)`.

No bridge-local digest bundle is theorem-facing by default.

If a concrete recursive or compression backend requires an additional minimal
public handle such as `U_N^pub`, that handle shall be justified in
`riscv-recursive-instantiation.md` and shall not be treated as part of the
canonical semantic theorem statement.

### 7.3 Classification Rule for Current Auxiliary Fields

Every field in `aux_RV64IM,current` and every derived repo-local claim bundle
shall be classified into exactly one of the following buckets:

1. **semantic public**:
   survives into `stmt_RV64IM` or `stmt_RV64IM,out`;
2. **recursive-internal public**:
   used inside `u_i'`, `u_i`, `U_i`, or the recursive/compression verifier, but
   not exposed in the normalized theorem-facing API;
3. **private witness**:
   moved into `w_i`, `W_i`, or recursive/compression witness material;
4. **audit-only**:
   retained only in a non-normative proof-complete or debug export.

For the current bridge-owned auxiliary fields, the default classification is:

- `root_params_id`, `fold_schedule`, and `chunk_count` are
  recursive-internal public,
- `public_step_count` is semantic public via `n`,
- `final_pc` and `halted` are semantic public only if they are part of the
  chosen public machine-state projection; otherwise they are recursive-internal
  public during migration,
- `stage_claims_digest`, `stage_packages_digest`, `kernel_opening_digest`,
  `prepared_step_bindings_digest`, `execution_digest`,
  `final_state_digest`, `transcript_final_digest`,
  `main_lane_surface_digest`, and `root_lane_columns_digest`
  are recursive-internal public by default in the current bridge and may later
  be demoted to private witness material if the concrete recursive backend does
  not require them as public chunk inputs,
- and current derived claim bundles such as accepted-proof, main-lane,
  kernel-opening, joint-opening, and root0 claim objects are audit-only in the
  normalized end-state API, but during migration any such bundle still required
  to make the bridge fail closed shall remain recursive-internal public until
  the concrete recursive/compression relation absorbs that safety function.

No field may remain theorem-facing solely because the current bridge exports it.

## 8. Export Boundary Requirements

### 8.1 Required exported data

The final theorem-facing proof may export only the following categories of data:

- the public statement,
- the compressed proof,
- and any minimal public accumulator handle required by the outer compression
  verifier.

### 8.2 Prohibited exported data in the final proof

The final theorem-facing proof shall not export the raw accumulator witness
material, including:

- full execution rows,
- full stage witness families,
- raw fold transcripts,
- per-step Fiat-Shamir challenges,
- duplicated proof bundles,
- or raw recursive witness objects.

Those may exist only in an optional **audit artifact** that is explicitly
non-normative.

## 9. Audit Artifact

An implementation may define an optional audit/export bundle carrying:

- raw trace projections,
- raw stage projections,
- raw folding transcripts,
- terminal accumulator objects,
- and debug witnesses.

Such an artifact is not the theorem-facing proof.

The verifier for the theorem-facing proof shall not depend on the availability
of that audit artifact.

## 10. Consequence for the Current Direct Verifier Path

A design in which the final external verifier recomputes Fiat-Shamir challenges
by replaying the folding transcript from raw public sidecar data is conforming
only as:

- a direct-verifier reference path,
- an audit path,
- or a transitional proof-complete artifact.

It is non-conforming as the final exported recursive proof format.

The current `neo-fold-next` RV64IM public proof surface falls into this
transitional category. In particular, it still exports verifier-checked
auxiliary obligation digests and derived claim bundles for stage claims, stage
packages, kernel openings, prepared-step bindings, and current main-lane/root-
lane bridge bindings.

Those bindings are legitimate current implementation obligations. They should be
understood as transitional bridge-owned exports that a later recursive or
compression layer may absorb, rather than as the final desired theorem-facing
public format.

The current proof object also carries projection bundles and packaged claim/
opening proofs used by that bridge path. Those are likewise transitional
proof-complete sidecars, not the target compressed recursive export.

## 11. Compression Direction (Informative)

The preferred compression direction for this codebase is the one explicitly
enabled by the cited papers:

- a Goldilocks-native outer proof system,
- with SNARK-friendly hashing,
- and accumulator compression in the style anticipated by SuperNeo.

The local paper basis explicitly names Spartan with a FRI-based PCS as an
efficient compression direction for SuperNeo accumulators over Goldilocks.

This specification does not mandate a concrete compressor implementation beyond
the following invariant:

- the compressor must prove the recursive verifier relation, thereby binding the
  carried accumulator semantics through the recursive step circuit, without
  requiring the final external verifier to replay the full folding verifier from
  raw per-step or per-row public material.

## 12. Ownership Clarifications

For avoidance of doubt:

- this file owns the **export boundary** between the inner accumulator, the
  recursive verifier layer, and the final compressed proof,
- this file is therefore **not** a standalone SuperNeo backend implementation
  spec and shall be read together with `riscv-kernel.md`,
- `riscv-kernel.md` owns the concrete SuperNeo backend contract, including the
  Goldilocks parameterization, challenge domain, and chunk-local reduction role
  split,
- `riscv-recursive-instantiation.md` owns the concrete recursive backend
  choice, compression backend choice, canonical encodings of `U_i`, `W_i`,
  `u_i`, and `w_i`, the canonical public accumulator handle if one is used, the
  adapter from the current chunked proof path, and the final public Rust API,
- this file intentionally abstracts over the concrete representation of `U_i`
  only up to a boundary-safe level; it does **not** permit weakening the
  carried accumulator semantics below the paper's `CE(b, L)^k` meaning under the
  backend contract,
- raw accumulators remain valid audit or debug artifacts, but they are not the
  codebase's target exported theorem-facing proof format,
- and an implementation that matches the outer recursive shape while changing
  the inner carried accumulator semantics away from the backend-grounded
  SuperNeo relation is non-conforming.
