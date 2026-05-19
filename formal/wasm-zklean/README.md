# wasm-zklean

Formal verification of the **wasm zkVM's R1CS arithmetization** against
[zkLean](https://github.com/GaloisInc/zkLean) (Galois). zkLean is a Lean 4
DSL for expressing zero-knowledge circuits and reasoning about them.

This package is a **leaf** in the `formal/` subtree: it is independent of the
existing SuperNeo / Twist-Shout / Nightstream Lean packages and does not share
their toolchain.

---

## Why a separate Lake package?

zkLean pins Lean to `v4.25.2` (and mathlib to `v4.25.2`). The rest of the
`formal/` subtree pins to `v4.28.0`. Lean 4 minor releases are not
binary-compatible, and mathlib has API churn across three minor versions, so a
single Lake closure containing both is not feasible without a port on one side
or the other.

A separate package with its own `lean-toolchain` is the cheap, low-risk
option: `elan` picks the correct compiler when working in this directory,
and no other formal package depends on us, so the mismatch is contained.

If the Rust crates and zkLean ever need to share Lean theorems, the boundary
must be artifact-level (e.g. extracted JSON / hex), not Lean import.

---

## What zkLean actually provides

(Verified against `zkLean` commit `014fa397fa30`, files
[`AST.lean`](https://github.com/GaloisInc/zkLean/blob/main/zkLean/zkLean/AST.lean),
[`Builder.lean`](https://github.com/GaloisInc/zkLean/blob/main/zkLean/zkLean/Builder.lean),
[`Semantics.lean`](https://github.com/GaloisInc/zkLean/blob/main/zkLean/zkLean/Semantics.lean).)

- **`ZKExpr f`** — inductive AST of expressions over a field `f`:
  `Field | Add | Sub | Neg | Mul`. Comes with an evaluator
  `ZKExpr.eval : ZKExpr f → f`.
- **`ZKBuilder f`** — a `FreeM`-based monad over a small set of primitive ops
  (`AllocWitness`, `ConstrainEq`, **`ConstrainR1CS a b c`**, lookup-table ops,
  RAM ops). Note that `ConstrainR1CS` already takes `a, b, c : ZKExpr f`
  — i.e., the DSL is *row-shaped*: one builder op corresponds to one R1CS
  row `(Σaᵢxᵢ)·(Σbᵢxᵢ) = (Σcᵢxᵢ)`.
- **Semantics** (`Semantics.lean`): `semantics : ZKBuilder f α → List f → Bool`
  interprets a builder against a concrete witness list and returns whether
  every primitive constraint is satisfied. This is *direct interpretation*,
  not compilation to matrices.
- **Predicate-transformer / Hoare-style framework** for reasoning about
  builders (`Formalism.lean`).

### What zkLean does **not** provide

- A function `ZKBuilder f a → R1CSMatrix f` (i.e., a verified compiler from
  the DSL to a flat matrix representation) with an accompanying soundness
  theorem. There is no first-class `Matrix` data type in zkLean.
- A `ZKField` instance for any concrete field. We will need to either supply
  one for the wasm zkVM's field (Goldilocks, `Fq`) ourselves, or work
  generically over `[ZKField f]` and instantiate at the very end.

The first point matters: it shapes the verification strategy below.

---

## Verification approach

The Rust wasm zkVM produces R1CS as flat sparse matrices (`A`, `B`, `C`)
plus per-row tags (opcode scope + label) via `WasmTaggedR1csBuilder`
(see `crates/neo-fold-next/src/wasm/tagged_r1cs_builder.rs`). The high-level
gadgets it composes from (conditional-select, zero-test, u32 byte
decomposition, …) are *not* preserved as structure post-build — only the
flattened rows survive.

To verify against zkLean, we need to bridge from "flat rows + tags" on the
Rust side to a structured representation on the Lean side that supports
per-gadget reasoning — without introducing an unverified
`flat rows ⟶ structured DSL` step. The next subsection explains why the
obvious "extract rows from a `ZKBuilder`" framing does not survive contact
with zkLean's semantics, and pivots to the design we actually propose.

### Why we don't extract rows from a `ZKBuilder` directly

An earlier draft of this design called for a "small fold"
`extract : ZKBuilder f Unit → Array (ZKExpr f × ZKExpr f × ZKExpr f)` that
walked witness allocations and `constrainR1CS` ops, recovering symbolic R1CS
rows from the builder. **That approach does not work directly against
zkLean as written.** zkLean's `AllocWitness` semantics (`Semantics.lean`) is

```
| ZKOp.AllocWitness => …; .pure (ZKExpr.Field (<- st.witness[idx]?))
```

— the continuation receives `ZKExpr.Field <concrete witness value>`, not a
symbolic variable reference. `ZKExpr` has no `Var Nat` constructor; its
leaves are always concrete field elements. So once a `ZKBuilder` is run, the
expression trees are evaluated against the supplied witness, and there is
no general fold that recovers symbolic rows without one of:

- forking `ZKExpr` (or `ZKOp`) to add a variable constructor,
- wrapping the field type with a symbolic layer (which then fails the
  `Field f` / `ZKField f` instance obligations), or
- specializing the witness, which loses the universal quantification we
  need.

The cleanest pivot is to introduce a coarser intermediate representation
that Rust and Lean both speak, and stage the cross-check at that level.

### Generated content: gadget-level `Instr` trace + flat rows

The Rust exporter emits, per circuit, three pieces of data into the
`.lean` file:

1. **`instructions : List Instr`** — a structured trace of gadget calls
   plus raw R1CS rows. `Instr` is defined *in Lean* in
   `WasmCircuit/Gadgets.lean` and is **non-parametric**: column indices
   are plain `Nat`, and the working field appears only at row-evaluation
   time. Today's surface (will grow as more gadgets are instrumented):
   ```lean
   inductive Instr where
     | ZeroTest : (value invWitness isZero : Nat) → Instr
     | Raw      : (a b c : SparseRow) → Instr   -- escape hatch
   ```
   Naming witness columns as `Nat` sidesteps the value-vs-variable problem
   entirely. Witness allocation is implicit (column indices are absolute).

2. **`actualRows : List Row`** — the literal sparse rows the Rust matrix
   builder produced, with column indices preserved.

3. **A row-level cross-check theorem**, proved by `native_decide` against a
   closed-form expansion `instrToRows : Instr → List Row` (also in
   `Gadgets.lean`):
   ```lean
   theorem trace_matches_actual :
       instructions.flatMap instrToRows = actualRows := by native_decide
   ```
   If the Rust gadget for, say, `ZeroTest` drifts from `instrToRows
   .ZeroTest`, this fails at `lake build` time.

`instrToRows` is the "small fold" the original design wanted — but it
operates on the coarser `Instr` type we control, not on a `ZKBuilder`, so
the symbolic-variable problem doesn't arise. There is no symbolic
extraction from zkLean's free monad; the symbolic naming lives in `Instr`
from the start.

#### Coefficients: `Int`, not generic over a field

`SparseRow := List (Nat × Int)`. Choosing `Int` for coefficients (rather
than `SparseRow f` over a generic field) lets the cross-check theorem
ride on plain `DecidableEq` for `List (Nat × Int)`, no `ZKField Fq`
required. The wasm zkVM gadgets only emit small integer literals (`1`,
`-1`, `1 << 8`, `1 << 16`, `1 << 24`, …), and the exporter does
balanced-residue conversion from Goldilocks back to signed `Int` before
writing the `.lean` file. Lifting `Int → f` for soundness proofs uses
`IntCast` (every `Field f` has one).

#### Witness: `Nat → f`, not `List f`

The witness is modelled as a total function `Nat → f`, not a `List f` or
`Array f`. This keeps soundness proofs purely field-theoretic with no
`Option`/`getD`/`getElem?` indirection. Bridging to zkLean's `List f`
witness happens later, when `instrToBuilder` lands.

### Lifting to zkLean's semantics for spec-level proofs

`Gadgets.lean` defines a builder lifter:

```lean
def instrToBuilder (alloc : Array (ZKExpr Fq)) (i : Instr) :
    ZKBuilder Fq PUnit :=
  (instrToRows i).forM (rowToBuilder alloc)
```

This *uses* zkLean primitives (`constrainR1CS`) and is where zkLean's
surface is brought in. It is derived mechanically from `instrToRows`
through a generic `rowToBuilder` — so `instrToRows` remains the single
source of truth for the row structure of every gadget, and
`instrToBuilder` cannot drift from it by construction.

Per-gadget soundness lemmas (in `Bridge.lean`) are stated directly
against zkLean's `semantics`:

```lean
theorem zeroTest_sound (alloc : Array (ZKExpr Fq)) (v inv iz : Nat)
    (witness : List Fq)
    (h_one : (alloc[constOneCol]!).eval = 1)
    (h_sat : semantics (instrToBuilder alloc (.ZeroTest v inv iz))
               witness = true) :
    ((alloc[v]!).eval = 0 → (alloc[iz]!).eval = 1) ∧
    ((alloc[v]!).eval ≠ 0 → (alloc[iz]!).eval = 0)
```

The proof goes through two small bridge lemmas in `Bridge.lean`:
`constrainR1CS_semantics_iff` (one constraint passes iff its R1CS equation
holds at `eval`-level) and `two_constraints_semantics_iff` (its
specialisation to a two-row sequence, which is what `Instr.ZeroTest`
lowers to). Future gadgets with row-count > 2 will get a generic `forM`
version of the bridge.

### Why this works

The expensive direction (recovering structure from flat matrix rows) is
sidestepped: Rust already knows the gadget at the call site, and the fix
is to *emit* the structured tag rather than discard it. Existing tag
infrastructure (`WasmConstraintCatalog.row_tags`) likely covers most of
what's needed, so the Rust-side instrumentation is small.

The Lean-side `instrToRows` and `instrToBuilder` are written once per
gadget (currently 3–6 gadgets) and proven once. Adding a new gadget adds
one `Instr` constructor + one `instrToRows` case + one `instrToBuilder`
case + one soundness lemma — bounded, mechanical work.

---

## Trust model

What is trusted, in shrinking order:

1. **Lean 4 kernel + `native_decide`**: standard; same surface as any other
   Lean verification.
2. **zkLean's semantics + theorems we reuse**: external dependency, but
   peer-reviewed Galois code; we pin a commit.
3. **`instrToRows` and `instrToBuilder`** (in `WasmCircuit/Gadgets.lean`):
   one closed-form case per `Instr` constructor. Reviewable; not large.
4. **Per-gadget soundness lemmas** in `WasmCircuit/Gadgets.lean`: one per
   Rust gadget we instrument; each proven once against zkLean's semantics.
5. **The bridge lemma** linking `semantics (instrToBuilder i)` to the
   row-form check on `instrToRows i`. Written once, mechanical.
6. **Rust trace tags identifying which gadget produced which rows**: small
   instrumentation (push an `Instr::CondSelect { … }` next to existing
   `push_row` calls). The cross-check theorem catches any mismatch.

What is **not** trusted:

- The Rust constraint-row generation itself — it must equal what the Lean
  gadget definitions produce, or `native_decide` fails.
- The Rust matrix builder's high-level correctness — it is reduced to
  per-row equality with Lean-side definitions.

---

## Directory layout

```
formal/wasm-zklean/
├── lakefile.toml             # package config
├── lean-toolchain            # pinned to v4.25.2 (zkLean's toolchain)
├── README.md                 # this file
└── WasmCircuit/
    ├── Gadgets.lean          # HAND-WRITTEN. `Instr` IR, `instrToRows`
    │                         # (data lowering), `instrToBuilder` (zkLean
    │                         # lowering, derived from `instrToRows` via
    │                         # `rowToBuilder` — single source of truth).
    ├── Bridge.lean            # HAND-WRITTEN. Unwinds `semantics ... = true`
    │                         # to field-level row equalities. Per-gadget
    │                         # soundness theorems live here.
    ├── Columns.lean          # AUTO-GENERATED; gitignored. Column-index
    │                         # `def`s derived from the Rust `define_columns!`
    │                         # macro. Imported by `Generated.lean` for
    │                         # readable references; NOT imported by
    │                         # `Gadgets.lean` (hardcoded `constOneCol := 0`
    │                         # plus a generated sanity-check theorem keeps
    │                         # the proof module independent of generated
    │                         # files).
    ├── Generated.lean        # AUTO-GENERATED; gitignored. Per-circuit
    │                         # `instructions`/`actualRows` + the
    │                         # `trace_matches_actual` cross-check.
    ├── Field.lean            # HAND-WRITTEN. `Fq := ZMod p` for Goldilocks
    │                         # plus the `ZKField Fq` instance. Primality of
    │                         # `p = 2^64 − 2^32 + 1` is a trust-debt axiom.
    └── Lemmas.lean           # TBD. Hand-written spec-level theorems about
                              # the full circuit, chaining gadget soundness.
```

`Gadgets.lean` is buildable on a clean checkout without running the
exporter. `Generated.lean` (and the `Columns.lean` it imports) require a
prior `cargo run --bin export_wasm_zklean`. The `lean_lib WasmCircuit`
glob in `lakefile.toml` picks all three up when present.

---

## Rust-side wiring

The exporter lives at
[`crates/neo-fold-next/src/bin/export_wasm_zklean.rs`](../../crates/neo-fold-next/src/bin/export_wasm_zklean.rs).
It currently emits a tiny demo circuit: one call to the real
`push_zero_test_gadget` (now `pub` in `wasm::gadgets`), plus the matching
`Instr.ZeroTest` trace and the `trace_matches_actual` cross-check. Future
work: instrument more gadget call sites so the trace is recovered from
the live `WasmConstraintCatalog` rather than constructed alongside it in
the exporter.

```bash
# Regenerate WasmCircuit/Generated.lean:
cargo run --bin export_wasm_zklean --release

# Then, from formal/wasm-zklean/:
lake build
```

The exporter is intentionally placed in `neo-fold-next` (where
`WasmTaggedR1csBuilder` lives) rather than a separate crate, because its
real input is the wasm constraint catalog produced by that crate.

---

## Build instructions

From this directory:

```bash
# First time: fetch zkLean and its transitive deps (mathlib, cslib, bvmod_eq)
lake update

# Then build:
lake build
```

`elan` will pick the v4.25.2 toolchain automatically from `lean-toolchain`.

---

## Open questions / next steps

1. **`ZKField Fq` instance.** zkLean's `ZKField` class requires field
   operations plus `field_to_bits` and `field_to_nat`. Goldilocks supports
   both, but we have to write the instance ourselves
   (planned home: `WasmCircuit/Field.lean`).
2. **Bridge lemma: builder semantics ↔ row-form satisfaction.** Proving
   that `semantics (instrToBuilder i) w = true ↔ ∀ (a,b,c) ∈ instrToRows i,
   a·w * b·w = c·w` is mechanical but not trivial — needs evaluation of
   `ZKExpr` against witness rows tied to sparse-row dot products.
   Confirm size/feasibility before committing further.
3. **Witness allocation ordering.** `Instr` constructors name columns by
   absolute index. Rust must allocate witnesses in a deterministic order
   and `instrToBuilder` must prepend matching `let _ ← witness` calls so
   column indices align. Bookkeeping is straightforward but is a real
   correctness obligation, not invisible.
4. **`native_decide` performance** at the cross-check theorem. With ~413
   wasm constraint rows and Goldilocks-field arithmetic, this should be
   sub-second, but worth benchmarking before relying on it in CI.
5. **Trace instrumentation strategy in Rust.** Tag-based recovery from the
   existing `WasmConstraintCatalog.row_tags` vs. source-level instrumentation
   in each gadget. Start with tag-based; reassess if any tag becomes
   ambiguous.
6. **CI cadence.** Because `Generated.lean` is gitignored, CI runs the
   exporter then `lake build` on this package; success implies
   `trace_matches_actual` held under `native_decide`. This catches
   Rust↔Lean drift without requiring contributors to keep a generated
   file in sync by hand.
