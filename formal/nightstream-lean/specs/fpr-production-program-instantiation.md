# Fixed-one production-program instantiation

```text
profile:
  fixed-one
  plain
  270-coordinate public carrier

initial target:
  complete DataCodecs
  + six certified physical CallRecipes
  + internally constructed SourceAuthority
  -> complete ProductionProgram
  -> ProductionProgram acceptance iff canonical checker acceptance

initial result:
  blocked at the mandatory physical-owner audit

authorized follow-up:
  re-derive one owner-aligned program from existing production phases

current status:
  precisely blocked before program selection by the absent whole-current owner
  artifact and the missing selected terminal refinement
```

## Scope and authority

This audit is the first gate for the single shipping-profile instantiation. It
does not change the frozen paper semantics, the typed lowering vocabulary,
Rust, or generated artifacts.

`FixedOneLoweringAdapter.CallAlignment` supplies model-level semantics for all
six open calls. A physical `CallRecipe` is stronger: it must have exact row
count, unique receipt ownership, complete column support, active soundness,
active honest completeness, and inactive satisfiability. A broad enclosing row
range, a profiler label, or a theorem with an additional bad-event branch is
not such a recipe.

## Exact call-owner map

### `step`

- Typed semantics:
  `FixedOneLoweringAdapter.CallAlignment.step`.
- Semantic operation:
  `FixedOneCanonicalAdapter.application`. It totalizes the complete native
  fixed-one application, including fold-variant matching, initial/active entry
  checks, recursive NIFS verification, nonempty output, semantic advance,
  Nebula advance, and the outgoing fresh link.
- Current Rust rows:
  the behavior is distributed across the complete base and recursive F-prime
  branch trees:
  `fprime.base.{prelude,source,initial,advance,output}` and
  `fprime.recursive.{prelude,transcript,nifs,prior_link,nebula,accumulator,
  counter,output}`, followed by selector composition and finalization in
  `frontends/r1cs_f_prime/full_relation.rs`.
- Important non-owner:
  `fprime.{base,recursive}.finalize.application` is the application-specific
  relation `F_j`; it is not the typed lowering's complete
  `FixedOneCanonicalAdapter.application`.
- Required codecs:
  `Option DirectState` and `AdapterWitness`.
- Missing refinement:
  no disjoint call-local row owner implements the complete typed operation.
  Giving the whole recursive branch to this call would also give it the NIFS,
  prior-hash/link, and next-hash rows required by later typed calls, violating
  unique receipt ownership. Giving it only a proper subset cannot establish
  the exact totalized output. No state or witness codec exists.

Result: **no existing complete physical owner**.

### `hashPrior`

- Typed semantics:
  `FixedOneLoweringAdapter.CallAlignment.hashPrior`.
- Semantic operation:
  totalized `paperHash` at iteration `i`, including presence of `current` and
  exact iteration, initial-state, running, and program-counter alignment.
- Current Rust rows:
  the recursive step computes the prior XOut inside
  `fprime.recursive.step.prior_link.digest`, then constrains its encoding and
  fresh public link in the rest of `fprime.recursive.step.prior_link`.
  At the terminal boundary Rust reuses the last step's already-computed
  `x_out_bits`; `terminal.latest_link` does not recompute the hash.
- Existing Lean physical slice:
  `FPrimeFullHistoryXOutSpongeReceipts.priorReceipt` owns only the nonoptional
  23-field Poseidon2 sponge core. `ProductionXOutSponge23InputAlignment`
  aligns that core's source vector. `TerminalLink.PlacementRefinement` owns the
  fused terminal prior-public equality.
- Required codecs:
  state, running, and optional digest; bounded `Nat` is already generic.
- Missing refinement:
  no owner supplies the optional presence coordinate, failed-preimage
  alignment behavior, complete call-frame serialization, and one reusable
  receipt for both the Step and Terminal occurrences. The terminal placement
  theorem explicitly owns `priorLinkAccepted`, not a `hashPrior` call.

Result: **no existing complete physical owner**.

### `hashNext`

- Typed semantics:
  `FixedOneLoweringAdapter.CallAlignment.hashNext`.
- Semantic operation:
  the same totalized `paperHash`, but at iteration `i + 1` and with the next
  state and running result.
- Current Rust rows:
  the base and recursive producers compute the next XOut inside
  `fprime.base.step.output` and `fprime.recursive.step.output`.
- Existing Lean physical slices:
  `FPrimeFullHistoryXOutSpongeReceipts.{baseReceipt,
  recursiveOutputReceipt}` own the two nonoptional 23-field sponge cores.
- Required codecs:
  state, running, and optional digest.
- Missing refinement:
  the same presence/alignment wrapper, serialization, frame placement, and
  exact receipt obligations as `hashPrior`. No current whole-program owner
  identifies either output phase with the typed `hashNext` occurrence.

Result: **no existing complete physical owner**.

### `nifsVerify`

- Typed semantics:
  `FixedOneLoweringAdapter.CallAlignment.nifsVerify`.
- Semantic operation:
  the selected fixed-one verifier on `(running, fresh, proof)`, returning the
  folded running value or rejecting.
- Current Rust rows:
  `fprime.recursive.step.nifs`, with nested PiCCS, PiRLC, PiDEC, and point
  binding owners. `terminal.nifs` is a separate final-fold operation and is
  not this Step call.
- Existing Lean semantics/artifact bridge:
  `FPrimeConcreteNifs.recursive_rows_nifsVerify_or_badRoot` reaches the
  concrete callback or the named projection-root event.
- Required codecs:
  running, adapter fresh input, and `Step.FoldProof`/NIFS proof.
- Missing refinement:
  no concrete lowering configuration selects these semantic carriers; no
  codecs or call-frame placement exist; no `CallRecipe` turns the physical
  owner into the exact partial `callEval`. The existing artifact theorem has a
  named `BatchBadRoot` alternative, whereas `CallRecipe.activeSoundness`
  requires the exact call result.

Result: a physical NIFS owner exists, but it **cannot be wrapped directly as
the required recipe**.

### `runningCheck`

- Typed semantics:
  `FixedOneLoweringAdapter.CallAlignment.runningCheck`.
- Semantic operation:
  the selected unary `terminalChecks.runningCheck` at the fixed verifier key.
- Current Rust candidates:
  `terminal.running_link` is only accumulator-digest continuity.
  `decider.terminal_ce` directly closes a family of final NIFS-output CE claims
  against their opened witnesses.
- Required codecs:
  running, running witness, and the generic Boolean result.
- Missing refinement:
  there is no production `FixedOneLoweringAdapter.Configuration` selecting
  concrete terminal relations/checkers or witness types. The family-level
  direct CE owner has not been refined to the selected unary call frame, and
  no corresponding codecs or receipt exist.

Result: an adjacent physical relation owner exists, but it **cannot be wrapped
directly as the required recipe**.

### `freshCheck`

- Typed semantics:
  `FixedOneLoweringAdapter.CallAlignment.freshCheck`.
- Semantic operation:
  the independent unary `terminalChecks.freshCheck`; the canonical terminal
  program performs no NIFS fold.
- Current Rust rows:
  no standalone owner. `terminal.latest_link` is only the prior-public
  equality and is already proved not to be `freshCheck`. The final fresh
  relation is consumed inside the broad `terminal.nifs` final-fold execution,
  which also owns running/folding work and produces a new accumulator.
- Required codecs:
  adapter fresh input, fresh witness, and the generic Boolean result.
- Missing refinement:
  no selected terminal relation/checker, no witness codec, no unary Boolean
  owner, and no nonoverlapping call receipt. Reusing `terminal.nifs` would
  change the semantic call and conflate it with operations outside the typed
  Terminal program.

Result: **no existing complete physical owner**.

## Codec audit

The selected generic family already has canonical field, Boolean, and bounded
natural codecs. The production fixed-one directory adds only:

- four-lane digest;
- optional digest;
- compact adapter `Encoded`.

No production codec exists for:

- `Option DirectState`;
- `AdapterWitness`;
- running;
- adapter fresh input;
- `Step.FoldProof`/NIFS proof;
- running witness;
- fresh witness.

Consequently the existing slices do not compose into a complete
`DataCodecs`, `Profile`, or `DirectProfile`.

## Hash-sharing audit

`FPrimeFullHistoryXOutSpongeReceipts.pureExecutions_equal` proves that the
three captured nonoptional sponge cores compute the same pure function when
given an equal 23-field vector. It does not prove equality of the distinct
prior/next preimages, the optional `paperHash` wrapper, alignment checks,
call-frame ownership, or current placement.

Therefore `hashPrior` and `hashNext` may eventually share a generic sponge
subrecipe, but the complete call recipes cannot currently be shared or
constructed from that theorem.

## Terminal and NIFS wrapping audit

- The recursive NIFS rows have a real physical owner, but exact typed codecs,
  frame placement, and bad-root handling are still missing.
- The terminal prior-link owner wraps only the fused public equality.
- The terminal direct CE owner is family-valued and is downstream of a final
  NIFS fold.
- The canonical typed Terminal program instead performs no NIFS and requires
  two independent unary relation checks.

Thus the existing terminal and NIFS row owners cannot be wrapped directly
into all three required recipes without changing semantics or inventing a new
row partition.

## Source authority

`PaperBoundary.SourceAuthority.ofCanonicalOpening` is the correct model-level
constructor. No production use currently proves its `contextExact` premise;
repository-wide uses are limited to its definition, frozen export, and focused
checks.

Gate 2 was not entered because Gate 1 failed. A future production theorem must
construct this receipt from decoded authoritative opening data and prove the
selected context identity internally. It must not add the receipt back as a
caller premise or move source authority into `nifsVerify`.

## Stop decision

The mandatory stop conditions are reached:

1. `step`, both complete hash calls, and `freshCheck` have no existing
   complete nonoverlapping physical owner.
2. the concrete state, witness, running, fresh, proof, and terminal-witness
   types lack canonical production codecs;
3. wrapping `terminal.nifs` as `freshCheck` or interpreting
   `finalize.application` as the typed `step` would change the semantic call;
4. assigning enclosing F-prime/NIFS ranges to multiple recipes would violate
   exact receipt ownership.

No new recipe, `DataCodecs`, `ProductionProgram`, source-authority production
theorem, Rust equality, or row artifact is constructed by this slice.
`DirectCalls.certifiedSubset` remains intentionally live; it cannot be retired
in favor of `allRecipes`.

This is an exact current-interface/source-owner obstruction, not a
kernel-checked impossibility theorem about every future encoding. Resolution
requires either new production physical owners matching the typed calls or an
explicitly authorized re-derivation of the typed program/owner partition from
the frozen checker. Neither is authorized in this slice.

## Owner-aligned Route 2 follow-up

Route 2 was subsequently authorized: describe the constraints production
actually owns rather than add six physical call owners to make the canonical
direct-call encoding fit. The canonical direct-call program and the current
production owner partition remain different constructions.

The current Rust owner vocabulary is phase-shaped:

- base and recursive branch phases;
- recursive transcript, NIFS, prior-link, Nebula, accumulator, counter, and
  output phases;
- selector and finalization phases;
- terminal NIFS, running-link, parent-link, latest-link, accumulator, public
  pinning, and terminal-CE phases.

These are not six missing `CallRecipe`s under new names. A production program
must reuse the existing block, branch, program, and receipt machinery and must
prove semantics for each selected physical phase. No parallel generic IR or
new row owner is authorized.

### Kernel-checked owner-alignment boundary

`FPrimeProductionOwnerProgramBoundary.AlignmentOpacity.
not_attemptedOwnerAlignmentBridge` reaches the exact structural interface
provided by `SourceAlignment.AlignedReceiptProgram`. It exhibits an aligned
receipt program whose complete physical rows are satisfied while its indexed
typed source rejects.

The obstruction is deliberately narrow: exact occurrence owners, receipt
well-scoping, and row conservation do not themselves prove the semantics of
an instruction or phase. It does not say that a production phase refinement
is impossible. It identifies the missing primitive as a phase-specific
soundness and honest-completeness theorem relating the existing physical rows
to the frozen operation they implement.

### Current-artifact hard gate

The only checked complete full-history artifact is the captured historical
program with:

```text
totalRows = 4,193,134
terminal public-link width = 257
```

The bounded current terminal-link placement instead starts at row
`9,673,389` and owns 270 public-link rows. Theorems
`currentTerminalLink_starts_after_historicalProgram` and
`currentTerminalLink_not_in_historicalProgram` prove that this current range
lies outside the historical whole-program interval.

Therefore the historical manifest cannot certify exact ownership of the
current shipping rows. No checked whole-current manifest or complete current
row program is present from which to prove:

- every selected current row has exactly one phase owner;
- current phase ranges do not overlap;
- no current row lies outside the selected program.

Artifact regeneration remains sequenced after selecting and refining the
typed owner program, so this audit does not manufacture a replacement
manifest or relabel the historical one as current.

### Terminal-selection hard gate

The frozen fixed-one terminal verifier selects explicit
`TerminalRelations` and `RelationChecks` and requires independent unary
running and fresh checks. It performs no NIFS fold. The physical terminal
shell instead reconstructs `TerminalFacts`: final-NIFS acceptance, digest
links, accumulator preservation, continuity, public pins, and terminal CE
facts.

`terminalFacts_do_not_select_relationChecks` proves the exact interface
opacity: the same `TerminalFacts` value is independent of the selected frozen
relations and checkers and can coexist with one accepting and one rejecting
exact fixed-one checker instantiation. This is not a proof that the production
terminal is unsound. It proves that the current terminal fact carrier alone
does not identify or refine the frozen terminal relations.

The missing concrete theorem must select the production relations/checkers
and derive both frozen terminal obligations from the existing physical
terminal facts, or return an exact registered production failure. Reusing the
final NIFS fold as `freshCheck`, suppressing `BatchBadRoot`, weakening the
terminal relation, or adding a generic refinement failure is forbidden.

### Stop result

The mandatory Phase 1 gate cannot be established from the current checked
assets:

1. the structural aligned-receipt layer does not own phase semantics;
2. the complete checked row partition is historical and does not contain the
   current 270-row terminal-link placement;
3. the physical terminal facts do not select or refine the frozen independent
   running/fresh checks.

Consequently this slice constructs no typed production owner program, no
production source-authority theorem, no Rust-program equality, and no current
row equality. Phase 2 was not entered, so
`SourceAuthority.ofCanonicalOpening` remains a model-level constructor whose
production `contextExact` discharge is still required.

`FPR-DIRECT-CALL-RECIPES` remains five of eleven and
`DirectCalls.certifiedSubset` remains live. The current production owner
partition has no physical-minimality claim; only the previously proved
semantic-obligation minimality and exact ownership of the captured historical
artifact remain available.

---

## FPRIME-NIFSVERIFY-RECIPE-TRANSPORT

```text
claim:
  The missing nifsVerify CallRecipe is a representation-transport obligation,
  not a protocol-design one. Every semantic ingredient is already proved.
status: WITHDRAWN 2026-07-26 (cycle 235). The transport target is artifact-
        owned, so transporting it would take the canonical row layout from a
        generated artifact — banned by §4.1. The semantic chain below survives
        as evidence; the row-reuse plan does not.
```

### Why this was withdrawn

`RecursiveRows` is not a Lean-owned row program. It is a conjunction of
`Satisfies <generated row family> assignment`, one field per artifact:

```lean
transcript : Satisfies FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows assignment
projectionGlue : Satisfies recursiveGlueRows assignment
feSumcheck : Satisfies recursiveFeRows assignment
...
```

`ConcreteNifs`' own header says it "records the exact verifier facts proved by
the **generated** recursive and terminal row families". Mechanically: its
transitive closure is 496 modules, of which **396 are `Artifacts` modules**.

So `recursive_rows_verify` is excellent *semantic* evidence — it says what the
verifier computes — but its rows cannot define the canonical recipe. Doing so
would inherit the artifact's row layout, which §4.1 bans outright.

**What survives.** The semantic chain (`callEval` → adapter → `stepSemantics`
→ `recursiveNativeVerify`) is unaffected and remains the specification the
canonical rows must meet. The `BadRoot` obstruction
(`FPRIME-NIFSVERIFY-SOUNDNESS-SHAPE`) also survives unchanged: the event is
intrinsic to checking a polynomial identity at one sampled challenge, so a
Lean-owned PiRLC projection will expose the same event.

**What replaces it.** Canonical `nifsVerify` rows must be *emitted* from the
Lean-owned PiCCS/PiRLC/PiDEC verifier obligations, exactly as
`canonicalProgram` is emitted for Poseidon2. The generated families become a
later comparison target, never a definition.


### Why this is not a §7 escalation

Constructing `nifsVerify` looked like it required selecting a folding rule,
which would change a verifier branch and force a stop. It does not. The chain
from the lowering call down to a frozen, row-forced verifier already exists and
is closed at every link:

| link | theorem | what it gives |
|---|---|---|
| lowering call → adapter | `FixedOneLoweringAdapter.nifsVerify` | `callEval Call.nifsVerify` **is** `adapter.step.nifsVerify`, by `rfl` on the accepting branch |
| adapter → F′ semantics | `FPrimeConcreteNifs.stepSemantics` | the callback is *definitionally* `recursiveNativeVerify`; callers cannot substitute one |
| semantics → decoded fold | `FPrimeConcreteNifs.stepSemantics_nifsVerify` | accepted rows drive the callback to `recursiveAccumulator proof` |
| rows → verify | `ConcreteNifs.recursive_rows_verify` | `RecursiveRows` forces `recursiveVerify proof = some (recursiveAccumulator proof)` |

So the verifier Lean must own is already owned. No folding rule is chosen here,
and nothing in this obligation touches a mapped paper definition.

### What is actually missing

`CallRecipe` demands rows in the lowering's own representation — `List
OwnedRow` under a `CallFrame` — together with `rowCount` against
`footprints.nifsVerify`, `rowsOwned`, `rowIdsNodup`, `rowsSupported`,
`activeSoundness`, `activeHonestCompleteness`, and inactive satisfiability.
`recursive_rows_verify` is stated over the `FPrimeFullHistoryProjection` row
family, a different representation.

The obligation is therefore: transport that row family into `OwnedRow` under a
frame, and discharge the seven `CallRecipe` fields against the transported
program. The audit above already names the same three gaps from the other
direction — "exact typed codecs, frame placement, and bad-root handling".

### Bad-root handling is the one real branch

`recursive_rows_nifsVerify_or_badRoot` is a disjunction: accepted rows give the
decoded accumulator **or** expose the projection-root event.
`activeSoundness` admits no disjunction — it demands `callEval = some outputs`.
So the transport must either discharge the `BadRoot` alternative inside the
frame or carry it as an exact named event. Silently dropping it would be
`A → ¬B` presented as a classification, which §3 forbids.

### Scale

The five direct recipes cost ~6,000 lines for equality, affine and zero-test
calls. This is the protocol's folding verifier and is larger than all five. It
is a multi-cycle obligation, not a single-cycle one, and it should not be
started by writing rows — it should start by fixing the frame placement and the
`BadRoot` treatment, because those two decide the row program's shape.

---

## FPRIME-NIFSVERIFY-SOUNDNESS-SHAPE

```text
claim:
  CallRecipe.activeSoundness, as typed, cannot be discharged for nifsVerify.
status: OBSTRUCTION 2026-07-26 (cycle 232), kernel-checked at production ops.
        Needs a contract decision, recorded below.
```

### The mismatch

`activeSoundness` demands an unconditional conclusion — satisfying the rows
gives `callEval call inputs = some outputs`. Every recipe built so far meets it,
because equality, affine maps, zero tests and Poseidon2 are exact arithmetic.

`nifsVerify` is the first call whose soundness is statistical. The honest
statement is a disjunction: accepted rows give the decoded accumulator **or**
expose `BatchBadRoot`, which is `¬Exact ∧ eval lhs beta = eval rhs beta`. No row
program can exclude it — rows check only the evaluation at `beta`, which is the
entire point of the projection optimization.

### Why the disjunction cannot be collapsed

`NifsRecipeShape.badRoot_at_production_ops` exhibits `BadRoot` at
`ProjectionProgram.K.ops`, the real Goldilocks-quadratic operations: the
coefficient vectors `X` and `1` differ yet both evaluate to `1` at `beta = 1`.
The fixture is also `Accepted` — a check the verifier passes — and perturbing one
coefficient breaks acceptance, so the check discriminates.

**Scope.** This is a hand-built identity, not one arising from `BatchIdentity
recursiveTraces` on a real proof. It says nothing about whether F′ is
attackable and is not a production defect. It rules out exactly one thing:
discharging `activeSoundness` by proving the event impossible.

### The decision

Two routes, and only one is admissible:

1. **`CallRecipe` gains a named-event disjunct.** `activeSoundness` becomes
   "outputs decode, or this exact named event holds". Honest, and it matches
   what the protocol actually proves. Cost: it touches the shared contract and
   the nine constructed recipes, which would discharge the new disjunct
   vacuously since their arithmetic is exact.

2. **`nifsVerify` carries a no-bad-root premise.** Rejected. §3 forbids it — no
   real consumer can construct that premise, because constructing it *is* the
   difficulty. It would move the obligation rather than close it.

Route 1 is the only one that closes anything. It is a contract change, not a
protocol change: no paper definition, frozen relation or verifier branch moves,
and the probability bound that makes the event negligible remains a separate M6
obligation, exactly as `ProjectionCheck`'s own header already states.

### Corrected event design (cycle 234)

The first attempt made the event a `CallRecipe` **field**:

```lean
badEvent : (ColumnId -> Field) -> Prop
```

That is wrong, and was reverted. A recipe author can set
`badEvent := fun _ => True` and discharge `activeSoundness` with
`Or.inr trivial`, proving nothing. It is exactly the generic-acceptance escape
`DirectCalls.RemainingRecipes` documents its typed boundary to prevent.

The event must be **closed, typed, and owned above the recipe** — by the F′
signature, keyed on the call — so a recipe cannot widen its own escape:

```lean
inductive FPrimeCallEvent (call : Call) (assignment : ColumnId -> Field) where
  | nifsBatchBadRoot :
      BatchBadRoot ProjectionProgram.K.ops identity -> FPrimeCallEvent .nifsVerify assignment
```

Every call other than `nifsVerify` has an **uninhabited** event type, so the
nine exact recipes discharge the disjunct by `Or.inl` and nothing about their
soundness weakens. The event must also depend on the exact call occurrence and
decoded inputs, not merely on the assignment, so the identity it names is the
one that call actually checked.

Propagation is then:

```text
physical Step satisfaction
  ⇒ frozen Step acceptance ∨ exact NIFS BatchBadRoot
```

Measured blast radius of the reverted attempt: 19 errors across 10 files — nine
recipes at two each (missing field, missing `Or.inl`) plus `SelectedBranch`.
Mechanical. At cycle 234, `lake build Nightstream` also had 36 errors in three
unrelated correspondence modules. That was a historical gate result, not an
exemption from the full build. Later maintenance removed the unimported
`ProductionPaperOuter` experiment and repaired `PrivateDecoder/Coverage` and
`PiDecStrictProductionCompiler/Differential`; current aggregate status must
still come from a new full build.

### The event must be bound to the call occurrence (cycle 237)

The corrected design above — a closed family keyed on the call — is necessary
but **not sufficient**. An event carrying an unquantified identity:

```lean
| nifsBatchBadRoot (identity : Identity K) (bad : BadRoot K.ops identity) :
    CallEvent .nifsVerify assignment
```

is closed, is uninhabited for every other call, and is *still* a free pass.
`NifsRecipeShape.unbound_event_is_inhabited` proves it: for **any** assignment
there is a bad root, witnessed outright by `collidingIdentity`. A `nifsVerify`
recipe could discharge `activeSoundness` with `Or.inr` and prove nothing — the
same defect as the reverted `badEvent := fun _ => True`, one level down.

Note the irony: cycle 232's witness, which proved the disjunction cannot be
collapsed, is exactly what makes an unbound event vacuous. The same theorem
closes one route and opens a hole in the next.

So the identity must be **bound** — it must be the identity this call
occurrence's rows actually checked, derived from the assignment by the row
program's own trace function.

**Consequence: the queue was in the wrong order.** That trace function is
defined *by* the Lean-owned NIFS row program. So the event family cannot be
defined before the row program exists, and the `CallRecipe` widening cannot be
finalized before the event family. The correct order is:

1. construct the Lean-owned NIFS row program and its projection traces;
2. define the event bound to those traces;
3. widen `CallRecipe` against it;
4. propagate through primitive, branch and Step soundness.

Items 2 and 3 were previously listed first. They are not startable.

---

## KHORNER-SOUNDNESS

```text
claim:
  A satisfying assignment of hornerRows computes hornerValue.
status: CLOSED 2026-07-26 (cycle 241). `KHorner.hornerRows_sound`.
```

**Closed.** The induction went through in one attempt, for a reason worth
recording: the reference values are *derived from the assignment*
(`carriedValue z`) rather than supplied alongside it. That removes the
agreement hypothesis and the parallel-list relation the obvious formulation
would have needed, and with it the `List.Forall₂` that is unavailable without
Mathlib.

**Soundness does not need frame disjointness.** Each step's `outLow_sound` uses
only that step's own rows, so overlapping frames would over-constrain the
system without breaking this direction. Disjointness is required for honest
completeness, ownership and conservation — all still open.

`KHorner` emits the row program for one polynomial evaluation and derives its
count — `3d` rows for degree `d`, three per multiplication, none for the
additions. What it does **not** yet prove is that a satisfying assignment
computes `hornerValue`.

`KMul.outLow_sound` and `KMul.outHigh_sound` are the per-step ingredients. What
is missing is the assembly across the recursion, which additionally needs frame
disjointness — distinct steps must allocate distinct columns, or one step's
products could be read as another's.

Until this closes, `KHorner` supplies a **count, not a correct evaluation**. The
module header says so explicitly; it originally claimed the soundness proof and
was corrected in the same cycle.

## KHORNER-K-BRIDGE

```text
claim:
  KHorner.Pair arithmetic is ProjectionProgram.K arithmetic.
status: CLOSED 2026-07-26 (cycle 242). `KBridge.toPair_eval`.
```

**Closed.** `toPair` maps `K` coordinates to `Pair`, and `toPair_add`,
`toPair_mul`, `toPair_eval` prove it carries extension addition,
multiplication and Horner evaluation. This upgrades cycle 238's numerical spot
check of `KMul`'s formulas against `K.mul` into a theorem.

The `[c]` base case is discharged by `K_mul_zero` and `K_add_zero`: `eval`'s
literal `c + point · 0` equals `c`, so skipping that multiplication loses
nothing — which is what makes `KHorner`'s `3d` count honest rather than a
shortcut.

---

## PROJECTION-EQUALITY-ROW-COST

```text
claim:
  A K-valued equality lowers to a known number of base-field R1CS rows.
status: CLOSED 2026-07-26 (cycle 250). `KEquality.rows_length` — two, derived
        from the emitted list.
```

**Two rows, and zero columns.** Each coordinate is emitted as `left · 1 =
right`, reading the constant-one wire in the `B` operand. Nothing is allocated,
so an equality costs rows but no columns.

That asymmetry is worth stating: a `Typed.Cost` for the projection block cannot
assume rows and columns move together, because this component moves one and not
the other.

**Consequence for the projection identity.** Two Horner evaluations plus one
`K`-equality is `3·d_lhs + 3·d_rhs + 2` rows, not `+ 1`. For a degree-3
identity on both sides that is 20 rows, and the PiRLC batch checks one identity
per native output component — so the undercount would have compounded.

I have repeatedly described the projection identity check as "two Horner
evaluations plus **one** equality row". That phrasing is a trap. A `K`-valued
equality is *two* Goldilocks coordinate equalities, low and high, so it cannot
lower to one physical row. One semantic equality must not be counted as one
R1CS row.

The exact cost depends on how equality is emitted — `a · 1 = b` is one row per
coordinate, giving two; a difference-times-one form is also two. Either way the
number is two, not one, and it must be *derived* from the emitted rows like
every other count in this project rather than asserted here.

This is recorded now so the identity check is written against it, rather than
discovered after a row count has been published.

---

## FPRIME-EVENT-INVENTORY

```text
claim:
  BatchBadRoot is the complete list of named soundness events for the canonical
  nifsVerify recipe.
status: PARTIAL 2026-07-26 (cycle 252). Settled for the projection layer;
        open for every other layer.
```

**Settled for the projection layer.** `SuperNeo.ProjectionCheck` is a closed
module — four definitions, five theorems, one event structure — and `BadRoot`
is the only event in it. `batchAccepted_implies_exact_or_badRoot` is the frozen
batch statement and its disjunction has exactly two arms. So for this layer the
inventory is complete at one event, and that is now a reading of the module
rather than an assumption.

**Open for every other layer.** PiCCS, PiDEC and the sumcheck rounds have their
own vocabularies and their own events; none is audited. Freezing
`FPrimeCallEvent` still requires those audits, and the layer-by-layer result
here must not be mistaken for the whole inventory.

Cycles 232 and 237 established that the event family must be closed and
occurrence-indexed. Neither established *what belongs in it*. `BatchBadRoot` is
the event the projection check exposes; a complete PiCCS/PiRLC/PiDEC verifier
recipe may expose others — commitment binding, transcript collision, or
sumcheck round failures each have their own named events elsewhere in this
tree.

Freezing `FPrimeCallEvent` with one constructor because one is all that has
been noticed would be exactly the failure mode of assuming a witness isolates
what it claims. The inventory must be audited against the complete verifier
obligations *before* the family is frozen, not after.

The canonical track evaluates through `lcEval : (Nat → Nat) → LinComb → Nat`,
so `KHorner`'s reference is on `Pair`, two `Nat` coordinates. `ProjectionCheck`
and `ProjectionProgram.K` work over `Fin goldilocksP` pairs. Both describe the
same arithmetic and `KMul`'s formulas were checked against `K.mul` numerically,
but no theorem relates the two representations.

The bridge belongs with the identity check rather than with the gadgets, since
that is the first place a `ProjectionCheck.Identity` is actually mentioned.

---

## KFRAMES-DISJOINT-ALLOCATION

```text
claim:
  Distinct Horner steps allocate disjoint columns, and the allocation is an
  exact contiguous block.
status: CLOSED 2026-07-26 (cycle 243). `KFrames.frameColumns_nodup`,
        `KFrames.frameColumn_step_disjoint`.
```

Step `s` takes `base + 3s`, `base + 3s + 1`, `base + 3s + 2`, in the order
`KMul.Frame` declares them. `count` multiplications occupy exactly
`[base, base + 3·count)`, so a caller places the whole allocator with one
number.

**Why this precedes completeness rather than following it.**
`KHorner.hornerRows_sound` assumes nothing about frames — each step's
`outLow_sound` reads only its own rows, so overlapping frames over-constrain
without breaking soundness. Honest completeness is the opposite: building a
witness means *writing* a value to every frame column, and two steps sharing a
column would have to hold two different products. Ownership and conservation
need the same fact.

**The slot bound is load-bearing, not decoration.** `frameColumn base 0 3` and
`frameColumn base 1 0` are the same column. Without `slot < 3` the disjointness
theorem is false, which is why it is a hypothesis rather than an omitted
detail.

Written as a mapped range rather than a `flatMap` over steps, which is what
makes `Nodup` provable using the local `nodup_map` — `List.Nodup.map` is
unavailable without Mathlib.

---

## KMUL-HONEST-COMPLETENESS

```text
claim:
  An honest execution satisfies the K-multiplication rows.
status: CLOSED 2026-07-26 (cycle 244). `KMulHonest.witness_satisfies`.
```

The witness writes each product to its frame column and leaves every other
column alone. Two premises make it work, and neither is decoration:

- **Freshness** — the operands must not mention the frame. Otherwise writing
  the products would change the operands' own values and the rows could fail.
  A disjoint allocator delivers exactly this, which is why `KFrames` had to
  precede this module.
- **Distinctness** — the witness is an `if`-chain on three columns. If two
  coincided the chain would silently drop one product and write another twice.

`canonical_distinct` discharges the second from `KFrames`, so it is a premise a
real consumer constructs rather than an obligation moved — the §3 test for any
new premise.

**End-to-end check.** On operands `⟨3,5⟩` and `⟨2,7⟩` the witness writes `6`,
`35`, `72` and leaves the operand columns untouched. Karatsuba recovery then
gives `72 − 6 − 35 = 31` and `6 + 7·35 = 251` — exactly the
`K.mul ⟨3,5⟩ ⟨2,7⟩ = ⟨251, 31⟩` that cycle 238 measured directly. Soundness,
the Karatsuba rewrite, the bridge and the witness all agree on one concrete
value.

---

## KMUL-OWNERSHIP-CONSERVATION

```text
claim:
  Exact row ownership and conservation for one K multiplication.
status: CLOSED 2026-07-26 (cycle 245). `KMulOwnership.rows_eq_map_owners`,
        `KMulOwnership.rows_conservation`.
```

**Ownership is positional.** The emitted program *is* the receipt list's image,
so position `i` is emitted by receipt `i` and no other. That is stronger than
proving the three row values pairwise distinct — the shape the Poseidon2 track
settled on after `POSEIDON2-ROW-OWNERSHIP-UNIQUENESS` showed value-distinctness
does not establish exactly-one-owner.

**Conservation names the operands.** A `K` multiplication reads four operand
combinations and writes three frame columns; `rows_conservation` proves nothing
else is reachable. The `cross` row is the case needing care: its operands are
`sumComb` concatenations, so a column it mentions comes from either coordinate
of that side.

With this, the multiplication atom satisfies six of §2's ten items —
constructive program, derived count, row ownership, column ownership,
conservation, soundness, honest completeness. It still lacks a `Typed.Cost`
receipt and its own spec/ledger pairing as a *recipe*, which only make sense
once it is a call rather than a gadget.

---

## KHORNER-COMPLETENESS

```text
claim:
  An honest execution satisfies the Horner evaluation rows.
status: CLOSED 2026-07-26 (cycle 248). `KHornerHonest.hornerWitness_satisfies`.
```

Proved this cycle, in `KHornerSupport`:

- `satisfies_extend` — satisfaction survives an extension that misses every
  column the rows reference. The reusable core of any inside-out witness.
- `hornerCarried_mentions` — a carried Horner value reaches only the
  coefficients and the frames at or after its own step. Applied to the suffix
  at `step + 1`, this gives frames strictly later than `step`, which is what
  discharges freshness at each extension under the canonical allocator's
  ordering.

**What is not proved.** The induction itself: building the witness inside out,
extending at each step, and showing the whole program holds. These two lemmas
are its ingredients and nothing more.

They are deliberately in their own module. This project has five recorded
instances of ingredients being reported as an assembled result — conservation,
row ownership, sponge conservation, the absorption cost claim, and the
`SpongeLayout` predicate — and a module boundary is the cheapest way to keep
the distinction visible rather than relying on prose.

### Third ingredient (cycle 247)

`KHornerHonest` derives freshness of every allocated frame from a single
placement hypothesis: **every operand column is below `base`**.

Stating it over the concrete `KFrames.frameAt base` rather than an abstract
`frames : Nat → Frame` turns frame distinctness, pairwise frame disjointness
and operand freshness from premises into theorems. One checkable hypothesis
replaces a bundle — and fewer premises means fewer places for §3's "obligation
moved rather than closed" to hide.

`suffix_fresh` is the load-bearing one: the carried suffix's columns are either
coefficients, which sit below the base, or frames strictly later than the
current step, which the consecutive layout separates. Both cases land outside
the enclosing frame.

**Still not written:** the inside-out witness and the induction that assembles
`satisfies_extend`, `hornerCarried_mentions` and `suffix_fresh` into honest
completeness. All three ingredients now exist; none of them is the result.

### Assembled (cycle 248)

`hornerWitness` builds the witness inside out — deepest frame first, each
enclosing step extending it — and `hornerWitness_satisfies` proves it satisfies
every emitted row, from one hypothesis: **every operand column is below
`base`**.

A fourth ingredient was needed and had not been anticipated:
`KHornerSupport.hornerRows_mentions`, the row analogue of
`hornerCarried_mentions`. Without it there is no way to show the already-
satisfied inner rows survive an enclosing extension, because nothing bounds
which columns those rows reference.

With this the evaluation program has soundness, a derived count, and honest
completeness. Ownership and conservation remain.

---

## KHORNER-OWNERSHIP

```text
claim:
  Exact row ownership for a Horner evaluation.
status: CLOSED 2026-07-26 (cycle 249).
        `KHornerOwnership.hornerRows_eq_map_receipts`.
```

The emitted evaluation is its receipt list's image, so with `receipts_length`
position `i` is emitted by receipt `i` and by no other.

**The receipt carries an offset, not just a slot.** For a single multiplication
the receipt is a slot. Here it must also say *which step*, because the row a
step emits depends on the value carried by the steps after it — and that
depends on how much of the coefficient list remains. `receiptRow` reconstructs
the suffix with `List.drop`; the lockstep between `hornerRows` decrementing the
list and incrementing the step is what makes the equality provable.

`receipts` is defined by that same recursion rather than as a `range` flatMap.
The flatMap form is the natural phrasing and forces range arithmetic through
the whole induction; the recursive form aligns structurally and the proof
collapses to one arithmetic goal.

**Conservation is not restated here.** It is already
`KHornerSupport.hornerRows_mentions` — every column of every emitted row is a
`beta` column, a coefficient column, or a frame at or after this step. Giving
it a second name in this module would have been a rename presented as a result.

With this the evaluation program has soundness, a derived count, honest
completeness, row ownership and conservation.

---

## KIDENTITY-PROJECTION-CHECK

```text
claim:
  One projection identity, checked at a challenge, as emitted rows.
status: CLOSED 2026-07-26 (cycle 251). `KIdentity.identityRows_sound`,
        `KIdentity.identityRows_is_projection_eval`.
```

The first composite: two Horner evaluations over disjoint frame blocks plus one
`K`-equality.

```
3·(|lhs| − 1) + 3·(|rhs| − 1) + 2
```

For a degree-`d` identity on both sides that is **6d + 2** rows. Checked at
both ends: degree 3 emits 20, degree 0 emits 2 — the base case emits only the
equality, no multiplication.

`identityRows_is_projection_eval` composes with `KBridge.toPair_eval`, so what
the rows force is agreement of the frozen `ProjectionCheck.eval`, not of a
private reimplementation.

### What this deliberately does not prove

**Not `Identity.Exact`.** Agreement at one challenge is what rows can check;
coefficient equality is what the protocol wants. The gap is exactly the
`BadRoot` event, shown non-vacuous at production operations by
`NifsRecipeShape.badRoot_at_production_ops`. That gap is intrinsic to the
projection optimization, not an artifact of this encoding — which is why the
`nifsVerify` recipe will need a named event rather than an unconditional
soundness statement.

**Nothing about where `beta` came from.** A checked identity is not a sound one
until the challenge is bound to a verifier transcript. That is a separate
obligation and remains open.

---

## KBATCH-PROJECTION-BATCH

```text
claim:
  A batch of projection identities as one emitted row program, with cost a
  fold over per-identity receipts.
status: CLOSED 2026-07-26 (cycle 252). `KBatch.batchRows_length`,
        `KBatch.batchRows_sound`, `KBatch.batch_exact_or_badRoot`.
```

**Cost is a fold, not a formula.** The PiRLC batch checks one identity per
native output component and those need not share a degree, so a uniform-degree
formula would be a subtotal presented as a total. Checked with a mixed batch: a
degree-3 identity (20 rows) and a degree-1 identity (8 rows) emit 28, and the
fold gives 28. A uniform formula would have produced 40 or 16.

**The frozen disjunction is now reached from emitted rows.**
`batch_exact_or_badRoot` composes the row program's agreement with the frozen
`batchAccepted_implies_exact_or_badRoot`. Cycle 235 withdrew the plan to reach
that statement by wrapping artifact-owned rows; this reaches it from a
Lean-owned program instead.

**The split that keeps the row program honest.** `BatchAccepted` needs
well-formedness *and* agreement. Well-formedness is a property of the
coefficient data that rows cannot enforce; agreement is what the rows deliver.
`batchAccepted_of_rows` takes them separately rather than letting the row
program appear to establish both.

---

## KBATCH-TRACE-FUNCTION

```text
claim:
  The map from decoded NIFS inputs to projection identity specs.
status: OPEN 2026-07-26 (cycle 253), and scoped. The identity set is NOT mine
        to choose — it must express operations already frozen elsewhere.
```

The batch checks whatever identities it is handed. What turns it from a generic
checker into a verifier fragment is the *trace function*: the map from decoded
authoritative NIFS inputs to the list of identity specs. That map is also what
the occurrence-indexed event must bind to, so it gates
`FPRIME-NIFSVERIFY-SOUNDNESS-SHAPE`.

### Why this is not a free choice

Π_RLC's `Equations` are three combine-equalities plus four structural
conditions. The combine operations are **abstract** in
`SuperNeo/Folding/PiRLC.lean` — `Algebra` constrains them only by
`commit_hom`, `publicInput_hom`, `evaluations_hom` and `norm_growth`, not by
any formula.

But a concrete instance is already frozen:
`SuperNeo/Concrete/Phi81Relation/PiRLCAlgebra/Algebra.lean`, whose
`combineEvaluations` is `PiRLCFinite.combineEvaluations`, with the
evaluation-homomorphism content in
`SuperNeo/Folding/PiCCS/OutputClaims/EvaluationHomomorphism/`. Its own header
states every algebra operation is executable and every law is a theorem over
that operation.

So the trace function must express **those** operations. Writing the obvious
random-linear-combination instead — `Σ challengeⁱ · inputᵢ` — would be
inventing a verifier branch, which §7 makes a stop rather than a choice, and
would silently diverge from the frozen algebra even where it looked right.

### Scale

44 modules under `Concrete/Phi81Relation`, 14 under the
evaluation-homomorphism subsystem. Reading enough of them to state the trace
faithfully is the next substantial obligation, and it is reading before writing
— exactly the step cycle 231 skipped when it assumed `RecursiveRows` was
Lean-owned.

### What the frozen algebra actually says (cycle 254)

Read rather than assumed. Four facts, and they change the picture.

**1. `RingK = Fin ringDegree → K`, `RingF = Fin ringDegree → F`.** The semantic
combine operates on cyclotomic *ring* elements — vectors of `ringDegree`
coordinates — not on scalars.

**2. `BatchIdentity` produces `List (ProjectionCheck.Identity K)`** — over plain
`K`, the Goldilocks quadratic extension. So the projection layer is a scalar
layer sitting under a ring layer.

**3. Projection is exactly polynomial evaluation.** A `RingK` element is a
coefficient vector; projecting it to `K` means evaluating that vector at the
challenge. `KHorner` is therefore not an arbitrary encoding choice — it *is*
the projection map, and the `3d` row cost is the cost of projecting one ring
element.

This retroactively justifies the whole `KMul`/`KHorner`/`KIdentity`/`KBatch`
tower's targeting: it is over `K`, which is what `BatchIdentity` requires.

**4. The challenges are independent per input, not powers of one challenge.**
`combineEvaluation` recurses head-first over `Fin count -> RingF`, computing
`Σᵢ embedChallenge(cᵢ) · xᵢ` with a *separate* `cᵢ` per input. Anyone encoding
this as `Σᵢ βⁱ · xᵢ` would produce a different verifier.

### What the trace must therefore do

Per matrix (`shape.matrixCount` of them) and per combine-equation, emit one
projection identity whose two sides are the coefficient vectors of the
`RingK`-level output and of `Σᵢ embedChallenge(cᵢ) · xᵢ`. `KBatch` then checks
them all, and `batch_exact_or_badRoot` delivers the frozen disjunction.

Ring multiplication is convolution, so the identity's coefficient lists are not
the operands directly — deriving them is the remaining work, and it is where a
degree bound and `maxDegree` must be pinned.

### Three algebras, and an existing owner (cycle 255)

**Three `K`-like algebras are in play, and two of them are distinct Lean types
with identical definitions:**

| algebra | layer | bridged? |
|---|---|---|
| `KHorner.Pair` | canonical row layer, `Nat` coordinates | yes — `KBridge` |
| `ProjectionProgram.K` | R1CS projection, `Fin goldilocksP` | — |
| `Concrete.K` | semantic / paper, `Fin goldilocksModulus` | **no** |

`goldilocksModulus` and `goldilocksP` are both `18446744069414584321`, so the
last two are isomorphic but not the same type. The frozen combine lives over
`Concrete.K`; `BatchIdentity` lives over `ProjectionProgram.K`. A bridge
between them is required and does not yet exist in this track.

**The trace's semantic target already has an owner.**
`Correspondence/FPrimeFullHistory/NifsPaper/PiRlc.lean` states it owns "the
Phi81 public combination, typed attempt construction, and the implication from
exact leaf equations to `PiRLC.Equations`" — which is precisely what the trace
function must establish.

**And it imports `FPrimeFullHistoryProjectionArtifact`.** So the cycle-235 rule
applies again, in the same shape: its *semantic* content is reusable as
evidence and as a specification, its *row layout* is not. Before building a
third bridge, the artifact-free part of that module must be separated from the
artifact-dependent part — the same provenance check that cycle 231 skipped and
cycle 253 reinstated.

Its own header also disclaims costs and row removal, which is consistent: it is
a refinement argument, not a row program.

---

## KRINGPROJECTION-COST

> **PARTLY WITHDRAWN 2026-07-27 (cycle 291)** by
> `KRINGPROJECTION-ROOT-IMPOSSIBLE`. `projectionRows_length` (159 rows for one
> Horner evaluation) stands. The combine-equation figures — `combineEquationCost`
> and the 803-row total — are deleted: a declared formula rather than an emitted
> program, and a subtotal that omitted the quotient projection and the
> evaluation of `Phi81`. No replacement number is recorded.

```text
claim:
  The row cost of projecting a cyclotomic ring element, and of one projected
  combine equation.
status: CLOSED 2026-07-26 (cycle 256). `KRingProjection.projectionRows_length`,
        `KRingProjection.combineEquationCost_two`.
```

**Projecting one `RingK` element costs 159 rows** — 54 coefficients, 53
multiplications, three rows each. Verified by emitting the program on a
54-element vector.

**No convolution appears anywhere.** `RingK` multiplication is convolution
reduced modulo `Φ₈₁`, which encoded directly would be quadratic per product. It
is not needed: projection is evaluation at the challenge, and evaluation at a
*root of the modulus* is a ring homomorphism, so a ring product becomes one
`KMul` — three rows — once both operands are projected.

That is why the `3d` Horner evaluation is load-bearing rather than convenient,
and it is the whole content of the projection optimization.

**One projected combine equation costs `321·count + 161` rows**: one output
projection, `2·count` operand projections, `count` scalar products, one
`K`-equality. At the production arity of two inputs that is **803**.

### The premise, carried not assumed

The homomorphism holds only when `Φ₈₁(beta) = 0`. At a non-root the reduction
step is invisible to the evaluation and a prover could exploit the difference,
so this is a real condition rather than a formality.

`RootOfModulus` carries it as a hypothesis. Discharging it belongs to challenge
binding — the challenge must come from the verifier's sampler over the strong
set. Nothing in this module proves it, and the module says so.

`modulusCoefficients` was checked to be `Φ₈₁` and not merely plausible: length
55 with ones at exactly positions `[0, 27, 54]`.

---

## KRINGPROJECTION-HOMOMORPHISM

> **WITHDRAWN 2026-07-27 (cycle 291)** by `KRINGPROJECTION-ROOT-IMPOSSIBLE`.
> Obligation 1 (`polyEval_polyMul`) and obligation 3
> (`rawMulCoeffK_eq_coeffAt_polyMul`) stand and are load-bearing on the
> replacement route. Obligation 2 is unreachable: no challenge in `K` is a root
> of `Phi81`. The projection is *not* a ring homomorphism here, and the quotient
> term is carried rather than deleted — which is what production always did.

```text
claim:
  Projecting a cyclotomic ring product at a root of the modulus gives the
  product of the projections.
status: OPEN 2026-07-26 (cycle 257). Vocabulary built; no part of the theorem
        is proved.
```

This is the theorem `KRINGPROJECTION-COST` relies on structurally. Without it,
"a ring product costs one `KMul`" is a plan rather than a fact, and the 803-row
combine-equation figure is the cost of a program that has not been shown to
compute the right thing.

### Three obligations, none discharged

1. **Evaluation is multiplicative on convolution.** One induction step given
   three ring laws for `mulPair`/`addPair` — distributivity, associativity, the
   zero law. None is written, and none is free: both operations reduce modulo
   the prime, so each needs the modular plumbing `KMul.karatsuba_identity`
   needed.
2. **Reduction is invisible at a root.** Subtracting a multiple of `Φ₈₁` leaves
   the evaluation unchanged when `Φ₈₁(beta) = 0`.
3. **`polyMul` agrees with the frozen `rawMulCoeffK`.** Until this, anything
   proved is about *this* multiplication, not the protocol's. The two are the
   same mathematics; sameness of mathematics is not sameness of definition.

### What was built

`KPolyHom` supplies the vocabulary: `polyAdd`, `polyScale`, `polyMul`, and
`polyEval` — uniform Horner with no special case. The list representation makes
`(a₀ :: a') · b = a₀·b + X·(a'·b)` the definition, so obligation 1 becomes a
one-step induction instead of index arithmetic over `rawMulCoeffK`'s fold.

`polyEval_singleton` and `hornerValue_singleton` record exactly where the
algebra and the row program diverge: `hornerValue` skips a multiply-by-zero,
which is right for rows and wrong for induction.

Checked that `polyMul` convolves rather than multiplying coefficientwise:
`(1+X)² ` evaluates at `X = 3` to `16`, not `4`.

---

## KPAIR-RING-LAWS

```text
claim:
  Commutativity, the zero law, and distributivity for Pair arithmetic.
status: CLOSED 2026-07-26 (cycle 258) for those three.
        KPAIR-ASSOCIATIVITY remains OPEN.
```

`mulPair` and `addPair` reduce modulo the prime at every step, so each law is a
congruence, not a syntactic identity. The pattern: strip inner reductions with
`mul_congr`/`add_congr`, expand with `Nat.add_mul`/`Nat.mul_add`, generalize
each monomial so nothing nonlinear remains, close with `omega`. `ring` would do
it in one line and is unavailable.

Distributivity is the load-bearing one — `polyMul`'s head-splitting identity
becomes `polyEval`'s multiplicativity only through it.

## KPAIR-ASSOCIATIVITY

```text
claim:
  mulPair is associative modulo the prime.
status: CLOSED 2026-07-26 (cycle 259). `KPairLaws.mulPair_assoc`.
```

**Closed by normalizing before comparing.** The obstacle was real: `generalize`
then `omega` compares `x.low * y.low * z.low` against
`x.low * (y.low * z.low)`, so generalizing one side leaves the other
ungeneralized. Adding `Nat.mul_assoc`, `Nat.mul_left_comm` and `Nat.mul_comm`
lets `simp` put both sides in an AC normal form first; `omega` then closes the
linear residue.

`Nat.mul_comm` was the missing one — with only `mul_left_comm` the ordered
rewriting has no way to fix operand order, and the goal survives.

Checked where it bites: `(X·X)·X` and `X·(X·X)` both give `7X`, since `X² = 7`.
A grouping that mishandled the extension rule would differ exactly there.

The `generalize`-then-`omega` pattern does **not** close it. Associativity
compares `x.low * y.low * z.low` against `x.low * (y.low * z.low)`; these are
distinct terms, so generalizing one side's monomials leaves the other side
ungeneralized and `omega` sees products of variables.

It needs explicit reassociation before generalizing, or the eight triple
products handled uniformly. The arithmetic is true — checked by hand on both
coordinates, where the `X² = 7` terms line up — but a hand check is not a proof
and it is not recorded as one.

---

## KPOLYEVAL-DISTRIBUTION

```text
claim:
  Evaluation distributes over polynomial addition and commutes with scaling.
status: CLOSED 2026-07-26 (cycle 260). `KPolyEval.polyEval_polyAdd`,
        `KPolyEval.polyEval_polyScale`.
```

The two structural facts under the multiplicativity induction, assembled from
the four ring laws rather than reproved from modular arithmetic.

**Canonicity is load-bearing, not bookkeeping.** `polyAdd`'s base cases return
the longer list unchanged, so `polyEval (polyAdd [] q) = polyEval q` — but the
statement demands `addPair ⟨0,0⟩ (polyEval q)`, and `addPair` *reduces*. The
two agree only because `polyEval` always returns residues. Without
`polyEval_canonical` those base cases are **false as stated**, not merely hard.

Checked on the case that exercises it: `(1 + 2X) + (3 + 4X + 5X²)` at `X = 10`
gives 564 whether added before or after evaluating — different lengths, so the
tail-keeping branch is the one taken.

**Module ordering.** `KPairLaws` imports `KPolyHom`, so the ring laws cannot be
used inside `KPolyHom`. Anything combining vocabulary with laws sits above both,
which is why `KPolyEval` exists as a third module. Found by trying the other
order and getting an unknown namespace.

---

## KPOLYEVAL-MULTIPLICATIVITY

```text
claim:
  Evaluation is multiplicative on polynomial convolution.
status: CLOSED 2026-07-26 (cycle 261). `KPolyEval.polyEval_polyMul`.
        First of KRINGPROJECTION-HOMOMORPHISM's three obligations.
```

Cycle 257 predicted this would be one induction step given the four ring laws.
It was, and the proof names each law it uses: `polyEval_polyAdd` splits the
head-splitting definition, `polyEval_polyScale` handles the scaled head,
`addPair_zero_left_canonical` discharges the shift's leading zero,
`mulPair_addPair_distrib_right` expands the right side, and `mulPair_assoc` is
what finally makes the two sides identical.

**Checked where the extension bites.** At a scalar point both sides give 16. At
the point `⟨0,1⟩` — where `X² = 7` is exercised — both give `⟨8, 2⟩`: the
convolution route evaluates `1 + 2X + X²`, the product route squares `⟨1,1⟩`.
A scalar-only check would not have distinguished a `mulPair` that dropped the
`7`.

### Two obligations remain before the projection homomorphism

- **Reduction is invisible at a root.** Subtracting a multiple of `Φ₈₁` leaves
  the evaluation unchanged when `Φ₈₁(beta) = 0`.
- **`polyMul` agrees with the frozen `rawMulCoeffK`.** Until then this is a
  theorem about *this* multiplication, not the protocol's.

---

## KPOLYEVAL-ROOT-REDUCTION

> **WITHDRAWN 2026-07-27 (cycle 291)** by `KRINGPROJECTION-ROOT-IMPOSSIBLE`.
> The lemma was true in general — the `X² − 4` fixture below is a real root —
> but `Phi81` has no root in `K`, so on the only modulus this track uses it was
> vacuous. Theorem and guard deleted. Replaced by
> `KPolyEval.polyEval_quotientForm`, which keeps the quotient term and needs no
> hypothesis on the point.

```text
claim:
  Adding a multiple of the modulus leaves the evaluation unchanged at a root.
status: CLOSED 2026-07-26 (cycle 262).
        `KPolyEval.polyEval_add_multiple_of_root`.
        Second of KRINGPROJECTION-HOMOMORPHISM's three obligations.
```

Immediate once multiplicativity is available: a multiple of the modulus
evaluates to zero at a root, so adding one changes nothing. This is what lets a
*reduced* cyclotomic product be checked by the *unreduced* convolution's
evaluation — the step the 803-row combine-equation cost depends on.

Checked with a genuine root rather than a trivial one: `X² − 4` has root 2;
base `5 + 3X` evaluates to 11; base plus `(7 + 9X)·(X² − 4)` still evaluates to
11. A zero quotient would have passed vacuously.

## KROOT-HYPOTHESIS-FORM

> **WITHDRAWN 2026-07-27 (cycle 291)** by `KRINGPROJECTION-ROOT-IMPOSSIBLE`.
> It proved two *unsatisfiable* conditions equivalent. `hornerValue_eq_polyEval`
> and `modulusCoefficients_canonical`, which it was built from, are kept — they
> reconcile the row-level and polynomial-level evaluators and the quotient route
> needs them.

```text
claim:
  KRingProjection.RootOfModulus and the polyEval root hypothesis are the same
  condition.
status: CLOSED 2026-07-26 (cycle 263).
        `KPolyEval.polyEval_root_of_RootOfModulus`.
```

`hornerValue_eq_polyEval` proves the two evaluators agree whenever the
coefficients are residues, and `modulusCoefficients_canonical` supplies that for
the modulus vector, which holds only zeros and ones. A caller supplying
`RootOfModulus` in the `hornerValue` form now gets the `polyEval` form for
free — no parallel unused hypothesis.

**The canonicity condition is load-bearing, and was checked.** On the
non-canonical singleton `⟨p+1, 0⟩` the two evaluators give
`18446744069414584322` and `1` respectively — they genuinely differ. On a
canonical singleton both give the coefficient. Without the hypothesis the
theorem is false, not merely unprovable.

`RootOfModulus` is stated over `hornerValue`; the reduction lemma needs the
`polyEval` form. The two differ on the final coefficient, because `hornerValue`
skips a multiply-by-zero that `polyEval` performs.

Until reconciled, a caller must supply the `polyEval` form directly and
`RootOfModulus` is unused by the theorem that needs it. That is a premise-shape
mismatch, not a gap in the mathematics — but an unused hypothesis structure is
exactly the shape §3 warns about, so it is named rather than left to be noticed.

---

## KCONCRETE-BRIDGE

```text
claim:
  SuperNeo.Concrete.K arithmetic is KHorner.Pair arithmetic.
status: CLOSED 2026-07-26 (cycle 264). `KConcreteBridge.ofConcrete_mul`.
```

The third of the three `K`-like algebras cycle 255 mapped is now connected.
`KBridge` joined the row layer to the projection layer; this joins the semantic
layer, where the frozen combine lives.

**`goldilocksModulus` and `goldilocksP` are separate `def`s** holding the same
literal, so `moduli_eq` has to be stated explicitly wherever the layers meet.
The types are isomorphic and not identical, and a conversion has to be written
rather than assumed — which is the whole reason the module exists.

**One parenthesization difference.** `Concrete.K.mul` writes `7 * a.c1 * b.c1`
left-associated; `mulPair` writes `7 * (a.c1 * b.c1)`. Same value, different
term, so the proof reassociates rather than closing by `rfl`. Exactly the kind
of difference that looks like nothing and blocks a `simp`.

**Cross-layer check.** `Concrete.K.mul ⟨3,5⟩ ⟨2,7⟩ = (251, 31)` — the same
value `ProjectionProgram.K.mul` gave in cycle 238 and the Karatsuba honest
witness produced in cycle 244. Four independent routes across three algebras
agree on one concrete value.

---

## KPOLYMUL-RAWMULCOEFF-AGREEMENT

```text
claim:
  polyMul's coefficients are the frozen rawMulCoeffK's.
status: CLOSED 2026-07-27 (cycle 275).
        `KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul`.
```

**Closed.** `polyMul`'s coefficients are the frozen `rawMulCoeffK`'s, so every
theorem in the `K` tower is a statement about the protocol's multiplication
rather than about a definition of my own.

The composition is a chain of existing theorems with no new mathematics:

```text
rawMulCoeffK left right degree
  ofConcrete_foldl          → a Pair-fold from ⟨0,0⟩
  foldl_step_from_zero      → sumOver over List.range ringDegree
  sumOver_range_truncate    → sumOver over List.range (degree+1)
  sumOver_congr_guard       → the guard is vacuous there
  convolution_eq_sumOver    → convolution (toList l) (toList r) degree
  coeffAt_polyMul           → coeffAt (polyMul …) degree
```

Two guard facts do the work at the joins: past the degree `index ≤ degree`
fails, so the tail is silent; below it, `degree − index < ringDegree` follows
from `degree < ringDegree`, so the guard is vacuous.

Checked on `(1+X)²` over the frozen `RingK`: coefficient 1 is **2** by both
routes, coefficients 0 and 2 are 1. The middle coefficient is the one a wrong
convolution gets wrong.


**`coeffAt_polyMul` proves `polyMul`'s coefficients are the convolution.** The
proof is structural because `convolution` was defined by the same recursion
`polyMul` uses — head times the whole right side, plus a shift — rather than as
a `foldl` over a range.

Canonicity had to be threaded through: `coeffAt_polyAdd` needs both sides'
coefficients to be residues, so `canonical_polyScale`, `canonical_polyAdd`,
`canonical_cons_zero` and `canonical_polyMul` establish that every constructor
preserves it.

Checked on the coefficient that discriminates: `(1+X)² ` gives `(1, 2, 1)` by
both routes. The middle `2` is the only one a coefficientwise product would get
wrong.

**Still open:** matching this against `rawMulCoeffK`, which folds over
`List.range ringDegree` with the guard `i ≤ degree ∧ degree − i < ringDegree`.
That is a different shape of sum over a different index set, and relating them
is the remaining step.

The last of `KRINGPROJECTION-HOMOMORPHISM`'s three obligations, and the one that
makes cycles 258–264 statements about the protocol's multiplication rather than
about definitions of my own.

**Built this cycle:** `coeffAt`, the lengths of `polyAdd`, `polyScale` and the
coefficient of a scaling.

**Not built:** the coefficient of a *product* — the convolution
characterization `coeffAt (polyMul a b) k = Σᵢ aᵢ · b_{k−i}`. That induction is
the remaining content.

### Why lengths came first

`rawMulCoeffK` folds over `List.range ringDegree` guarding each term with
`i ≤ degree ∧ degree − i < ringDegree`. Matching that against a list needs to
know exactly where `coeffAt` starts returning the default.

`polyAdd` keeps the longer list, so its length is a **maximum, not a sum** —
easy to state wrongly, and the reason `polyMul`'s length is not
`a.length + b.length`. Checked: adding a 2- and a 3-coefficient polynomial gives
3, not 5.

### One statement strengthened before landing

`coeffAt_polyScale` was first written as a disjunction with an out-of-range
case. That case is not an exception: `coeffAt` returns zero there and
`mulPair scalar ⟨0,0⟩ = ⟨0,0⟩`, so the same equation holds. A disjunction whose
second arm is implied by the first is weaker for no reason, and it was replaced
by the unconditional form.

### The frozen ring as a list (cycle 267)

`toList` converts `RingK = Fin ringDegree → K` into `List Pair`, and
`coeffAt_toList` proves it respects coefficients **at every index**, in range
and out.

That totality is the useful part: `ringKCoeff` returns `K.zero` beyond the
degree and `coeffAt` returns `⟨0,0⟩`, and `ofConcrete K.zero = ⟨0,0⟩`, so the
two defaults agree and no case split reaches the caller. The index-set argument
against `rawMulCoeffK`'s guarded fold can therefore ignore the boundary
entirely.

Checked with a genuinely varying element — `1` at index 27, zero elsewhere:
54 coefficients, index 27 gives 1, indices 0 and 53 give 0, and index 54 gives
0. A constant element would have made the conversion look correct regardless.

---

## KFOLDSUM-ACCUMULATOR

```text
claim:
  A guarded foldl from init equals init plus the same fold from zero.
status: CLOSED 2026-07-26 (cycle 268).
        `KFoldSum.foldl_step_accumulator`, `KFoldSum.foldl_step_from_zero`.
```

The crux of relating `rawMulCoeffK` to `convolution`. The frozen definition
accumulates left-to-right over `List.range ringDegree`, guarding each term;
`convolution` nests to the right, head first. They are the same sum in
different association, and every direct attempt runs into the accumulator —
`foldl` threads a partial result the recursion does not have.

Peeling it out needs `addPair`'s **associativity**, not just commutativity,
because the accumulator sits on the left of every step. That is what cycle
259's associativity proof is for; without it this lemma has no route.

**Canonicity is load-bearing for the fourth time in this tower.** The zero-start
form equals the accumulated form only up to `addPair`'s reduction, so `init`
must be a residue.

Checked that both halves do real work: guard `i < 3` over `range 6` sums to 3,
not 15 — so the guard filters — and starting from `init = 10` gives 13, so the
accumulator is genuinely additive rather than discarded.

### Translating the frozen definition (cycle 269)

`rawMulCoeffK`'s guard and term are stated in the semantic vocabulary. Two
lemmas restate them in the polynomial vocabulary, which is what lets the
accumulator lemma apply to it at all.

**`guard_collapses`** — `degree − index < ringDegree` follows from
`index ≤ degree` once `degree < ringDegree`, so the frozen two-part guard is
just `index ≤ degree` on the range that matters.

The hypothesis is load-bearing, and was checked: at degree 60 the two-part
guard is *false* while `index ≤ degree` is *true*, so above the degree the
second conjunct does real work. That is why the frozen definition carries it.

**`term_is_polynomial`** — one `ofConcrete_mul` and two `coeffAt_toList`s.

Neither is the agreement. After both, the remaining step is showing the guarded
fold over `List.range ringDegree` visits exactly the terms `convolution` does.

### Splitting the range (cycle 270)

`rawMulCoeffK` folds over all of `List.range ringDegree`, but its guard fires
only below the degree. `sumOver_append` splits the sum over concatenation and
`sumOver_of_no_guard` shows an inert stretch contributes zero, so the tail can
be discarded — which makes the fold's length independent of `ringDegree` and
comparable to a convolution that stops at the degree.

Checked: with guard `i < 3` and terms `i`, range 6 and range 3 both sum to 3
while the tail `[3,4,5]` sums to 0; and `[0,1] ++ [2,3]` gives 6 = 1 + 5.

What remains of `KPOLYMUL-RAWMULCOEFF-AGREEMENT` is the index-visiting
induction proper: that the retained head of the range visits exactly
`convolution`'s terms, in the same association. The shape difference to close is
a trailing `addPair … ⟨0,0⟩` that `sumOver` produces and `convolution`'s base
case does not.

### The trailing zero, isolated (cycle 271)

`convolution_add_zero` closes the exact shape difference cycle 270 recorded:
`sumOver` bottoms out at `⟨0,0⟩` and adds it, while `convolution` bottoms out at
a bare product. They agree because every `convolution` branch produces a
residue — `convolution_canonical`.

The condition is necessary, not bookkeeping. Absorption fails for a
non-canonical value: `addPair ⟨p+1, 0⟩ ⟨0,0⟩` gives `1`, not `p+1`. And
`convolution` really is canonical on inputs where reduction matters:
`[p−1, p−1]` squared at index 1 gives 2, since `(p−1)² ≡ 1`.

With this, every stated obstacle to `KPOLYMUL-RAWMULCOEFF-AGREEMENT` is
discharged. What remains is the index re-association itself: the tail of
`List.range (k+2)` carries the shifted terms `convolution` visits on the
tail list. That is the `List.range_succ_eq_map` manipulation that failed in
cycle 249 and needed a recursive redefinition instead — the same treatment is
likely to apply.

### Re-indexing, and a failed attempt (cycle 272)

`sumOver_map` proves sums commute with re-indexing: mapping the index list is
the same as composing guard and term with the map. This is what lets
`List.range_succ_eq_map`'s shifted tail be matched against a recursion that
drops a list head. Checked: `[0,1,2]` mapped by `(+10)` sums to 33 either way.

**The re-association itself was attempted and withdrawn.** With `sumOver_map`
in hand the step case lines up, but the two base cases do not fall out: the
empty-left case needs an induction of its own over the degree, and the
zero-degree case has to reconcile `[0]` against a bare product. The attempt was
removed rather than left partial.

The obstacle is now narrower than cycle 271 recorded — it is the *base cases*,
not the step — and that is the useful part of the failure.

### The shape reconciliation closes (cycle 273)

`convolution_eq_sumOver` proves the convolution is the sum over its own index
range. Cycle 272 narrowed the obstacle from "the induction" to "its two base
cases", and both fell:

- **empty left list** — every term is a product with a missing coefficient,
  hence zero, so `sumOver_of_zero_terms` applies whatever the guard does;
- **zero degree** — a singleton range against a bare product, closed by the
  canonicity absorption from cycle 271.

The step case needed `sumOver_map` from cycle 272 plus one rewrite-ordering
fix: `List.range_succ_eq_map` must target the goal's range explicitly, or it
rewrites the induction hypothesis's range instead.

Checked at both the middle coefficient of `(1+X)²`, where both forms give 2,
and at a degree past the left list, where both give 0 — the case that exercises
the empty-tail branch rather than the generic one.

**What remains of `KPOLYMUL-RAWMULCOEFF-AGREEMENT`** is now only assembly:
`foldl_step_from_zero` turns the frozen fold into `sumOver`, `guard_collapses`
and `term_is_polynomial` restate its guard and term, `sumOver_append` and
`sumOver_of_no_guard` discard the inert tail, and this closes the shape. Every
piece exists; none has been composed.

### The fold transfers (cycle 274)

`ofConcrete_foldl` carries the frozen `K`-accumulation to the `Pair`-accumulation
of the transferred terms, with the accumulator generalized since it changes at
every step. This was the last *algebra* gap: `rawMulCoeffK` accumulates with
`K.add`/`K.mul` while everything above accumulates in `Pair`.

With it, the composition is a chain of existing lemmas:

```text
rawMulCoeffK a b k
  →  ofConcrete_foldl        a Pair-fold from ofConcrete K.zero = ⟨0,0⟩
  →  foldl_step_from_zero    sumOver over List.range ringDegree
  →  split at k+1            sumOver_append, sumOver_of_no_guard
  →  guard_collapses         the guard is vacuous below k+1
  →  convolution_eq_sumOver  convolution (toList a) (toList b) k
  →  coeffAt_polyMul         coeffAt (polyMul …) k
```

**The one piece still missing** is the range split itself:
`List.range ringDegree = (List.range ringDegree).take (k+1) ++ drop (k+1)` is
`List.take_append_drop`, but relating `(List.range n).take m` to
`List.range m` is not yet stated. That is the whole remaining content of
`KPOLYMUL-RAWMULCOEFF-AGREEMENT`.

---

## KRINGMUL-REDUCTION-QUOTIENT

```text
claim:
  ringKMul's coefficient formula is the mod-Φ₈₁ reduction, expressible as
  raw convolution plus a multiple of the modulus.
status: OPEN 2026-07-27 (cycle 276). Found while assembling the homomorphism.
```

`KRINGPROJECTION-HOMOMORPHISM`'s three obligations are all closed, but they do
not compose directly, and the reason is worth stating precisely.

`polyEval_add_multiple_of_root` is generic: it handles a reduction *expressed
as* `base + quotient · modulus`. `ringKMul` does not have that shape. It is a
coefficient formula:

```lean
folded := if i < 27 then rawMulCoeffK a b (i+54) else rawMulCoeffK a b (i+27)
twice  := if i + 81 ≤ 106 then rawMulCoeffK a b (i+81) else 0
result := rawMulCoeffK a b i - folded + twice
```

Bridging requires **exhibiting the quotient** — the polynomial that, multiplied
by `Φ₈₁`, accounts for exactly this rearrangement.

### The reduction is two-stage, and that is why there are three terms

Verified numerically: a product of two degree-53 elements reaches degree **106**.
One fold by `X⁵⁴ = −X²⁷ − 1` sends degree 106 to 79 and 52 — and **79 is still
above 53**, so a second fold is needed. The `twice` branch fires for exactly the
26 indices with `i + 81 ≤ 106`, which is the second reduction of degrees 81
through 106, exactly as `ringFMul`'s docstring states.

So the quotient is not a single monomial multiple; it has two stages. The
generic lemma applies twice, but each application needs its own quotient
exhibited.

This is the last mathematical gap between the closed obligations and the
homomorphism statement.

### The quotient, derived and validated (cycle 277)

`Φ₈₁ = X⁵⁴ + X²⁷ + 1`, so `X⁵⁴ ≡ −X²⁷ − 1`. Two cases:

| degree | reduces to | because |
|---|---|---|
| `54 ≤ d ≤ 80` | `−X^{d−27} − X^{d−54}` | `X^d + X^{d−27} + X^{d−54} = X^{d−54}·Φ₈₁` |
| `81 ≤ d ≤ 106` | `X^{d−81}` | `X^d − X^{d−81} = X^{d−81}(X²⁷−1)·Φ₈₁` |

The second uses `X⁸¹ − 1 = Φ₈₁·(X²⁷ − 1)`, which is why degrees at and above 81
collapse in one step rather than two: the second fold cancels the first.

Collecting by target index reproduces the code exactly:

```text
r_i = c_i − (c_{i+54} if i ≤ 26) − (c_{i+27} if 27 ≤ i ≤ 53) + (c_{i+81} if i ≤ 25)
```

which is `folded` split at 27 and `twice` guarded by `i + 81 ≤ 106`.

**Validated against the frozen code**, both stages:
`X⁵³ · X = X⁵⁴` gives coefficients 0 and 27 equal to `p−1` and nothing else —
that is `−X²⁷ − 1`. `X⁵³ · X²⁸ = X⁸¹` gives coefficient 0 equal to 1 and
coefficient 27 equal to 0 — that is `X⁸¹ ≡ 1`.

So the quotient is
`q = Σ_{54≤d≤80} c_d X^{d−54} + Σ_{81≤d≤106} c_d X^{d−81}(X²⁷ − 1)`.

What remains is constructing `q` as a `List Pair` and proving the polynomial
identity — mechanical given the above, but not yet written.

### Power vocabulary (cycle 278)

The per-degree reduction argument compares `β^d` against `β^{d−27}` and
`β^{d−54}`, which needs powers and their additivity. `powPair` and
`powPair_add` supply them; `mulPair_one_left` is the identity law the base case
needs, and it is conditional on canonicity like everything else in this tower.

Checked in the extension rather than on scalars: `X² = (7,0)`, `X³ = (0,7)`,
`X⁴ = (49,0)`, and `X⁵ = X²·X³ = (0,49)`. A scalar-only check would not
distinguish a `mulPair` that dropped the `X² = 7` rule.

With this, the two identities the reduction needs are expressible:
`β^{d−54} · Φ₈₁(β) = 0` for `54 ≤ d ≤ 80`, and `β^{d−81} · (β⁸¹ − 1) = 0` for
`d ≥ 81`. Neither is stated yet.

### The first reduction identity (cycle 279)

`reduction_single_fold` proves that for `54 ≤ d ≤ 80` the three powers
`β^d`, `β^{d−27}`, `β^{d−54}` sum to zero at a root, which is what folds those
degrees in one step. It rests on `β^d = β^{d−54}·β⁵⁴` and
`β^{d−27} = β^{d−54}·β²⁷` from `powPair_add`, then factoring the shared
`β^{d−54}` out and applying the root.

`mulPair_add_self` is the factoring step, extracted because rewriting the
trailing summand in place hits every occurrence of `β^{d−54}` rather than only
the one intended.

**The root hypothesis is a real constraint, and that was checked.** At
`base = 1` the expression is 3 and at `base = 0` it is 1 — neither zero. So the
theorem is not vacuous. It is *also* not shown satisfiable here: exhibiting a
`β` with `Φ₈₁(β) = 0` is the challenge-binding obligation, and the strong set
exists precisely because such roots do.

**Still open:** the second identity, `β⁸¹ = 1` for degrees 81 through 106.
Multiplying the root by `β²⁷` gives `β⁸¹ + β⁵⁴ + β²⁷ = 0`, which against the
root `β⁵⁴ + β²⁷ + 1 = 0` yields `β⁸¹ = 1` by additive cancellation — a lemma
this tower does not yet have.

### The second reduction identity (cycle 280)

`powPair_eightyOne` proves `β⁸¹ = 1` at a root, which is why degrees 81 and
above collapse in one step rather than two.

**No additive cancellation was needed.** Cycle 279 recorded that the obvious
route — subtracting the root from `β²⁷` times the root — requires a
cancellation lemma this tower does not have. Adding the root, which is zero, and
regrouping reaches the same place with associativity alone:

```text
β⁸¹ = β⁸¹ + (β⁵⁴ + β²⁷ + 1)   = (β⁸¹ + β⁵⁴ + β²⁷) + 1   = 0 + 1   = 1
```

The recorded obstacle turned out to be avoidable rather than hard, which is a
different outcome from the previous four times a precise record led to a short
next cycle — there the obstacle fell, here it dissolved.

**The expansion was checked independently of the root.**
`β²⁷·(β⁵⁴ + β²⁷ + 1) = β⁸¹ + β⁵⁴ + β²⁷` must hold for *any* base, and it does at
a scalar base and at an extension base with a nonzero high coordinate. That
tests the `powPair_add` and distributivity chain without the hypothesis doing
any work.

Both reduction identities are now proved. What remains for
`KRINGMUL-REDUCTION-QUOTIENT` is applying them per-degree across the 107
coefficients of the raw convolution.

### Two routes, and a planning error (cycle 281)

There are two ways to close `KRINGMUL-REDUCTION-QUOTIENT`, and I built pieces
of both before settling which one to use.

**Route A — the quotient.** Prove the *root-free* polynomial identity

```text
polyMul (toList a) (toList b)
  = polyAdd (toList (ringKMul a b)) (polyMul q Φ₈₁)
```

then apply `polyEval_add_multiple_of_root`, which was already proved in cycle
262. The root is used only at the evaluation step. Degree-consistent:
`deg q ≤ 52`, so `deg(q·Φ₈₁) ≤ 106`, matching the raw convolution.

**Route B — per-degree evaluation.** Evaluate both sides and apply the two
reduction identities degree by degree. This needs `powPair`, `powPair_add`,
`reduction_single_fold` and `powPair_eightyOne`, plus a bridge from `polyEval`'s
`foldr` to a power sum that does not exist.

**Cycles 279 and 280 built Route B's identities before Route A was recognised
as available.** They are correct theorems and they are guarded, but they are
not on the path Route A takes — Route A needs only the coefficient identity and
a lemma from cycle 262.

The error was proceeding from "what is the next obstacle" without asking "which
route is shortest given what is already proved". `polyEval_add_multiple_of_root`
had been sitting proved since cycle 262 and its applicability was not checked
against the reduction until now.

### The quotient in closed form (cycle 282)

The two sums combine. The second contributes `+c_d` at index `d−54` for
`81 ≤ d ≤ 106`, i.e. indices 27 through 52 — exactly the range the first sum
leaves empty. So:

```text
q_j = c_{j+54} − (c_{j+81} if j ≤ 25 else 0)      for j = 0 … 52
```

one subtraction, no case split on the positive part.

**Boundary cases checked by hand**, since they are where an off-by-one would
hide:

| index | `folded − twice` | `q_i + q_{i−27} + q_{i−54}` |
|---|---|---|
| `i ≤ 25` | `c_{i+54} − c_{i+81}` | `q_i` alone, matching |
| `i = 26` | `c₈₀ − 0` (twice is off: `107 > 106`) | `q₂₆ = c₈₀` |
| `27 ≤ i ≤ 52` | `c_{i+27}` | `c_{i+54} + c_{i+27} − c_{i+54}` |
| `i = 53` | `c₈₀` | `0 + c₈₀ + 0` (`q₅₃` out of range) |
| `54 ≤ i ≤ 80` | — (`r_i = 0`) | `c_{i+27} + c_i − c_{i+27} = c_i` |

**Validated end to end** on `a = X⁵³`, `b = X`, where `c = X⁵⁴`: the formula
gives `q = 1`, so `q·Φ₈₁ = 1 + X²⁷ + X⁵⁴`, and the reduced product's two `p−1`
coefficients at 0 and 27 cancel against it exactly, leaving `X⁵⁴`. That case
exercises both the fold and the cancellation.

What remains is expressing `q` as a `List Pair` — which needs a `subPair` this
tower does not have — and proving the coefficient identity.

### Subtraction (cycle 283)

`subPair` supplies the operation the quotient's coefficients need. Every
operation in this tower until now has been `addPair`, `mulPair` or a
`goldilocksP − 1` multiplier.

`Nat` has no negatives, so the complement is explicit: `p − y % p`, safe because
`y % p < p`. `complement_add` is the whole content — `(p − v%p) + v` is exactly
`p · (1 + v/p)`, a multiple of the prime, which the final reduction discards.

`addPair_subPair` is the defining law and holds only for canonical `x` — the
sixth place in this tower where that condition is load-bearing (260, 263, 266,
268, 278, 283).

**Checked on the wraparound case**, which is where naive `Nat` subtraction
fails by truncating to zero: `3 − 10` gives `p − 7`, and adding 10 back recovers
3. A check using only `10 − 3` would not have distinguished the two.

### The quotient as a list (cycle 284)

`reductionQuotient` realises the closed form from cycle 282 as a 53-element
coefficient list, using `subPair` from cycle 283. Its coefficients are stated
in range and out, and every one is a residue.

**Validated against the algebra, not just the formula.** On `raw = X⁸¹` the
definition yields `q₀ = p−1` and `q₂₇ = 1` — that is `q = X²⁷ − 1`, exactly the
factor in `X⁸¹ − 1 = Φ₈₁·(X²⁷ − 1)` that cycle 277 derived by hand. The list
definition reproduces the algebra independently of the derivation that
motivated it.

On `raw = X⁵⁴` it yields the constant `1`, matching cycle 282's worked case.
The two cases exercise both branches of the conditional: index 0 takes the
subtraction, index 27 does not.

What remains for `KRINGMUL-REDUCTION-QUOTIENT` is the coefficient identity
`raw = reduced + q·Φ₈₁` itself.

### The modulus is sparse (cycle 285)

`Φ₈₁` has three nonzero coefficients out of 55. Convolving against it picks out
three terms and kills the rest, which is what makes the coefficient identity a
three-way sum rather than a 53-term one.

`coeffAt_modulus` states the lookup at *every* index, in range and out — past 54
both the list default and the condition give zero, so no case split reaches the
caller. That totality is the same property `coeffAt_toList` needed in cycle 267,
and for the same reason.

Verified by sweeping 60 indices: exactly `[0, 27, 54]` are nonzero, the
neighbours 26 and 28 are zero, and index 55 — past the list entirely — is zero.

**Still open:** the coefficient identity itself. With sparsity, what has to be
shown is that `convolution q Φ₈₁ i` reduces to `q_i + q_{i−27} + q_{i−54}`. The
convolution recurses over all 53 of `q`'s entries, so extracting the three
surviving terms needs a way to discard the vanishing ones — `sumOver` with
`convolution_eq_sumOver` from cycle 273 is the likely route, since
`mulPair _ ⟨0,0⟩ = ⟨0,0⟩` makes the other terms inert.

### Two routes to the three-way reduction (cycle 286)

`convolution q Φ₈₁ i` must reduce to
`q_i + (q_{i−27} if i ≥ 27) + (q_{i−54} if i ≥ 54)`. The conditionals are not
cosmetic: `Nat` subtraction truncates, so `coeffAt q (i − 27)` at `i < 27` would
read `q₀` rather than vanish.

**Route (a) — term extraction.** Via `convolution_eq_sumOver` (cycle 273), the
convolution is a `sumOver` across `List.range (i+1)` whose terms vanish except
at `j ∈ {i, i−27, i−54}`, because `mulPair _ ⟨0,0⟩ = ⟨0,0⟩` and the modulus is
sparse. Needs a lemma extracting the surviving terms from an otherwise-zero sum.

**Route (b) — convolve the other way.** `convolution Φ₈₁ q i` recurses on the
*modulus*, so its own recursion walks the sparse list and the zero entries drop
out step by step. But it needs `polyMul_comm`, which is not proved.

**Assessment: neither is shorter.** `polyMul_comm` is an induction of
comparable size to the extraction lemma, and it adds a general theorem to the
tower rather than a targeted one. Route (a) is preferred on that ground alone,
not on proof length.

Commutativity was checked to *hold* numerically before being ruled out on cost
— `[1,2,3]·[4,5]` and `[4,5]·[1,2,3]` both give `[4,13,22,15,0]` — so route (b)
is genuinely available if the extraction lemma turns out worse than expected.

### Discarding vanishing terms (cycle 287)

`sumOver_filter` proves that dropping indices whose term vanishes leaves the sum
unchanged. That is route (a)'s extraction mechanism, and it avoids splitting the
range around each survivor: filter first, then evaluate a short list.

Checked on exactly the shape the reduction needs — a sum over 107 indices with
three nonzero terms. The filtered list has length 3 and both sums give the same
value. A check on a short list would not have exercised the case where the vast
majority of terms are discarded.

**What remains for the three-way reduction:** instantiate this with
`keep j = decide (i − j ∈ {0, 27, 54})`, show the filtered list is exactly the
surviving indices, and evaluate. The guards on the truncating subtractions
(cycle 286) belong in that step.

### Terms vanish off the survivors (cycle 288)

`modulus_term_vanishes` supplies `sumOver_filter`'s hypothesis: at any index
whose modulus offset is not 0, 27 or 54, the product is zero because the modulus
coefficient is.

**The Nat-subtraction subtlety, recorded.** Over `List.range (degree + 1)` every
index satisfies `j ≤ degree`, so `degree − j = 0` holds exactly at `j = degree`
and truncation never fires. *Outside* that range it would: at `j > degree` the
difference truncates to 0 and the term would be wrongly retained.
`offset_zero_iff` states the in-range fact so downstream proofs do not have to
rediscover it.

**Survivor sets verified across all three regimes**, since the count depends on
whether the degree clears 27 and 54:

| degree | survivors |
|---|---|
| 60 | `[6, 33, 60]` — three |
| 30 | `[3, 30]` — two |
| 10 | `[10]` — one |

A check at a single degree would have exercised only one regime. The three-way
reduction's case analysis is exactly this split, so all three had to be seen.

### Identifying the survivors: the low regime (cycle 289)

`filter_survivors_low` proves that below 27 exactly one term survives — the
diagonal — because `degree − j` never reaches 27 when `degree` itself does not.

**The bound is tight, and that was checked.** At degree 26 the survivor list is
`[26]`; at degree 27, where the hypothesis fails, it is `[0, 27]` — a second
survivor appears precisely at the boundary. A hypothesis of `degree < 54` would
also have been true here but would have been wrong, and the boundary check is
what distinguishes them.

`survives` is named once so the three regimes share the predicate rather than
each restating the disjunction.

**The remaining two regimes need a different technique.** The low regime peels
one element off the end of the range with `List.range_succ`. The middle regime's
second survivor sits at `degree − 27`, in the interior, so peeling from the end
does not reach it without 27 steps. Extracting an interior element needs either
a range split at that position or a characterisation of the filter by membership
rather than by construction.

### The survivors are an arithmetic progression (cycle 290)

Cycle 289 framed the remaining work as two more regimes, each needing interior
extraction. Sampling the filter across all three shows a more uniform structure:

| degree | survivors |
|---|---|
| 30 | `[3, 30]` |
| 40 | `[13, 40]` |
| 53 | `[26, 53]` |
| 54 | `[0, 27, 54]` |
| 80 | `[26, 53, 80]` |
| 106 | `[52, 79, 106]` |

The survivors are always `degree − 27k` for `k = 0, 1, 2` with `27k ≤ degree` —
an arithmetic progression of difference 27, descending from the diagonal. The
"three regimes" are just how many of the three offsets fit below the degree.

That suggests one statement parameterised by the number of admissible `k`,
rather than three regime lemmas with different extraction techniques. It is
also the shape `sumOver_map` (cycle 272) is built for: a shifted index list.

**Route reassessment.** Cycle 281 chose route (a) over route (b) on scope
grounds. Both now need index-shifting of comparable difficulty — route (a) to
extract interior survivors, route (b) for the `polyEval`-to-power-sum bridge.
`sumOver_map` is the shared tool, and whichever route is taken should use it
rather than re-deriving a shift.

---

## KRINGPROJECTION-ROOT-IMPOSSIBLE

```text
claim:
  No challenge in K is a root of Phi81, so the root-based projection
  homomorphism is unreachable and the quotient term cannot be deleted.
status: CLOSED-NEGATIVE 2026-07-27 (cycle 291). Arithmetic witness measured
        outside Lean; not yet kernel-checked. See "What is not proved" below.
supersedes:
  KRINGPROJECTION-HOMOMORPHISM (obligation 2), KPOLYEVAL-ROOT-REDUCTION,
  KROOT-HYPOTHESIS-FORM, and the cost figures in KRINGPROJECTION-COST.
```

### What was wrong

Cycles 232–290 carried `KRingProjection.RootOfModulus beta`, the premise
`Phi81(beta) = 0`, on the reasoning that evaluation at a root of the modulus is
a ring homomorphism, so a cyclotomic product would cost one `KMul` — three rows
— instead of a convolution. `KRINGPROJECTION-COST`'s 803-row combine equation
was computed under that assumption, and `KPOLYEVAL-ROOT-REDUCTION` and
`KROOT-HYPOTHESIS-FORM` existed only to serve it.

The premise is **unsatisfiable**. `Phi81`'s roots are primitive 81st roots of
unity, so a root in `K` requires `81 | |K*| = p^2 - 1`. For Goldilocks
`p = 2^64 - 2^32 + 1`:

| quantity | value |
|---|---|
| `v3(p - 1)` | 1 (from `2^32 - 1 = 3 * 5 * 17 * 257 * 65537`) |
| `v3(p + 1)` | 0 (`p + 1 = 2 mod 3`) |
| `v3(p^2 - 1)` | 1, and `81 = 3^4` |
| `ord_81(p)` | 27 |

So `Phi81` splits over `F_p` into two irreducible factors of degree 27 and its
roots live in `F_{p^27}`, never in the quadratic extension the verifier works
over. Every theorem carrying `RootOfModulus` was vacuously true, and a search
of the whole tree confirms the structure had **no constructor anywhere** — it
could not have had one.

This is the trap in prompt section 3, "a premise that moved rather than
closed", in its strongest form: not merely a premise no consumer happens to
construct, but one no consumer can construct.

### What production actually checks

`ProjectionProgram.ProjectionTrace.identity`
(`Nightstream/Implementation/R1CS/Core/Projection/Polynomial.lean:124`) never
made the assumption. It checks the quotient form

```text
sum_i rho_i * x_i  =  q * Phi81 + pad(out)
```

as a *coefficient* identity of fixed width, tested at one challenge, with `q`
supplied by the prover as committed columns. `quotientColumns.length = 53` and
`maxDegree = 106` throughout the recursive and terminal profiles. That needs no
condition on the challenge; its soundness is the frozen
`ProjectionCheck.accepted_implies_exact_or_badRoot`, whose single named event
is `BadRoot`.

The canonical track reaches the same two numbers independently:
`KQuotient.quotientLength = 53`, derived from the requirement that `q * Phi81`
reach the raw convolution's degree 106 given `deg Phi81 = 54`. Those figures
are derived here, not read from the artifact.

### What changed in code

Deleted, with their axiom guards:

- `KRingProjection.RootOfModulus`
- `KPolyEval.polyEval_root_of_RootOfModulus`
- `KPolyEval.polyEval_multiple_of_root`
- `KPolyEval.polyEval_add_multiple_of_root`
- `KRingProjection.combineEquationCost`, `combineEquationCost_eq`,
  `combineEquationCost_two`, `batchCombineCost`, `batchCombineCost_uniform`,
  and the local `sum_replicate` they used

Added: `KPolyEval.polyEval_quotientForm`, the evaluated production right-hand
side, with **no hypothesis on the point**:

```text
polyEval p (polyAdd out (polyMul q m))
  = addPair (polyEval p out) (mulPair (polyEval p q) (polyEval p m))
```

Two rewrites, `polyEval_polyAdd` and `polyEval_polyMul`, both already proved.
The route that needed an impossible premise is replaced by one that needs none.

### Why the cost figures went rather than being adjusted

`combineEquationCost 2 = 803` had two independent defects, either fatal:

- **A count without a construction.** `803` was arithmetic on a `def`, not the
  length of an emitted row list.
- **A subtotal presented as a total.** It priced a ring product at one `KMul`
  and omitted the quotient projection and the evaluation of `Phi81` entirely —
  precisely the two terms the impossible premise was hiding.

No replacement number is recorded. The replacement must be a fold over an
emitted quotient-identity program, and that program does not exist yet.

### What is not proved

The impossibility argument above is **measured, not derived**: `v3(p^2 - 1)`
and `ord_81(p)` were computed outside Lean. It is not kernel-checked and is not
claimed to be.

Kernel-checking it would need the order of `K*` and a Lagrange argument, or the
Frobenius identity `beta^(p^2) = beta` for the concrete `K` — a real
sub-project without Mathlib. It is deliberately not attempted, because nothing
downstream depends on it: the replacement route carries the quotient term and
therefore needs no statement about roots at all. The impossibility is recorded
to explain why the old route was abandoned, not to support the new one.

`KRINGPROJECTION-COST` is left with no cost claim rather than a corrected one.

---

## KQUOTIENT-SURVIVORS

```text
claim:
  The modulus convolution's surviving terms are exactly degree - 27k for the
  admissible k, and the convolution collapses to a sum over them.
status: PROVED 2026-07-27 (cycle 291). filter_survivors and
        convolution_modulus_eq_survivor_sum, both guarded.
```

Cycle 290 predicted one statement parameterised by the admissible `k` would
replace three regime lemmas. It did, and the parameterisation is a list rather
than an index:

```text
survivorList degree =
  (if 54 <= degree then [degree - 54] else [])
    ++ (if 27 <= degree then [degree - 27] else [])
    ++ [degree]
```

`filter_survivors` proves `(List.range (degree + 1)).filter (survives degree) =
survivorList degree` at **every** degree, by induction on the degree rather
than by case split. The step is exact: raising the degree by one shifts every
survivor up by one (`survivorList_succ`), and a new survivor appears at index 0
exactly when the degree reaches 27 or 54, where the two boundary cases close by
`decide` on closed lists. `filter_map_succ` is the small helper that lets
`List.range_succ_eq_map`'s shifted tail be filtered.

`convolution_modulus_eq_survivor_sum` then discards the other 104 terms in one
step via `sumOver_filter`, whose per-index hypothesis is exactly
`modulus_term_vanishes`. The regime `if`s survive only inside `survivorList`;
everything above that line is regime-free.

This is the work `KRINGPROJECTION-ROOT-IMPOSSIBLE` promotes from a stepping
stone to the main line: with the root route closed, the quotient identity is
the encoding, and this is its coefficient-level core.

**Remaining for `KRINGMUL-REDUCTION-QUOTIENT`.** The sum over `survivorList` is
still a `sumOver` of `mulPair` against `Phi81` coefficients; turning it into the
three-term form `q[degree] + q[degree - 27] + q[degree - 54]` needs
`mulPair_one_left` at each survivor plus the canonicity side conditions. Then
the coefficient identity `raw = reduced + q * Phi81` follows, and
`polyEval_quotientForm` carries it to the challenge.

---

## KQUOTIENT-IDENTITY-HIGH

```text
claim:
  Above the reduced range the quotient reproduces the raw coefficient exactly,
  which is the upper half of raw = reduced + q * Phi81.
status: PROVED 2026-07-27 (cycle 292). KQuotient.quotient_identity_high,
        guarded. Lower half (degree < 54) not yet written.
```

### The frozen reduction, matched

`SuperNeo.Concrete.ringKMul` reduces by `X^54 = -X^27 - 1`:

```text
reduced[i] = raw[i] - (raw[i+54] if i < 27 else raw[i+27])
                    + (raw[i+81] if i + 81 <= 106 else 0)
```

Reducing `X^j` by hand for each range gives, for `j` in 54..80,
`X^j = -X^{j-27} - X^{j-54}`, and for `j` in 81..106 a second pass collapses to
`X^j = X^{j-81}`. Collecting contributions at index `i` reproduces the frozen
definition exactly, including both guard boundaries. The quotient

```text
q_j = c_{j+54} - (c_{j+81} if j <= 25 else 0),   j = 0..52
```

is the matching one. That check is the reason the identity below came out
first try rather than needing the definition adjusted.

### The upper half

For `degree >= 54` the reduced polynomial has run out, so the quotient must
carry the raw coefficient alone. Three survivors contribute and they telescope:

| survivor | value | why |
|---|---|---|
| `q[degree]` | `0` | past the quotient's 53 entries |
| `q[degree - 27]` | `raw[degree + 27]` | subtrahend guard is false at these degrees |
| `q[degree - 54]` | `raw[degree] - raw[degree + 27]` | subtrahend guard is true |

`addPair_subPair` collapses the last two. The statement has **no case split by
degree**; the splits live in `quotientCoeff_shift27` and
`quotientCoeff_shift54`, and each collapses for the same reason — past index
106 the raw convolution is empty, so "the quotient ran out" and "the raw
coefficient is zero" are the same fact.

### The length bound has a constructor

`quotient_identity_high` takes `raw.length <= 107`. That is what makes the
boundary cases agree, so it must not be an unconstructed premise. It is not:
`KPolyCoeff.rawProduct_length` proves `(polyMul (toList a) (toList b)).length =
107` for any two `RingK` elements, via a new `polyMul_length`
(`m + n - 1`, both operands nonempty). `Canonical raw` is likewise constructed
by `canonical_polyMul`.

Both hypotheses were checked against prompt section 6 item 5 before this was
reported, which is what turned up the missing `polyMul_length`.

### What remains

The lower half, `degree < 54`, where the reduced coefficient is nonzero and the
identity is `raw[d] = reduced[d] + (q * Phi81)[d]`. Two regimes there
(`quotientSum_low` and `quotientSum_middle` are already proved and supply the
sums), and the arithmetic is `addPair`/`subPair` cancellation of the same shape
as above. Then `polyEval_quotientForm` carries the whole identity to the
challenge with no condition on it.

---

## KQUOTIENT-IDENTITY

```text
claim:
  raw = reduced + q * Phi81, at every coefficient, as lists, and evaluated at
  any point.
status: PROVED 2026-07-27 (cycle 293) at the list level.
        KQuotient.coefficient_identity,
        KQuotient.raw_eq_reduced_add_quotient_multiple,
        KQuotient.polyEval_raw_eq_quotientForm. All guarded.
        NOT yet connected to the frozen ringKMul - see "The one remaining link".
```

### The three ranges

| range | survivors | why it closes |
|---|---|---|
| `d < 27` | `q[d]` | the reduction's second-pass term is live; the quotient's own subtrahend cancels it, then the result cancels the reduction's subtraction |
| `27 <= d < 54` | `q[d-27]`, `q[d]` | the shifted survivor cancels the top one, leaving `raw[d+27]`, which is what the reduction subtracted |
| `54 <= d` | three | reduced is empty; the three telescope to `raw[d]` (cycle 292) |

`coefficient_identity` glues them with no degree in the conclusion.

### Up to lists, and to the point

Both sides have length 107: the raw convolution by `rawProduct_length`, and
`polyAdd reduced (polyMul q Phi81)` because `polyMul` of a 53-entry quotient
and a 55-entry modulus is `53 + 55 - 1 = 107`. That coincidence is not luck -
reaching 107 is what fixed `quotientLength = 53` in the first place.

`list_ext_coeffAt` upgrades the coefficient identity to a list equality (the
length hypothesis is not redundant: `coeffAt` returns zero past the end, so `[x]`
and `[x, 0]` agree at every index while being different lists). Then
`polyEval_quotientForm` gives

```text
polyEval p raw = polyEval p reduced + polyEval p q * polyEval p Phi81
```

with **no hypothesis on `p`**. This is the shape the frozen
`ProjectionTrace.identity` tests, and it is what the withdrawn root route
(cycle 291) was trying to reach by assuming `Phi81(p) = 0`, which is impossible
over `K`.

### Both hypotheses have constructors

`Canonical raw` from `canonical_polyMul`; `raw.length = 107` from
`rawProduct_length`. Checked against prompt section 6 item 5 before reporting.

### The one remaining link — CLOSED 2026-07-27 (cycle 294)

> Closed by `KQUOTIENT-FROZEN-LINK` below: `KQuotient.toList_ringKMul` proves
> `reducedList` of the raw convolution **is** `ringKMul`. Both prerequisites
> named here were written — `KConcreteBridge.ofConcrete_sub` and
> `KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_all`. The text below records what
> was open at the time.

`reducedCoeff` is written in `ringKMul`'s exact shape - including its
`index + 81 <= 106` guard rather than the equivalent `index <= 25`, so the
correspondence is visible rather than argued - and it was derived by reducing
`X^54 = -X^27 - 1` by hand, the same derivation that produced `ringKMul`. But
**the two are connected only by inspection.** Two things are needed:

1. `ofConcrete` must carry `K.sub`, like it already carries `K.add` and `K.mul`.
2. `KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul` must extend past degree 53. It
   currently carries `below : degree < ringDegree`, because its proof truncates
   `List.range ringDegree` to `List.range (degree + 1)`, which needs
   `degree + 1 <= 54`. `ringKMul` reads `rawMulCoeffK` at degrees 54 through
   106 - every one of its off-diagonal reads is outside the proved range.

That second point is the substantive one and it is not a formality. The
agreement does hold at every degree: for `degree >= 54` the frozen fold's guard
`degree - i < 54` fires exactly where `convolution`'s terms are nonzero, because
`toList` is zero outside 0..53. The proof shape is: split
`List.range (degree + 1)` at 54 with `List.range_add`, kill the tail with
`sumOver_of_zero_terms` since every index there exceeds `toList`'s width, and
relate the frozen guard to vanishing terms with a new `sumOver_of_guard_zero`.

Until that is written, everything in `KQuotient` is a theorem about the
list-level convolution and its reduction - not yet about the protocol's ring
multiplication. `KPOLYMUL-RAWMULCOEFF-AGREEMENT` covers only `degree < 54`.

---

## KQUOTIENT-FROZEN-LINK

```text
claim:
  reducedList of the raw convolution is the frozen ringKMul, so the quotient
  identity is a statement about the protocol's ring multiplication.
status: PROVED 2026-07-27 (cycle 294). KQuotient.toList_ringKMul and
        KQuotient.polyEval_ringKMul_quotientForm, both guarded.
closes: KRINGMUL-REDUCTION-QUOTIENT; extends KPOLYMUL-RAWMULCOEFF-AGREEMENT to
        every degree.
```

### Why the agreement had to be extended

`KPOLYMUL-RAWMULCOEFF-AGREEMENT` (cycle 275) carried `degree < ringDegree`.
`ringKMul` reads `rawMulCoeffK` three times per coefficient — at `index`, at
`index + 27` or `index + 54`, and at `index + 81` — and **only the first is ever
below the ring degree.** Without the extension the link could not even be
stated, let alone proved.

### The two proofs are mirror images

| | below the ring degree | at or above it |
|---|---|---|
| first | shorten the range, since the guard is silent past the degree | drop the guard, since it is false exactly where a coefficient is out of range |
| then | drop the guard, now true everywhere | lengthen the range, since the added terms are zero |

The swap is forced, not stylistic. Below the degree the guard is false at
indices whose terms are **not** zero — any `index` in `(degree, 54)` reads
`coeffAt right 0`, which is generally nonzero — so the guard cannot be dropped
first. Above the degree the frozen `List.range ringDegree` is already the
shorter list, so it cannot be shortened further.

That asymmetry is why `sumOver_of_guard_zero` and
`sumOver_range_truncate_terms` are new lemmas rather than reuses:
`sumOver_range_truncate` shortens a range when the *guard* goes silent; these
two remove a guard whose false positions cost nothing, and shorten a range where
the *terms* vanish.

### `ofConcrete` carries subtraction

`Fin.sub` is `(n - b + a) % n`; `subPair` is `(a + (p - b % p)) % p`. Same value,
different term order, and the `% p` on the subtrahend is redundant only because
`b.val < n`. `ofConcrete_sub` is one commutation and one `Nat.mod_eq_of_lt`.

### The payoff, in the protocol's vocabulary

```text
polyEval p (polyMul (toList a) (toList b))
  = polyEval p (toList (ringKMul a b))
      + polyEval p (reductionQuotient (polyMul (toList a) (toList b)))
        * polyEval p Phi81
```

Every symbol on the right is either a frozen definition or the quotient this
track constructs, and there is **no hypothesis on `p`**. This is the algebraic
content of what `ProjectionProgram.ProjectionTrace.identity` tests, derived in
Lean rather than read from an artifact.

### What this does not give

A row program, a cost, or a challenge. This is the algebra the Pi_RLC recipe
must encode, not the recipe. `KRINGPROJECTION-COST` still records no
combine-equation number, and will not until a program emits one.

---

## KQUOTIENTIDENTITY-RECIPE

```text
claim:
  The row program for one projected quotient identity, and its derived cost.
status: PARTIAL 2026-07-27 (cycle 295). Construction, derived row count and
        atom soundness proved and guarded. Ownership, conservation,
        equation-level soundness, honest completeness and Typed.Cost NOT
        written - this is not yet a complete recipe under the ten-item
        checklist.
replaces: the combine-equation figures withdrawn by
        KRINGPROJECTION-ROOT-IMPOSSIBLE.
```

### What is encoded

The frozen check is a coefficient identity between fixed-width vectors, tested
at one challenge. The encoding never materialises the degree-106 vectors; it
uses two facts the `K` tower proves, neither of which needs any condition on the
challenge:

- `KPolyEval.polyEval_polyMul` — `eval(rho_i * x_i) = eval(rho_i) * eval(x_i)`;
- `KQuotient.polyEval_ringKMul_quotientForm` — the quotient form evaluates to
  `eval(out) + eval(q) * eval(Phi81)`.

So the program checks
`sum_i eval(rho_i) * eval(x_i) = eval(out) + eval(q) * eval(Phi81)`
with every `eval` a Horner projection.

### The derived cost

| part | rows | derived from |
|---|---|---|
| one input pair (atom) | 321 | `hornerRows_length` twice at 54 coefficients, plus `KMul.rows_length` |
| output projection | 159 | `hornerRows_length` at 54 |
| quotient projection | 156 | `hornerRows_length` at 53 |
| modulus evaluation | 162 | `hornerRows_length` at 55 |
| `q(beta) * Phi81(beta)` | 3 | `KMul.rows_length` |
| `K`-equality | 2 | `KEquality.rows_length` - two coordinate rows, not one |

`identityRows_length` gives `321 * pairs.length + 482`, and at the production
arity of two, `identityRows_length_production` gives **1124**.

Every one of these is the `List.length` of an emitted program, reached through
the gadget length lemmas. None is declared, measured, or read from an artifact.

### The check against the withdrawn number

`1124 - 803 = 321 = 156 + 162 + 3` - exactly the quotient projection, the
modulus evaluation, and their product. That is an independent confirmation of
`KRINGPROJECTION-ROOT-IMPOSSIBLE`'s diagnosis: the old figure was the same
program with precisely the three terms the impossible root assumption hid.

### Why Phi81 costs 162 rows

Its coefficients are constants but `Phi81(beta)` is not - it is a degree-54
polynomial in the challenge, and Horner spends one `K` multiplication per step.
An addition chain would compute `beta^27` in about seven multiplications and
`Phi81(beta) = (beta^27)^2 + beta^27 + 1` in one more, so roughly 24 rows.

That optimisation is deliberately not taken. The number above is derived from
the program actually emitted; a cheaper program must be emitted before a cheaper
number may be recorded. Named here as `KQUOTIENTIDENTITY-MODULUS-CHAIN`.

### What is NOT proved — updated 2026-07-27 (cycle 296)

> Column ownership and equation-level soundness, listed below as open, are now
> **proved**: `identityColumns_nodup`, `atoms_disjoint`, `atom_inside`,
> `tail_blocks_separated`, `tail_inside`, `pairsRows_sound` and
> `identityRows_sound`. See `KQUOTIENTIDENTITY-OWNERSHIP-SOUNDNESS` below.
> Conservation, honest completeness, `Typed.Cost` and the decoder remain open.
> The text below records what was open at the time.


- **Column ownership and conservation.** The layout is constructed - sequential
  blocks of `atomWidth`, then 159, 156, 162, 3 - and the widths are derived from
  `KFrames.frameColumns_length`, but no disjointness theorem is written.
  `KHorner` already defers frame disjointness to the caller, so this is the
  first place it must actually be discharged.
- **Equation-level soundness.** `productRows_sound` proves the atom: satisfaction
  forces the frame's carried output to be the product of the two projections.
  Composing the atoms into
  `sum_i eval(rho_i) * eval(x_i) = eval(out) + eval(q) * eval(Phi81)` needs an
  induction over the pair list with `lcEval_append` for the free summation, and
  is not written.
- **Honest completeness** and a **`Typed.Cost`** tuple.

Until those are written this is a construction with a derived cost, not a
recipe. It must not be counted toward Phase 4's `nifsVerify`.

---

## KQUOTIENTIDENTITY-OWNERSHIP-SOUNDNESS

```text
claim:
  The quotient-identity program's column allocation is collision-free and
  gapless, and satisfaction forces the projected quotient identity.
status: PROVED 2026-07-27 (cycle 296). Fifteen theorems, all guarded.
        Conservation, honest completeness and Typed.Cost still NOT written.
```

### Column ownership

Every block the program allocates is a `KFrames.frameColumns` run, and those are
contiguous by construction, so the whole allocation is itself one contiguous run
and ownership becomes arithmetic rather than combinatorial:
`frameColumns_mem_iff` turns membership into an interval and `omega` closes it.

- `identityColumns base pairCount = frameColumns base (107 * pairCount + 160)`
- `identityColumns_length` = `321 * pairCount + 480`
- `identityColumns_nodup` — no column is allocated twice
- `atoms_disjoint` — distinct atoms share no column
- `atom_blocks_separated` — inside an atom, the two projections and the product
  frame occupy `[0,159)`, `[159,318)` and `[318,321)`
- `atom_inside` — every atom lies inside the allocation
- `tail_blocks_separated` — each gap between the output, quotient, modulus and
  product blocks is exactly the preceding block's width, so the allocation is
  **gapless** as well as collision-free
- `tail_inside` — the last block ends exactly at the allocation's end, so the
  allocation is exhausted rather than merely contained

**A consistency check that was not built in.** Columns are `321n + 480` and rows
are `321n + 482`. The difference is exactly two — the `K`-equality's two
coordinate rows, which allocate nothing. The two counts were derived
independently, one from `List.length` of the emitted rows and one from
`frameColumns_length`, and they differ by precisely the gadget that emits rows
without columns.

This is the first place `KHorner`'s deferred frame disjointness is actually
discharged rather than passed on.

### Equation-level soundness

`identityRows_sound`: for any assignment satisfying every emitted row, with the
constant wire set,

```text
sum_i eval(rho_i) * eval(x_i) = eval(out) + eval(q) * eval(Phi81)
```

which is the equation `KQuotient.polyEval_ringKMul_quotientForm` proves the
frozen ring multiplication satisfies. It is unconditional in the vector lengths
— soundness needs no sizing hypothesis, only `z 0 = 1`.

The composition step is `carriedValue_concat`: the left-hand side is a
concatenation of combinations, so its value is the `Pair` sum of the atom
outputs and the summation costs nothing. That is the same fact that makes the
row count `321` per pair with no term for the sum.

### Still open

- **Conservation** — CLOSED 2026-07-27 (cycle 298), see `KQUOTIENTIDENTITY-CONSERVATION`. Earlier note from cycle 297: The atom and the
  left-hand side are proved conservative (`productRows_conservation`,
  `pairsRows_conservation`); composing the six parts of `identityRows` is not.
  Original text: that no emitted row touches a column outside the allocation
  plus the declared shared reads (the challenge and the coefficient carriers).
  `KHornerSupport.hornerRows_mentions` and `KMulOwnership.rows_conservation`
  are the pieces; composing them over this layout is not written.
- **Honest completeness** — an honest execution yields a satisfying assignment.
- **`Typed.Cost`** — the four-component tuple.
- **The decoder** from a frozen `ProjectionTrace` to `Carried` lists. This is
  what would construct the length hypotheses `identityRows_length` takes, and
  what would carry `identityRows_sound` to `ProjectionCheck.Accepted`. Until it
  exists the recipe is sound about *its own* inputs, not yet about the frozen
  trace's.

---

## KQUOTIENTIDENTITY-CONSERVATION

```text
claim:
  No emitted row reaches outside the allocation plus the declared shared reads.
status: PROVED 2026-07-27 (cycle 298). identityRows_conservation, guarded.
        Atom and left-hand side were cycle 297; the six-part composition and
        the two carried-value lemmas are cycle 298.
```

### The blocker was in the witness, not the layout

`KHornerSupport` stated what an emitted Horner row may mention as
`FrameAtOrAfter frames step column` — "some frame at or after this step", with
**no upper bound**. That is unusable for conservation: a column with no ceiling
cannot be shown to lie inside a finite block. The layout arithmetic was never
the hard part; the witness was too weak to state the conclusion.

The definition is now `FrameOfRun frames coefficients step column`, which adds
`later + 1 < step + coefficients.length` — an `n`-coefficient evaluation
performs `n − 1` multiplications, so its frames are steps `step` through
`step + n − 2`. The recursion preserves the bound exactly: the suffix at
`step + 1` with `n − 1` coefficients has the same ceiling as the whole run.

`KHornerHonest`, the only other consumer, never needed the bound — its freshness
argument uses the lower bound alone. It took two extra binders in two `rcases`
patterns.

**The gate caught the consequence.** `hornerCarried_mentions` gained a
`Quot.sound` dependency from the added `omega` calls, and its axiom guard
failed. That is the fail-closed guard doing its job on a change made three
modules away.

### What is proved

- `frameOfRun_interval` — a run's frames lie in `[blockBase, blockBase + 3(n−1))`
- `hornerBlock_conservation` — one Horner block mentions only the challenge, a
  coefficient, or its own block
- `hornerCarried_conservation` — the same for a carried result, needed because
  the product rows read the evaluations' carried values rather than their rows
- `productRows_conservation` — the atom stays inside `[atomBase, atomBase + 321)`
- `pairsRows_conservation` — by induction, the left-hand side stays inside
  `[base, base + 321·pairs.length)`
- `allocated_iff` — the allocation as an interval, which is the form the
  arithmetic wants

### The composition — done

`identityRows_conservation` covers all six parts. The two lemmas cycle 297
predicted were needed are `productCarried_mentions` (a product frame's carried
output mentions only that frame) and `pairsCarried_mentions` (the left-hand
side's carried value stays inside the atoms' block, by induction over the
first).

**The conclusion has an arm cycle 297 did not anticipate: `column = 0`.**
`KEquality`'s rows write a literal one on the constant wire, so the constant
wire is reachable. It allocates nothing and is shared by every program in the
system, so it belongs in the conclusion as its own arm rather than being folded
into the allocation or hidden in `SharedRead`. A conservation statement that
omitted it would have been false.

### A note on `maxRecDepth`

`KQuotientIdentity` now sets `maxRecDepth 8000`. The conservation induction
rewrites through 321-column atom blocks and the default depth is not enough.
Two other modules in this tree already set it higher. It is an elaboration
budget, not a soundness option.

---

## KQUOTIENTIDENTITY-HONEST

```text
claim:
  An honest execution yields an assignment satisfying the emitted program.
status: PROVED 2026-07-27 (cycle 302). identityRows_honest, guarded.
        Atom (299), left-hand side (300), primitives (301), assembly (302).
        Lives in KQuotientIdentityHonest.lean.
```

### The missing primitive

`KHornerHonest` had `hornerWitness_satisfies` — one block's witness satisfies
that block — but nothing saying **where the witness writes**. Composition needs
the second: to extend block A's witness with block B's without disturbing A, B's
writes must miss everything A's rows mention.

`hornerWitness_off_block` supplies it: every column the witness touches is a
frame column of step `step` or later, so anything strictly below `base + 3·step`
keeps its value. Three lines of `witness_off_frame` plus the recursion.

### Completeness consumes conservation

The atom's proof is built left to right — the left evaluation's witness,
extended by the right evaluation's, extended by the product's — and each
extension is justified by a bound that **conservation already proved**:

- `hornerBlock_conservation` bounds what a block's rows mention, which is the
  hypothesis `satisfies_extend` needs;
- `hornerCarried_conservation` bounds the carried results, which is the
  freshness `KMulHonest.witness_satisfies` needs.

Neither bound had to be re-derived. That is the payoff for having stated
conservation in interval form rather than as membership in a column list.

### Sizing is needed here but not for soundness

`productRows_honest` takes `left.length = 54` and `right.length = 54`;
`productRows_sound` takes neither. Soundness is a statement about whatever
assignment satisfies the rows, so the layout is irrelevant to it. Completeness
must place the blocks, so it needs the widths. The asymmetry is real and worth
noting: a recipe can be sound at any sizing and complete only at the intended
one.

### The left-hand side — done (cycle 300)

`pairsRows_honest` extends each atom's witness by the atoms to its right. The
new primitive is `atomWitness_off_block` (an atom's witness writes only inside
its atom), matching `hornerWitness_off_block` one level up; the justification at
each step is `productRows_conservation`, again reused rather than re-derived.

### What remains

The four tail blocks, the product and the equality — three more Horner
extensions in the same shape, then the multiplication. The `KEquality` part is
the only one that is not more of the same: its rows are equalities, not writes,
so the honest assignment has to make the two sides *equal*. That is
`identityRows_sound`'s equation in reverse, and an honest prover's quotient
satisfies it by `KQuotient.polyEval_ringKMul_quotientForm`.

---

## KQUOTIENTIDENTITY-COST

```text
claim:
  The recipe's Typed.Cost, with every nonzero component a receipt.
status: PROVED 2026-07-27 (cycle 300). identityCost, identityCost_rows,
        identityCost_columns, identityCost_gap. All guarded.
```

```text
identityCost n = { recurringRows := 321n + 482, committedColumns := 0,
                   publicColumns := 0, auxiliaryColumns := 321n + 480 }
```

`identityCost_rows` is `identityRows_length`; `identityCost_columns` is
`identityColumns_length`. Neither number is restated — the theorems *are* the
receipts.

### Why two components are zero

Not a gap: a statement about ownership. This recipe allocates only the
intermediate products of its Horner ladders and multiplications. The vectors it
**reads** — the challenge, the two operand vectors per pair, the output, the
quotient and the modulus — are preallocated inputs, referenced through the
`Carried` lists the caller supplies. Counting them here would double-count them
against whatever recipe allocates them, which is prompt section 4.4's trap
exactly.

`identityCost_gap` records that rows exceed columns by exactly two: the
`K`-equality emits two rows and allocates nothing.

---

## KQUOTIENTIDENTITY-HONEST-PRIMITIVES

```text
claim:
  The whole-program witness, its placement bound, and the preservation lemma
  that lets the honest identity be stated about the caller's assignment.
status: PROVED 2026-07-27 (cycle 301). identityWitness,
        identityWitness_off_block, projected_preserved. All guarded.
```

`identityWitness` is the five writing parts, left to right. The `K`-equality
writes nothing — it is the one part whose satisfaction is a fact about the
*values* rather than about where the witness put them, which is why it will need
the honest identity rather than another placement lemma.

`identityWitness_off_block` says every input the caller placed below `base`
keeps its value. `projected_preserved` turns that into: a projection computed
under the witness equals the projection computed under `z`. Together they let
the honest hypothesis be stated about the caller's assignment rather than about
the constructed one — which matters because the caller is the only party who can
supply it.

### The file split

`KQuotientIdentity.lean` reached 1390 lines and the assembly would have breached
the project's 1500-line cap. The honest-completeness material moved to
`KQuotientIdentityHonest.lean` (1030 + 399 lines).

It **continues the same namespace** rather than opening a new one. This is a
split for size, not a responsibility boundary: the row program and its witness
are one recipe, and renaming nine theorems to record a file boundary would have
been churn with no reader benefit. Guard entries are unchanged; only the guard
file's import list grew.

### The assembly, precisely

Five satisfaction facts and ten preservation steps:

| part | satisfied at | preserved through |
|---|---|---|
| `pairsRows` | `w₁` | `w₂ w₃ w₄ w₅` |
| output block | `w₂` | `w₃ w₄ w₅` |
| quotient block | `w₃` | `w₄ w₅` |
| modulus block | `w₄` | `w₅` |
| the product | `w₅` | — |

Every preservation step is `satisfies_extend` with two inputs already proved: a
bound on what the part's rows mention (conservation) and an agreement below the
next block's base (`hornerWitness_off_block`, `witness_off_frame`). The chain of
bases is `base ≤ outBase ≤ quotientBase ≤ modulusBase ≤ frame`, so an earlier
part's bound clears every later stage at once.

The sixth part, `KEquality.rows_complete`, needs
`carriedValue w₅ (pairsCarried …) = carriedValue w₅ (concatCarried …)`, which is
`identityRows_sound`'s equation read in the other direction and supplied by
`projected_preserved` from the caller's honest identity.

---

## KQUOTIENTIDENTITY-RECIPE — ten of ten

```text
status: COMPLETE against the section 2 checklist 2026-07-27 (cycle 302),
        for the recipe's own inputs. See the scope note below.
```

| item | where |
|---|---|
| constructive row program | `identityRows` |
| derived row count | `identityRows_length`, `identityRows_length_production` |
| exact row ownership | the six-part append; each part's rows come from one gadget |
| exact column ownership, no collision | `identityColumns_nodup`, `atoms_disjoint`, `atom_inside`, `tail_blocks_separated`, `tail_inside` |
| conservation | `identityRows_conservation` |
| soundness | `identityRows_sound` |
| honest completeness | `identityRows_honest` |
| `Typed.Cost` | `identityCost` + `identityCost_rows` + `identityCost_columns` |
| fail-closed axiom guard | `tests/Axioms/CanonicalKQuotientIdentity.lean` |
| spec and ledger entry | this file; cycles 295–302 |

### The scope note, which is the whole caveat

Every theorem above quantifies over the `Carried` lists the **caller supplies**.
The recipe is sound, conservative and complete about *those*. What does not
exist is the decoder from a frozen `ProjectionProgram.ProjectionTrace` to those
lists. Until it does:

- the sizing hypotheses (`output.length = 54` and friends) have no constructor
  from production data;
- `identityRows_sound`'s conclusion is an equation about projections, not
  `ProjectionCheck.Accepted`;
- `KBatch`'s `agrees` premise stays moved rather than closed.

So this is a complete recipe for a program, not yet a complete encoding of the
frozen check. Saying otherwise would be the "premise that moved rather than
closed" trap one level up.

### What the assembly cost

Five satisfaction facts, ten preservation steps, and one value fact. The
preservation steps were free in the sense that mattered: each is
`satisfies_extend` fed by a conservation bound and an off-block agreement, both
already proved. The value fact — the `KEquality` part — is the only one that
needed anything new, and what it needed was `projected_preserved`, so that the
honest identity could be a hypothesis about the caller's assignment rather than
about the constructed witness.

---

## KTRACEDECODER

```text
claim:
  A frozen projection trace's coefficient columns decode into the recipe's
  carriers, with the recipe's sizing hypotheses following from the trace's
  layout validity.
status: PROVED 2026-07-27 (cycle 304). Decoding, lengths, denotation,
        placement and the evaluation bridge. All guarded.
```

### What closed

`ProjectionTrace.LayoutValid` already pins `outputColumns.length = 54`,
`quotientColumns.length = 53` and `maxDegree = 106`. `decodeVector` preserves
length, so `decoded_output_sized` and `decoded_quotient_sized` derive two of the
recipe's sizing hypotheses **from the trace being well-formed** rather than from
a caller's promise. `decodeModulus_length` gives the third.

`decodeVector_belowBase` reduces the recipe's `BelowBase` hypotheses to
`∀ column ∈ columns, column < base` — a checkable property of the layout rather
than an assumption about combinations. `decodeModulus_belowBase` needs only
`0 < base`, since the modulus carriers mention nothing but the constant wire.

### Base-field coefficients in a K-valued vector

The trace's coefficient columns hold base-field elements; the recipe's carriers
are `K`-valued. Production embeds with `K.ofBase`, so a decoded coefficient is
the column in the low coordinate and **nothing** in the high one.
`decodeBase` is asymmetric for that reason, and it is not an approximation:
`carriedValue_decodeBase` proves the decoded carrier denotes exactly
`KBridge.toPair (K.ofBase (baseAt z column))`, high coordinate genuinely zero.

### The evaluation bridge — done (cycle 304)

`projected_decodeVector` and `projected_decodeModulus`. Both fell out of
`KBridge.toPair_eval`, which already said the canonical Horner reference
computes what `ProjectionCheck.eval` computes; the decoding only had to put the
coefficient lists in the same form.

The modulus version needs `z 0 = 1`. That is not bookkeeping: `Φ₈₁`'s carriers
are constants written on the constant wire, so without the hypothesis a prover
could set column 0 freely and the modulus would evaluate to something else.

### A search that was wrong, corrected in the same cycle

The module header first said the `Polynomial.eval` algebra needed to reach
`ProjectionCheck.Accepted` was missing. It is not: `eval_add`, `eval_scale`,
`eval_mul`, `eval_sum` and `eval_padRight` are all already proved in
`Core/Projection/Polynomial.lean`. The claim was written before grepping for
them — prompt section 4.2's trap, caught and corrected before reporting.

What is actually left is narrower: injectivity of `KBridge.toPair` (to get from
a `Pair` equation back to a `K` one), and the assembly.


---

## KTRACEDECODER-FROZEN-EVAL

```text
claim:
  The recipe's equation is the frozen check's evaluation component, for the
  trace's own identity.
status: PROVED 2026-07-27 (cycle 305). equation_reaches_frozen_eval, with
        KBridge.toPair_injective, pairSum_toPair, eval_identity_lhs and
        eval_identity_rhs. All guarded.
```

### The trip back

The recipe's equation lives in `Pair`; `ProjectionCheck.Accepted` lives in `K`.
`KBridge.toPair_injective` is what makes the return trip possible — without it
the row layer could only push facts forward, never conclude anything about the
frozen relation. It is two `Fin.ext`s.

The rest is bookkeeping over algebra that already existed: `eval_sum` and
`eval_mul` reduce the frozen left-hand side to a fold of products; `eval_add`,
`eval_mul` and `eval_padRight` reduce the right-hand side to
`q(beta)·Phi81(beta) + out(beta)`; `pairSum_toPair` matches the recipe's sum to
the frozen fold. The one place the two sides genuinely differ is the order of
the final addition, which `addPair_comm` fixes.

### The other half of `Accepted` — CLOSED 2026-07-27 (cycle 306)

> `WellFormed` did not need to be proved: `ProjectionTrace.identity_wellFormed_of_widths`
> was already in `Core/ProjectionLengths.lean`, along with `length_add`,
> `length_mul`, `length_sum_eq` and `length_padRight`. Cycle 305 recorded a
> caution to *check* rather than assume, and checking is what found it. See
> `KTRACEDECODER-FROZEN-RELATION`.


---

## KTRACEDECODER-FROZEN-RELATION

```text
claim:
  The emitted row program's equation gives ProjectionCheck.Accepted, and
  therefore coefficient-exactness or the frozen BadRoot event.
status: PROVED 2026-07-27 (cycle 306). accepted_of_equation and
        exact_or_badRoot_of_equation, both guarded.
```

`Accepted` is `WellFormed ∧ eval lhs beta = eval rhs beta`. Cycle 305 supplied
the second conjunct; the first turned out to be already proved upstream, so this
cycle is a composition rather than a construction.

`exact_or_badRoot_of_equation` is the destination the canonical track has been
walking toward since cycle 291: **the frozen soundness statement, reached from a
row program whose every count, column, witness and cost was derived in Lean**.
`BadRoot` is the only escape and it is `SuperNeo.ProjectionCheck`'s own event,
not one invented for this path.

### The pattern worth naming

Three of the last four cycles ended with "the thing I planned to prove already
existed" — `KBridge.toPair_eval` (304), the `Polynomial.eval` algebra (304),
`identity_wellFormed_of_widths` (306). Once (304) that cost a wrong claim in a
module header, written before grepping. The habit that fixed it — record the
caution to check, then check — is cheap and has now paid twice.

### The batch — CONDITIONALLY BRIDGED 2026-07-27 (cycle 307; corrected cycle 308)

`KTraceDecoder.batchAccepted_of_traces` constructs `BatchAccepted` for the
identities of a list of frozen traces from an explicit `TraceAccepts` hypothesis
for every trace. `TraceAccepts.equation` is the quotient equation itself; it is
not yet constructed from satisfaction of the emitted NIFS rows.
`batchExact_or_badRoot_of_traces` is therefore a conditional transport theorem,
not an unconditional operational soundness theorem.


---

## KTRACE-PUBLIC-PIRLC-OCCURRENCE

```text
claim:
  One selected artifact-free public PiRLC occurrence constructs its exact
  traces, emitted quotient rows, row subtotal, and occurrence-bound event.
status: MODEL-PROVED 2026-07-27. `KTraceProgram.{rows_length,
        batchAccepted_of_rows,Occurrence.exact_or_badRoot}` and
        `KPiRlcTrace.{occurrence,occurrence_rows_length,
        occurrence_exact_or_badRoot}`. All guarded.
```

Cycle 308 remains the historical correction to cycle 307. The new selected
program closes the obligation cycle 308 left open: `KTraceProgram.rows`
instantiates `KQuotientIdentity.identityRows` for the minimal canonical traces,
and satisfaction constructs `BatchAccepted` without an external
`TraceAccepts`. `KPiRlcTrace` constructs the public-role trace batch directly
from decoded NIFS coefficient columns. Its exact row subtotal is

```text
(23 + 2 * matrixCount) * (321 * arity + 482).
```

This is not the complete `nifsVerify` cost. It excludes PiCCS, PiDEC,
transcript, point-binding, accumulator, residual, and call-framing rows.

The event is closed over the occurrence's own identity list.
`KTraceBadRootFixture.occurrence_badRoot_of_satisfied_rows` supplies a
satisfying emitted-row witness taking that exact event branch, while
`not_eventFreeOccurrenceSoundness` rules out silently restoring the current
exact-only soundness shape.

Two independent boundaries remain kernel-checked:

- `NifsCompletionBoundary.publicOccurrence_does_not_determine_completeNifs`
  proves that the public occurrence alone cannot determine a setup-selected
  HyperNova verifier result. A complete recipe must construct every omitted
  verifier program.
- `DeploymentSelectionBoundary.fixedProfile_does_not_select_step_or_nifs_cost`
  proves that the fixed call footprints leave the setup-owned `step` and
  `nifsVerify` costs independent.

Therefore the event cannot yet be propagated through `CallRecipe`, branches,
and Step: there is no complete call occurrence to bind it to, and the current
contract demands an unconditional output. Likewise `allRecipes`, a unique
numeric deployment cost, and Rust replacement remain open. Constructing any of
them now would invent either the omitted verifier rows or a deployment
application.

---

## NIFSCOMPLETIONBOUNDARY-VACUITY

```text
claim:
  publicOccurrence_does_not_determine_completeNifs proved nothing about the
  projection occurrence and is withdrawn.
status: WITHDRAWN 2026-07-27 (cycle 310). Replaced by
        setupVerifier_is_a_real_choice, guarded.
```

### What was wrong

The theorem read

```text
¬ ∃ decode : Occurrence → Option Unit,
    ∀ verifier, decode occurrence = verifier.verify () () () ()
```

and its docstring said "the public projection occurrence is not a complete NIFS
program". The `∀ verifier` sits **inside**, with `decode occurrence` already
fixed, so the contradiction is only that one value cannot equal two different
values.

Checked rather than argued: the identical statement and proof were re-derived
over an arbitrary type,

```text
theorem holds_for_anything {A : Type} (a : A) : ¬ DeterminesEveryVerifier a
theorem holds_even_when_type_contains_the_verifier
    (v : Verifier Unit Unit Unit Unit) : ¬ DeterminesEveryVerifier v
```

Both compile. The statement would equally "prove" that a value which *is* the
verifier is not a complete NIFS program.

### The structural lesson

A statement of the form "X does not determine Y", where X and Y are independent
inputs, is provable for **every** X and therefore says nothing about any of
them. Content requires holding something fixed.

`DeploymentSelectionBoundary.fixedProfile_does_not_select_step_or_nifs_cost` is
the non-vacuous form of the same idea: it holds all eight fixed call footprints
equal and varies only `step` and `nifsVerify`. That theorem earns its
conclusion; the withdrawn one did not.

### What replaced it

`setupVerifier_is_a_real_choice`: two legitimate HyperNova verifiers over the
same carriers give opposite results. That is a fact about the verifiers, stated
as such, and it is the honest reason `nifsVerify` must be *selected*. Nothing
about the projection occurrence is claimed.

The conclusion the withdrawn theorem asserted is very probably true. It is not
proved, and no theorem in the tree now asserts it.

---

## PIDEC-PROFILE-AUDIT

```text
claim:
  What Pi_DEC's verifier checks, and which parts are fixed by protocol
  constants versus parameterised by the application.
status: AUDITED 2026-07-27 (cycle 314) from
        crates/neo-fold-clean/src/paper/reductions/pi_dec.rs::verify.
        No Lean row program written.
```

### The thirteen checks, in order

| # | check | shape fixed by |
|---|---|---|
| 1 | `validate_child_count` — exactly `k_rho` children | **constant** `K_RHO = 14` |
| 2 | `validate_fold_digest_canonical`, parent and each child | constant |
| 3 | `validate_r_shape` — `r.len() = log2(next_pow2(n).max 2)` | **application** (`s.n`) |
| 4 | `validate_y_ring_shape` | application (structure) |
| 5 | `validate_inactive_x_zero` — `X` zero on `[ceil(m_in/D), cols)` | application (`m_in`), `D = 54` |
| 6 | `validate_child_x_low_norm` — centered alphabet | **constant** `B_BASE = 2` |
| 7 | `validate_supported_sidecars` | constant |
| 8 | `validate_s_col_shape` | application |
| 9 | `validate_s_col_consistency` | — |
| 10 | `validate_ct_consistency` | — |
| 11 | `validate_y_ring_padding_zero` | see the carrier-270 note below |
| 12 | `validate_fold_digest_consistency` | — |
| 13 | `validate_adv_recomposition` — base-`b` recomposition under the `DecMixer` | **constant** `B_BASE = 2`, `K_RHO = 14` |

then `engine::verify_pi_dec` — the core algebraic check.

### What this means for construction

Π_DEC is mostly protocol-constant-shaped. Only `r.len()`, `m_in` and the
structure-derived shapes are application data, so the row program can be built
**parameterised on `ell` and `m_in`** exactly as `KQuotientIdentity` was built
parameterised on `arity` and `matrixCount`. It does **not** need the application
choice to start.

### A trap to carry in

Check 11 is `validate_y_ring_padding_zero`. The recorded finding
`carrier-270-pirlc-non-closure` says the thirteen "padding" coordinates are ring
lanes 41–53 and are **not** inert. Any Lean encoding of this check must treat
them as live; assuming padding is inert is the exact error that finding names.

### Scale

Thirteen checks plus a core algebraic verify. Π_RLC took roughly thirteen cycles
to reach all ten checklist items. Π_DEC should be budgeted comparably, and it is
one of four sub-programs (Π_CCS, Π_RLC, Π_DEC ×2, transcript).

---

## ARITH-GOLDILOCKS-CERTIFICATE

```text
claim:
  The Lucas certificate arithmetic for the Goldilocks modulus, kernel-checked.
status: PROVED 2026-07-27 (cycle 315). Nine theorems, all guarded, all with
        EMPTY axiom sets. This is the certificate data only, NOT primality.
supports: ARITH-GOLDILOCKS-FIELD (still `planned`).
```

`7` has multiplicative order exactly `q - 1` modulo `q = 2^64 - 2^32 + 1`:
`7^(q-1) = 1`, and `7^((q-1)/p)` is a named residue other than `1` for each
prime `p` in the full factorisation `q - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537`.

Every theorem is closed by `decide`, so reduction happens in the kernel, and
`#print axioms` reports **no axioms at all** — not even `propext`. No
`native_decide`.

### The fail-closed design, and the bug that forced it

`powMod` recurses on a fuel counter so the kernel can reduce it structurally.
Fuel exhaustion returns `0`, a **poison** value.

Every condition is therefore stated as an *exact residue*, never as `≠ 1`. Under
a `≠ 1` formulation an exhausted fuel budget returns `0`, and `0 ≠ 1` holds — so
an under-fuelled call would pass while computing nothing.

This is not a hypothetical. The first draft of the probe had the arguments
swapped (`fuel = 7`, `base = 70` against a 64-bit exponent), hit the exhaustion
branch, and returned `1` — precisely the value a Fermat test accepts. It was
caught only because `#eval` then reported `7` a quadratic residue, which
contradicts an independent computation. A `= 2` negative control had already
"passed" and proved nothing, because it showed only that `decide` runs.

The exact-residue formulation makes the poison value fail every condition.

### What this does not establish

Not primality, and not `EuclidPrime goldilocksP`. The remaining step is Lucas's
theorem: order `q - 1` gives `q - 1` distinct powers of `7`, hence every nonzero
residue is a unit, hence a proper factorisation is impossible. That derivation
needs no Mathlib but it is not written, and until it is, the 297 occurrences of
`EuclidPrime goldilocksP` remain typed hypotheses.

---

## ARITH-GOLDILOCKS-LUCAS-COST

```text
claim:
  What the Lucas derivation from ARITH-GOLDILOCKS-CERTIFICATE actually costs.
status: ASSESSED 2026-07-27 (cycle 316). Route open, not blocked, but it is
        foundational library work rather than protocol work.
```

### The derivation, in full

From `ord(7) = q - 1`: the powers `7^0 … 7^(q-2)` are `q - 1` distinct residues,
each a unit (since `7^k · 7^(ord-k) ≡ 1`). Units are a subset of `{1,…,q-1}`,
which has exactly `q - 1` elements, so **every** element of `{1,…,q-1}` is a
unit. If `q = a·b` with `1 < a < q` then `a` is a unit, so `a·c ≡ 1`, so
`b ≡ b·a·c ≡ (a·b)·c ≡ 0`, contradicting `1 < b < q`. Hence `q` is prime, and
Euclid's lemma follows.

### The one expensive step

"`q - 1` distinct elements inside a set of size `q - 1`, therefore the set is
exhausted" — an injective map between finite sets of equal cardinality is
surjective.

The tree has **no** `Finset`, no `Fintype.card`, and no pigeonhole lemma; every
`List.Nodup` use in it is list-level and local. Searched by concept and by name
across `Nightstream/`, not by one filename. So this step has to be built.

It cannot be `decide`d either: the set has ~1.8 × 10^19 elements, so it must be
a genuine proof, not a kernel computation. That is the opposite of the
certificate arithmetic, which reduces in 0.19 s.

### Verdict

The route is **open, not blocked** — pigeonhole is strictly easier than
primality, so this is not a reduction to a statement of comparable strength.
But it is a few hundred lines of general-purpose finite-cardinality development
with no protocol content, and it sits off the Phase 4 critical path.

Recorded so the next iteration prices it before committing, rather than
discovering the cost partway in.

### The standing alternative

Keep `EuclidPrime goldilocksP` as a typed hypothesis, which is what all 297
occurrences already do. Every soundness theorem downstream stays conditional on
a field fact that is assumed everywhere and proved nowhere — an accurate
description of the tree today, and one `ARITH-GOLDILOCKS-FIELD` already records
at `planned`.

---

## PIDEC-LOWNORM-ROWS

```text
claim:
  The emitted row program for Pi_DEC's b = 2 low-norm check, with derived
  count and column allocation.
status: COMPLETE against the ten-item checklist 2026-07-27 (cycle 321).
```

`validate_child_x_low_norm` requires every active packed entry of a child's `X`
to satisfy `within_nc_bound(v, b)`. The deployment fixes `b = 2` — the params
module is named `goldilocks_paper_b2` — so the window is `{-1, 0, 1}` and the
constraint is `x·(x-1)·(x+1) = 0`, equivalently `x³ = x`.

`b = 2` is a **protocol constant**, so this atom needed no application choice.

### The receipt

Two rows, one auxiliary column. `x · x = s` then `s · x = x`; the second row
writes back to the operand, which is a shared read rather than an allocation.
Both counts come from the emitted list.

### The algebra was already there

`PaperJoint.NormRange` already proves
`cubicResidual_eq_zero_iff_strictNormTwo` — the roots of `(z+1)z(z-1)` are
exactly the strict centered window, with `representedRoots_nodup` giving the
three canonical residues `q-1, 0, 1`. Checking before building is what found
it; the job here was the row program and the bridge, not the mathematics.

### What is missing, precisely

Only the translation. Soundness here is `lcEval`-over-`Nat`: satisfaction forces
`x³ ≡ x` and needs **no field premise**. Concluding `x ∈ {-1,0,1}` is
`NormRange`'s theorem and carries `BaseFieldNoZeroDivisors`, derivable from
`EuclidPrime goldilocksP` via `baseFieldNoZeroDivisors_of_modulusEuclid`.

So this atom adds no *new* premise to the tree — it inherits the one that 297
occurrences already carry, and `ARITH-GOLDILOCKS-CERTIFICATE` is the work
toward discharging it.


---

## PIDEC-LOWNORM-HONEST

```text
claim:
  An in-window value satisfies both low-norm rows.
status: PROVED 2026-07-27 (cycle 318). lowNormRows_honest, guarded.
```

The hypothesis of `lowNormRows_honest` is *exactly* the conclusion of
`lowNormRows_sound` — `x³ ≡ x` in the row layer's `Nat` vocabulary. So the check
is complete for precisely the values it accepts: no gap in either direction, and
no separate characterisation of the accepted set that could drift from the
soundness statement.

### Freshness is load-bearing

`lowNormRows_honest` requires `¬ Mentions value squareColumn`. That is not
bookkeeping. If the checked combination read its own square column, writing the
square would change the value being squared, and neither row need hold. The
witness is only honest for a combination that does not observe the column the
witness writes.

This is the same shape as `KHornerHonest`'s freshness obligations, and it is why
`lcEval_lowNormWitness` is stated separately: it is the step that consumes
freshness, and everything after it is arithmetic.

### Remaining for this atom

The bridge to `Concrete.F`, conservation, and a `Typed.Cost`. Seven of the ten
checklist items are now met.


---

## PIDEC-LOWNORM-BRIDGE

```text
claim:
  A row-layer residue whose cube is itself is a root of the frozen cubic.
status: PROVED 2026-07-27 (cycle 320). cubicResidual_eq_zero_of_cube, guarded.
```

Composing with `NormRange.cubicResidual_eq_zero_iff_strictNormTwo` carries
`lowNormRows_sound` to the frozen root classification — the strict `b = 2`
centered window. That is the soundness-to-the-frozen-relation obligation for
this atom.

### How it was done without guessing

`(NormRange.cubicResidual ⟨x, h⟩).val` reduces by `rfl` to

```text
((x + 1) % n * x % n) * ((n - 1 + x) % n) % n
```

That normal form was established by *probing* — stating it and checking `rfl`
— rather than derived on paper and then fought with. With the form known, the
proof is: strip the inner reductions with `Nat.mul_mod`, then apply
`cubic_expansion_multiple` at `m = n - 1`.

The subtraction-free expansion from cycle 319 is what makes the last step
mechanical; `Fin`'s `-1` is the complement `n - 1`, and without that lemma the
goal is a thicket of `Nat` subtraction.

### The premise it inherits

The final composition carries `BaseFieldNoZeroDivisors`, derivable from
`EuclidPrime goldilocksP`. No *new* premise is introduced — this atom inherits
the one already carried on 297 occurrences, and
`ARITH-GOLDILOCKS-CERTIFICATE` is the work toward discharging it.


---

## PIDEC-LOWNORM-RECIPE — ten of ten

```text
status: COMPLETE against the section 2 checklist 2026-07-27 (cycle 321).
```

| item | where |
|---|---|
| constructive row program | `lowNormRows` |
| derived row count | `lowNormRows_length` (2) |
| exact row ownership | the two-element emitted list |
| exact column ownership, no collision | `lowNormColumns_length` (1), `lowNormColumns_nodup` |
| conservation | `lowNormRows_conservation` |
| soundness to the frozen relation | `lowNormRows_sound` + `cubicResidual_eq_zero_of_cube` + `NormRange.cubicResidual_eq_zero_iff_strictNormTwo` |
| honest completeness | `lowNormRows_honest` |
| `Typed.Cost` | `lowNormCost` + `lowNormCost_rows` + `lowNormCost_columns` |
| fail-closed axiom guard | `tests/Axioms/CanonicalKLowNorm.lean` |
| spec and ledger entry | this file; cycles 317–321 |

### Two notes on the cost

`committedColumns` and `publicColumns` are zero because the checked value is a
**read**: the entry being range-checked belongs to whatever recipe allocated it
— here Π_DEC's child claim — and counting it again would double-count, which is
prompt section 4.4's trap.

Conservation reaches the *constant wire* in no row. Unlike `KEquality`, neither
of these rows carries a literal, so the `column = 0` arm that
`identityRows_conservation` needed does not arise here. That asymmetry is worth
noticing rather than copying the earlier statement by habit.

### Scope

This is one of Π_DEC's thirteen checks. The other twelve — child count, fold
digest canonicity and consistency, `r` shape, `y`-ring shape and padding,
inactive-`X` zero, sidecars, `s_col` shape and consistency, `ct` consistency,
and recomposition — are not written.

---

## PIDEC-YRINGPADDING-CHECK

```text
claim:
  The check-level row program for Pi_DEC's y_ring padding obligation, with a
  folded cost.
status: PROVED 2026-07-27 (cycle 324). paddingRows and its receipts, guarded.
```

`validate_y_ring_padding_zero` ranges over every lane past `D = 54` in every
`y_ring` row. The check-level program is the per-lane atom concatenated.

### The cost is a fold, deliberately

`paddingRows_length` is stated as a fold over per-lane receipts, with
`paddingRows_length_eq` evaluating it to `2 * lanes.length` separately. The
number of padded lanes is a property of the claim's shape, not a protocol
constant, so leading with a closed formula would be a subtotal presented as a
total — the same reason `KBatch` folds rather than assuming a uniform degree.

### Two rows per lane, nothing allocated

A `K` zero is two physical rows, and no lane allocates a column, so the
auxiliary component stays zero however many lanes there are. That is why the
cost has a variable row count and a constant zero column count — worth stating,
because a recipe whose column count grew with its input would need a layout and
this one does not.

### Scope

This is the row program for one of Pi_DEC's thirteen checks. Enumerating *which*
lanes are padded from a claim's shape is decoder work and is not written here.

---

## PIDEC-CHECK-CLASSIFICATION

```text
claim:
  Which of Pi_DEC's validators are row obligations and which are decoder-side
  shape tests.
status: DETERMINED 2026-07-27 (cycle 326) by reading each validator body.
```

Five of the thirteen validators are **pure length assertions**:

| validator | test |
|---|---|
| `validate_child_count` | `children.len() == k_rho` |
| `validate_r_shape_one` | `claim.r.len() == expected` |
| `validate_y_ring_shape_one` | `claim.y_ring.len() == s.t()` |
| `validate_s_col_shape_one` | `claim.s_col.len() == split_nc_column_point_len(s.m)` |
| `validate_ct_consistency_one` (first half) | `claim.ct.len() == claim.y_ring.len()` |

These are **decoder-side**, like `KProjectionTrace.Trace.Valid`. List lengths are
structural facts about the decoded witness layout, not field constraints.
**Emitting rows for them would fabricate constraints the protocol does not
have** — prompt section 3's fabricated-recipe trap.

### The re-accounting

"Thirteen checks" overstates the row work. The genuine row obligations are:

| obligation | status |
|---|---|
| `inactive_x_zero` | built (`KZeroCheck`, ten of ten) |
| `child_x_low_norm` | built (`KLowNorm`, ten of ten) |
| `y_ring_padding_zero` | built (`KZeroCheck.paddingRows`) |
| `s_col_consistency` | built (`KConsistency`) |
| `ct` value equality | built (`KConsistency`) |
| `fold_digest_canonical` / `_consistency` | not built — needs Poseidon2 |
| `adv_recomposition` | not built — needs the Ajtai commitment layer |
| `supported_sidecars` | unexamined |
| `engine::verify_pi_dec` | read in full; see below |

### The core is elsewhere

`engine::verify_pi_dec` delegates to `neo_reductions::api::dec::verify_dec_public`,
which re-checks `r` agreement across children, bounds `m_in`, and then performs
the actual decomposition algebra. **Enumerating the `validate_*` helpers was
enumerating the guards, not the check.** Any claim about Π_DEC's row cost that
counts only the validators is a subtotal presented as a total.

### Superseded 2026-07-27 (cycle 327)

`verify_dec_public` has now been read in full. Its algebraic content is a single
relation — `parent = Σ_i b^i · childᵢ` — instantiated on four carriers, and it
is owned by `PIDEC-RECOMPOSITION` in `specs/pidec-recomposition.md`. The
correction to the line above is one of *size*, not direction: the core is real,
and it is smaller than its guard surface.
