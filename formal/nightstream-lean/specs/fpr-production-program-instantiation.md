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
Mechanical. Separately, `lake build Nightstream` has **36 pre-existing errors**
in `ProductionPaperOuter`, `PrivateDecoder/Coverage` and
`PiDecStrictProductionCompiler/Differential`, none of which reach `CallRecipe`.
`lake build tests.Axioms` does not cover them, so the aggregate gate being green
is not evidence that the whole library builds.

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
