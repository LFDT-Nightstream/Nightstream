import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality

/-!
Contract: record that `runningCheck` and `freshCheck` are setup selections, so
no row program for them can be derived from this encoding.

## The result is an obstruction, and it reaches the interface

`Vocabulary.callEval` sends `Call.runningCheck` and `Call.freshCheck` to
`parameters.terminalChecks.runningCheck` and `.freshCheck`. Those are **fields**
of `CanonicalTerminalVerifier.RelationChecks`, a structure whose `_iff` fields
tie them to `TerminalRelations.runningHolds` and `.freshHolds` — themselves
setup-supplied `Prop`s.

So the pair `(relations, checks)` is an input, not a derivation. This module
exhibits two legitimate inhabitants over the same carriers that disagree on the
same argument, which is a statement about the checks rather than about anything
the encoding could have computed.

`NifsCompletionBoundary.setupVerifier_is_a_real_choice` is the same fact for
`nifsVerify`. The three of them plus `step` are the encoding's selection
surface.

## Why this is not the vacuous shape

`NifsCompletionBoundary` records a theorem it withdrew: "X does not determine Y"
for independent inputs X and Y is provable for every X and says nothing. The
statements here are not of that shape. They fix everything except the checks and
compare two concrete inhabitants at one concrete argument, so what varies is
exactly the object the claim is about.

## What a reader should know before trusting a future recipe

The one place in this tree where these checks are instantiated against native
F′ conformance is `Implementation/Rust/CanonicalConformance/OneSlot.lean`, and
there `runningCheck` is `receipt.accepted` conjoined with a field match — a
**carried verdict**, not a recomputation. That is recorded as
`TERMINAL-CHECK-RECEIPT-CARRIED` rather than encoded around, because the project
rule is that digests compress but never authorise, and a check that returns a
carried `accepted` bit is on the wrong side of that line for anything that
crosses a trust boundary.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary

open Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
open Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
open Nightstream.HyperNova.Construction2.Paper

/-- Terminal relations that hold of everything. -/
def acceptingRelations :
    TerminalRelations Unit Unit Unit Unit Unit 1 where
  runningHolds := fun _ _ _ _ => True
  freshHolds := fun _ _ _ _ => True

/-- Terminal relations that hold of nothing. -/
def rejectingRelations :
    TerminalRelations Unit Unit Unit Unit Unit 1 where
  runningHolds := fun _ _ _ _ => False
  freshHolds := fun _ _ _ _ => False

/-- A legitimate setup-selected checker: everything passes. -/
def acceptingChecks : RelationChecks acceptingRelations where
  runningCheck := fun _ _ _ _ => true
  freshCheck := fun _ _ _ _ => true
  runningCheck_iff := by simp [acceptingRelations]
  freshCheck_iff := by simp [acceptingRelations]

/-- A legitimate setup-selected checker over the same carriers: nothing
passes. -/
def rejectingChecks : RelationChecks rejectingRelations where
  runningCheck := fun _ _ _ _ => false
  freshCheck := fun _ _ _ _ => false
  runningCheck_iff := by simp [rejectingRelations]
  freshCheck_iff := by simp [rejectingRelations]

/-- A lawful relation in which only the running unary relation holds. -/
def runningOnlyRelations :
    TerminalRelations Unit Unit Unit Unit Unit 1 where
  runningHolds := fun _ _ _ _ => True
  freshHolds := fun _ _ _ _ => False

/-- Exact executable checks for `runningOnlyRelations`. -/
def runningOnlyChecks : RelationChecks runningOnlyRelations where
  runningCheck := fun _ _ _ _ => true
  freshCheck := fun _ _ _ _ => false
  runningCheck_iff := by simp [runningOnlyRelations]
  freshCheck_iff := by simp [runningOnlyRelations]

/-- A lawful relation in which only the fresh unary relation holds. -/
def freshOnlyRelations :
    TerminalRelations Unit Unit Unit Unit Unit 1 where
  runningHolds := fun _ _ _ _ => False
  freshHolds := fun _ _ _ _ => True

/-- Exact executable checks for `freshOnlyRelations`. -/
def freshOnlyChecks : RelationChecks freshOnlyRelations where
  runningCheck := fun _ _ _ _ => false
  freshCheck := fun _ _ _ _ => true
  runningCheck_iff := by simp [freshOnlyRelations]
  freshCheck_iff := by simp [freshOnlyRelations]

/-- **`runningCheck` is a real choice.**

Two legitimate `RelationChecks` over the same carriers give opposite verdicts on
the same argument. A recipe for `runningCheck` must therefore *select* one; no
amount of encoding-side reasoning derives it.

Scope, precisely: this is a statement about the two checkers. It does not say
the terminal program is incomplete, and it does not license leaving the recipe
unbuilt once a selection exists. -/
theorem runningCheck_is_a_real_choice :
    acceptingChecks.runningCheck ⟨0, by decide⟩ () () ()
      ≠ rejectingChecks.runningCheck ⟨0, by decide⟩ () () () := by
  decide

/-- **`freshCheck` is a real choice.**  Same statement, other relation. -/
theorem freshCheck_is_a_real_choice :
    acceptingChecks.freshCheck ⟨0, by decide⟩ () () ()
      ≠ rejectingChecks.freshCheck ⟨0, by decide⟩ () () () := by
  decide

/-- The running relation can be valid while the fresh relation is invalid.
This mutation isolates the running branch. -/
theorem running_valid_fresh_invalid :
    runningOnlyChecks.runningCheck ⟨0, by decide⟩ () () () = true /\
      runningOnlyChecks.freshCheck ⟨0, by decide⟩ () () () = false := by
  decide

/-- The fresh relation can be valid while the running relation is invalid.
This mutation isolates the fresh branch. -/
theorem fresh_valid_running_invalid :
    freshOnlyChecks.runningCheck ⟨0, by decide⟩ () () () = false /\
      freshOnlyChecks.freshCheck ⟨0, by decide⟩ () () () = true := by
  decide

/-- A valid final NIFS output does not imply the fresh unary terminal
relation. The NIFS verifier accepts this exact input and returns its running
output, while the independent fresh check rejects the same terminal
candidate. -/
theorem nifs_accepts_while_fresh_terminal_is_false :
    FixedOne.Minimality.Model.setup.nifs.verify () () false () = some () /\
      FixedOne.Minimality.Model.relationChecks.freshCheck
          selected () false true =
        false := by
  decide

/-- **The disagreement survives the `_iff` fields.**

Both checkers are fully lawful — each is tied to its own relations by
`runningCheck_iff` — so the disagreement is not an artefact of an unconstrained
field. It is the relations that differ, and the relations are the input. -/
theorem lawful_checkers_still_disagree :
    (acceptingChecks.runningCheck ⟨0, by decide⟩ () () () = true
      ↔ acceptingRelations.runningHolds ⟨0, by decide⟩ () () ())
    ∧ (rejectingChecks.runningCheck ⟨0, by decide⟩ () () () = true
      ↔ rejectingRelations.runningHolds ⟨0, by decide⟩ () () ())
    ∧ acceptingChecks.runningCheck ⟨0, by decide⟩ () () ()
        ≠ rejectingChecks.runningCheck ⟨0, by decide⟩ () () () :=
  ⟨acceptingChecks.runningCheck_iff _ _ _ _,
    rejectingChecks.runningCheck_iff _ _ _ _,
    runningCheck_is_a_real_choice⟩

end Nightstream.Implementation.R1CS.Canonical.TerminalCheckSelectionBoundary
