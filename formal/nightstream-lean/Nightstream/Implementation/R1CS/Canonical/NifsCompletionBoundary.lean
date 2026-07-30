import Nightstream.HyperNova.NonInteractiveMultiFold

/-!
Contract: record that the HyperNova NIFS verifier is a setup-selected input, so
a complete `nifsVerify` recipe must select one.

## A withdrawn theorem, and why

This module previously carried
`publicOccurrence_does_not_determine_completeNifs`:

```text
¬ ∃ decode : Occurrence → Option Unit,
    ∀ verifier, decode occurrence = verifier.verify () () () ()
```

read as "the public projection occurrence is not a complete NIFS program". It
is withdrawn, because it proves nothing about the occurrence. The `∀ verifier`
sits *inside*, with `decode occurrence` already fixed, so the contradiction is
only that one value cannot equal two different values. The identical statement
and proof go through with `Occurrence` replaced by any type at all — including
a type that literally **is** the verifier. Checked, not argued.

The defect is structural, not a matter of wording: a statement of the form "X
does not determine Y", where X and Y are independent inputs, is provable for
every X and says nothing about any of them. Content requires holding something
fixed.

`Encoding.DeploymentSelectionBoundary.footprint_fields_do_not_determine_step_or_nifs_rows`
is a narrower record-independence fact: it holds all eight fixed raw footprint
fields equal and varies only `step` and `nifsVerify`. It is not a statement
about certified deployments.

## What is left

That two legitimate setup-selected verifiers over the same carriers disagree —
a fact about the verifiers. It is why `nifsVerify` must be *selected* rather
than derived, and it is deliberately not dressed up as a statement about the
projection occurrence.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.NifsCompletionBoundary

open Nightstream.HyperNova.NonInteractiveMultiFold

/-- A setup-selected verifier that accepts the unique unit transition. -/
def acceptingVerifier : Verifier Unit Unit Unit Unit where
  verify := fun _ _ _ _ => some ()

/-- A setup-selected verifier with the same carriers that rejects it. -/
def rejectingVerifier : Verifier Unit Unit Unit Unit where
  verify := fun _ _ _ _ => none

theorem acceptingVerifier_result :
    acceptingVerifier.verify () () () () = some () :=
  rfl

theorem rejectingVerifier_result :
    rejectingVerifier.verify () () () () = none :=
  rfl

/-- **The setup-selected verifier is a real choice.**

Two legitimate HyperNova verifiers over the same carriers give opposite results
on the same transition. This is a statement about the verifiers and nothing
else: a complete `nifsVerify` recipe must select one, and no amount of
projection-side reasoning substitutes for that selection.

Scope, precisely: this does **not** say the public PiRLC occurrence is
incomplete. It says the verifier is an independent input. The corresponding
raw-footprint independence fact is
`Encoding.DeploymentSelectionBoundary.footprint_fields_do_not_determine_step_or_nifs_rows`.
It does not replace selection and construction of proof-carrying recipes. -/
theorem setupVerifier_is_a_real_choice :
    acceptingVerifier.verify () () () ()
      ≠ rejectingVerifier.verify () () () () := by
  rw [acceptingVerifier_result, rejectingVerifier_result]
  exact Option.some_ne_none ()

end Nightstream.Implementation.R1CS.Canonical.NifsCompletionBoundary
