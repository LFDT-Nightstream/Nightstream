import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

/-!
Kernel obstruction to treating the typed Fiat--Shamir schedule as a
random-oracle contract.

Owns: one finite oracle that follows the exact typed call interface but
ignores the initial context, every challenge label, and every round message.
Two distinct public contexts therefore derive identical coins.

Does not own: a concrete Poseidon2 encoding, a random-oracle model, a
collision bound, a multi-forking theorem, NIFS soundness, Rust, R1CS,
artifacts, or costs.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

/-- A one-round paper shape is enough to expose context, label, and
round-message omission.  The other dimensions are irrelevant to the
transcript schedule. -/
def shape : Shape where
  cubeVariables := 1
  freshCount := 0
  runningCount := 0
  matrixCount := 0
  coefficientCount := 0

/-- This oracle satisfies the typed interface while discarding every input. -/
def constantOracle : Oracle Bool Bool Unit shape where
  initialState := fun _context => ()
  absorbRound := fun _state _round _message => ()
  squeeze := fun _state _label => (false, ())

/-- A typed public-input absorption call can likewise discard both public
inputs unless a separate random-oracle/refinement contract binds them. -/
def constantPublicInputAbsorber : Unit -> Bool -> Bool -> Unit :=
  fun _state _running _fresh => ()

/-- One syntactically valid finite one-round certificate. -/
def certificate : Certificate Bool shape where
  rounds := fun _round => { coefficients := [false] }

/-- Distinct public contexts receive the same complete derived coin view.
Typed ordering alone therefore cannot discharge transcript binding or the
random-oracle hypothesis. -/
theorem distinct_contexts_same_derived :
    (true : Bool) ≠ false /\
      derive constantOracle true certificate =
        derive constantOracle false certificate := by
  constructor
  · decide
  · rfl

/-- Even distinct semantic labels can be ignored by the abstract squeeze
operation. -/
theorem distinct_labels_same_squeeze :
    let alpha : ChallengeLabel shape := .alpha ⟨0, by decide⟩
    alpha ≠ .gamma /\
      constantOracle.squeeze () alpha =
        constantOracle.squeeze () .gamma := by
  decide

/-- Distinct running/fresh public pairs can reach the same pre-challenge state
through a function satisfying the typed absorption interface. -/
theorem distinct_public_inputs_same_bound_state :
    ((true, true) : Bool × Bool) ≠ (false, false) /\
      constantPublicInputAbsorber () true true =
        constantPublicInputAbsorber () false false := by
  decide

end Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction
