import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionRound

/-!
Exact algebraic view of the production quartic combined-NC round.

Assurance tier: model-level over the concrete five-coefficient materialized
round.

Owns: identification of the concrete production acceptance predicate with
the generic constant-first quartic claimed-round equations.

Does not own: transcript transport, generated-row refinement, terminal
semantics, SumCheck soundness, parent or raw-child authority, costs, or row
removal.

Emits constraints: none.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.QuarticEmbedding

open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-- The two verifier equations for one constant-first quartic message. -/
def ClaimedRoundAccepted
    (coefficients : List K)
    (claimIn challenge claimOut : K) : Prop :=
  coefficients.length = ProductionRound.degree + 1 ∧
    claimIn =
      K.add
        (Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients K.zero)
        (Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients K.one) ∧
    claimOut =
      Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients challenge

/-- The concrete production predicate is exactly the generic quartic
claimed-round predicate instantiated with its five assignment columns. -/
theorem productionAccepted_iff_claimedRoundAccepted
    (assignment : Nat → Nat) :
    ProductionRound.Accepted assignment ↔
      ClaimedRoundAccepted (ProductionRound.coefficientValues assignment)
        (ProductionRound.claimInValue assignment)
        (ProductionRound.challengeValue assignment)
        (ProductionRound.claimOutValue assignment) := by
  constructor
  · intro accepted
    exact ⟨by simpa [ProductionRound.coefficientValues] using
        ProductionRound.coefficient_count,
      accepted.initial, accepted.terminal⟩
  · intro accepted
    exact ⟨accepted.2.1, accepted.2.2⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.QuarticEmbedding
