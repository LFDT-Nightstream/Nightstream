import NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedGeometry
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryPlan
import NightstreamFPrime.Layout.Stage1.Poseidon2HashChainCompactWitness

/-! Connect the compact application's rows to its selected step and pilot preimages. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonSoundness

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ApplicationPoseidonRetainedBlock ApplicationPoseidonRetainedGeometry

variable {application : Stage1.Application.Program} {certificate : Certificate application}
  {columns : Nat}

theorem input_form_eq_pilot (geometry : Geometry application certificate columns)
    (lane : Fin 4) :
    (interface geometry).priorState lane =
      (PiRLCPoseidonGeometry.priorInputBlock application).form
        (PiRLCPoseidonGeometry.priorInputStart application)
        (PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry))
        (ApplicationOrdinaryPlan.Location.preimageWord lane) := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCPoseidonGeometry.priorInputStart application + 35 * 41 + lane.val * 41 =
      PiRLCPoseidonGeometry.priorInputStart application + (35 + lane.val) * 41
    omega

theorem output_form_eq_pilot (geometry : Geometry application certificate columns)
    (lane : Fin 4) :
    (interface geometry).digest lane =
      (PiRLCPoseidonGeometry.outputInputBlock application).form
        (PiRLCPoseidonGeometry.outputInputStart application)
        (PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry))
        (ApplicationOrdinaryPlan.Location.preimageWord lane) := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCPoseidonGeometry.outputInputStart application + 35 * 41 + lane.val * 41 =
      PiRLCPoseidonGeometry.outputInputStart application + (35 + lane.val) * 41
    omega

/-- Arbitrary accepted coordinates bind the actual pilot states to the
selected step. No encoding or honest-witness premise is used. -/
theorem rowsZero_implies_step (geometry : Geometry application certificate columns)
    (assignment : Assignment F columns)
    (one : assignment (oneColumn geometry) = 1)
    (rows : (plan geometry).RowsZero assignment) :
    (List.ofFn fun lane => ((interface geometry).digest lane).eval assignment) =
      application.step
        (List.ofFn fun lane => ((interface geometry).priorState lane).eval assignment)
        (List.ofFn fun lane => ((interface geometry).message lane).eval assignment) := by
  apply (application.hashChain_relation_iff certificate _ _ _).mpr
  exact Stage1.Poseidon2HashChainCompact.soundness (interface geometry) assignment one rows

/-- The existing constructive S-box values satisfy this placed plan for
every valid selected application step. -/
theorem complete_of_encoding (geometry : Geometry application certificate columns)
    (assignment : Assignment F columns) (prior message : Fin 4 → F)
    (one : assignment (oneColumn geometry) = 1)
    (priorEq : ∀ lane, ((interface geometry).priorState lane).eval assignment = prior lane)
    (messageEq : ∀ lane, ((interface geometry).message lane).eval assignment = message lane)
    (step : (List.ofFn fun lane => ((interface geometry).digest lane).eval assignment) =
      application.step (List.ofFn prior) (List.ofFn message))
    (encoded : ∀ invocation row, ((interface geometry).sbox invocation row).eval assignment =
      Stage1.Poseidon2HashChainCompactWitness.witness prior message invocation row) :
    (plan geometry).RowsZero assignment := by
  apply Stage1.Poseidon2HashChainCompactWitness.complete_of_encoding
    (interface geometry) assignment prior message one priorEq messageEq _ encoded
  exact (application.hashChain_relation_iff certificate prior message _).mp step

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonSoundness
