import NightstreamFPrime.Lifecycle.BoundedDepth.CompressionSlot
import NightstreamFPrime.Spec.Compression.Fresh

/-! Π_CCS on the fresh instance alone (SPEC.md, Algorithm 9, step 5, and Algorithm 10, step 7).

`statement` is the Π_CCS statement of the fresh instance under its circuit's relation: one fresh
source, no running source (`oneShape`), with the ring source of the relation. An accepted run
whose output claims hold, on a path that avoids the bad sets of `Spec.Compression.Fresh.game`,
gives the fresh instance under `Spec.CCS.Holds` as `Lifecycle.Relation.freshStatement`
(`game_avoids`). Each bad set has density at most `38 / q²` (`game_densityLe`, `ccsError_eq`).
When `Relation` of the argument holds for the fresh claim with the values of the run's output,
the output claims hold (`fresh_ambient`). -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.CompressionFresh

open Finset NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Verifier NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Claim NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Lifecycle.PaperAlgebra NightstreamFPrime.Lifecycle.BoundedDepth.CompressionSlot

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- The shape of Π_CCS on the fresh instance alone. -/
abbrev oneShape : Folding.PiCCS.PaperJoint.Shape :=
  RingMatrixSource.ringShape cubeVariables 1 0 productionProfile.ccsMatrices

/-- The Π_CCS statement of the fresh instance with commitment `c` and public input `x` under
`relation`. -/
def statement (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (c : PaperAlgebra.Commitment)
    (x : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    StrongReduction.Statement K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) oneShape
      (RingCarrierLayout.carrierWidth logicalWidth)
      (ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth logicalWidth)) baseOps
    where
  cubeLayout := Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
    (RingCarrierLayout.carrierWidth logicalWidth) relation.cubeFits
  matrixSource := RingMatrixSource.source cubeVariables 1 0 productionProfile.ccsMatrices
    logicalWidth relation.system.matrices relation.system.constraintPolynomial
  commitments := fun _ => c
  publicInputs := fun _ => x
  priorPoint := ⟨List.replicate cubeVariables K.zero, List.length_replicate⟩
  claimedPadCoefficient := fun coordinate => coordinate.running.elim0
  claimedMatrixCoefficient := fun coordinate => coordinate.running.elim0

/-- The witness of the one source. -/
def witnessOf (z : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    OutputWitness oneShape (RingCarrierLayout.carrierWidth logicalWidth) :=
  ⟨fun _ => z⟩

theorem degree_le (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (c : PaperAlgebra.Commitment)
    (x : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ((statement relation c x).verifierInput K.embed).sumcheckDegreeBound ≤ 8 :=
  (ProductionKey.derivedDegreeBound_eq relation).le

/-- The input claim of the run is the fresh instance under `Spec.CCS.Holds`. -/
theorem ccs_of_source (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (z : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : SourceHolds extensionOps K.embed (openingMaps ajtai) productionGlobalParams
      (statement relation (fresh.commitments ⟨0, by decide⟩) (fresh.publicInputs ⟨0, by decide⟩))
      (witnessOf z)) :
    CCS.Holds (semantics ajtai) productionGlobalParams (freshStatement relation fresh) z :=
  ((SourceMembership.sourceHolds_iff_memberships extensionOps extensionLaws K.embed
    (openingMaps ajtai) productionGlobalParams rfl _ _).mp source).1 ⟨0, by decide⟩

/-- On each path of the run that avoids its bad sets, an accepted run whose output claims hold
gives the fresh instance under `Spec.CCS.Holds`. -/
theorem game_avoids (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (z : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (Compression.Fresh.game (statement relation (fresh.commitments ⟨0, by decide⟩)
        (fresh.publicInputs ⟨0, by decide⟩)) (witnessOf z) 8).Avoids fun out =>
      ∀ probe, out = some probe →
        AmbientOutputHolds extensionOps K.embed (openingMaps ajtai) productionGlobalParams
          (statement relation (fresh.commitments ⟨0, by decide⟩)
            (fresh.publicInputs ⟨0, by decide⟩)) probe (witnessOf z) →
          CCS.Holds (semantics ajtai) productionGlobalParams (freshStatement relation fresh) z :=
  (Compression.Fresh.game_avoids _ _ 8 rfl RingCoefficientKernel.ringConstantTermLaw
    (degree_le relation _ _)).mono fun _ holds probe equal ambient =>
      ccs_of_source ajtai relation fresh z (holds probe equal ambient)

theorem ccsError_eq : Compression.Fresh.ccsError oneShape 8 = 38 / (fieldModulus : ℝ) ^ 2 := by
  have count : oneShape.cubeVariables + oneShape.jointCoefficientCount + 8 = 38 := rfl
  unfold Compression.Fresh.ccsError
  rw [count, card_K]
  push_cast
  rfl

/-- Each bad set of the run has density at most `38 / q²`. -/
theorem game_densityLe (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (c : PaperAlgebra.Commitment)
    (x : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (z : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (Compression.Fresh.game (statement relation c x) (witnessOf z) 8).DensityLe
      (38 / (fieldModulus : ℝ) ^ 2) :=
  ccsError_eq ▸ Compression.Fresh.game_densityLe _ _ 8 (degree_le relation c x)

theorem centeredMagnitude_lt_ambient (value : F) :
    centeredMagnitude value < productionGlobalParams.ambientBound := by
  have := value.isLt
  change min value.val (fieldModulus - value.val) < fieldModulus / 2 + 1
  omega

/-- When `Relation` holds for the fresh claim with the values of the run's output, the output
claims of the run hold for the fresh witness slice. -/
theorem fresh_ambient {key : ℕ → ℕ → ℕ → F} {matrices : Fin 3 → Matrices F}
    {depths : Fin 2 → ℕ} {layout : Layout} {st : Compression.Verifier.Statement (params key matrices depths layout)}
    {W : ℕ → F} (holds : Relation (params key matrices depths layout) st W)
    (wf : WellFormed (params key matrices depths layout))
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (keyEq : key = keyOf ajtai) (represents : Represents (matrices 2) relation.matrices)
    (c : PaperAlgebra.Commitment)
    (x : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (probe : Probe K oneShape)
    (values : st (2 : Fin 3) = claimValues (logicalWidth := logicalWidth) (publicFits := publicFits)
      ⟨c, x, probe.coins.roundPoint, ⟨probe.response.fullOutput.padCoordinate ⟨0, by decide⟩,
        probe.response.fullOutput.matrixCoordinate ⟨0, by decide⟩⟩⟩) :
    AmbientOutputHolds extensionOps K.embed (openingMaps ajtai) productionGlobalParams
      (statement relation c x) probe
      (witnessOf (slotWitness (logicalWidth := logicalWidth) (publicFits := publicFits) matrices
        layout W 2)) := by
  intro source
  have zero : source = ⟨0, by decide⟩ := Fin.ext (by
    have := source.isLt
    change (source : ℕ) < 1 at this
    change (source : ℕ) = 0
    omega)
  subst zero
  refine ⟨⟨slot_commitment holds ajtai _ 2 keyEq represents.cover values,
    slot_publicInput holds _ wf 2 values,
    fun column => centeredMagnitude_lt_ambient _⟩, trivial, ?_⟩
  apply congrArg (fun family => #[family])
  show (EvaluationFamily.mk _ _ : EvaluationFamily K oneShape) =
    ⟨probe.response.fullOutput.padCoordinate ⟨0, by decide⟩,
      probe.response.fullOutput.matrixCoordinate ⟨0, by decide⟩⟩
  congr 1
  · funext o
    exact relation_pad holds _ 2 relation represents values o
  · funext m o
    exact relation_matrix holds _ 2 relation represents values m o

end NightstreamFPrime.Lifecycle.BoundedDepth.CompressionFresh
