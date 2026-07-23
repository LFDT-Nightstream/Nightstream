import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows

/-!
Production execution refinement for the raw-old-block projection rows.

This leaf instantiates the generic row-to-terminal seam with the fixed Rust
emitter layout.  Its only execution premise is satisfaction of every actual
physical row under the transparent assignment decoded from pending state,
the ordered full `WitnessMat` family, and compiler-derived witness values.
No child projection sidecar, digest, desired projection equation, column-map
premise, execution-audit premise, or implementation-failure event occurs.

Owns: the fixed execution theorem turning satisfaction of every actual
production projection row under the transparent assignment, together with
same-family terminal CE, into `ProjectionOpeningAccepted`.

Does not own: construction of a satisfying internal witness, terminal-CE row
refinement, native commitment-key binding, transcript generation, whole-circuit
acceptance, costs, or row-removal authority.

Emits constraints: no; consumes the exact generated production rows.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.execution.rows` | all fixed physical projection rows hold under the generated inverse and transparent source assignment | checked execution premise |
| `f_prime.pi_ccs_nc.delayed.execution.audit` | fixed placement, column provenance, row semantics, pending reads, and raw-child reads construct the generic terminal audit | derived |
| `f_prime.pi_ccs_nc.delayed.execution.opening` | row-derived projection plus same-family terminal CE yields authoritative terminal projection opening | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Protocol
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionTensorPrefix
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjectionRows
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt
open PackedWitness

private abbrev activeSemanticShape := ProductionDomain.semanticShape

universe uState

private theorem columnsValue_eq_concrete
    (assignment : Nat -> Nat) (columns : KColumns) (value : Concrete.K)
    (c0 : assignment columns.c0 = value.c0.val)
    (c1 : assignment columns.c1 = value.c1.val) :
    toConcreteK (columns.value assignment) = value := by
  rcases value with ⟨value0, value1⟩
  have c0Exact : assignment columns.c0 = value0.val := by
    simpa using c0
  have c1Exact : assignment columns.c1 = value1.val := by
    simpa using c1
  simp only [KColumns.value, baseAt, residue, toConcreteK, toConcreteField,
    Concrete.K.mk.injEq]
  constructor
  · apply Fin.ext
    change assignment columns.c0 % goldilocksP = value0.val
    rw [c0Exact, Nat.mod_eq_of_lt]
    simpa [goldilocksP, goldilocksModulus] using value0.isLt
  · apply Fin.ext
    change assignment columns.c1 % goldilocksP = value1.val
    rw [c1Exact, Nat.mod_eq_of_lt]
    simpa [goldilocksP, goldilocksModulus] using value1.isLt

private theorem productionFactoredOldBlockColumns
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F) :
    ((oldBlockValues productionLayout
        (canonicalAssignment pending finalWitnesses internalWitness) ++
      [productionFactoredLayout.factor.finalPoint.value
        (canonicalAssignment pending finalWitnesses internalWitness)]).map
        toConcreteK) =
      pending.oldBlock.coordinates := by
  apply List.ext_get
  · simpa [oldBlockValues] using pending.oldBlock.dimension.symm
  · intro index leftWithin rightWithin
    have indexLt19 : index < 19 := by
      simpa [oldBlockValues] using leftWithin
    by_cases indexLt18 : index < 18
    · let bit : Fin productionLayout.blockVariables := ⟨index, by
        change index < 18
        exact indexLt18⟩
      have source0 := canonicalAssignment_oldBlock_c0 pending finalWitnesses
        internalWitness bit
      have source1 := canonicalAssignment_oldBlock_c1 pending finalWitnesses
        internalWitness bit
      have pendingGetD :
          pending.oldBlock.coordinates.getD index Concrete.K.zero =
            pending.oldBlock.coordinates.get ⟨index, rightWithin⟩ := by
        simp [List.getD_eq_getElem?_getD, rightWithin]
      rw [pendingGetD] at source0 source1
      simpa [oldBlockValues, bit, indexLt18] using
        columnsValue_eq_concrete
          (canonicalAssignment pending finalWitnesses internalWitness)
          (productionLayout.oldBlock bit)
          (pending.oldBlock.coordinates.get ⟨index, rightWithin⟩)
          source0 source1
    · have indexEq : index = 18 := by omega
      subst index
      have source0 := canonicalAssignment_factoredPoint_c0 pending
        finalWitnesses internalWitness
      have source1 := canonicalAssignment_factoredPoint_c1 pending
        finalWitnesses internalWitness
      have pendingGetD :
          pending.oldBlock.coordinates.getD 18 Concrete.K.zero =
            pending.oldBlock.coordinates.get ⟨18, rightWithin⟩ := by
        simp [List.getD_eq_getElem?_getD, rightWithin]
      rw [pendingGetD] at source0 source1
      simpa [oldBlockValues, productionLayout] using
        columnsValue_eq_concrete
          (canonicalAssignment pending finalWitnesses internalWitness)
          productionFactoredLayout.factor.finalPoint
          (pending.oldBlock.coordinates.get ⟨18, rightWithin⟩)
          source0 source1

private theorem productionParentColumn
    (pending : ProductionDelayedBlockLane)
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (lane : Fin productionLayout.activeLanes) :
    toConcreteK
        ((productionLayout.parent lane).value
          (canonicalAssignment pending finalWitnesses internalWitness)) =
      pending.parentYZcol lane := by
  exact columnsValue_eq_concrete
    (canonicalAssignment pending finalWitnesses internalWitness)
    (productionLayout.parent lane) (pending.parentYZcol lane)
    (canonicalAssignment_parent_c0 pending finalWitnesses internalWitness lane)
    (canonicalAssignment_parent_c1 pending finalWitnesses internalWitness lane)

/-- Satisfaction of the actual fixed Rust-emitted production rows derives
the terminal projection-opening authority over exactly the same ordered raw
`WitnessMat` family opened by terminal CE.  There is no refinement-failure
branch: emitter placement, column inversion, pending-state reads, and raw
witness reads are all theorems instantiated inside this proof. -/
theorem productionRows_projectionOpeningAccepted
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= activeSemanticShape.carrierWidth}
    {context : FixedActive.Context activeSemanticShape State
      publicRingColumns publicFits verifierRows}
    {certificate : FixedActive.Certificate context}
    (finalWitnesses :
      Fin productionGlobalParams.k -> PackedWitness.Matrix activeSemanticShape)
    (internalWitness : Nat -> ProjectionProgram.F)
    (rows : forall row : Fin totalRows,
      RowHolds
        (physicalAssignment
          (DelayedProduction.outgoingPending context certificate)
          finalWitnesses internalWitness)
        (actualRow productionEmitterLayout row))
    (terminalCE : TerminalCE.Holds
      (ProductionTerminal.TerminalCEBridge.semantics context)
      (ProductionTerminal.TerminalCEBridge.terminalInstance context certificate
        (fun child => PackedWitness.unpack (finalWitnesses child)))) :
    ProductionTerminal.ProjectionOpeningAccepted context certificate
      (fun child => PackedWitness.unpack (finalWitnesses child)) := by
  let pending := DelayedProduction.outgoingPending context certificate
  let assignment := canonicalAssignment pending finalWitnesses internalWitness
  have canonicalRows : ArtifactRowsSatisfied productionArtifactContract
      assignment := by
    intro row
    apply (rowHolds_pull_iff
      (physicalAssignment pending finalWitnesses internalWitness)
      (emitterColumnMap productionEmitterLayout) (artifactRow row)).mpr
    simpa [actualRow, assignment, canonicalAssignment, pending] using rows row
  let audit : FactoredTerminalExecutionAudit context certificate
      productionFactoredLayout
      artifactRow assignment finalWitnesses :=
    { artifact := productionArtifactContract
      rows := canonicalRows
      canonical := by
        intro column
        exact canonicalAssignment_lt pending finalWitnesses internalWitness column
      constantOne := canonicalAssignment_constantOne pending finalWitnesses
        internalWitness
      oldBlockColumns := by
        change
          ((oldBlockValues productionLayout assignment ++
            [productionFactoredLayout.factor.finalPoint.value assignment]).map
              toConcreteK) =
            (DelayedProduction.outgoingPending context certificate).oldBlock.coordinates
        simpa [assignment, pending] using
          productionFactoredOldBlockColumns pending finalWitnesses internalWitness
      parentColumns := by
        intro lane
        change
          toConcreteK ((productionFactoredLayout.base.parent lane).value assignment) =
            (DelayedProduction.outgoingPending context certificate).parentYZcol lane
        simpa [assignment, pending] using
          productionParentColumn pending finalWitnesses internalWitness lane
      rawWitnessColumns := by
        intro child coordinate
        change
          assignment (rawWitnessColumn productionLayout child coordinate) =
            (PackedWitness.unpack (finalWitnesses child) coordinate).val
        simpa [assignment] using
          canonicalAssignment_rawWitness pending finalWitnesses internalWitness
            child coordinate
      terminalCE := terminalCE }
  exact audit.projectionOpeningAccepted

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
