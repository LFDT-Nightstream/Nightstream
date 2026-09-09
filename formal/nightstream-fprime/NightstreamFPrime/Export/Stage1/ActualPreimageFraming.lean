import NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness
import NightstreamFPrime.Export.Stage1.PilotDecodedEnvironment
import NightstreamFPrime.Layout.Stage1.StateDecoder

/-!
Owns row-derived canonical framing for the actual pilot hash preimages.
The decoded source environment evaluates the compiled sparse forms directly.
No canonical coordinate encoding or caller-supplied representation is assumed.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPreimageFraming

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiCCSAssignmentSoundness

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def priorState (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (word : Nat) : F :=
  if bounded : word < PilotProduction.stateHashWords then
    ((PiRLCPoseidonGeometry.priorInputBlock program).form
      (PiRLCPoseidonGeometry.priorInputStart program)
      (PiRLCPoseidonGeometry.priorInputFits
        (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry))
      ⟨word, bounded⟩).eval assignment
  else 0

def outputState (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (word : Nat) : F :=
  if bounded : word < PilotProduction.stateHashWords then
    ((PiRLCPoseidonGeometry.outputInputBlock program).form
      (PiRLCPoseidonGeometry.outputInputStart program)
      (PiRLCPoseidonGeometry.outputInputFits
        (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry))
      ⟨word, bounded⟩).eval assignment
  else 0

theorem priorWord_eq
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (word : Fin PilotProduction.stateHashWords) :
    (Spartan.pullback (decodedEnv geometry assignment))
        (PilotProduction.priorPreimageStart + word.val) =
      priorState geometry assignment word.val := by
  have mapped := decodedEnv_location geometry assignment
    (PiCCSOrdinaryDirectPlan.Location.priorInput word)
  rw [PiCCSOrdinaryDirectPlan.Location.priorInput_form_eq_pilot] at mapped
  simpa only [Spartan.pullback, PiCCSOrdinaryDirectPlan.Location.sourceColumn,
    priorState, dif_pos word.isLt] using mapped

theorem outputWord_eq
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (word : Fin PilotProduction.stateHashWords) :
    (Spartan.pullback (decodedEnv geometry assignment))
        (PilotProduction.outputPreimageStart + word.val) =
      outputState geometry assignment word.val := by
  have mapped := decodedEnv_location geometry assignment
    (PiCCSOrdinaryDirectPlan.Location.outputInput word)
  rw [PiCCSOrdinaryDirectPlan.Location.outputInput_form_eq_pilot] at mapped
  simpa only [Spartan.pullback, PiCCSOrdinaryDirectPlan.Location.sourceColumn,
    outputState, dif_pos word.isLt] using mapped

/-- Accepted ordinary rows enforce canonical framing on the preimages that
the pilot hashes, for every logical assignment. -/
theorem rowsZero_implies_actualPreimageCanonical
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiCCSOrdinaryDirectPlan.plan relation geometry).RowsZero assignment) :
    StateDecoder.Canonical (priorState geometry assignment) ∧
      StateDecoder.Canonical (outputState geometry assignment) := by
  let env := decodedEnv geometry assignment
  have arithmetic := rowsZero_implies_arithmeticSpecs relation geometry assignment one rows
  have statement := arithmetic.statementBinding
  have framed : PiCCS.v1_1.StateBinding.SpecHolds
      (PiCCS.v1_1.Formal.statementBindingInterface
        (PiCCSArithmetic.sharedInterface relationLogicalWidth relationPublicFits)).state
      PiCCSArithmetic.statementBindingLogicalStart (Spartan.pullback env) :=
    statement.state
  constructor
  · intro word member
    have bounded : word.index < PilotProduction.stateHashWords := by
      rw [PilotProduction.stateHashWords_eq]
      exact PiCCS.v1_1.StateBinding.fixedWord_index_lt word member
    rw [← priorWord_eq geometry assignment ⟨word.index, bounded⟩]
    exact framed.priorCanonical word member
  · intro word member
    have bounded : word.index < PilotProduction.stateHashWords := by
      rw [PilotProduction.stateHashWords_eq]
      exact PiCCS.v1_1.StateBinding.fixedWord_index_lt word member
    rw [← outputWord_eq geometry assignment ⟨word.index, bounded⟩]
    exact framed.outputCanonical word member

/-- Accepted state-binding rows derive both typed preimage representations
for the actual pilot hash inputs. The caller supplies no representation or
canonical-coordinate encoding. -/
theorem rowsZero_implies_preimageRepresentations
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PilotOrdinaryDirectPlan.oneColumn geometry) = 1)
    (rows : (PiCCSOrdinaryDirectPlan.plan relation
      (PilotOrdinaryDirectPlan.piCcsGeometry geometry)).RowsZero assignment) :
    PriorStateHash.RepresentsPreimage PilotProduction.priorInterface
        PilotProduction.witnessOffset
        (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))
        (StateDecoder.preimage relationLogicalWidth relationPublicFits
          (priorState (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)) ∧
      OutputHash.RepresentsPreimage PilotProduction.outputInterface
        (Lifecycle.Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)
        (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))
        (StateDecoder.preimage relationLogicalWidth relationPublicFits
          (outputState (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment)) := by
  have canonical := rowsZero_implies_actualPreimageCanonical relation
    (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment one rows
  constructor
  · apply StateDecoder.priorRepresents relationLogicalWidth relationPublicFits
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))
      (priorState (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment) canonical.1
    intro word
    rw [PilotDecodedEnvironment.priorWord_agrees, priorWord_eq]
  · apply StateDecoder.outputRepresents relationLogicalWidth relationPublicFits
      (PilotSpartan.pullback (PilotDecodedEnvironment.env geometry assignment))
      (outputState (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment) canonical.2
    intro word
    rw [PilotDecodedEnvironment.outputWord_agrees, outputWord_eq]

/-- Accepted state-binding rows preserve the complete four-word context key
between the two actual hashed preimages. -/
theorem rowsZero_implies_contextKeys
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiCCSOrdinaryDirectPlan.plan relation geometry).RowsZero assignment) :
    StateDecoder.keyDigest (outputState geometry assignment) =
      StateDecoder.keyDigest (priorState geometry assignment) := by
  have arithmetic := rowsZero_implies_arithmeticSpecs relation geometry assignment one rows
  unfold StateDecoder.keyDigest
  apply StateDecoder.slice_congr
  intro lane
  have bounded : PiCCS.v1_1.StateBinding.contextWordStart + lane.val <
      PilotProduction.stateHashWords := by
    have bound := lane.isLt
    norm_num [PiCCS.v1_1.StateBinding.contextWordStart,
      PilotProduction.digestWords, PilotValues.digestWords,
      PilotProduction.stateHashWords_eq] at bound ⊢
    omega
  rw [← outputWord_eq geometry assignment
      ⟨PiCCS.v1_1.StateBinding.contextWordStart + lane.val, bounded⟩,
    ← priorWord_eq geometry assignment
      ⟨PiCCS.v1_1.StateBinding.contextWordStart + lane.val, bounded⟩]
  simpa only [PiCCSArithmetic.sharedInterface, PiCCSArithmetic.parentInterface,
    PiCCS.v1_1.Formal.statementBindingInterface, PiCCS.v1_1.Formal.atOffset,
    PiCCSInputs.interface, PiCCSInputs.priorStateWord, PiCCSInputs.outputStateWord,
    Expr.eval] using arithmetic.statementBinding.state.contextPreserved lane

end NightstreamFPrime.Export.Stage1.ActualPreimageFraming
