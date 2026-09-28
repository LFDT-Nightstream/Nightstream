import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
import NightstreamFPrime.Layout.Poseidon2.HashInvocationRows
import NightstreamFPrime.Layout.Stage1.Spartan

/-!
Owns projection of the actual pilot physical rows to its ordinary compiler
rows. The original lowering supplies every auxiliary value; the existing
source support controls the change from the complete to the pilot layout.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryPhysicalCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Package

private theorem prior_extra_source (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) :
    R1CS.RowsHold (Spartan.pullback target)
      (R1CS.lowerConstraints (PilotData.priorExtraConstraints ())
        PilotValues.logicalColumnCount).rows := by
  change R1CS.RowsHold (Spartan.pullback target)
    (R1CS.lowerConstraints
      (NightstreamFPrime.Layout.Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset)
      (NightstreamFPrime.Layout.Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset)).rows at rows
  rw [NightstreamFPrime.Layout.Pilot.logicalConstraints, R1CS.lowerConstraints_append_rows] at rows
  have prior := ((R1CS.rowsHold_append (Spartan.pullback target) _ _).mp rows).1
  have split : NightstreamFPrime.Layout.Pilot.priorConstraints PilotProduction.interface PilotProduction.witnessOffset =
      PilotProduction.priorRawConstraints () ++ PilotData.priorExtraConstraints () := by
    rw [PilotProduction.priorConstraints_decomposition,
      PilotData.priorExtraConstraints_eq, List.append_assoc]
  rw [split, R1CS.lowerConstraints_append_rows, PilotProduction.priorHash_freshCount,
    Nat.add_zero] at prior
  have extra := ((R1CS.rowsHold_append (Spartan.pullback target) _ _).mp prior).2
  have start : NightstreamFPrime.Layout.Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset =
      PilotValues.logicalColumnCount := by
    rw [PilotProduction.logicalColumnCount_eq]
    rfl
  simpa only [start] using extra

private theorem priorRows_of_physical (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) :
    R1CS.RowsHold (fun column => target (Spartan.liftPilotColumn column))
      ((PilotData.priorExtraRows ()).map Rows.CompiledRow.toR1CS) := by
  rw [PilotOrdinaryDirectSource.priorRows_eq]
  apply (PilotSpartan.remapRows_hold
    (fun column => target (Spartan.liftPilotColumn column)) _).mpr
  apply R1CS.rowsHold_of_agree _ PilotOrdinaryDirectSource.PhysicalSource
    (Spartan.pullback target)
    (PilotSpartan.pullback (fun column => target (Spartan.liftPilotColumn column)))
    PilotOrdinaryDirectSource.priorLoweredRows_varsSatisfy _ (prior_extra_source target rows)
  intro column supported
  have bounded : column < Spartan.pilotSourceColumnCount :=
    PilotOrdinaryDirectSource.physicalSource_lt column supported
  change target (Spartan.liftPilotColumn (PilotSpartan.sourceToSpartan column)) =
    target (Spartan.sourceToSpartan column)
  rw [Spartan.sourceToSpartan, if_pos bounded]

private theorem outputState_pilotColumn (lane : Fin 4) :
    PilotSpartan.sourceToSpartan (PilotOrdinaryDirectPlan.Location.outputState lane).sourceColumn =
      PilotData.outputChain.witnessStart + PilotData.outputChain.absorbCount * 592 + 584 + lane.val := by
  change PilotSpartan.sourceToSpartan
      (PilotProduction.lifecycleOutputOffset + PilotValues.absorbCount * 592 + 584 + lane.val) = _
  rw [PilotOrdinaryDirectSource.outputState_targetColumn]
  rfl

private theorem outputDigest_pilotColumn (lane : Fin 4) :
    PilotSpartan.sourceToSpartan (PilotOrdinaryDirectPlan.Location.outputDigest lane).sourceColumn =
      PilotData.outputChain.digestStart + lane.val := by
  have bounded := lane.isLt
  change PilotSpartan.sourceToSpartan (PilotSpartan.outputDigestStart + lane.val) =
    PilotSpartan.secondPublicStart + lane.val
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by
    rw [PilotSpartan.outputDigestStart_value, PilotSpartan.priorPublicStart_value]
    omega)]
  rw [if_neg (by
    rw [PilotSpartan.outputDigestStart_value, PilotSpartan.outputPreimageStart_value]
    omega)]
  rw [if_neg (by omega)]
  rw [if_pos (by
    rw [PilotSpartan.outputDigestStart_value, PilotSpartan.witnessStart_value]
    omega)]
  rw [Nat.add_sub_cancel_left]

private theorem outputDigest_value_of_physical (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) (lane : Fin 4) :
    target (Spartan.liftPilotColumn
      (PilotData.outputChain.witnessStart + PilotData.outputChain.absorbCount * 592 + 584 + lane.val)) =
    target (Spartan.liftPilotColumn (PilotData.outputChain.digestStart + lane.val)) := by
  have logical := R1CS.lowerConstraints_sound (Spartan.pullback target)
    (NightstreamFPrime.Layout.Pilot.logicalConstraints PilotProduction.interface PilotProduction.witnessOffset)
    (NightstreamFPrime.Layout.Pilot.logicalColumnCount PilotProduction.interface PilotProduction.witnessOffset) rows
  have outputRows := ((constraintsHold_append (Spartan.pullback target) _ _).mp logical).2
  rw [NightstreamFPrime.Layout.Pilot.outputConstraints_eq] at outputRows
  rw [← PilotProduction.lifecycleOutputOffset_matches_layout] at outputRows
  have computed := NightstreamFPrime.Layout.Poseidon2.HashInvocationRows.digest_eq_expected_of_hashConstraints
    (OutputHash.hashInterface PilotProduction.outputInterface)
    PilotProduction.lifecycleOutputOffset (Spartan.pullback target) outputRows lane
  simp only [PilotProduction.outputHashInterface_input,
    NightstreamFPrime.Layout.Poseidon2.hash_compile_output_eq,
    PilotProduction.outputPreimage_chunkCount, PilotProduction.outputHashInterface_expected,
    Hash.digestE, Permutation.freshState, Expr.eval_var] at computed
  change target (Spartan.sourceToSpartan (PilotOrdinaryDirectPlan.Location.outputState lane).sourceColumn) =
    target (Spartan.sourceToSpartan (PilotOrdinaryDirectPlan.Location.outputDigest lane).sourceColumn) at computed
  rw [PilotOrdinaryDirectPlan.Location.stage1Map,
    PilotOrdinaryDirectPlan.Location.stage1Map,
    outputState_pilotColumn, outputDigest_pilotColumn] at computed
  exact computed

/-- The four package output-digest rows follow from the actual pilot physical
rows. The hash owner supplies the stored-output assertion and existing maps
identify both package columns. No digest or source-row premise is supplied. -/
private theorem outputDigestRows_of_physical (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) :
    ∀ row ∈ PilotData.digestRows PilotData.outputChain,
      row.Holds (fun column => target (Spartan.liftPilotColumn column)) := by
  intro row member
  rw [PilotData.digestRows, List.mem_ofFn'] at member
  obtain ⟨lane, rfl⟩ := member
  have value := outputDigest_value_of_physical target rows lane
  have zeroCoefficient : fieldValue 0 = (0 : F) := rfl
  have oneCoefficient : fieldValue 1 = (1 : F) := rfl
  simpa only [SparseRow.Holds, PilotData.digestRow, SparseCombination.eval,
    PilotData.oneCombination, PilotData.zeroCombination, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, zeroCoefficient, oneCoefficient, Rows.fieldValue_val,
    zero_add, add_zero, one_mul, mul_one, neg_one_mul, sub_eq_add_neg] using
      (sub_eq_zero.mpr value)


/-- Actual pilot physical rows imply all ordinary package rows under the
existing pilot-to-Stage-1 column lift, including the original lowered witness
values and the four output-digest assertions. -/
theorem rows_of_physical (target : Env)
    (rows : NightstreamFPrime.Layout.Pilot.PhysicalHolds PilotProduction.interface
      PilotProduction.witnessOffset (Spartan.pullback target)) :
    R1CS.RowsHold (fun column => target (Spartan.liftPilotColumn column))
      PilotOrdinaryDirectSource.sourceRows := by
  let env : Env := fun column => target (Spartan.liftPilotColumn column)
  have prior := (Rows.compiledRows_hold_iff (PilotData.priorExtraRows ()) env).mp
    (priorRows_of_physical target rows)
  have digest := outputDigestRows_of_physical target rows
  apply (R1CS.rowsHold_append env PilotOrdinaryDirectSource.instructionRows
    PilotOrdinaryDirectSource.assertionRows).mpr
  constructor
  · intro row member
    obtain ⟨instruction, instructionMember, rfl⟩ := List.mem_map.mp member
    apply (witnessInstruction_toR1CS_holds instruction env).mpr
    apply prior.1 instruction
    simpa only [PilotData.witnessInstructions, Rows.witnessInstructionsTR_eq] using instructionMember
  · intro row member
    obtain ⟨assertion, assertionMember, rfl⟩ := List.mem_map.mp member
    apply (sparseRow_holds assertion env).mpr
    rw [PilotData.assertionRows, List.mem_append] at assertionMember
    rcases assertionMember with before | output
    · apply prior.2 assertion
      simpa only [Rows.assertionRowsTR_eq] using before
    · exact digest assertion output

end NightstreamFPrime.Export.Stage1.PilotOrdinaryPhysicalCompleteness
