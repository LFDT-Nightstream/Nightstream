import NightstreamFPrime.Export.Stage1.Wide.GridSupport
import NightstreamFPrime.Export.Stage1.Wide.PiCCSPoseidonSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport MatrixProgram
open RunningTransitionReducedMatrixProgram

private theorem shared_wire (program : Program) (wire : RetainedBlock)
    (lower : RetainedLayout.sharedStart program ≤ wire.start)
    (upper : wire.start + wire.coordinateCount ≤ RetainedLayout.sharedEnd program) :
    WireSupported (Live program) wire := by
  intro column low high
  exact Or.inr (Or.inl ⟨by omega, by omega⟩)

theorem running_wires (program : Program) :
    WireSupported (Live program) (stateWire program) ∧
    WireSupported (Live program) (outputWire program) ∧
    WireSupported (Live program) (piDecWire program) ∧
    WireSupported (Live program) (poseidonWire program) ∧
    WireSupported (Live program) (inverseWire program) ∧
    WireSupported (Live program) (flagWire program) := by
  let within : RunningTransitionRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
    ⟨by rw [RunningTransitionRetainedGeometry.completeLogicalWidth_eq,
      (RetainedLayout.boundaries program).2.2.1]; decide⟩
  have stateLower : RetainedLayout.sharedStart program ≤ (stateWire program).start := by
    change RetainedLayout.sharedStart program ≤ RetainedLayout.sharedStart program + 28 * 41
    omega
  have outputLower : RetainedLayout.sharedStart program ≤ (outputWire program).start := by
    change RetainedLayout.sharedStart program ≤ RetainedLayout.sharedStart program + _
    omega
  have piDecLower : RetainedLayout.sharedStart program ≤ (piDecWire program).start := by
    change RetainedLayout.sharedStart program ≤ PiRLCPoseidonGeometry.pilotLogicalWidth program
    rw [(RetainedLayout.boundaries program).2.1, PiRLCPoseidonGeometry.pilotLogicalWidth_eq]
    decide
  have inverseLower : RetainedLayout.sharedStart program ≤ (inverseWire program).start := by
    change RetainedLayout.sharedStart program ≤ (piDecWire program).start + _
    omega
  have flagLower : RetainedLayout.sharedStart program ≤ (flagWire program).start := by
    change RetainedLayout.sharedStart program ≤ (inverseWire program).start + 41
    omega
  refine ⟨shared_wire program _ stateLower (RunningTransitionRetainedGeometry.stateFits within),
    shared_wire program _ outputLower (RunningTransitionRetainedGeometry.outputFits within),
    shared_wire program _ piDecLower (RunningTransitionRetainedGeometry.piDecFits within),
    ?_, shared_wire program _ inverseLower (RunningTransitionReducedMatrixSemantics.inverseFits within),
    shared_wire program _ flagLower (RunningTransitionReducedMatrixSemantics.flagFits within)⟩
  intro column _ high
  left
  have endpoint : (poseidonWire program).start + (poseidonWire program).coordinateCount =
      RetainedLayout.hashEnd program := by
    change PiCCSPoseidonPlan.retainedStart program + (PiCCSPoseidonPlan.retainedBlock program).coordinateCount = _
    rw [PiCCSPoseidonPlan.retainedBlock_coordinateCount]
    exact (LaterPoseidonRetainedBlocks.samplerStart_eq program).symm
  exact lt_of_lt_of_eq high endpoint

theorem running_blocks (program : Program) (oneColumn : Nat)
    (one : ∀ column, column.val = oneColumn → Live program column)
    (block : MatrixProgram.Block) (member : block ∈ (matrixProgram program oneColumn).blocks)
    (ordinal : Nat) (forms : RowForms (PerApplicationFixedPoint.logicalWidth program))
    (loaded : block.row? _ (fun _ => none) ordinal = some forms) : ∀ port, Form program (forms port) := by
  obtain ⟨state, output, piDec, poseidon, inverse, flag⟩ := running_wires program
  have grid (selected : MultiplicationGrid.Block)
      (oneEq : selected.oneColumn = oneColumn)
      (left : AffineSupported (Live program) selected.left)
      (right : AffineSupported (Live program) selected.right)
      (out : AffineSupported (Live program) selected.output)
      (loaded : (MatrixProgram.Block.multiplicationGrid selected).row? _ (fun _ => none) ordinal = some forms) :
      ∀ port, Form program (forms port) := by
    obtain ⟨decoded, decodedLoaded, same⟩ := Option.bind_eq_some_iff.mp loaded
    cases Option.some.inj same
    exact ordinary_form _ (multiplication_grid selected (fun column equal => one column (equal.trans oneEq))
      left right out ordinal decoded decodedLoaded)
  simp only [matrixProgram, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl <;>
    apply grid _ rfl _ _ _ loaded <;>
    simp only [flagGrid, bindingGrid, pointHeaderGrid, pointGrid, groupsGrid, stateGrid,
      flagProgram, baseProgram, AffineSupported, List.mem_cons, List.not_mem_nil, or_false,
      List.mem_map, forall_eq_or_imp, forall_eq, retained, constant, TermSupported]
  all_goals try simp only [state, output, piDec, inverse, flag, and_self, false_implies]
  all_goals try exact fun _ => True.intro
  rintro rule ⟨term, _, rfl⟩
  exact poseidon

theorem running (program : Program)
    (geometry : RunningTransitionRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    Plans program (RunningTransitionReducedPlan.plan geometry) := by
  intro row port
  let matrices := matrixProgram program (RunningTransitionRetainedGeometry.oneColumn geometry).val
  change Form program ((matrices.row? _ (fun _ => none) row.val).getD (fun _ => .empty) port)
  cases loaded : matrices.row? _ (fun _ => none) row.val with
  | none => exact empty _
  | some forms =>
    exact matrix_program matrices (fun _ => none)
      (running_blocks program _ (one program)) row.val forms loaded port

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
