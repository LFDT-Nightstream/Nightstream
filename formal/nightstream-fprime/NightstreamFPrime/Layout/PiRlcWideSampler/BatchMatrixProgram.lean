import NightstreamFPrime.Layout.PiRlcWideSampler.PoseidonMatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.Indexed

/-! One compact matrix program for all 34 permutations and 17 wide reductions.
The range template is serialized from the certified direct compiler. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.BatchMatrix

open MatrixProgram ProductionRelation

def rangeBlock {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    (source : Fin 17) : MatrixProgram.Block :=
  .ordinaryTemplate (MatrixRows.block compiled interface.oneColumn (MatrixRows.poseidonBlock interface)
    (MatrixRows.poseidonSlot source) (BatchPlan.rangeStart interface source) 0) (OrdinaryTemplate.ofSemantic compiled.rows.get)

def ranges {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    : MatrixProgram.Program :=
  Program.indexed (rangeBlock compiled interface)

theorem ranges_exact {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (ranges compiled interface) (BatchPlan.rangeFamily compiled interface) sourceRow := by
  apply Exact.indexed
  · intro source
    rfl
  · intro source row
    have custody : ∀ index : Fin compiled.rows.length,
        OrdinaryTemplate.row? (OrdinaryTemplate.ofSemantic compiled.rows.get) (0 + index.val) =
          some (compiled.rows.get index) := by
      intro index
      simpa only [Nat.zero_add] using OrdinaryTemplate.row?_ofSemantic compiled.rows.get index
    have known := MatrixRows.rangeRow compiled interface source 0
      (OrdinaryTemplate.row? (OrdinaryTemplate.ofSemantic compiled.rows.get)) custody row
    change (Program.mk [.ordinary (MatrixRows.block compiled interface.oneColumn
      (MatrixRows.poseidonBlock interface) (MatrixRows.poseidonSlot source)
      (BatchPlan.rangeStart interface source) 0)]).row? columns _ row.val = _ at known
    rw [Program.singleton_row?, if_pos (show row.val < (MatrixProgram.Block.ordinary
      (MatrixRows.block compiled interface.oneColumn (MatrixRows.poseidonBlock interface)
        (MatrixRows.poseidonSlot source) (BatchPlan.rangeStart interface source) 0)).rowCount from row.isLt)] at known
    exact known

def program {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    : MatrixProgram.Program :=
  (PoseidonMatrix.program interface).append (ranges compiled interface)

/-- The compact program decodes exactly the complete sampler plan. -/
theorem exact {columns : Nat} (compiled : RangePlan.Compiled) (interface : BatchPlan.Interface columns)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (program compiled interface) (BatchPlan.plan compiled interface) sourceRow :=
  (PoseidonMatrix.exact interface sourceRow).append
    (ranges_exact compiled interface sourceRow) _

end NightstreamFPrime.Layout.PiRlcWideSampler.BatchMatrix
