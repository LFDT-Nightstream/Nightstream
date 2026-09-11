import NightstreamFPrime.Layout.Stage1.Spartan
import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition
import NightstreamFPrime.Layout.Stage1.SpartanValues
import NightstreamFPrime.Layout.R1CS.ColumnMap

/-!
Owns the complete Stage 1 prefix rows in Spartan order, their fixed-domain
padding, and the preservation theorems for those rows.
-/

namespace NightstreamFPrime.Layout.Stage1.Spartan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Layout
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def sourceRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  PilotPiCCSPiRLCPiDECRunningTransition.physicalRows relation

def remappedRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  remapRows (sourceRows relation)

theorem remappedRows_hold
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) :
    R1CS.RowsHold target (remappedRows relation) ↔
      PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation
        (pullback target) := by
  exact remapRows_hold target (sourceRows relation)

theorem sourceColumnCount_matches
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PilotPiCCSPiRLCPiDECRunningTransition.physicalColumnCount relation =
      SourceColumnCount := by
  rw [PilotPiCCSPiRLCPiDECRunningTransition.physicalColumnCount_eq relation,
    sourceColumnCount_eq]

theorem sourceRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows relation).length = 29218024 := by
  exact PilotPiCCSPiRLCPiDECRunningTransition.physicalRowCount_eq relation

theorem sourceRowCount_bounds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows relation).length ≤ domainSize := by
  rw [sourceRowCount_eq relation, domainSize_eq]
  norm_num

private theorem padCombination_eval (target : Env)
    (combination : R1CS.LinearCombination) :
    (padCombination combination).eval target =
      combination.eval (paddedPullback target) := by
  exact R1CS.mapCombinationColumns_eval spartanToPadded combination target

private theorem padRow_holds (target : Env) (row : R1CS.Row) :
    (padRow row).Holds target ↔ row.Holds (paddedPullback target) := by
  exact R1CS.mapRowColumns_holds spartanToPadded row target

private theorem padRows_hold (target : Env) (rows : List R1CS.Row) :
    R1CS.RowsHold target (padRows rows) ↔
      R1CS.RowsHold (paddedPullback target) rows := by
  constructor
  · intro holds row member
    have padded := holds (padRow row) (by
      rw [padRows, List.mem_map]
      exact ⟨row, member, rfl⟩)
    exact (padRow_holds target row).mp padded
  · intro holds row member
    rw [padRows, List.mem_map] at member
    rcases member with ⟨source, sourceMember, rfl⟩
    exact (padRow_holds target source).mpr (holds source sourceMember)

private theorem zeroRows_hold (target : Env) (count : Nat) :
    R1CS.RowsHold target (List.replicate count zeroRow) := by
  intro row member
  have equals : row = zeroRow := by
    simpa using (List.eq_of_mem_replicate member)
  subst row
  simp [zeroRow, R1CS.Row.Holds]

def paddedRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  padRows (remappedRows relation) ++
    List.replicate
      (domainSize - (sourceRows relation).length) zeroRow

theorem paddedRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (paddedRows relation).length = domainSize := by
  unfold paddedRows padRows remappedRows remapRows
  rw [List.length_append, List.length_map, List.length_map,
    List.length_replicate,
    Nat.add_sub_of_le (sourceRowCount_bounds relation)]

/-- The padded direct-Spartan rows preserve and reflect the complete current
Stage 1 prefix through the running transition. -/
theorem paddedRows_hold
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env) :
    R1CS.RowsHold target (paddedRows relation) ↔
      PilotPiCCSPiRLCPiDECRunningTransition.PhysicalHolds relation
        (pullback (paddedPullback target)) := by
  change R1CS.RowsHold target (paddedRows relation) ↔
    R1CS.RowsHold (pullback (paddedPullback target))
      (sourceRows relation)
  constructor
  · intro holds
    have split := (R1CS.rowsHold_append target _ _).mp holds
    exact (remappedRows_hold relation (paddedPullback target)).mp
      ((padRows_hold target (remappedRows relation)).mp split.1)
  · intro holds
    apply (R1CS.rowsHold_append target _ _).mpr
    exact ⟨
      (padRows_hold target (remappedRows relation)).mpr
        ((remappedRows_hold relation (paddedPullback target)).mpr
          holds),
      zeroRows_hold target _⟩

end NightstreamFPrime.Layout.Stage1.Spartan
