import NightstreamFPrime.Export.Stage1.PiCCSPhysicalPackets
import NightstreamFPrime.Export.Stage1.PiCCSCompleteness
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns the ordinary-row projection from the actual PiCCS physical lowering.
The existing twelve child lists and their owned fresh starts determine each
of the eight exported arithmetic packets. No child rows are reconstructed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryPhysicalCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open PiCCSArithmetic

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem prefixStart_getD (start : Nat) (deltas : List Nat)
    (index : Nat) (bounded : index < deltas.length) :
    (PiCCSStarts.prefixStarts start deltas).getD index 0 =
      start + (deltas.take index).sum := by
  induction deltas generalizing start index with
  | nil => simp at bounded
  | cons delta rest induction =>
      cases index with
      | zero => simp [PiCCSStarts.prefixStarts]
      | succ index =>
          have restBound : index < rest.length := by simpa using bounded
          simpa only [PiCCSStarts.prefixStarts, List.getD_cons_succ,
            List.take_succ_cons, List.sum_cons, Nat.add_assoc] using
            induction (start + delta) index restBound

private theorem child_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (index : Fin 12) :
    R1CS.RowsHold (Spartan.pullback target)
      (R1CS.lowerConstraints
        ((NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists relation
          (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset).getD index.val [])
        (PiCCSStarts.freshStarts.getD index.val 0)).rows := by
  let children := NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  let first := NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset
  have bounded : index.val < children.length := index.isLt
  have segments := R1CS.LoweringPlan.rowsHold_segments_of_constraints
    (NightstreamFPrime.Layout.PiCCS.v1_1.plan relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset)
    (Spartan.pullback target) children
    (NightstreamFPrime.Layout.PiCCS.v1_1.logicalConstraints_eq_flatten relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset) physical
  have selected := R1CS.segmentsHold_get (Spartan.pullback target) children first
    segments ⟨index.val, bounded⟩
  have counts : (List.ofFn fun current : Fin children.length =>
      R1CS.totalFreshCount (children.get current)) = children.map R1CS.totalFreshCount := by
    change List.ofFn (R1CS.totalFreshCount ∘ children.get) = _
    rw [← List.map_ofFn, List.ofFn_get]
  rw [counts] at selected
  have fresh : PiCCSStarts.freshStarts.getD index.val 0 =
      first + ((children.map R1CS.totalFreshCount).take index.val).sum := by
    rw [PiCCSStarts.freshStarts_eq_layout relation]
    rw [prefixStart_getD _ _ index.val (by
      change index.val < (children.map R1CS.totalFreshCount).length
      simpa only [List.length_map] using bounded)]
    rw [PiCCSStarts.logicalFreshBase_eq_layout relation]
    rfl
  rw [fresh, List.getD_eq_get children [] ⟨index.val, bounded⟩]
  exact selected

private theorem packet_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (index : Fin 12) (rowStart freshStart : Nat) (constraints : List Expr)
    (source : (NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset).getD index.val [] = constraints)
    (start : PiCCSStarts.freshStarts.getD index.val 0 = freshStart) :
    R1CS.RowsHold target
      ((compilePacket rowStart freshStart constraints).map Rows.CompiledRow.toR1CS) := by
  rw [compilePacket_toR1CS, Spartan.remapRows_hold]
  have selected := child_rows relation target physical index
  rw [source, start] at selected
  exact selected

private theorem packets_of_physical
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    PacketHolds logicalWidth publicFits target := by
  constructor
  · apply packet_rows relation target physical (0 : Fin 12)
    · rfl
    · rfl
  · apply packet_rows relation target physical (4 : Fin 12)
    · conv_lhs =>
        rw [show (4 : Fin 12).val = 4 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.initialClaimConstraints_eq logicalWidth publicFits
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl
  · apply packet_rows relation target physical (5 : Fin 12)
    · conv_lhs =>
        rw [show (5 : Fin 12).val = 5 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.sumcheckConstraints_eq logicalWidth publicFits
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl
  · apply packet_rows relation target physical (6 : Fin 12)
    · conv_lhs =>
        rw [show (6 : Fin 12).val = 6 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.evalKConstraints_eq logicalWidth publicFits
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl
  · apply packet_rows relation target physical (7 : Fin 12)
    · conv_lhs =>
        rw [show (7 : Fin 12).val = 7 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.evalAConstraints_eq logicalWidth publicFits
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl
  · apply packet_rows relation target physical (8 : Fin 12)
    · conv_lhs =>
        rw [show (8 : Fin 12).val = 8 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.ccsConstraints_eq relation
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl
  · apply packet_rows relation target physical (9 : Fin 12)
    · conv_lhs =>
        rw [show (9 : Fin 12).val = 9 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.normConstraints_eq relation
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl
  · apply packet_rows relation target physical (10 : Fin 12)
    · conv_lhs =>
        rw [show (10 : Fin 12).val = 10 from rfl]
        simp only [NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists,
          List.getD_cons_succ, List.getD_cons_zero]
      have source := PiCCSCompleteness.finalIdentityConstraints_eq relation
      dsimp only [PiCCSInvocations.parentInterface] at source
      exact source
    · rfl

/-- Every ordinary compact PiCCS row is the same row of the actual child
lowering at its owned fresh start. The completed target supplies all rows. -/
theorem ordinaryRows_of_physical
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    R1CS.RowsHold target
      ((arithmeticRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS) := by
  have packets := packets_of_physical relation target physical
  unfold arithmeticRows
  simp only [List.map_append, R1CS.rowsHold_append]
  exact ⟨⟨⟨⟨⟨⟨⟨packets.statementBinding, packets.initialClaim⟩,
    packets.sumcheck⟩, packets.eval_K⟩, packets.eval_A⟩, packets.ccs⟩,
    packets.norm⟩, packets.finalIdentity⟩

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryPhysicalCompleteness
