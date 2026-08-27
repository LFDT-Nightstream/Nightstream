import NightstreamFPrime.Export.Stage1.Data

/-!
Owns the bounded preparation plan for ordinary Stage 1 rows.

The eight PiCCS packets, 544 PiRLC digest lanes, 17 selector-final packets,
and one PiDEC packet are independent immutable blocks. The emitter may prepare
them concurrently, but it writes completed blocks in this Lean-owned order.
The expansion and classifier theorems prove that segmentation does not change
any package row or witness instruction.
-/

namespace NightstreamFPrime.Export.Stage1.OrdinaryRowPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

inductive Block where
  | statementBinding
  | initialClaim
  | sumcheck
  | evalK
  | evalA
  | ccs
  | norm
  | finalIdentity
  | piRlcLane (source round : Nat) (lane : Fin 4)
  | piRlcSelectorFinal (source : Nat)
  | explicitRows (values : List Rows.CompiledRow)
deriving Repr

def Block.rows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Block → List Rows.CompiledRow
  | .statementBinding =>
      PiCCSArithmetic.statementBindingRows logicalWidth publicFits
  | .initialClaim => PiCCSArithmetic.initialClaimRows logicalWidth publicFits
  | .sumcheck => PiCCSArithmetic.sumcheckRows logicalWidth publicFits
  | .evalK => PiCCSArithmetic.evalKRows logicalWidth publicFits
  | .evalA => PiCCSArithmetic.evalARows logicalWidth publicFits
  | .ccs => PiCCSArithmetic.ccsRows logicalWidth publicFits
  | .norm => PiCCSArithmetic.normRows logicalWidth publicFits
  | .finalIdentity =>
      PiCCSArithmetic.finalIdentityRows logicalWidth publicFits
  | .piRlcLane source round lane =>
      PiRLCSamplerOrdinaryRows.laneRows
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane
  | .piRlcSelectorFinal source =>
      PiRLCSamplerOrdinaryRows.selectorFinalRows source
  | .explicitRows values => values

def piCcsBlocks (_unit : Unit) : List Block :=
  [.statementBinding, .initialClaim, .sumcheck, .evalK, .evalA, .ccs,
    .norm, .finalIdentity]

def piRlcWindowBlocks (source round : Nat) : List Block :=
  (List.finRange 4).map (Block.piRlcLane source round)

def piRlcSourceBlocks (source : Nat) : List Block :=
  (List.range PiRLCSamplerOrdinaryRows.digestRoundCount).flatMap
      (piRlcWindowBlocks source) ++
    [.piRlcSelectorFinal source]

def piRlcBlocks (_unit : Unit) : List Block :=
  (List.range PiRLCSamplerOrdinaryRows.sourceCount).flatMap
    piRlcSourceBlocks

def piDecBlock (_unit : Unit) : Block :=
  .explicitRows
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows

def piDecBlocks (_unit : Unit) : List Block :=
  [piDecBlock ()]

def canonicalBlocks (_unit : Unit) : List Block :=
  piCcsBlocks () ++ piRlcBlocks () ++ piDecBlocks ()

private theorem flatMap_map_rows {Alpha : Type}
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (values : List Alpha) (make : Alpha → Block) :
    (values.map make).flatMap (Block.rows logicalWidth publicFits) =
      values.flatMap fun value =>
        (make value).rows logicalWidth publicFits := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [inductionHypothesis]

private theorem flatMap_flatMap_rows {Alpha : Type}
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (values : List Alpha) (blocks : Alpha → List Block) :
    (values.flatMap blocks).flatMap (Block.rows logicalWidth publicFits) =
      values.flatMap fun value =>
        (blocks value).flatMap (Block.rows logicalWidth publicFits) := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [inductionHypothesis]

theorem piCcsBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (piCcsBlocks ()).flatMap (Block.rows logicalWidth publicFits) =
      PiCCSArithmetic.arithmeticRows logicalWidth publicFits := by
  unfold piCcsBlocks PiCCSArithmetic.arithmeticRows
  simp only [List.flatMap_cons, List.flatMap_nil, Block.rows,
    List.append_nil, List.append_assoc]

theorem piRlcWindowBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) :
    (piRlcWindowBlocks source round).flatMap
        (Block.rows logicalWidth publicFits) =
      PiRLCSamplerOrdinaryRows.windowRows
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round := by
  unfold piRlcWindowBlocks PiRLCSamplerOrdinaryRows.windowRows
  rw [flatMap_map_rows]
  rfl

theorem piRlcSourceBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (source : Nat) :
    (piRlcSourceBlocks source).flatMap
        (Block.rows logicalWidth publicFits) =
      PiRLCSamplerOrdinaryRows.sourceRows
        (logicalWidth := logicalWidth) (publicFits := publicFits) source := by
  unfold piRlcSourceBlocks PiRLCSamplerOrdinaryRows.sourceRows
  rw [List.flatMap_append, flatMap_flatMap_rows]
  simp_rw [piRlcWindowBlocks_expand]
  rfl

theorem piRlcBlocks_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (piRlcBlocks ()).flatMap (Block.rows logicalWidth publicFits) =
      PiRLCSamplerOrdinaryRows.rows
        (logicalWidth := logicalWidth) (publicFits := publicFits) := by
  unfold piRlcBlocks PiRLCSamplerOrdinaryRows.rows
  rw [flatMap_flatMap_rows]
  simp_rw [piRlcSourceBlocks_expand]

private theorem explicitRowsBlock_expand
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (values : List Rows.CompiledRow) :
    ([Block.explicitRows values]).flatMap
        (Block.rows logicalWidth publicFits) = values := by
  simp only [List.flatMap_cons, List.flatMap_nil, Block.rows,
    List.append_nil]

theorem piDecBlocks_expand :
    (piDecBlocks ()).flatMap
        (Block.rows Data.logicalWidth Data.publicFits) =
      (PiDECArithmetic.canonicalPlan
        Data.logicalWidth Data.publicFits).rows := by
  exact explicitRowsBlock_expand Data.logicalWidth Data.publicFits _

theorem canonicalBlocks_expand :
    (canonicalBlocks ()).flatMap
        (Block.rows Data.logicalWidth Data.publicFits) =
      Data.arithmeticRows () := by
  unfold canonicalBlocks Data.arithmeticRows
  rw [List.flatMap_append, List.flatMap_append, piCcsBlocks_expand,
    piRlcBlocks_expand, piDecBlocks_expand]

private theorem witnessInstructions_flatMap
    (blocks : List Block) :
    Rows.witnessInstructionsTR
        (blocks.flatMap (Block.rows Data.logicalWidth Data.publicFits)) =
      blocks.flatMap fun block =>
        Rows.witnessInstructionsTR
          (block.rows Data.logicalWidth Data.publicFits) := by
  induction blocks with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      simp [Rows.witnessInstructionsTR_append, inductionHypothesis]

private theorem assertionRows_flatMap (blocks : List Block) :
    Rows.assertionRowsTR
        (blocks.flatMap (Block.rows Data.logicalWidth Data.publicFits)) =
      blocks.flatMap fun block =>
        Rows.assertionRowsTR
          (block.rows Data.logicalWidth Data.publicFits) := by
  induction blocks with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      simp [Rows.assertionRowsTR_append, inductionHypothesis]

theorem canonicalWitnessInstructions_expand :
    (canonicalBlocks ()).flatMap (fun block =>
        Rows.witnessInstructionsTR
          (block.rows Data.logicalWidth Data.publicFits)) =
      Rows.witnessInstructionsTR (Data.arithmeticRows ()) := by
  rw [← witnessInstructions_flatMap, canonicalBlocks_expand]

theorem canonicalAssertionRows_expand :
    (canonicalBlocks ()).flatMap (fun block =>
        Rows.assertionRowsTR
          (block.rows Data.logicalWidth Data.publicFits)) =
      Rows.assertionRowsTR (Data.arithmeticRows ()) := by
  rw [← assertionRows_flatMap, canonicalBlocks_expand]

end NightstreamFPrime.Export.Stage1.OrdinaryRowPlan
