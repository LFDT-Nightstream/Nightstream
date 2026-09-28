import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSource

/-!
Select one existing PiCCS ordinary packet before constructing its row list.
The packet order, source columns, and R1CS rows remain unchanged. Only the
selected row is decoded from the existing compiled-row representation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPiCCSPacketSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout

private theorem append_getElem? {Alpha : Type} {left : List Alpha} {count : Nat}
    (lengthEq : left.length = count) (right : List Alpha) (index : Nat) :
    (left ++ right)[index]? =
      if index < count then left[index]? else right[index - count]? := by
  by_cases bounded : index < count
  · rw [if_pos bounded, List.getElem?_append_left (by simpa only [lengthEq] using bounded)]
  · rw [if_neg bounded, List.getElem?_append_right (by
      simpa only [lengthEq] using Nat.le_of_not_gt bounded), lengthEq]

/-- The branch lengths are the existing packet lengths. Packet construction
occurs inside the selected branch. The final packet's existing optional list
lookup also rejects indices beyond the complete ordinary source domain. -/
def row? (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) (ordinal : Nat) :
    Option R1CS.Row :=
  let selected : Option Rows.CompiledRow :=
    if ordinal < 160 then
      (PiCCSArithmetic.statementBindingRows logicalWidth publicFits)[ordinal]?
    else
      let ordinal := ordinal - 160
      if ordinal < 116631 then
        (PiCCSArithmetic.initialClaimRows logicalWidth publicFits)[ordinal]?
      else
        let ordinal := ordinal - 116631
        if ordinal < 424657 then
          (PiCCSArithmetic.sumcheckRows logicalWidth publicFits)[ordinal]?
        else
          let ordinal := ordinal - 424657
          if ordinal < 8542 then
            (PiCCSArithmetic.evalKRows logicalWidth publicFits)[ordinal]?
          else
            let ordinal := ordinal - 8542
            if ordinal < 109630 then
              (PiCCSArithmetic.evalARows logicalWidth publicFits)[ordinal]?
            else
              let ordinal := ordinal - 109630
              if ordinal < 20794 then
                (PiCCSArithmetic.ccsRows logicalWidth publicFits)[ordinal]?
              else
                let ordinal := ordinal - 20794
                if ordinal < 752 then
                  (PiCCSArithmetic.normRows logicalWidth publicFits)[ordinal]?
                else
                  (PiCCSArithmetic.finalIdentityRows logicalWidth publicFits)[ordinal - 752]?
  selected.map Rows.CompiledRow.toR1CS

/-- Total optional-output equality to the existing source list. The proof
uses only its append spine and the owner's packet-length theorems; it does
not expand any packet, constraint, matrix, or package. The relation supplies
the same shape certificate as the existing indexed source authority. -/
theorem row?_eq_sourceRows_getElem?
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ordinal : Nat) :
    row? logicalWidth publicFits ordinal =
      (PiCCSOrdinaryDirectSource.sourceRows logicalWidth publicFits)[ordinal]? := by
  unfold row? PiCCSOrdinaryDirectSource.sourceRows
  rw [List.getElem?_map]
  apply congrArg (Option.map Rows.CompiledRow.toR1CS)
  simp only [PiCCSArithmetic.arithmeticRows, List.append_assoc,
    append_getElem? (PiCCSArithmetic.statementBindingRows_length
      logicalWidth publicFits relation),
    append_getElem? (PiCCSArithmetic.initialClaimRows_length
      logicalWidth publicFits relation),
    append_getElem? (PiCCSArithmetic.sumcheckRows_length
      logicalWidth publicFits relation),
    append_getElem? (PiCCSArithmetic.evalKRows_length
      logicalWidth publicFits relation),
    append_getElem? (PiCCSArithmetic.evalARows_length
      logicalWidth publicFits relation),
    append_getElem? (PiCCSArithmetic.ccsRows_length
      logicalWidth publicFits relation),
    append_getElem? (PiCCSArithmetic.normRows_length
      logicalWidth publicFits relation)]

/-- Every valid ordinal returns the exact existing programRow, with no
source-value, assignment-validity, or matrix-agreement premise. -/
theorem row?_eq_programRow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 811669) :
    row? logicalWidth publicFits index.val =
      some (PiCCSOrdinaryDirectSource.programRow relation index) := by
  rw [row?_eq_sourceRows_getElem? relation]
  have bounded : index.val <
      (PiCCSOrdinaryDirectSource.sourceRows logicalWidth publicFits).length := by
    rw [PiCCSOrdinaryDirectSource.sourceRows_length relation]
    exact index.isLt
  rw [List.getElem?_eq_getElem bounded]
  rfl

end NightstreamFPrime.Export.Stage1.PiDECPiCCSPacketSource
