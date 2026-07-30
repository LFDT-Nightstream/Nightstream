import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
import Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding

/-!
Contract: allocate the verifier-computed public `Pi_RLC` output once and reuse
those exact columns as the public portion of the strict-`Pi_DEC` parent.

The allocation is Lean-owned and matrix-count-parametric.  It contains one
Phi81 coefficient block for every public role, in `publicOrder` order.  The
non-public strict-`Pi_DEC` fields are supplied by `ParentSidecars`; this module
does not fabricate columns for fields that `Pi_RLC` does not compute.

Owns:
- the public-role ordinal and its injectivity;
- the exact output coefficient columns and their contiguous allocation;
- exact output widths, non-collision, and allocation coverage;
- interleaving the two evaluation limbs into strict-`Pi_DEC` row order; and
- the column-identity `ParentArtifact`.

Does not own:
- values carried by the allocated columns;
- quotient rows or their witness;
- construction or validation of the strict-`Pi_DEC` sidecars;
- the delayed old-point bridge;
- a complete `PiDecRecipe.Decomposition`; or
- Rust/artifact correspondence.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcParentBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.SuperNeo.Concrete

/-! ## Public output allocation -/

/-- Zero-based ordinal of one public role, in `publicOrder` order. -/
def roleIndex {matrixCount : Nat} : PublicRole matrixCount → Nat
  | .commitment lane => lane.val
  | .x block => 18 + block.val
  | .yRing row limb => 23 + 2 * row.val + limb.val

theorem roleIndex_lt {matrixCount : Nat} (role : PublicRole matrixCount) :
    roleIndex role < 23 + 2 * matrixCount := by
  cases role with
  | commitment lane =>
      have := lane.isLt
      simp only [roleIndex] at *
      omega
  | x block =>
      have := block.isLt
      simp only [roleIndex] at *
      omega
  | yRing row limb =>
      have rowLt := row.isLt
      have limbLt := limb.isLt
      simp only [roleIndex] at *
      omega

theorem roleIndex_injective {matrixCount : Nat} :
    Function.Injective (@roleIndex matrixCount) := by
  intro left right equal
  cases left with
  | commitment left =>
      cases right with
      | commitment right =>
          apply congrArg PublicRole.commitment
          apply Fin.ext
          simpa [roleIndex] using equal
      | x right =>
          have leftLt := left.isLt
          have rightLt := right.isLt
          simp only [roleIndex] at equal leftLt rightLt
          omega
      | yRing right limb =>
          have leftLt := left.isLt
          have rightLt := right.isLt
          have limbLt := limb.isLt
          simp only [roleIndex] at equal leftLt rightLt limbLt
          omega
  | x left =>
      cases right with
      | commitment right =>
          have leftLt := left.isLt
          have rightLt := right.isLt
          simp only [roleIndex] at equal leftLt rightLt
          omega
      | x right =>
          apply congrArg PublicRole.x
          apply Fin.ext
          simpa [roleIndex] using equal
      | yRing right limb =>
          have leftLt := left.isLt
          have rightLt := right.isLt
          have limbLt := limb.isLt
          simp only [roleIndex] at equal leftLt rightLt limbLt
          omega
  | yRing left leftLimb =>
      cases right with
      | commitment right =>
          have leftLt := left.isLt
          have limbLt := leftLimb.isLt
          have rightLt := right.isLt
          simp only [roleIndex] at equal leftLt limbLt rightLt
          omega
      | x right =>
          have leftLt := left.isLt
          have limbLt := leftLimb.isLt
          have rightLt := right.isLt
          simp only [roleIndex] at equal leftLt limbLt rightLt
          omega
      | yRing right rightLimb =>
          have leftLt := left.isLt
          have rightLt := right.isLt
          have leftLimbLt := leftLimb.isLt
          have rightLimbLt := rightLimb.isLt
          have rows : left.val = right.val := by
            simp only [roleIndex] at equal
            omega
          have limbs : leftLimb.val = rightLimb.val := by
            simp only [roleIndex] at equal
            omega
          have rowEqual : left = right := Fin.ext rows
          have limbEqual : leftLimb = rightLimb := Fin.ext limbs
          cases rowEqual
          cases limbEqual
          rfl

/-- Numeric column of one public output coefficient. -/
def coefficientColumn
    (base : Nat) {matrixCount : Nat}
    (role : PublicRole matrixCount) (coefficient : Fin ringDegree) : Nat :=
  base + roleIndex role * ringDegree + coefficient.val

/-- Complete public output carrier in the exact role-indexed shape. -/
def outputColumns (base matrixCount : Nat) : ProjectionColumns matrixCount where
  commitment lane :=
    List.ofFn fun coefficient =>
      coefficientColumn base
        (.commitment lane : PublicRole matrixCount) coefficient
  x block :=
    List.ofFn fun coefficient =>
      coefficientColumn base (.x block : PublicRole matrixCount) coefficient
  yRing row limb :=
    List.ofFn fun coefficient => coefficientColumn base (.yRing row limb) coefficient

@[simp] theorem outputColumns_width
    (base matrixCount : Nat) (role : PublicRole matrixCount) :
    ((outputColumns base matrixCount).at role).length = ringDegree := by
  cases role <;> simp [outputColumns, ProjectionColumns.at]

/-- The exact contiguous block occupied by all public output coefficients. -/
def outputAllocation (base matrixCount : Nat) : List Nat :=
  List.range' base ((23 + 2 * matrixCount) * ringDegree)

@[simp] theorem outputAllocation_length (base matrixCount : Nat) :
    (outputAllocation base matrixCount).length =
      (23 + 2 * matrixCount) * ringDegree := by
  simp [outputAllocation]

theorem outputAllocation_nodup (base matrixCount : Nat) :
    (outputAllocation base matrixCount).Nodup := by
  exact List.nodup_range'

theorem coefficientColumn_mem_allocation
    (base : Nat) {matrixCount : Nat}
    (role : PublicRole matrixCount) (coefficient : Fin ringDegree) :
    coefficientColumn base role coefficient ∈
      outputAllocation base matrixCount := by
  rw [outputAllocation, List.mem_range']
  refine ⟨roleIndex role * ringDegree + coefficient.val, ?_, by
    simp [coefficientColumn, Nat.add_assoc]⟩
  have roleLt := roleIndex_lt role
  have coefficientLt := coefficient.isLt
  simp only [ringDegree] at roleLt coefficientLt ⊢
  omega

theorem coefficientColumn_injective
    (base : Nat) {matrixCount : Nat} :
    Function.Injective
      (fun owner : PublicRole matrixCount × Fin ringDegree =>
        coefficientColumn base owner.1 owner.2) := by
  rintro ⟨leftRole, leftCoefficient⟩ ⟨rightRole, rightCoefficient⟩ equal
  have leftRoleLt := roleIndex_lt leftRole
  have rightRoleLt := roleIndex_lt rightRole
  have leftCoefficientLt := leftCoefficient.isLt
  have rightCoefficientLt := rightCoefficient.isLt
  have coefficients : leftCoefficient.val = rightCoefficient.val := by
    have payload :
        roleIndex leftRole * ringDegree + leftCoefficient.val =
          roleIndex rightRole * ringDegree + rightCoefficient.val := by
      simp only [coefficientColumn, Nat.add_assoc] at equal
      exact Nat.add_left_cancel equal
    have modular := congrArg (fun value => value % ringDegree) payload
    simpa [Nat.mul_add_mod_of_lt leftCoefficientLt,
      Nat.mul_add_mod_of_lt rightCoefficientLt] using modular
  have roles : roleIndex leftRole = roleIndex rightRole := by
    have payload :
        roleIndex leftRole * ringDegree + leftCoefficient.val =
          roleIndex rightRole * ringDegree + rightCoefficient.val := by
      simp only [coefficientColumn, Nat.add_assoc] at equal
      exact Nat.add_left_cancel equal
    have divided := congrArg (fun value => value / ringDegree) payload
    simpa [Nat.mul_comm,
      Nat.mul_add_div (by decide : 0 < ringDegree),
      Nat.div_eq_of_lt leftCoefficientLt,
      Nat.div_eq_of_lt rightCoefficientLt] using divided
  have roleEqual := roleIndex_injective roles
  have coefficientEqual := Fin.ext coefficients
  cases roleEqual
  cases coefficientEqual
  rfl

/-! ## Strict-PiDEC public-parent view -/

/-- Columns owned by strict `Pi_DEC` but not computed by public `Pi_RLC`.

Every field is explicit.  In particular, the twenty padding limbs of each
evaluation row are not silently replaced by zero-column references. -/
structure ParentSidecars (matrixCount : Nat) where
  commitmentD : Nat
  commitmentKappa : Nat
  xInactive : Nat
  xRows : Nat
  xWidth : Nat
  xRowsColumn : Nat
  xWidthColumn : Nat
  inputWidth : Nat
  inputWidthColumn : Nat
  evaluationPadding : Fin matrixCount → List Nat
  ct : List (Nat × Nat)
  sColumn : List (Nat × Nat)
  foldDigest : List Nat

/-- Active 108-limb evaluation row, interleaved low/high by coefficient. -/
def activeEvaluationRow
    (base : Nat) {matrixCount : Nat} (row : Fin matrixCount) : List Nat :=
  List.ofFn fun position : Fin (2 * ringDegree) =>
    coefficientColumn base
      (.yRing row
        ⟨position.val % 2, Nat.mod_lt _ (by decide)⟩)
      ⟨position.val / 2, by
        have positionLt := position.isLt
        simp only [ringDegree] at positionLt ⊢
        omega⟩

@[simp] theorem activeEvaluationRow_length
    (base : Nat) {matrixCount : Nat} (row : Fin matrixCount) :
    (activeEvaluationRow base row).length = 2 * ringDegree := by
  simp [activeEvaluationRow]

theorem activeEvaluationRow_getD
    (base : Nat) {matrixCount : Nat} (row : Fin matrixCount)
    (coefficient : Fin ringDegree) (limb : Fin 2) :
    (activeEvaluationRow base row).getD
        (2 * coefficient.val + limb.val) 0 =
      coefficientColumn base (.yRing row limb) coefficient := by
  have positionLt :
      2 * coefficient.val + limb.val <
        (activeEvaluationRow base row).length := by
    rw [activeEvaluationRow_length]
    have coefficientLt := coefficient.isLt
    have limbLt := limb.isLt
    omega
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem positionLt]
  simp only [activeEvaluationRow, List.getElem_ofFn, Option.getD_some]
  congr 1
  · apply congrArg (PublicRole.yRing row)
    apply Fin.ext
    simpa [Nat.mul_comm] using
      Nat.mul_add_mod_of_lt limb.isLt
  · apply Fin.ext
    change (2 * coefficient.val + limb.val) / 2 = coefficient.val
    rw [Nat.mul_add_div (by decide : 0 < 2),
      Nat.div_eq_of_lt limb.isLt, Nat.add_zero]

private theorem getD_append_left
    {Carrier : Type} (left right : List Carrier) (index : Nat)
    (default : Carrier) (indexLt : index < left.length) :
    (left ++ right).getD index default = left.getD index default := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by
      rw [List.length_append]
      omega),
    List.getElem?_eq_getElem indexLt,
    List.getElem_append_left]

private theorem getD_ofFn
    {Carrier : Type} {count : Nat}
    (items : Fin count → Carrier) (index : Fin count) (default : Carrier) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp),
    List.getElem_ofFn]
  simp

/-- Full strict-`PiDEC` row.  The active public prefix is followed by the
explicit sidecar-owned padding suffix. -/
def evaluationRow
    (base : Nat) {matrixCount : Nat}
    (sidecars : ParentSidecars matrixCount) (row : Fin matrixCount) : List Nat :=
  activeEvaluationRow base row ++ sidecars.evaluationPadding row

theorem evaluationRow_getD_active
    (base : Nat) {matrixCount : Nat}
    (sidecars : ParentSidecars matrixCount) (row : Fin matrixCount)
    (coefficient : Fin ringDegree) (limb : Fin 2) :
    (evaluationRow base sidecars row).getD
        (2 * coefficient.val + limb.val) 0 =
      coefficientColumn base (.yRing row limb) coefficient := by
  have indexLt :
      2 * coefficient.val + limb.val <
        (activeEvaluationRow base row).length := by
    rw [activeEvaluationRow_length]
    have coefficientLt := coefficient.isLt
    have limbLt := limb.isLt
    omega
  rw [evaluationRow,
    getD_append_left _ _ _ _ indexLt,
    activeEvaluationRow_getD]

/-- Full strict-`PiDEC` parent layout with one public projection owner and
explicit owners for every excluded sidecar. -/
def parentClaim
    (base : Nat) {matrixCount : Nat}
    (point : PointColumns) (sidecars : ParentSidecars matrixCount) :
    ClaimLayout where
  commitment := {
    dCol := sidecars.commitmentD
    kappaCol := sidecars.commitmentKappa
    dataCols := assembleCommitmentColumns (outputColumns base matrixCount)
  }
  adv := none
  xActiveCols := assembleXColumns (outputColumns base matrixCount)
  xInactiveCol := sidecars.xInactive
  xRows := sidecars.xRows
  xWidth := sidecars.xWidth
  xRowsCol := sidecars.xRowsColumn
  xWidthCol := sidecars.xWidthColumn
  mIn := sidecars.inputWidth
  mInCol := sidecars.inputWidthColumn
  yRingCols := List.ofFn fun row => evaluationRow base sidecars row
  ctCols := sidecars.ct
  rCols := point.r
  sColCols := sidecars.sColumn
  foldDigestCols := sidecars.foldDigest

/-- Public output and strict-parent fields are identical columns, not equal
values asserted by a caller. -/
def batchColumns
    {params : Nightstream.SuperNeo.GlobalParams}
    {arity : Nightstream.SuperNeo.Folding.BatchArity params}
    (base : Nat) {matrixCount : Nat}
    (point : PointColumns) (sidecars : ParentSidecars matrixCount)
    (challenges : Fin arity.total → List Nat)
    (inputs : Fin arity.total → ProjectionColumns matrixCount) :
    BatchColumns params arity matrixCount := {
      parentClaim := parentClaim base point sidecars
      challenges := challenges
      inputs := inputs
      output := outputColumns base matrixCount
      inputPoints := fun _ => point
      outputPoint := point
    }

theorem parentArtifact
    {params : Nightstream.SuperNeo.GlobalParams}
    {arity : Nightstream.SuperNeo.Folding.BatchArity params}
    (base : Nat) {matrixCount : Nat}
    (point : PointColumns) (sidecars : ParentSidecars matrixCount)
    (challenges : Fin arity.total → List Nat)
    (inputs : Fin arity.total → ProjectionColumns matrixCount) :
    ParentArtifact
      (batchColumns base point sidecars challenges inputs) := by
  refine {
    commitment := rfl
    x := rfl
    evaluationRows := by simp [batchColumns, parentClaim]
    yRing := ?_
    r := rfl
  }
  intro row limb
  unfold batchColumns
  simp only [outputColumns]
  apply congrArg List.ofFn
  funext coefficient
  simp only [parentClaim]
  rw [getD_ofFn]
  exact (evaluationRow_getD_active
    base sidecars row coefficient limb).symm

/-! ## Placement in the one call-local namespace -/

/-- One public-output segment inside the call's declared temporary suffix. -/
structure FrameLayout
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (matrixCount : Nat) where
  offset : Nat
  fits :
    offset + (23 + 2 * matrixCount) * ringDegree ≤
      frame.temporaries.ids.length

namespace FrameLayout

def base
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {matrixCount : Nat}
    (layout : FrameLayout frame matrixCount) : Nat :=
  temporarySource frame layout.offset

def columns
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {matrixCount : Nat}
    (layout : FrameLayout frame matrixCount) :
    ProjectionColumns matrixCount :=
  outputColumns layout.base matrixCount

def sourceIndex
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {matrixCount : Nat}
    (layout : FrameLayout frame matrixCount)
    (role : PublicRole matrixCount) (coefficient : Fin ringDegree) : Nat :=
  layout.offset + roleIndex role * ringDegree + coefficient.val

theorem sourceIndex_lt
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {matrixCount : Nat}
    (layout : FrameLayout frame matrixCount)
    (role : PublicRole matrixCount) (coefficient : Fin ringDegree) :
    layout.sourceIndex role coefficient < frame.temporaries.ids.length := by
  have roleLt := roleIndex_lt role
  have coefficientLt := coefficient.isLt
  have fits := layout.fits
  simp only [sourceIndex, ringDegree] at *
  omega

theorem coefficientColumn_eq_temporarySource
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {matrixCount : Nat}
    (layout : FrameLayout frame matrixCount)
    (role : PublicRole matrixCount) (coefficient : Fin ringDegree) :
    coefficientColumn layout.base role coefficient =
      temporarySource frame (layout.sourceIndex role coefficient) := by
  simp [base, sourceIndex, coefficientColumn, temporarySource,
    Nat.add_assoc]

theorem columnMap_coefficientColumn
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {matrixCount : Nat}
    (layout : FrameLayout frame matrixCount)
    (role : PublicRole matrixCount) (coefficient : Fin ringDegree) :
    columnMap frame (coefficientColumn layout.base role coefficient) =
      frame.temporaries.ids[layout.sourceIndex role coefficient]'(
        layout.sourceIndex_lt role coefficient) := by
  rw [layout.coefficientColumn_eq_temporarySource]
  exact columnMap_temporarySource frame
    (layout.sourceIndex_lt role coefficient)

end FrameLayout

@[simp] theorem selected_outputAllocation_length (base : Nat) :
    (outputAllocation base 13).length = 2646 := by
  simp [outputAllocation, ringDegree]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcParentBinding
