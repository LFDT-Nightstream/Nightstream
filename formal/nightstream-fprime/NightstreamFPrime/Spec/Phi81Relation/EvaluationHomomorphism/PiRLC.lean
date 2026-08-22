import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.Embedding
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws

/-! Provenance: adapted from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/EvaluationHomomorphism/PiRLC.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces were renamed and
the v1.1 explicit-matrix evaluation section was added for canonical Pad. -/

/-!
Orchestration of the typed Phi81 `Pi_RLC` evaluation map.

Protocol: SuperNeo Theorem 5, evaluation-homomorphism branch of `Pi_RLC`.
Phase: one complete-carrier action through matrix rows, Boolean MLE, and every
canonical matrix.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: the exact local product-order predicate and its discharge by executable
Phi81 multiplication; exact assembly of the existing carrier, embedding, and
extension-evaluation leaves; and one-source matrix homomorphism theorems.

Does not own: commitments, public input projection, finite-source RingF
combination, norm preservation, transcript challenges, `Pi_RLC` acceptance,
Rust, R1CS, row removal, or counts.

Emits constraints: no.

Authority boundary: matrices, complete assignments, points, and challenges
are explicit typed values. Every derived row, lane, matrix, and array is
computed from the canonical Phi81 source. `ProductOrderLaw challenge` names
exactly the reassociation `bar * (rho * z) = rho * (bar * z)` and is discharged
for every challenge by the symbolic executable-ring proof in `RingFLaws`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.evaluation_hom.product_order` | `bar * (rho * z) = rho * (bar * z)` for one row basis and block | derived | `productOrderLaw` |
| `nifs.pi_rlc.verify.evaluation_hom.row.blocks` | complete flat carrier equals the block/lane contraction of the canonical basis-defined kernel | derived | `rowRing_eq_blockSum` |
| `nifs.pi_rlc.verify.evaluation_hom.row.selector` | a zero or one-hot original matrix row selects exactly zero or one basis-kernel image | derived | `rowRing_eq_zero_of_padded_row_zero`, `rowRing_eq_kernelImage_of_unit_padded_row` |
| `nifs.pi_rlc.verify.evaluation_hom.row.action` | the local law distributes through matrix-row coefficients and blocks | derived | `rowRing_act` with `productOrderLaw` |
| `nifs.pi_rlc.verify.evaluation_hom.mle` | coefficientwise embedding and Boolean MLE preserve the action | derived | `matrixEvaluation_act` with `productOrderLaw` |
| `nifs.pi_rlc.verify.evaluation_hom.matrices` | every canonical matrix evaluation preserves the action | derived | `evaluations_act` with `productOrderLaw` |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLC

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource


/-- The exact product-order identity needed for one challenge. It is strictly
weaker than exposing global associativity and commutativity as separate
premises. -/
def ProductOrderLaw (challenge : RingF) : Prop :=
  forall (row : Fin ringDegree) (block : RingF),
    ringFMul (Phi81CoefficientKernel.barBasis row)
        (ringFMul challenge block) =
      ringFMul challenge
        (ringFMul (Phi81CoefficientKernel.barBasis row) block)

/-- The executable Phi81 quotient multiplication satisfies the exact local
product-order predicate for every verifier challenge. -/
theorem productOrderLaw (challenge : RingF) : ProductOrderLaw challenge :=
  fun row block =>
    RingFLaws.ringFMul_barBasis_productOrder row challenge block

/-! ## Canonical finite sums and carrier regrouping -/

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_comm⟩

private theorem sumRange_zero (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps 0 term = 0 := by
  rfl

private theorem sumRange_succ (count : Nat) (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps (count + 1) term =
      sumRange ConcreteCarrier.baseOps count term + term count := by
  rfl

private theorem fadd_mul (left middle right : F) :
    (left + middle) * right = left * right + middle * right := by
  calc
    (left + middle) * right = right * (left + middle) := Fin.mul_comm _ _
    _ = right * left + right * middle := Lean.Grind.Fin.left_distrib _ _ _
    _ = left * right + middle * right := by
      rw [Fin.mul_comm right left, Fin.mul_comm right middle]

private theorem sumRange_add
    (count : Nat) (left right : Nat -> F) :
    sumRange ConcreteCarrier.baseOps count
        (fun index => left index + right index) =
      sumRange ConcreteCarrier.baseOps count left +
        sumRange ConcreteCarrier.baseOps count right := by
  induction count with
  | zero =>
      rw [sumRange_zero, sumRange_zero, sumRange_zero]
      exact (ConcreteCarrier.baseLaws.zero_add (0 : F)).symm
  | succ count inductionHypothesis =>
      rw [sumRange_succ, sumRange_succ, sumRange_succ,
        inductionHypothesis]
      ac_rfl

private theorem sumRange_mul_left
    (factor : F) (count : Nat) (term : Nat -> F) :
    factor * sumRange ConcreteCarrier.baseOps count term =
      sumRange ConcreteCarrier.baseOps count
        (fun index => factor * term index) := by
  induction count with
  | zero =>
      rw [sumRange_zero, sumRange_zero]
      exact Fin.mul_zero _
  | succ count inductionHypothesis =>
      rw [sumRange_succ, sumRange_succ,
        Lean.Grind.Fin.left_distrib, inductionHypothesis]

private theorem sumRange_mul_right
    (count : Nat) (term : Nat -> F) (factor : F) :
    sumRange ConcreteCarrier.baseOps count term * factor =
      sumRange ConcreteCarrier.baseOps count
        (fun index => term index * factor) := by
  induction count with
  | zero =>
      rw [sumRange_zero, sumRange_zero]
      exact Fin.zero_mul _
  | succ count inductionHypothesis =>
      rw [sumRange_succ, sumRange_succ, fadd_mul,
        inductionHypothesis]

private theorem sumRange_swap
    (outerCount innerCount : Nat) (term : Nat -> Nat -> F) :
    sumRange ConcreteCarrier.baseOps outerCount (fun outer =>
        sumRange ConcreteCarrier.baseOps innerCount (fun inner =>
          term outer inner)) =
      sumRange ConcreteCarrier.baseOps innerCount (fun inner =>
        sumRange ConcreteCarrier.baseOps outerCount (fun outer =>
          term outer inner)) := by
  induction outerCount with
  | zero =>
      rw [sumRange_zero]
      symm
      apply sumRange_eq_zero ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
      intro inner innerLt
      rfl
  | succ outerCount inductionHypothesis =>
      rw [sumRange_succ, inductionHypothesis]
      symm
      calc
        sumRange ConcreteCarrier.baseOps innerCount (fun inner =>
            sumRange ConcreteCarrier.baseOps (outerCount + 1) (fun outer =>
              term outer inner)) =
          sumRange ConcreteCarrier.baseOps innerCount (fun inner =>
            sumRange ConcreteCarrier.baseOps outerCount (fun outer =>
                term outer inner) + term outerCount inner) := by
              apply sumRange_congr
              intro inner innerLt
              rw [sumRange_succ]
        _ = sumRange ConcreteCarrier.baseOps innerCount (fun inner =>
              sumRange ConcreteCarrier.baseOps outerCount (fun outer =>
                term outer inner)) +
            sumRange ConcreteCarrier.baseOps innerCount
              (fun inner => term outerCount inner) :=
          sumRange_add innerCount _ _

private theorem sumRange_append
    (leftCount rightCount : Nat) (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps (leftCount + rightCount) term =
      sumRange ConcreteCarrier.baseOps leftCount term +
        sumRange ConcreteCarrier.baseOps rightCount
          (fun index => term (leftCount + index)) := by
  induction rightCount with
  | zero =>
      rw [Nat.add_zero, sumRange_zero]
      exact (ConcreteCarrier.baseLaws.add_zero _).symm
  | succ rightCount inductionHypothesis =>
      rw [Nat.add_succ, sumRange_succ, sumRange_succ,
        inductionHypothesis]
      exact Lean.Grind.Fin.add_assoc _ _ _

/-- A complete block-major carrier is the consecutive concatenation of its
54-lane chunks. -/
private theorem sumRange_chunks
    (blockCount : Nat) (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps (blockCount * ringDegree) term =
      sumRange ConcreteCarrier.baseOps blockCount (fun block =>
        sumRange ConcreteCarrier.baseOps ringDegree (fun lane =>
          term (block * ringDegree + lane))) := by
  induction blockCount with
  | zero => rfl
  | succ blockCount inductionHypothesis =>
      rw [Nat.succ_mul, sumRange_append, inductionHypothesis, sumRange_succ]

private def listSum {Index : Type}
    (indices : List Index) (term : Index -> F) : F :=
  match indices with
  | [] => 0
  | index :: rest => term index + listSum rest term

private theorem foldl_eq_add_listSum
    {Index : Type} (indices : List Index) (term : Index -> F)
    (initial : F) :
    indices.foldl (fun accumulated index => accumulated + term index) initial =
      initial + listSum indices term := by
  induction indices generalizing initial with
  | nil => exact (ConcreteCarrier.baseLaws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem listSum_append
    {Index : Type} (left right : List Index) (term : Index -> F) :
    listSum (left ++ right) term = listSum left term + listSum right term := by
  induction left with
  | nil => exact (ConcreteCarrier.baseLaws.zero_add _).symm
  | cons index left inductionHypothesis =>
      simp only [List.cons_append, listSum, inductionHypothesis]
      exact (ConcreteCarrier.baseLaws.add_assoc _ _ _).symm

private theorem listSum_map
    {Left Right : Type} (indices : List Left) (map : Left -> Right)
    (term : Right -> F) :
    listSum (indices.map map) term =
      listSum indices (fun index => term (map index)) := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.map_cons, listSum, inductionHypothesis]

private theorem listSum_range_eq_sumRange
    (count : Nat) (term : Nat -> F) :
    listSum (List.range count) term =
      sumRange ConcreteCarrier.baseOps count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, listSum_append, listSum, inductionHypothesis,
        sumRange_succ]
      rw [listSum]
      rw [Fin.add_zero]

private def sumFinF {count : Nat} (term : Fin count -> F) : F :=
  sumRange ConcreteCarrier.baseOps count fun index =>
    if indexLt : index < count then term ⟨index, indexLt⟩ else 0

private theorem sumFinF_congr
    {count : Nat} (left right : Fin count -> F)
    (equal : forall index, left index = right index) :
    sumFinF left = sumFinF right := by
  unfold sumFinF
  apply sumRange_congr
  intro index indexLt
  rw [dif_pos indexLt, dif_pos indexLt]
  exact equal ⟨index, indexLt⟩

private theorem sumFinF_zero {count : Nat} :
    sumFinF (fun _ : Fin count => (0 : F)) = 0 := by
  unfold sumFinF
  apply sumRange_eq_zero ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
  intro index indexLt
  rw [dif_pos indexLt]
  rfl

private theorem sumFinF_select
    {count : Nat} (selected : Fin count) (term : Fin count -> F) :
    sumFinF (fun index => if index = selected then term index else 0) =
      term selected := by
  unfold sumFinF
  calc
    sumRange ConcreteCarrier.baseOps count (fun index =>
        if indexLt : index < count then
          if (⟨index, indexLt⟩ : Fin count) = selected then
            term ⟨index, indexLt⟩
          else
            0
        else
          0) =
      sumRange ConcreteCarrier.baseOps count (fun index =>
        if index = selected.val then term selected else 0) := by
      apply sumRange_congr
      intro index indexLt
      rw [dif_pos indexLt]
      by_cases equal : index = selected.val
      · have finEqual : (⟨index, indexLt⟩ : Fin count) = selected :=
          Fin.ext equal
        rw [if_pos equal, if_pos finEqual, finEqual]
      · have finDifferent : (⟨index, indexLt⟩ : Fin count) ≠ selected := by
          intro finEqual
          exact equal (congrArg Fin.val finEqual)
        rw [if_neg equal, if_neg finDifferent]
    _ = term selected := by
      exact MatrixCoefficientSource.sumRange_select ConcreteCarrier.baseOps
        ConcreteCarrier.baseLaws count selected.val (fun _ => term selected)
        selected.isLt

private theorem sumFinF_mul_left
    {count : Nat} (factor : F) (term : Fin count -> F) :
    factor * sumFinF term = sumFinF fun index => factor * term index := by
  unfold sumFinF
  rw [sumRange_mul_left]
  apply sumRange_congr
  intro index indexLt
  rw [dif_pos indexLt, dif_pos indexLt]

private theorem sumFinF_mul_right
    {count : Nat} (term : Fin count -> F) (factor : F) :
    sumFinF term * factor = sumFinF fun index => term index * factor := by
  unfold sumFinF
  rw [sumRange_mul_right]
  apply sumRange_congr
  intro index indexLt
  rw [dif_pos indexLt, dif_pos indexLt]

private theorem sumFinF_swap
    {outerCount innerCount : Nat}
    (term : Fin outerCount -> Fin innerCount -> F) :
    sumFinF (fun outer => sumFinF fun inner => term outer inner) =
      sumFinF (fun inner => sumFinF fun outer => term outer inner) := by
  let guarded : Nat -> Nat -> F := fun outer inner =>
    if outerLt : outer < outerCount then
      if innerLt : inner < innerCount then
        term ⟨outer, outerLt⟩ ⟨inner, innerLt⟩
      else
        0
    else
      0
  calc
    sumFinF (fun outer => sumFinF fun inner => term outer inner) =
        sumRange ConcreteCarrier.baseOps outerCount (fun outer =>
          sumRange ConcreteCarrier.baseOps innerCount (guarded outer)) := by
      unfold sumFinF
      apply sumRange_congr
      intro outer outerLt
      rw [dif_pos outerLt]
      apply sumRange_congr
      intro inner innerLt
      unfold guarded
      simp only [dif_pos outerLt, dif_pos innerLt]
    _ = sumRange ConcreteCarrier.baseOps innerCount (fun inner =>
          sumRange ConcreteCarrier.baseOps outerCount (fun outer =>
            guarded outer inner)) := sumRange_swap _ _ guarded
    _ = sumFinF (fun inner => sumFinF fun outer => term outer inner) := by
      unfold sumFinF
      apply sumRange_congr
      intro inner innerLt
      rw [dif_pos innerLt]
      apply sumRange_congr
      intro outer outerLt
      unfold guarded
      simp only [dif_pos outerLt, dif_pos innerLt]

/-! ## Ring-valued finite sums -/

private def ringFSumRange : Nat -> (Nat -> RingF) -> RingF
  | 0, _ => ringFZero
  | count + 1, term =>
      ringFAdd (ringFSumRange count term) (term count)

private theorem ringFSumRange_apply
    (count : Nat) (term : Nat -> RingF) (output : Fin ringDegree) :
    ringFSumRange count term output =
      sumRange ConcreteCarrier.baseOps count fun index => term index output := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [ringFSumRange, ringFAdd, sumRange, ConcreteCarrier.baseOps,
        inductionHypothesis]

private theorem ringFMul_ringFSumRange_right
    (left : RingF) (count : Nat) (term : Nat -> RingF) :
    ringFMul left (ringFSumRange count term) =
      ringFSumRange count fun index => ringFMul left (term index) := by
  induction count with
  | zero => exact CarrierAction.ringFMul_zero_right left
  | succ count inductionHypothesis =>
      rw [ringFSumRange, ringFSumRange, CarrierAction.ringFMul_add_right,
        inductionHypothesis]

private def sumRingF {count : Nat} (term : Fin count -> RingF) : RingF :=
  ringFSumRange count fun index =>
    if indexLt : index < count then term ⟨index, indexLt⟩ else ringFZero

private theorem sumRingF_apply
    {count : Nat} (term : Fin count -> RingF)
    (output : Fin ringDegree) :
    sumRingF term output = sumFinF fun index => term index output := by
  unfold sumRingF sumFinF
  rw [ringFSumRange_apply]
  apply sumRange_congr
  intro index indexLt
  rw [dif_pos indexLt, dif_pos indexLt]

private theorem sumRingF_congr
    {count : Nat} (left right : Fin count -> RingF)
    (equal : forall index, left index = right index) :
    sumRingF left = sumRingF right := by
  funext output
  rw [sumRingF_apply, sumRingF_apply]
  apply sumFinF_congr
  intro index
  exact congrFun (equal index) output

private theorem ringFMul_sumRingF_right
    {count : Nat} (left : RingF) (term : Fin count -> RingF) :
    ringFMul left (sumRingF term) =
      sumRingF fun index => ringFMul left (term index) := by
  unfold sumRingF
  rw [ringFMul_ringFSumRange_right]
  apply congrArg (ringFSumRange count)
  funext index
  by_cases indexLt : index < count
  · rw [dif_pos indexLt, dif_pos indexLt]
  · rw [dif_neg indexLt, dif_neg indexLt]
    exact CarrierAction.ringFMul_zero_right left

private theorem listSum_canonical_eq_sumFinF
    {count : Nat} (term : Fin count -> F) :
    listSum (canonicalFinIndices count) term = sumFinF term := by
  let natTerm : Nat -> F := fun index =>
    if indexLt : index < count then term ⟨index, indexLt⟩ else 0
  have termFromValue :
      (fun index : Fin count => natTerm index.val) = term := by
    funext index
    unfold natTerm
    rw [dif_pos index.isLt]
  calc
    listSum (canonicalFinIndices count) term =
        listSum (canonicalFinIndices count)
          (fun index => natTerm index.val) := by rw [termFromValue]
    _ = listSum ((canonicalFinIndices count).map Fin.val) natTerm :=
      (listSum_map (canonicalFinIndices count) Fin.val natTerm).symm
    _ = listSum (List.range count) natTerm := by
      rw [canonicalFinIndices_values]
    _ = sumRange ConcreteCarrier.baseOps count natTerm :=
      listSum_range_eq_sumRange count natTerm
    _ = sumFinF term := rfl

private theorem matrixVectorAt_eq_sumFinF
    {variables columns : Nat}
    (matrix : PaperLinearAlgebra.BooleanMatrix F variables columns)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (vertex : BooleanVertex variables) :
    PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps matrix assignment vertex =
      sumFinF (fun column => matrix vertex column * assignment column) := by
  unfold PaperLinearAlgebra.matrixVectorAt
  calc
    (canonicalFinIndices columns).foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * assignment column) 0 =
      0 + listSum (canonicalFinIndices columns)
        (fun column => matrix vertex column * assignment column) :=
          foldl_eq_add_listSum _ _ 0
    _ = listSum (canonicalFinIndices columns)
        (fun column => matrix vertex column * assignment column) :=
      ConcreteCarrier.baseLaws.zero_add _
    _ = _ := listSum_canonical_eq_sumFinF _

/-- Regrouping the complete carrier into its canonical 54-lane blocks changes
only the indexing, not the finite sum. -/
private theorem sumFinF_carrier_blocks
    {shape : Shape} (term : Fin shape.carrierWidth -> F) :
    sumFinF term =
      sumFinF fun block :
          Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
        sumFinF fun lane : Fin ringDegree =>
          term (CarrierAction.carrierColumn block lane) := by
  have widthEquality :
      shape.carrierWidth =
        Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree := by
    unfold Shape.carrierWidth
    rw [Phi81CarrierLayout.blockCount_carrierWidth]
    exact Phi81CarrierLayout.carrierWidth_eq shape.logicalWidth
  let natTerm : Nat -> F := fun index =>
    if indexLt : index < shape.carrierWidth then
      term ⟨index, indexLt⟩
    else
      0
  change
    sumRange ConcreteCarrier.baseOps shape.carrierWidth natTerm =
      sumRange ConcreteCarrier.baseOps
        (Phi81ColumnLayout.blockCount shape.carrierWidth) (fun blockIndex =>
          if blockLt :
              blockIndex < Phi81ColumnLayout.blockCount shape.carrierWidth then
            sumRange ConcreteCarrier.baseOps ringDegree (fun laneIndex =>
              if laneLt : laneIndex < ringDegree then
                term (CarrierAction.carrierColumn
                  ⟨blockIndex, blockLt⟩ ⟨laneIndex, laneLt⟩)
              else
                0)
          else
            0)
  calc
    sumRange ConcreteCarrier.baseOps shape.carrierWidth natTerm =
        sumRange ConcreteCarrier.baseOps
          (Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree)
          natTerm := by
      exact congrArg
        (fun count => sumRange ConcreteCarrier.baseOps count natTerm)
        widthEquality
    _ = sumRange ConcreteCarrier.baseOps
          (Phi81ColumnLayout.blockCount shape.carrierWidth)
          (fun blockIndex =>
            sumRange ConcreteCarrier.baseOps ringDegree (fun laneIndex =>
              natTerm (blockIndex * ringDegree + laneIndex))) :=
      sumRange_chunks _ natTerm
    _ = _ := by
      apply sumRange_congr
      intro blockIndex blockLt
      rw [dif_pos blockLt]
      apply sumRange_congr
      intro laneIndex laneLt
      rw [dif_pos laneLt]
      have flatLt :
          blockIndex * ringDegree + laneIndex < shape.carrierWidth := by
        rw [widthEquality]
        simp only [ringDegree] at laneLt ⊢
        omega
      unfold natTerm
      rw [dif_pos flatLt]
      apply congrArg term
      apply Fin.ext
      rfl

/-! ## Matrix-row decomposition -/

/-- Exact base-ring value of one derived matrix image at one Boolean row. -/
def rowRing {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables) : RingF :=
  fun output =>
    PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
      (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix output)
      assignment vertex

/-- Contribution of one complete 54-lane assignment block to one original
matrix row. The original matrix coefficient is applied only after the
canonical basis-kernel contraction. -/
def blockRowRing {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) : RingF :=
  fun output =>
    sumFinF fun row : Fin ringDegree =>
      system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
          matrix vertex block row *
        CarrierAction.kernelImage row
          (CarrierAction.assignmentBlock assignment block) output

/-- Canonical sum of the block contributions for one original matrix row. -/
def blockRowSum {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables) : RingF :=
  fun output =>
    sumFinF fun block :
        Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
      blockRowRing system assignment matrix vertex block output

/-- At a canonical block/lane column, the derived coefficient matrix is
exactly the finite contraction of the canonical basis-defined kernel against
that lane. -/
private theorem coefficientMatrix_carrierColumn
    {shape : Shape}
    (system : Structure shape) (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables)
    (output : Fin ringDegree)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
        matrix output vertex (CarrierAction.carrierColumn block lane) =
      sumFinF fun row : Fin ringDegree =>
        system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
            matrix vertex block row *
          Phi81CoefficientKernel.phi81Kernel.weight output row lane := by
  have decoded :
      system.matrixSource.columnLayout.decode
          (CarrierAction.carrierColumn block lane) = (block, lane) := by
    exact CarrierAction.decode_carrierColumn block lane
  unfold MatrixCoefficientSource.MatrixSource.coefficientMatrix
    MatrixCoefficientSource.MatrixSource.coefficientMatrixOf sumFinF
  rw [decoded]
  rfl

set_option maxRecDepth 100000 in -- fixed-size: ring degree 54, not artifact data
/-- The canonical basis-kernel contraction has the same explicit finite-sum
coordinate used by the connected matrix source. -/
theorem kernelImage_apply
    (row : Fin ringDegree) (block : RingF) (output : Fin ringDegree) :
    CarrierAction.kernelImage row block output =
      sumRange ConcreteCarrier.baseOps ringDegree fun index =>
        if indexLt : index < ringDegree then
          block ⟨index, indexLt⟩ *
            Phi81CoefficientKernel.phi81Kernel.weight output row
              ⟨index, indexLt⟩
        else
          0 := by
  rfl

set_option maxRecDepth 100000 in -- fixed-size: ring degree 54, not artifact data
private theorem kernelImage_eq_sumFinF
    (row : Fin ringDegree) (block : RingF) (output : Fin ringDegree) :
    CarrierAction.kernelImage row block output =
      sumFinF fun lane : Fin ringDegree =>
        block lane *
          Phi81CoefficientKernel.phi81Kernel.weight output row lane := by
  rw [kernelImage_apply]
  rfl

/-- The flat derived matrix-vector product is exactly the canonical
block/row/kernel tree. This is the structural connection used by the action
proof; it is not a second evaluator supplied by the caller. -/
theorem rowRing_eq_blockSum
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables) :
    rowRing system assignment matrix vertex =
      blockRowSum system assignment matrix vertex := by
  funext output
  unfold rowRing
  rw [matrixVectorAt_eq_sumFinF, sumFinF_carrier_blocks]
  unfold blockRowSum blockRowRing
  apply sumFinF_congr
  intro block
  let matrixCoefficient : Fin ringDegree -> F := fun row =>
    system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
      matrix vertex block row
  let assignmentBlock : RingF :=
    CarrierAction.assignmentBlock assignment block
  let weight : Fin ringDegree -> Fin ringDegree -> F := fun row lane =>
    Phi81CoefficientKernel.phi81Kernel.weight output row lane
  calc
    sumFinF (fun lane =>
        system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
            matrix output vertex (CarrierAction.carrierColumn block lane) *
          assignment (CarrierAction.carrierColumn block lane)) =
      sumFinF (fun lane =>
        sumFinF (fun row => matrixCoefficient row * weight row lane) *
          assignmentBlock lane) := by
      apply sumFinF_congr
      intro lane
      rw [coefficientMatrix_carrierColumn
        system matrix vertex output block lane]
      rfl
    _ =
      sumFinF (fun lane =>
        sumFinF (fun row =>
          (matrixCoefficient row * weight row lane) *
            assignmentBlock lane)) := by
      apply sumFinF_congr
      intro lane
      exact sumFinF_mul_right _ _
    _ = sumFinF (fun row =>
          sumFinF (fun lane =>
            (matrixCoefficient row * weight row lane) *
              assignmentBlock lane)) :=
      sumFinF_swap _
    _ = sumFinF (fun row =>
          matrixCoefficient row *
            sumFinF (fun lane => assignmentBlock lane * weight row lane)) := by
      apply sumFinF_congr
      intro row
      rw [sumFinF_mul_left]
      apply sumFinF_congr
      intro lane
      calc
        (matrixCoefficient row * weight row lane) * assignmentBlock lane =
            matrixCoefficient row *
              (weight row lane * assignmentBlock lane) :=
          Lean.Grind.Fin.mul_assoc _ _ _
        _ = matrixCoefficient row *
              (assignmentBlock lane * weight row lane) := by
          rw [Fin.mul_comm (weight row lane) (assignmentBlock lane)]
    _ = sumFinF (fun row =>
          matrixCoefficient row *
            CarrierAction.kernelImage row assignmentBlock output) := by
      apply sumFinF_congr
      intro row
      rw [kernelImage_eq_sumFinF]
    _ = _ := rfl

/-- A verifier-owned all-zero original matrix row produces the zero derived
Phi81 row. This is a local selector fact, not an authority claim. -/
theorem rowRing_eq_zero_of_padded_row_zero
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables)
    (rowZero : forall
      (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
      (lane : Fin ringDegree),
      system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
        matrix vertex block lane = 0) :
    rowRing system assignment matrix vertex = ringFZero := by
  rw [rowRing_eq_blockSum]
  funext output
  unfold blockRowSum blockRowRing ringFZero
  simp only [rowZero, Fin.zero_mul, sumFinF_zero]

/-- A verifier-owned one-hot original matrix row selects exactly the named
basis-kernel image of the named supplied complete-assignment block. This is a
local algebraic fact and does not establish opening authority. -/
theorem rowRing_eq_kernelImage_of_unit_padded_row
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables)
    (selectedBlock : Fin
      (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (selectedLane : Fin ringDegree)
    (unitRow : forall
      (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
      (lane : Fin ringDegree),
      system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
          matrix vertex block lane =
        if block = selectedBlock then
          if lane = selectedLane then 1 else 0
        else
          0) :
    rowRing system assignment matrix vertex =
      CarrierAction.kernelImage selectedLane
        (CarrierAction.assignmentBlock assignment selectedBlock) := by
  rw [rowRing_eq_blockSum]
  funext output
  unfold blockRowSum blockRowRing
  calc
    sumFinF (fun block =>
        sumFinF fun row =>
          system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
              matrix vertex block row *
            CarrierAction.kernelImage row
              (CarrierAction.assignmentBlock assignment block) output) =
      sumFinF (fun block =>
        if block = selectedBlock then
          sumFinF fun row =>
            if row = selectedLane then
              CarrierAction.kernelImage row
                (CarrierAction.assignmentBlock assignment block) output
            else
              0
        else
          0) := by
      apply sumFinF_congr
      intro block
      by_cases blockEqual : block = selectedBlock
      · rw [if_pos blockEqual]
        apply sumFinF_congr
        intro row
        rw [unitRow block row, if_pos blockEqual]
        split <;> simp only [Fin.one_mul, Fin.zero_mul]
      · rw [if_neg blockEqual]
        calc
          sumFinF (fun row =>
              system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
                  matrix vertex block row *
                CarrierAction.kernelImage row
                  (CarrierAction.assignmentBlock assignment block) output) =
            sumFinF (fun _ : Fin ringDegree => (0 : F)) := by
              apply sumFinF_congr
              intro row
              rw [unitRow block row, if_neg blockEqual, Fin.zero_mul]
          _ = 0 := sumFinF_zero
    _ = CarrierAction.kernelImage selectedLane
          (CarrierAction.assignmentBlock assignment selectedBlock) output := by
      rw [sumFinF_select selectedBlock, sumFinF_select selectedLane]

/-! ## Conditional assignment action -/

private theorem blockRowRing_eq_sumRingF
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    blockRowRing system assignment matrix vertex block =
      sumRingF fun row : Fin ringDegree =>
        CarrierAction.ringFScale
          (system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
            matrix vertex block row)
          (CarrierAction.kernelImage row
            (CarrierAction.assignmentBlock assignment block)) := by
  funext output
  unfold blockRowRing CarrierAction.ringFScale
  rw [sumRingF_apply]

private theorem blockRowSum_eq_sumRingF
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables) :
    blockRowSum system assignment matrix vertex =
      sumRingF fun block :
          Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
        blockRowRing system assignment matrix vertex block := by
  funext output
  unfold blockRowSum
  rw [sumRingF_apply]

private theorem weightedKernel_act
    (challenge : RingF) (law : ProductOrderLaw challenge)
    (scalar : F) (row : Fin ringDegree) (block : RingF) :
    CarrierAction.ringFScale scalar
        (CarrierAction.kernelImage row (ringFMul challenge block)) =
      ringFMul challenge
        (CarrierAction.ringFScale scalar
          (CarrierAction.kernelImage row block)) := by
  calc
    CarrierAction.ringFScale scalar
        (CarrierAction.kernelImage row (ringFMul challenge block)) =
      CarrierAction.ringFScale scalar
        (ringFMul (Phi81CoefficientKernel.barBasis row)
          (ringFMul challenge block)) := by
      rw [CarrierAction.kernelImage_eq_ringFMul]
    _ = CarrierAction.ringFScale scalar
          (ringFMul challenge
            (ringFMul (Phi81CoefficientKernel.barBasis row) block)) := by
      exact congrArg (CarrierAction.ringFScale scalar) (law row block)
    _ = CarrierAction.ringFScale scalar
          (ringFMul challenge (CarrierAction.kernelImage row block)) := by
      rw [CarrierAction.kernelImage_eq_ringFMul]
    _ = ringFMul challenge
          (CarrierAction.ringFScale scalar
            (CarrierAction.kernelImage row block)) :=
      (CarrierAction.ringFMul_scale_right challenge scalar
        (CarrierAction.kernelImage row block)).symm

/-- One canonical carrier block preserves its complete derived matrix-row
contribution under the RingF assignment action, conditional only on the exact
local product-order law. -/
theorem blockRowRing_act
    {shape : Shape}
    (system : Structure shape) (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    blockRowRing system (CarrierAction.act challenge assignment)
        matrix vertex block =
      ringFMul challenge
        (blockRowRing system assignment matrix vertex block) := by
  rw [blockRowRing_eq_sumRingF, blockRowRing_eq_sumRingF,
    CarrierAction.assignmentBlock_act]
  calc
    sumRingF (fun row : Fin ringDegree =>
        CarrierAction.ringFScale
          (system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
            matrix vertex block row)
          (CarrierAction.kernelImage row
            (ringFMul challenge
              (CarrierAction.assignmentBlock assignment block)))) =
      sumRingF (fun row : Fin ringDegree =>
        ringFMul challenge
          (CarrierAction.ringFScale
            (system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
              matrix vertex block row)
            (CarrierAction.kernelImage row
              (CarrierAction.assignmentBlock assignment block)))) := by
      apply sumRingF_congr
      intro row
      exact weightedKernel_act challenge law _ row _
    _ = ringFMul challenge
          (sumRingF fun row : Fin ringDegree =>
            CarrierAction.ringFScale
              (system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
                matrix vertex block row)
              (CarrierAction.kernelImage row
                (CarrierAction.assignmentBlock assignment block))) :=
      (ringFMul_sumRingF_right challenge _).symm

/-- Summing every canonical block preserves the same RingF action. -/
theorem blockRowSum_act
    {shape : Shape}
    (system : Structure shape) (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables) :
    blockRowSum system (CarrierAction.act challenge assignment)
        matrix vertex =
      ringFMul challenge (blockRowSum system assignment matrix vertex) := by
  rw [blockRowSum_eq_sumRingF, blockRowSum_eq_sumRingF]
  calc
    sumRingF (fun block :
        Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
      blockRowRing system (CarrierAction.act challenge assignment)
        matrix vertex block) =
      sumRingF (fun block :
          Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
        ringFMul challenge
          (blockRowRing system assignment matrix vertex block)) := by
      apply sumRingF_congr
      intro block
      exact blockRowRing_act system challenge law assignment matrix vertex block
    _ = ringFMul challenge
          (sumRingF fun block :
              Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
            blockRowRing system assignment matrix vertex block) :=
      (ringFMul_sumRingF_right challenge _).symm

/-- The canonical derived matrix row commutes with the complete RingF
assignment action. The theorem carries exactly one local algebraic premise. -/
theorem rowRing_act
    {shape : Shape}
    (system : Structure shape) (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.rowVariables) :
    rowRing system (CarrierAction.act challenge assignment) matrix vertex =
      ringFMul challenge (rowRing system assignment matrix vertex) := by
  rw [rowRing_eq_blockSum, rowRing_eq_blockSum]
  exact blockRowSum_act system challenge law assignment matrix vertex

/-- The existing typed matrix evaluator is exactly Boolean MLE of the
coefficientwise-embedded canonical row rings. -/
theorem matrixEvaluation_eq_evaluateRows
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (point : Point shape) (matrix : Fin shape.matrixCount) :
    matrixEvaluation system assignment point matrix =
      RingKAction.evaluateRows
        (fun vertex => RingKAction.embedChallenge
          (rowRing system assignment matrix vertex)) point := by
  rfl

/-- One complete Phi81 matrix evaluation commutes with the exact RingF
assignment action. All carrier and embedding work is discharged before the
existing Boolean-MLE action theorem is used. -/
theorem matrixEvaluation_act
    {shape : Shape}
    (system : Structure shape) (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (point : Point shape) (matrix : Fin shape.matrixCount) :
    matrixEvaluation system (CarrierAction.act challenge assignment)
        point matrix =
      ringKMul (RingKAction.embedChallenge challenge)
        (matrixEvaluation system assignment point matrix) := by
  rw [matrixEvaluation_eq_evaluateRows, matrixEvaluation_eq_evaluateRows]
  let rows : RingKAction.Rows shape.rowVariables := fun vertex =>
    RingKAction.embedChallenge (rowRing system assignment matrix vertex)
  calc
    RingKAction.evaluateRows
        (fun vertex => RingKAction.embedChallenge
          (rowRing system (CarrierAction.act challenge assignment)
            matrix vertex)) point =
      RingKAction.evaluateRows
        (RingKAction.actRows (RingKAction.embedChallenge challenge) rows)
        point := by
      apply congrArg (fun rowValues => RingKAction.evaluateRows rowValues point)
      funext vertex
      unfold RingKAction.actRows rows
      rw [rowRing_act system challenge law assignment matrix vertex,
        Embedding.embedChallenge_ringFMul]
    _ = ringKMul (RingKAction.embedChallenge challenge)
          (RingKAction.evaluateRows rows point) :=
      RingKAction.evaluateRows_embeddedChallenge_action challenge rows point
    _ = _ := rfl

/-! ## Explicit completed-matrix action

SuperNeo v1.1 owns `Pad` separately from the CCS matrix family. These
definitions generalize the stored-matrix proof above to one explicit
completed matrix while retaining the canonical Phi81 source, kernel, and
carrier layout. -/

namespace ExplicitMatrix

def rowRing
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables) : RingF :=
  fun output =>
    PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
      (system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps
        matrix output)
      assignment vertex

private def blockRowRing
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) : RingF :=
  fun output =>
    sumFinF fun row : Fin ringDegree =>
      system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix vertex
          block row *
        CarrierAction.kernelImage row
          (CarrierAction.assignmentBlock assignment block) output

private def blockRowSum
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables) : RingF :=
  fun output =>
    sumFinF fun block :
        Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
      blockRowRing system matrix assignment vertex block output

private theorem coefficientMatrix_carrierColumn
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (vertex : BooleanVertex shape.rowVariables)
    (output : Fin ringDegree)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps matrix
        output vertex (CarrierAction.carrierColumn block lane) =
      sumFinF fun row : Fin ringDegree =>
        system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix vertex
            block row *
          Phi81CoefficientKernel.phi81Kernel.weight output row lane := by
  have decoded :
      system.matrixSource.columnLayout.decode
          (CarrierAction.carrierColumn block lane) = (block, lane) :=
    CarrierAction.decode_carrierColumn block lane
  unfold MatrixCoefficientSource.MatrixSource.coefficientMatrixOf sumFinF
  rw [decoded]
  rfl

private theorem rowRing_eq_blockSum
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables) :
    rowRing system matrix assignment vertex =
      blockRowSum system matrix assignment vertex := by
  funext output
  unfold rowRing
  rw [matrixVectorAt_eq_sumFinF, sumFinF_carrier_blocks]
  unfold blockRowSum blockRowRing
  apply sumFinF_congr
  intro block
  let matrixCoefficient : Fin ringDegree → F := fun row =>
    system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix vertex
      block row
  let assignmentBlock : RingF :=
    CarrierAction.assignmentBlock assignment block
  let weight : Fin ringDegree → Fin ringDegree → F := fun row lane =>
    Phi81CoefficientKernel.phi81Kernel.weight output row lane
  calc
    sumFinF (fun lane =>
        system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps
            matrix output vertex (CarrierAction.carrierColumn block lane) *
          assignment (CarrierAction.carrierColumn block lane)) =
      sumFinF (fun lane =>
        sumFinF (fun row => matrixCoefficient row * weight row lane) *
          assignmentBlock lane) := by
      apply sumFinF_congr
      intro lane
      rw [coefficientMatrix_carrierColumn
        system matrix vertex output block lane]
      rfl
    _ = sumFinF (fun lane =>
        sumFinF (fun row =>
          (matrixCoefficient row * weight row lane) *
            assignmentBlock lane)) := by
      apply sumFinF_congr
      intro lane
      exact sumFinF_mul_right _ _
    _ = sumFinF (fun row =>
          sumFinF (fun lane =>
            (matrixCoefficient row * weight row lane) *
              assignmentBlock lane)) :=
      sumFinF_swap _
    _ = sumFinF (fun row =>
          matrixCoefficient row *
            sumFinF (fun lane => assignmentBlock lane * weight row lane)) := by
      apply sumFinF_congr
      intro row
      rw [sumFinF_mul_left]
      apply sumFinF_congr
      intro lane
      calc
        (matrixCoefficient row * weight row lane) * assignmentBlock lane =
            matrixCoefficient row *
              (weight row lane * assignmentBlock lane) :=
          Lean.Grind.Fin.mul_assoc _ _ _
        _ = matrixCoefficient row *
              (assignmentBlock lane * weight row lane) := by
          rw [Fin.mul_comm (weight row lane) (assignmentBlock lane)]
    _ = sumFinF (fun row =>
          matrixCoefficient row *
            CarrierAction.kernelImage row assignmentBlock output) := by
      apply sumFinF_congr
      intro row
      rw [kernelImage_eq_sumFinF]
    _ = _ := rfl

private theorem blockRowRing_eq_sumRingF
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    blockRowRing system matrix assignment vertex block =
      sumRingF fun row : Fin ringDegree =>
        CarrierAction.ringFScale
          (system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix
            vertex block row)
          (CarrierAction.kernelImage row
            (CarrierAction.assignmentBlock assignment block)) := by
  funext output
  unfold blockRowRing CarrierAction.ringFScale
  rw [sumRingF_apply]

private theorem blockRowSum_eq_sumRingF
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables) :
    blockRowSum system matrix assignment vertex =
      sumRingF fun block :
          Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
        blockRowRing system matrix assignment vertex block := by
  funext output
  unfold blockRowSum
  rw [sumRingF_apply]

private theorem blockRowRing_act
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    blockRowRing system matrix (CarrierAction.act challenge assignment)
        vertex block =
      ringFMul challenge
        (blockRowRing system matrix assignment vertex block) := by
  rw [blockRowRing_eq_sumRingF, blockRowRing_eq_sumRingF,
    CarrierAction.assignmentBlock_act]
  calc
    sumRingF (fun row : Fin ringDegree =>
        CarrierAction.ringFScale
          (system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix
            vertex block row)
          (CarrierAction.kernelImage row
            (ringFMul challenge
              (CarrierAction.assignmentBlock assignment block)))) =
      sumRingF (fun row : Fin ringDegree =>
        ringFMul challenge
          (CarrierAction.ringFScale
            (system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix
              vertex block row)
            (CarrierAction.kernelImage row
              (CarrierAction.assignmentBlock assignment block)))) := by
      apply sumRingF_congr
      intro row
      exact weightedKernel_act challenge law _ row _
    _ = ringFMul challenge
          (sumRingF fun row : Fin ringDegree =>
            CarrierAction.ringFScale
              (system.matrixSource.paddedEntry ConcreteCarrier.baseOps matrix
                vertex block row)
              (CarrierAction.kernelImage row
                (CarrierAction.assignmentBlock assignment block))) :=
      (ringFMul_sumRingF_right challenge _).symm

private theorem blockRowSum_act
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables) :
    blockRowSum system matrix (CarrierAction.act challenge assignment) vertex =
      ringFMul challenge (blockRowSum system matrix assignment vertex) := by
  rw [blockRowSum_eq_sumRingF, blockRowSum_eq_sumRingF]
  calc
    sumRingF (fun block :
        Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
      blockRowRing system matrix (CarrierAction.act challenge assignment)
        vertex block) =
      sumRingF (fun block :
          Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
        ringFMul challenge
          (blockRowRing system matrix assignment vertex block)) := by
      apply sumRingF_congr
      intro block
      exact blockRowRing_act system matrix challenge law assignment vertex block
    _ = ringFMul challenge
          (sumRingF fun block :
              Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
            blockRowRing system matrix assignment vertex block) :=
      (ringFMul_sumRingF_right challenge _).symm

private theorem rowRing_act
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (vertex : BooleanVertex shape.rowVariables) :
    rowRing system matrix (CarrierAction.act challenge assignment) vertex =
      ringFMul challenge (rowRing system matrix assignment vertex) := by
  rw [rowRing_eq_blockSum, rowRing_eq_blockSum]
  exact blockRowSum_act system matrix challenge law assignment vertex

/-- Complete 54-lane evaluation of one explicit completed matrix. -/
def evaluate
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (point : Point shape) : Evaluation :=
  fun output =>
    (BooleanTable.tabulate fun vertex =>
      K.embed (PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps
          matrix output)
        assignment vertex)).evaluate ConcreteCarrier.extensionOps point

private theorem evaluate_eq_evaluateRows
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (assignment : Assignment shape)
    (point : Point shape) :
    evaluate system matrix assignment point =
      RingKAction.evaluateRows
        (fun vertex => RingKAction.embedChallenge
          (rowRing system matrix assignment vertex)) point := by
  rfl

theorem evaluate_zero
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (point : Point shape) :
    evaluate system matrix BaseLinear.assignmentZero point =
      BaseLinear.evaluationZero := by
  funext lane
  unfold evaluate BaseLinear.assignmentZero BaseLinear.Raw.assignmentZero
    BaseLinear.evaluationZero ringKZero
  unfold BooleanTable.evaluate
  simpa only [BaseLinear.matrixVectorAt_zero] using
    (BaseLinear.evaluateTabulated_zero point)

theorem evaluate_add
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (left right : Assignment shape)
    (point : Point shape) :
    evaluate system matrix (BaseLinear.assignmentAdd left right) point =
      BaseLinear.evaluationAdd
        (evaluate system matrix left point)
        (evaluate system matrix right point) := by
  funext lane
  unfold evaluate BaseLinear.assignmentAdd BaseLinear.Raw.assignmentAdd
    BaseLinear.evaluationAdd
  unfold BooleanTable.evaluate
  simpa only [BaseLinear.matrixVectorAt_add,
    ConcreteCarrier.embed_add] using
    (BaseLinear.evaluateTabulated_add
      (fun vertex => K.embed (PaperLinearAlgebra.matrixVectorAt
        ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps
          matrix lane)
        left vertex))
      (fun vertex => K.embed (PaperLinearAlgebra.matrixVectorAt
        ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps
          matrix lane)
        right vertex))
      point)

theorem evaluate_scale
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (scalar : F)
    (assignment : Assignment shape)
    (point : Point shape) :
    evaluate system matrix (BaseLinear.assignmentScale scalar assignment) point =
      BaseLinear.evaluationScale scalar
        (evaluate system matrix assignment point) := by
  funext lane
  unfold evaluate BaseLinear.assignmentScale BaseLinear.Raw.assignmentScale
    BaseLinear.evaluationScale
  unfold BooleanTable.evaluate
  have embedScale (value : F) :
      K.embed (scalar * value) =
        K.mul (K.embed scalar) (K.embed value) := by
    simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
      (ConcreteCarrier.embed_mul scalar value)
  simpa only [BaseLinear.matrixVectorAt_scale, embedScale] using
    (BaseLinear.evaluateTabulated_scale (K.embed scalar)
      (fun vertex => K.embed (PaperLinearAlgebra.matrixVectorAt
        ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrixOf ConcreteCarrier.baseOps
          matrix lane)
        assignment vertex))
      point)

theorem evaluate_act
    {shape : Shape}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape)
    (point : Point shape) :
    evaluate system matrix (CarrierAction.act challenge assignment) point =
      ringKMul (RingKAction.embedChallenge challenge)
        (evaluate system matrix assignment point) := by
  rw [evaluate_eq_evaluateRows, evaluate_eq_evaluateRows]
  let rows : RingKAction.Rows shape.rowVariables := fun vertex =>
    RingKAction.embedChallenge (rowRing system matrix assignment vertex)
  calc
    RingKAction.evaluateRows
        (fun vertex => RingKAction.embedChallenge
          (rowRing system matrix (CarrierAction.act challenge assignment)
            vertex)) point =
      RingKAction.evaluateRows
        (RingKAction.actRows (RingKAction.embedChallenge challenge) rows)
        point := by
      apply congrArg (fun rowValues => RingKAction.evaluateRows rowValues point)
      funext vertex
      unfold RingKAction.actRows rows
      rw [rowRing_act system matrix challenge law assignment vertex,
        Embedding.embedChallenge_ringFMul]
    _ = ringKMul (RingKAction.embedChallenge challenge)
          (RingKAction.evaluateRows rows point) :=
      RingKAction.evaluateRows_embeddedChallenge_action challenge rows point
    _ = _ := rfl

/-- Evaluation of one explicit completed matrix commutes with the canonical
finite base-field combination used by PiDEC. -/
theorem evaluate_baseCombine
    {shape : Shape} {count : Nat}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape)
    (point : Point shape) :
    evaluate system matrix
        (BaseLinear.combineAssignments weights assignments) point =
      BaseLinear.combineEvaluations weights fun index =>
        evaluate system matrix (assignments index) point := by
  induction count with
  | zero => exact evaluate_zero system matrix point
  | succ count inductionHypothesis =>
      rw [BaseLinear.combineAssignments, BaseLinear.combineEvaluations,
        evaluate_add, evaluate_scale,
        inductionHypothesis
          (fun index => weights index.succ)
          (fun index => assignments index.succ)]

end ExplicitMatrix

/-- Array-level form of `matrixEvaluation_act`: every canonical matrix and
all 54 evaluation lanes preserve the same RingF action. -/
theorem evaluations_act
    {shape : Shape}
    (system : Structure shape) (challenge : RingF)
    (law : ProductOrderLaw challenge)
    (assignment : Assignment shape) (point : Point shape) :
    evaluations system (CarrierAction.act challenge assignment) point =
      Array.ofFn fun matrix : Fin shape.matrixCount =>
        ringKMul (RingKAction.embedChallenge challenge)
          (matrixEvaluation system assignment point matrix) := by
  apply Array.ext
  · simp [evaluations]
  · intro index leftLt rightLt
    let matrix : Fin shape.matrixCount :=
      ⟨index, by simpa [evaluations] using leftLt⟩
    simpa [matrix, evaluations] using
      matrixEvaluation_act system challenge law assignment point matrix

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLC
