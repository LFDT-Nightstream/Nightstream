import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.R1CS.Canonical.KPointEquality
import Nightstream.Implementation.R1CS.Canonical.KRecomposition
import Nightstream.Implementation.R1CS.Core.LinearSubstitution
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix

/-!
Contract: Lean-owned physical equality rows for the production `Pi_DEC`
radix fold.

The verifier fixes fourteen weights in the Goldilocks field.  This module
uses those exact semantic weights as row coefficients; it does not recover
them from Rust and does not replace them with a separately selected base.

The weighted combination is linear and therefore allocates no column.
One base-field coordinate costs one equality row.  One quadratic-extension
coordinate costs two equality rows.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RadixRows

open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-! ## The exact finite fold -/

/-- The semantic head-first finite base-field fold. -/
def combineField :
    {count : Nat} → (Fin count → F) → (Fin count → F) → F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        combineField
          (fun index => weights index.succ)
          (fun index => values index.succ)

/-- The same fold on row-layer linear combinations.  Scaling and appending
terms are coefficient rewrites, so this emits no row. -/
def combineComb :
    {count : Nat} → (Fin count → F) → (Fin count → LinComb) → LinComb
  | 0, _, _ => []
  | _ + 1, weights, values =>
      LinearSubstitution.scaleTerms (weights 0).val (values 0) ++
        combineComb
          (fun index => weights index.succ)
          (fun index => values index.succ)

/-- List presentation of the semantic fold. -/
theorem combineField_eq_foldr_zip_ofFn
    {count : Nat} (weights values : Fin count → F) :
    combineField weights values =
      ((List.ofFn values).zip (List.ofFn weights)).foldr
        (fun pair suffix => pair.2 * pair.1 + suffix) 0 := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      simp only [combineField, List.ofFn_succ, List.zip_cons_cons,
        List.foldr_cons]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => values index.succ)]

/-- The row module's semantic fold is definitionally the paper verifier's
production radix fold once the exact production weights are supplied. -/
theorem combineField_radix
    (values :
      Fin productionGlobalParams.k → F) :
    combineField PiDEC.radixWeight values =
      PiDECAlgebra.Radix.recomposeScalar values := by
  calc
    combineField PiDEC.radixWeight values =
        PiDECAlgebra.Radix.recomposeScalarList values :=
      combineField_eq_foldr_zip_ofFn PiDEC.radixWeight values
    _ = PiDECAlgebra.Radix.recomposeScalar values :=
      PiDECAlgebra.Radix.recomposeScalarList_eq values

/-- Evaluation of the row combination is exactly the same finite field
fold. -/
theorem residue_lcEval_combineComb
    (assignment : Nat → Nat) :
    ∀ {count : Nat} (weights : Fin count → F)
      (values : Fin count → LinComb),
      residue (lcEval assignment (combineComb weights values)) =
        combineField weights
          (fun index => residue (lcEval assignment (values index)))
  | 0, _, _ => rfl
  | _ + 1, weights, values => by
      rw [combineComb, KHorner.lcEval_append,
        KRecomposition.lcEval_scaleTerms, combineField]
      rw [show goldilocksP = Numeric.modulus by rfl]
      rw [residue_mod, residue_add, residue_mod, residue_mul,
        residue_field_val,
        residue_lcEval_combineComb assignment
          (fun index => weights index.succ)
          (fun index => values index.succ)]

/-! ## Quadratic-extension fold -/

/-- The same finite combination in the concrete extension carrier. -/
def combineK :
    {count : Nat} →
      (Fin count → F) →
      (Fin count → Nightstream.SuperNeo.Concrete.K) →
      Nightstream.SuperNeo.Concrete.K
  | 0, _, _ => Nightstream.SuperNeo.Concrete.K.zero
  | _ + 1, weights, values =>
      Nightstream.SuperNeo.Concrete.K.add
        (Nightstream.SuperNeo.Concrete.K.mul
          (Nightstream.SuperNeo.Concrete.K.embed (weights 0))
          (values 0))
        (combineK
          (fun index => weights index.succ)
          (fun index => values index.succ))

theorem combineK_c0 :
    ∀ {count : Nat} (weights : Fin count → F)
      (values : Fin count → Nightstream.SuperNeo.Concrete.K),
      (combineK weights values).c0 =
        combineField weights (fun index => (values index).c0)
  | 0, _, _ => rfl
  | _ + 1, weights, values => by
      simp only [combineK, Nightstream.SuperNeo.Concrete.K.add,
        Nightstream.SuperNeo.Concrete.K.mul,
        Nightstream.SuperNeo.Concrete.K.embed, Fin.zero_mul, Fin.mul_zero,
        Fin.add_zero]
      rw [combineK_c0
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      rfl

theorem combineK_c1 :
    ∀ {count : Nat} (weights : Fin count → F)
      (values : Fin count → Nightstream.SuperNeo.Concrete.K),
      (combineK weights values).c1 =
        combineField weights (fun index => (values index).c1)
  | 0, _, _ => rfl
  | _ + 1, weights, values => by
      simp only [combineK, Nightstream.SuperNeo.Concrete.K.add,
        Nightstream.SuperNeo.Concrete.K.mul,
        Nightstream.SuperNeo.Concrete.K.embed, Fin.zero_mul, Fin.mul_zero,
        Fin.add_zero]
      rw [combineK_c1
        (fun index => weights index.succ)
        (fun index => values index.succ)]
      rfl

/-- The coordinate fold used by the row program is exactly the semantic
evaluation fold, lane by lane. -/
theorem combineK_eq_combineEvaluations_apply :
    ∀ {count : Nat} (weights : Fin count → F)
      (values : Fin count → Evaluation) (lane : Fin ringDegree),
      combineK weights (fun index => values index lane) =
        BaseLinear.combineEvaluations weights values lane
  | 0, _, _, _ => rfl
  | _ + 1, weights, values, lane => by
      simp only [combineK, BaseLinear.combineEvaluations,
        BaseLinear.evaluationAdd, BaseLinear.evaluationScale]
      rw [combineK_eq_combineEvaluations_apply
        (fun index => weights index.succ)
        (fun index => values index.succ) lane]

/-- Production commitment recomposition is the scalar radix fold at every
row and Phi81 lane. -/
theorem recomposeCommitment_apply
    {verifierRows : Nat}
    (values :
      Fin productionGlobalParams.k →
        PiRLCAlgebra.Commitment.Value verifierRows)
    (row : Fin verifierRows) (lane : Fin ringDegree) :
    PiDECAlgebra.Commitment.recomposeCommitment values row lane =
      PiDECAlgebra.Radix.recomposeScalar
        (fun child => values child row lane) := by
  unfold PiDECAlgebra.Commitment.recomposeCommitment
  change
    PiDECAlgebra.Commitment.combineCommitments PiDEC.radixWeight values
        row lane =
      PiDECAlgebra.Radix.recomposeScalar
        (fun child => values child row lane)
  rw [← combineField_radix]
  congr 1

/-- Production evaluation recomposition is the extension-field radix fold at
every matrix and Phi81 lane. -/
theorem recomposeEvaluations_get
    {shape : Phi81Relation.Shape}
    (values :
      Fin productionGlobalParams.k → Array Evaluation)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    (PiDEC.recomposeEvaluations (shape := shape) values).getD
        matrix.val BaseLinear.evaluationZero lane =
      combineK PiDEC.radixWeight
        (fun child =>
          (values child).getD matrix.val BaseLinear.evaluationZero lane) := by
  have bound :
      matrix.val <
        (PiDEC.recomposeEvaluations (shape := shape) values).size := by
    simp [PiDEC.recomposeEvaluations]
  rw [Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem bound]
  simp only [PiDEC.recomposeEvaluations, Array.getElem_ofFn]
  exact
    (combineK_eq_combineEvaluations_apply PiDEC.radixWeight
      (fun child =>
        (values child).getD matrix.val BaseLinear.evaluationZero) lane).symm

/-- Row-layer carried form of the extension fold. -/
def combineCarried
    {count : Nat} (weights : Fin count → F)
    (values : Fin count → Carried) : Carried where
  low := combineComb weights (fun index => (values index).low)
  high := combineComb weights (fun index => (values index).high)

theorem residue_lcEval_eq_decoded_c0
    (assignment : Nat → Nat) (value : Carried) :
    residue (lcEval assignment value.low) =
      (KPointEquality.decoded assignment value).c0 := by
  apply Fin.ext
  change
    lcEval assignment value.low %
        Nightstream.SuperNeo.Concrete.goldilocksModulus =
      lcEval assignment value.low
  exact Nat.mod_eq_of_lt (by
    unfold lcEval
    exact Nat.mod_lt _ (by decide))

theorem residue_lcEval_eq_decoded_c1
    (assignment : Nat → Nat) (value : Carried) :
    residue (lcEval assignment value.high) =
      (KPointEquality.decoded assignment value).c1 := by
  apply Fin.ext
  change
    lcEval assignment value.high %
        Nightstream.SuperNeo.Concrete.goldilocksModulus =
      lcEval assignment value.high
  exact Nat.mod_eq_of_lt (by
    unfold lcEval
    exact Nat.mod_lt _ (by decide))

/-- Decoding the combined carried value gives the exact concrete extension
fold. -/
theorem decoded_combineCarried
    (assignment : Nat → Nat)
    {count : Nat} (weights : Fin count → F)
    (values : Fin count → Carried) :
    KPointEquality.decoded assignment (combineCarried weights values) =
      combineK weights
        (fun index => KPointEquality.decoded assignment (values index)) := by
  apply Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded]
  unfold combineCarried carriedValue
    Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete
  simp only [KHorner.Pair.mk.injEq]
  constructor
  · apply residue_injective_of_lt
      (by unfold lcEval; exact Nat.mod_lt _ (by decide))
      (combineK weights
        (fun index => KPointEquality.decoded assignment (values index))).c0.isLt
    rw [residue_lcEval_combineComb, combineK_c0, residue_field_val]
    apply congrArg (combineField weights)
    funext index
    exact residue_lcEval_eq_decoded_c0 assignment (values index)
  · apply residue_injective_of_lt
      (by unfold lcEval; exact Nat.mod_lt _ (by decide))
      (combineK weights
        (fun index => KPointEquality.decoded assignment (values index))).c1.isLt
    rw [residue_lcEval_combineComb, combineK_c1, residue_field_val]
    apply congrArg (combineField weights)
    funext index
    exact residue_lcEval_eq_decoded_c1 assignment (values index)

/-! ## Emitted coordinate checks -/

structure FCoordinate where
  children : Fin productionGlobalParams.k → LinComb
  parent : LinComb

structure KCoordinate where
  children : Fin productionGlobalParams.k → Carried
  parent : Carried

def fRow (coordinate : FCoordinate) : Row :=
  KEquality.equalityRow
    (combineComb PiDEC.radixWeight coordinate.children)
    coordinate.parent

def kRows (coordinate : KCoordinate) : List Row :=
  KEquality.rows
    (combineCarried PiDEC.radixWeight coordinate.children)
    coordinate.parent

def rows
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate) : List Row :=
  fCoordinates.map fRow ++ kCoordinates.flatMap kRows

theorem rows_length
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate) :
    (rows fCoordinates kCoordinates).length =
      fCoordinates.length + 2 * kCoordinates.length := by
  unfold rows
  rw [List.length_append, List.length_map, List.length_flatMap]
  have each :
      (kCoordinates.map fun coordinate => (kRows coordinate).length).sum =
        (kCoordinates.map fun _ => 2).sum := by
    apply congrArg List.sum
    apply List.map_congr_left
    intro coordinate _
    exact KEquality.rows_length _ _
  rw [each]
  have sumTwos : ∀ entries : List KCoordinate,
      (entries.map fun _ => 2).sum = 2 * entries.length := by
    intro entries
    induction entries with
    | nil => rfl
    | cons coordinate rest inductionHypothesis =>
        simp only [List.map_cons, List.sum_cons, List.length_cons,
          inductionHypothesis]
        omega
  rw [sumTwos]

/-- The production radix rows allocate no columns. -/
def columns : List Nat := []

theorem columns_length : columns.length = 0 := rfl

theorem columns_nodup : columns.Nodup := List.nodup_nil

/-! ## Support and conservation -/

/-- A carried radix combination cannot mention a column absent from every
source coordinate. -/
theorem mentions_combineComb :
    ∀ {count : Nat} (weights : Fin count → F)
      (values : Fin count → LinComb) (column : Nat),
      Mentions (combineComb weights values) column →
        ∃ index, Mentions (values index) column
  | 0, _, _, _, mentioned => by
      simp [combineComb, Mentions] at mentioned
  | _ + 1, weights, values, column, mentioned => by
      simp only [combineComb, Mentions, List.map_append,
        List.mem_append] at mentioned
      rcases mentioned with head | tail
      · refine ⟨0, ?_⟩
        simpa [LinearSubstitution.scaleTerms, Mentions] using head
      · obtain ⟨index, source⟩ :=
          mentions_combineComb
            (fun child => weights child.succ)
            (fun child => values child.succ) column tail
        exact ⟨index.succ, source⟩

/-- Every column in a radix row is the constant wire or belongs to one of the
authoritative child/parent coordinates named by that row's receipt. -/
theorem rows_conservation
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate)
    (row : Row) (member : row ∈ rows fCoordinates kCoordinates)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0
      ∨ (∃ coordinate ∈ fCoordinates,
          (∃ child, Mentions (coordinate.children child) column)
            ∨ Mentions coordinate.parent column)
      ∨ (∃ coordinate ∈ kCoordinates,
          (∃ child,
            Mentions (coordinate.children child).low column
              ∨ Mentions (coordinate.children child).high column)
            ∨ Mentions coordinate.parent.low column
            ∨ Mentions coordinate.parent.high column) := by
  unfold rows at member
  rcases List.mem_append.1 member with inF | inK
  · rcases List.mem_map.1 inF with ⟨coordinate, coordinateMember, rfl⟩
    simp only [fRow, KEquality.equalityRow] at mentioned
    rcases mentioned with inCombined | inOne | inParent
    · obtain ⟨child, source⟩ :=
        mentions_combineComb PiDEC.radixWeight coordinate.children column
          inCombined
      exact Or.inr (Or.inl
        ⟨coordinate, coordinateMember, Or.inl ⟨child, source⟩⟩)
    · exact Or.inl (by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using inOne)
    · exact Or.inr (Or.inl
        ⟨coordinate, coordinateMember, Or.inr inParent⟩)
  · rcases List.mem_flatMap.1 inK with
      ⟨coordinate, coordinateMember, rowMember⟩
    have support :=
      KEquality.rows_conservation
        (combineCarried PiDEC.radixWeight coordinate.children)
        coordinate.parent row rowMember column mentioned
    rcases support with isOne | inLow | inHigh | inParentLow | inParentHigh
    · exact Or.inl isOne
    · obtain ⟨child, source⟩ :=
        mentions_combineComb PiDEC.radixWeight
          (fun index => (coordinate.children index).low) column inLow
      exact Or.inr (Or.inr
        ⟨coordinate, coordinateMember,
          Or.inl ⟨child, Or.inl source⟩⟩)
    · obtain ⟨child, source⟩ :=
        mentions_combineComb PiDEC.radixWeight
          (fun index => (coordinate.children index).high) column inHigh
      exact Or.inr (Or.inr
        ⟨coordinate, coordinateMember,
          Or.inl ⟨child, Or.inr source⟩⟩)
    · exact Or.inr (Or.inr
        ⟨coordinate, coordinateMember, Or.inr (Or.inl inParentLow)⟩)
    · exact Or.inr (Or.inr
        ⟨coordinate, coordinateMember, Or.inr (Or.inr inParentHigh)⟩)

theorem satisfies_fRows
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows fCoordinates kCoordinates) assignment) :
    Satisfies (fCoordinates.map fRow) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

theorem satisfies_kRows
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows fCoordinates kCoordinates) assignment) :
    Satisfies (kCoordinates.flatMap kRows) assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

/-- Satisfaction forces one base-field parent coordinate to be the exact
production radix recomposition. -/
theorem rows_sound_f
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows fCoordinates kCoordinates) assignment)
    (coordinate : FCoordinate) (member : coordinate ∈ fCoordinates) :
    PiDECAlgebra.Radix.recomposeScalar
        (fun child => residue (lcEval assignment (coordinate.children child))) =
      residue (lcEval assignment coordinate.parent) := by
  have equal :=
    (KEquality.equalityRow_iff assignment _ coordinate.parent constantWire).1
      (satisfies_fRows fCoordinates kCoordinates assignment satisfied
        (fRow coordinate)
        (List.mem_map.2 ⟨coordinate, member, rfl⟩))
  have fieldEqual := congrArg residue equal
  rw [residue_lcEval_combineComb, combineField_radix] at fieldEqual
  exact fieldEqual

/-- Satisfaction forces one extension parent coordinate to be the exact
production radix recomposition. -/
theorem rows_sound_k
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows fCoordinates kCoordinates) assignment)
    (coordinate : KCoordinate) (member : coordinate ∈ kCoordinates) :
    combineK PiDEC.radixWeight
        (fun child =>
          KPointEquality.decoded assignment (coordinate.children child)) =
      KPointEquality.decoded assignment coordinate.parent := by
  have coordinateRows :
      Satisfies (kRows coordinate) assignment := by
    intro row rowMember
    exact satisfies_kRows fCoordinates kCoordinates assignment satisfied row
      (List.mem_flatMap.2 ⟨coordinate, member, rowMember⟩)
  have halves :=
    KEquality.rows_sound assignment
      (combineCarried PiDEC.radixWeight coordinate.children)
      coordinate.parent constantWire coordinateRows
  have decodedEqual :
      KPointEquality.decoded assignment
          (combineCarried PiDEC.radixWeight coordinate.children) =
        KPointEquality.decoded assignment coordinate.parent := by
    apply Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_injective
    rw [KPointEquality.ofConcrete_decoded,
      KPointEquality.ofConcrete_decoded]
    unfold carriedValue
    simp only [KHorner.Pair.mk.injEq]
    exact halves
  rw [decoded_combineCarried] at decodedEqual
  exact decodedEqual

/-- Honest production-radix equations satisfy all rows without extending the
assignment. -/
theorem rows_honest
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (fHonest : ∀ coordinate ∈ fCoordinates,
      PiDECAlgebra.Radix.recomposeScalar
          (fun child =>
            residue (lcEval assignment (coordinate.children child))) =
        residue (lcEval assignment coordinate.parent))
    (kHonest : ∀ coordinate ∈ kCoordinates,
      combineK PiDEC.radixWeight
          (fun child =>
            KPointEquality.decoded assignment (coordinate.children child)) =
        KPointEquality.decoded assignment coordinate.parent) :
    Satisfies (rows fCoordinates kCoordinates) assignment := by
  intro row member
  rcases List.mem_append.1 member with inF | inK
  · rcases List.mem_map.1 inF with ⟨coordinate, coordinateMember, rfl⟩
    refine
      (KEquality.equalityRow_iff assignment _ coordinate.parent
        constantWire).2 ?_
    apply residue_injective_of_lt
      (by unfold lcEval; exact Nat.mod_lt _ (by decide))
      (by unfold lcEval; exact Nat.mod_lt _ (by decide))
    rw [residue_lcEval_combineComb, combineField_radix]
    exact fHonest coordinate coordinateMember
  · rcases List.mem_flatMap.1 inK with
      ⟨coordinate, coordinateMember, rowMember⟩
    have decodedEqual :
        KPointEquality.decoded assignment
            (combineCarried PiDEC.radixWeight coordinate.children) =
          KPointEquality.decoded assignment coordinate.parent := by
      rw [decoded_combineCarried]
      exact kHonest coordinate coordinateMember
    have concreteCoordinates :
        lcEval assignment
              (combineCarried PiDEC.radixWeight coordinate.children).low =
            lcEval assignment coordinate.parent.low ∧
          lcEval assignment
              (combineCarried PiDEC.radixWeight coordinate.children).high =
            lcEval assignment coordinate.parent.high := by
      have pairs := congrArg
        Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete
        decodedEqual
      rw [KPointEquality.ofConcrete_decoded,
        KPointEquality.ofConcrete_decoded] at pairs
      unfold carriedValue at pairs
      simpa only [KHorner.Pair.mk.injEq] using pairs
    exact
      KEquality.rows_complete assignment
        (combineCarried PiDEC.radixWeight coordinate.children)
        coordinate.parent constantWire concreteCoordinates.1
        concreteCoordinates.2 row rowMember

/-! ## Cost -/

def cost
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate) :
    Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := fCoordinates.length + 2 * kCoordinates.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem cost_rows
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate) :
    (rows fCoordinates kCoordinates).length =
      (cost fCoordinates kCoordinates).recurringRows :=
  rows_length fCoordinates kCoordinates

theorem cost_columns
    (fCoordinates : List FCoordinate)
    (kCoordinates : List KCoordinate) :
    columns.length = (cost fCoordinates kCoordinates).auxiliaryColumns :=
  columns_length

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RadixRows
