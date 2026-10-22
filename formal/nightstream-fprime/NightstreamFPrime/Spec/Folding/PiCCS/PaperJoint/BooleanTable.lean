import Std
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

/-!
Canonical Boolean-table coefficient transform for the paper-level `Pi_CCS`
model.

Owns: the verifier-owned squarefree alpha basis, an explicit finite Boolean
table, the recursive coefficient transform `p0 + A * (p1 - p0)`, and
coefficient-zero equivalence with leafwise table zero.

Does not own: evaluation of the derived coefficients at Boolean points, an MLE
correctness theorem, construction of CCS or norm residual tables, range
semantics, SumCheck, transcript sampling, root counting, Rust, R1CS, or
constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

open NightstreamFPrime.Spec.SumCheck

universe uField

structure InterpolationOps (Field : Type uField)
    extends SumCheck.Finite.Ops Field where
  neg : Field -> Field

namespace InterpolationOps

def sub {Field : Type uField} (ops : InterpolationOps Field)
    (left right : Field) : Field :=
  ops.add left (ops.neg right)

end InterpolationOps

structure InterpolationZeroLaws
    {Field : Type uField}
    (ops : InterpolationOps Field) : Prop where
  add_zero : forall value, ops.add value ops.zero = value
  neg_zero : ops.neg ops.zero = ops.zero

def canonicalExponentVectors : Nat -> List (List Nat)
  | 0 => [[]]
  | variables + 1 =>
      (canonicalExponentVectors variables).map (fun exponents => 0 :: exponents) ++
        (canonicalExponentVectors variables).map (fun exponents => 1 :: exponents)

theorem canonicalExponentVectors_entry_length
    {variables : Nat}
    {exponents : List Nat}
    (member : exponents ∈ canonicalExponentVectors variables) :
    exponents.length = variables := by
  induction variables generalizing exponents with
  | zero =>
      simp [canonicalExponentVectors] at member
      simp [member]
  | succ variables inductionHypothesis =>
      simp only [canonicalExponentVectors, List.mem_append, List.mem_map] at member
      rcases member with ⟨prior, priorMember, rfl⟩ | ⟨prior, priorMember, rfl⟩
      · simp [inductionHypothesis priorMember]
      · simp [inductionHypothesis priorMember]

theorem canonicalExponentVectors_length (variables : Nat) :
    (canonicalExponentVectors variables).length = 2 ^ variables := by
  induction variables with
  | zero => simp [canonicalExponentVectors]
  | succ variables inductionHypothesis =>
      simp only [canonicalExponentVectors, List.length_append, List.length_map,
        inductionHypothesis, Nat.pow_succ]
      omega

theorem canonicalExponentVectors_nodup (variables : Nat) :
    (canonicalExponentVectors variables).Nodup := by
  induction variables with
  | zero => simp [canonicalExponentVectors]
  | succ variables inductionHypothesis =>
      rw [canonicalExponentVectors]
      rw [List.nodup_append]
      refine ⟨?_, ?_, ?_⟩
      · exact List.Pairwise.map _ (by
          intro left right notEqual equal
          exact notEqual (List.cons.inj equal).2) inductionHypothesis
      · exact List.Pairwise.map _ (by
          intro left right notEqual equal
          exact notEqual (List.cons.inj equal).2) inductionHypothesis
      · intro value lowMember other highMember
        rcases List.mem_map.mp lowMember with ⟨low, _, rfl⟩
        rcases List.mem_map.mp highMember with ⟨high, _, rfl⟩
        simp

def canonicalAlphaMonomials (shape : Shape) : List (AlphaMonomial shape) :=
  List.pmap
    (fun exponents arity =>
      { exponents := exponents
        arity := arity })
    (canonicalExponentVectors shape.cubeVariables)
    (fun _ member => canonicalExponentVectors_entry_length member)

theorem canonicalAlphaMonomials_length (shape : Shape) :
    (canonicalAlphaMonomials shape).length = 2 ^ shape.cubeVariables := by
  simp [canonicalAlphaMonomials, canonicalExponentVectors_length]

def canonicalAlphaBasis (shape : Shape) : AlphaBasis shape where
  monomials := canonicalAlphaMonomials shape
  nodup := by
    unfold canonicalAlphaMonomials
    apply List.Pairwise.pmap
      (canonicalExponentVectors_nodup shape.cubeVariables)
      (fun _ member => canonicalExponentVectors_entry_length member)
    intro left _ right _ notEqual equal
    exact notEqual (congrArg AlphaMonomial.exponents equal)

inductive BooleanTable (Field : Type uField) : Nat -> Type uField where
  | leaf (value : Field) : BooleanTable Field 0
  | branch {variables : Nat}
      (low high : BooleanTable Field variables) :
      BooleanTable Field (variables + 1)

namespace BooleanTable

def entries {Field : Type uField} :
    {variables : Nat} -> BooleanTable Field variables -> List Field
  | 0, .leaf value => [value]
  | _ + 1, .branch low high => low.entries ++ high.entries

theorem entries_length
    {Field : Type uField}
    {variables : Nat}
    (table : BooleanTable Field variables) :
    table.entries.length = 2 ^ variables := by
  induction table with
  | leaf => simp [entries]
  | branch low high lowInduction highInduction =>
      simp only [entries, List.length_append, lowInduction, highInduction,
        Nat.pow_succ]
      omega

def AllEntriesZero
    {Field : Type uField}
    {variables : Nat}
    (ops : InterpolationOps Field)
    (table : BooleanTable Field variables) : Prop :=
  forall value, value ∈ table.entries -> value = ops.zero

def interpolateCoefficients
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    {variables : Nat} -> BooleanTable Field variables -> List Field
  | 0, .leaf value => [value]
  | _ + 1, .branch low high =>
      let lowCoefficients := low.interpolateCoefficients ops
      let highCoefficients := high.interpolateCoefficients ops
      lowCoefficients ++
        List.zipWith ops.sub highCoefficients lowCoefficients

theorem interpolateCoefficients_length
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (table : BooleanTable Field variables) :
    (table.interpolateCoefficients ops).length = 2 ^ variables := by
  induction table with
  | leaf => simp [interpolateCoefficients]
  | branch low high lowInduction highInduction =>
      simp only [interpolateCoefficients, List.length_append,
        List.length_zipWith, lowInduction, highInduction, Nat.min_self,
        Nat.pow_succ]
      omega

private def ListZero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (values : List Field) : Prop :=
  forall value, value ∈ values -> value = ops.zero

private theorem listZero_append
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (left right : List Field) :
    ListZero ops (left ++ right) ↔ ListZero ops left ∧ ListZero ops right := by
  constructor
  · intro allZero
    refine ⟨?_, ?_⟩
    · intro value member
      exact allZero value (List.mem_append_left right member)
    · intro value member
      exact allZero value (List.mem_append_right left member)
  · rintro ⟨leftZero, rightZero⟩ value member
    rcases List.mem_append.mp member with leftMember | rightMember
    · exact leftZero value leftMember
    · exact rightZero value rightMember

private theorem listZero_zipWith_sub_iff
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    {left right : List Field}
    (sameLength : left.length = right.length)
    (rightZero : ListZero ops right) :
    ListZero ops (List.zipWith ops.sub left right) ↔ ListZero ops left := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => simp [ListZero]
      | cons rightHead rightTail => simp at sameLength
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at sameLength
      | cons rightHead rightTail =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          have headZero : rightHead = ops.zero :=
            rightZero rightHead (by simp)
          have tailZero : ListZero ops rightTail := by
            intro value member
            exact rightZero value (by simp [member])
          constructor
          · intro differencesZero
            have headDifferenceZero :
                ops.sub leftHead rightHead = ops.zero :=
              differencesZero (ops.sub leftHead rightHead) (by simp)
            have leftHeadZero : leftHead = ops.zero := by
              simpa [InterpolationOps.sub, headZero, laws.neg_zero,
                laws.add_zero] using headDifferenceZero
            have tailDifferencesZero :
                ListZero ops (List.zipWith ops.sub leftTail rightTail) := by
              intro value member
              exact differencesZero value (by simp [member])
            have leftTailZero :=
              (inductionHypothesis sameLength tailZero).mp tailDifferencesZero
            intro value member
            rcases List.mem_cons.mp member with rfl | tailMember
            · exact leftHeadZero
            · exact leftTailZero value tailMember
          · intro leftZero
            have leftHeadZero : leftHead = ops.zero :=
              leftZero leftHead (by simp)
            have leftTailZero : ListZero ops leftTail := by
              intro value member
              exact leftZero value (by simp [member])
            have tailDifferencesZero :=
              (inductionHypothesis sameLength tailZero).mpr leftTailZero
            intro value member
            rcases List.mem_cons.mp member with rfl | tailMember
            · simp [InterpolationOps.sub, leftHeadZero, headZero,
                laws.neg_zero, laws.add_zero]
            · exact tailDifferencesZero value tailMember

theorem interpolateCoefficients_zero_iff_allEntriesZero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    {variables : Nat}
    (table : BooleanTable Field variables) :
    ListZero ops (table.interpolateCoefficients ops) ↔
      table.AllEntriesZero ops := by
  induction table with
  | leaf value => simp [interpolateCoefficients, AllEntriesZero, entries, ListZero]
  | branch low high lowInduction highInduction =>
      rw [show (BooleanTable.branch low high).AllEntriesZero ops ↔
          low.AllEntriesZero ops ∧ high.AllEntriesZero ops by
        change ListZero ops (low.entries ++ high.entries) ↔
          ListZero ops low.entries ∧ ListZero ops high.entries
        exact listZero_append ops low.entries high.entries]
      rw [interpolateCoefficients, listZero_append]
      constructor
      · rintro ⟨lowZero, differenceZero⟩
        have sameLength :
            (high.interpolateCoefficients ops).length =
              (low.interpolateCoefficients ops).length := by
          rw [interpolateCoefficients_length, interpolateCoefficients_length]
        have highZero :=
          (listZero_zipWith_sub_iff ops laws sameLength lowZero).mp differenceZero
        exact ⟨lowInduction.mp lowZero, highInduction.mp highZero⟩
      · rintro ⟨lowEntriesZero, highEntriesZero⟩
        have lowZero := lowInduction.mpr lowEntriesZero
        have highZero := highInduction.mpr highEntriesZero
        refine ⟨lowZero, ?_⟩
        have sameLength :
            (high.interpolateCoefficients ops).length =
              (low.interpolateCoefficients ops).length := by
          rw [interpolateCoefficients_length, interpolateCoefficients_length]
        exact (listZero_zipWith_sub_iff ops laws sameLength lowZero).mpr highZero

def toAlphaPolynomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    (table : BooleanTable Field shape.cubeVariables) :
    AlphaPolynomial Field (canonicalAlphaBasis shape) where
  coefficients := table.interpolateCoefficients ops
  coefficientCount := by
    rw [interpolateCoefficients_length]
    exact (canonicalAlphaMonomials_length shape).symm

theorem toAlphaPolynomial_coefficientZero_iff_allEntriesZero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    {shape : Shape}
    (table : BooleanTable Field shape.cubeVariables) :
    (table.toAlphaPolynomial ops).CoefficientZero ops.toOps ↔
      table.AllEntriesZero ops := by
  exact interpolateCoefficients_zero_iff_allEntriesZero ops laws table

end BooleanTable

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
