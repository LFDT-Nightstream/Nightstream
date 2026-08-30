import NightstreamFPrime.Circuit.VariableSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain

/-!
Owns variable-support propagation for the fixed PiCCS SumCheck chain.
The chain has no local witness allocation.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def KSupported (value : KExpr) (allowed : Nat → Prop) : Prop :=
  value.c0.VarsSatisfy allowed ∧ value.c1.VarsSatisfy allowed

def RoundSupported {degree : Nat} (round : FixedChain.Round degree)
    (allowed : Nat → Prop) : Prop :=
  (∀ coefficient, KSupported (round.coefficient coefficient) allowed) ∧
    KSupported round.challenge allowed

private theorem add_supported (left right : KExpr) (allowed : Nat → Prop)
    (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    KSupported (KExpr.add left right) allowed :=
  ⟨⟨leftSupport.1, rightSupport.1⟩,
    ⟨leftSupport.2, rightSupport.2⟩⟩

private theorem mul_supported (left right : KExpr) (allowed : Nat → Prop)
    (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    KSupported (KExpr.mul left right) allowed := by
  unfold KExpr.mul KSupported
  simp only [Expr.VarsSatisfy]
  exact ⟨
    ⟨⟨leftSupport.1, rightSupport.1⟩,
      ⟨⟨trivial, leftSupport.2⟩, rightSupport.2⟩⟩,
    ⟨⟨leftSupport.1, rightSupport.2⟩,
      ⟨leftSupport.2, rightSupport.1⟩⟩⟩

private theorem evaluateCoefficients_supported (point : KExpr)
    (coefficients : List KExpr) (allowed : Nat → Prop)
    (pointSupport : KSupported point allowed)
    (coefficientsSupport : ∀ coefficient ∈ coefficients,
      KSupported coefficient allowed) :
    KSupported (FixedChain.evaluateCoefficients point coefficients) allowed := by
  induction coefficients with
  | nil =>
      exact ⟨trivial, trivial⟩
  | cons coefficient coefficients inductionHypothesis =>
      apply add_supported
      · exact coefficientsSupport coefficient (by simp)
      · apply mul_supported point _ allowed pointSupport
        apply inductionHypothesis
        intro current member
        exact coefficientsSupport current (by simp [member])

private theorem evaluateRound_supported {degree : Nat}
    (round : FixedChain.Round degree) (point : KExpr)
    (allowed : Nat → Prop) (roundSupport : RoundSupported round allowed)
    (pointSupport : KSupported point allowed) :
    KSupported (FixedChain.evaluateRound round point) allowed := by
  apply evaluateCoefficients_supported point round.coefficients allowed
    pointSupport
  intro coefficient member
  rw [FixedChain.Round.coefficients, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact roundSupport.1 index

private theorem equalities_supported (left right : KExpr)
    (allowed : Nat → Prop) (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    ∀ expression ∈ KExpr.equalities left right,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [KExpr.equalities, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact ⟨leftSupport.1, ⟨trivial, rightSupport.1⟩⟩
  · exact ⟨leftSupport.2, ⟨trivial, rightSupport.2⟩⟩

private theorem constraintsFrom_supported {degree : Nat}
    (current : KExpr) (rounds : List (FixedChain.Round degree))
    (allowed : Nat → Prop) (currentSupport : KSupported current allowed)
    (roundsSupport : ∀ round ∈ rounds, RoundSupported round allowed) :
    ∀ expression ∈ FixedChain.Owned.constraintsFrom current rounds,
      expression.VarsSatisfy allowed := by
  induction rounds generalizing current with
  | nil =>
      intro expression member
      simp [FixedChain.Owned.constraintsFrom] at member
  | cons round rounds inductionHypothesis =>
      intro expression member
      rw [FixedChain.Owned.constraintsFrom, List.mem_append] at member
      rcases member with headMember | tailMember
      · have roundSupport := roundsSupport round (by simp)
        have zeroSupport : KSupported KExpr.zero allowed :=
          ⟨trivial, trivial⟩
        have oneSupport : KSupported KExpr.one allowed :=
          ⟨trivial, trivial⟩
        have rightSupport := add_supported
          (FixedChain.evaluateRound round KExpr.zero)
          (FixedChain.evaluateRound round KExpr.one) allowed
          (evaluateRound_supported round KExpr.zero allowed roundSupport
            zeroSupport)
          (evaluateRound_supported round KExpr.one allowed roundSupport
            oneSupport)
        exact equalities_supported current _ allowed currentSupport rightSupport
          expression headMember
      · apply inductionHypothesis
        · have roundSupport := roundsSupport round (by simp)
          exact evaluateRound_supported round round.challenge allowed
            roundSupport roundSupport.2
        · intro later laterMember
          exact roundsSupport later (by simp [laterMember])
        · exact tailMember

private theorem outputFrom_supported {degree : Nat}
    (current : KExpr) (rounds : List (FixedChain.Round degree))
    (allowed : Nat → Prop) (currentSupport : KSupported current allowed)
    (roundsSupport : ∀ round ∈ rounds, RoundSupported round allowed) :
    KSupported (FixedChain.Owned.outputFrom current rounds) allowed := by
  induction rounds generalizing current with
  | nil => exact currentSupport
  | cons round rounds inductionHypothesis =>
      apply inductionHypothesis
      · have roundSupport := roundsSupport round (by simp)
        exact evaluateRound_supported round round.challenge allowed
          roundSupport roundSupport.2
      · intro later member
        exact roundsSupport later (by simp [member])

/-- Exact support of the fixed PiCCS SumCheck-chain rows. -/
theorem flatConstraints_varsSatisfy {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ roundIndex,
      RoundSupported (interface.round offset roundIndex) allowed) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  change ∀ expression ∈ flatConstraints (Circuit.ops
    (FixedChain.Owned.circuit (coreInterface interface offset)).main offset), _
  rw [FixedChain.Owned.flatConstraints_eq]
  apply constraintsFrom_supported (coreInterface interface offset).initial
    (coreInterface interface offset).rounds allowed
  · simpa [coreInterface] using initialSupport
  · intro round member
    rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨roundIndex, rfl⟩
    exact roundSupport roundIndex

/-- The final SumCheck claim preserves the exact support of the initial claim
and all verifier-bound rounds. -/
theorem output_varsSatisfy {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ roundIndex,
      RoundSupported (interface.round offset roundIndex) allowed) :
    KSupported (output interface offset) allowed := by
  unfold output FixedChain.Owned.output
  apply outputFrom_supported (coreInterface interface offset).initial
    (coreInterface interface offset).rounds allowed
  · simpa [coreInterface] using initialSupport
  · intro round member
    rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨roundIndex, rfl⟩
    exact roundSupport roundIndex

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain
