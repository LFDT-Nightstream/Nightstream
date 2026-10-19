import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain

/-!
Owns variable-support propagation for the fixed PiCCS SumCheck chain.
Its rows read the supported input wires and the chain's own stored interval.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain

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

private theorem coefficients_supported {degree : Nat}
    (round : FixedChain.Round degree) (allowed : Nat → Prop)
    (roundSupport : RoundSupported round allowed) :
    ∀ coefficient ∈ round.coefficients, KSupported coefficient allowed := by
  intro coefficient member
  rw [FixedChain.Round.coefficients, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact roundSupport.1 index

private theorem coefficientSum_supported (coefficients : List KExpr)
    (allowed : Nat → Prop)
    (support : ∀ coefficient ∈ coefficients, KSupported coefficient allowed) :
    KSupported (FixedChain.Owned.coefficientSum coefficients) allowed := by
  induction coefficients with
  | nil => exact ⟨trivial, trivial⟩
  | cons coefficient rest inductionHypothesis =>
      exact add_supported _ _ allowed (support coefficient (by simp))
        (inductionHypothesis fun current member =>
          support current (by simp [member]))

private theorem boundarySum_supported (coefficients : List KExpr)
    (allowed : Nat → Prop)
    (support : ∀ coefficient ∈ coefficients, KSupported coefficient allowed) :
    KSupported (FixedChain.Owned.boundarySum coefficients) allowed := by
  cases coefficients with
  | nil => exact ⟨⟨trivial, trivial⟩, ⟨trivial, trivial⟩⟩
  | cons coefficient rest =>
      exact add_supported _ _ allowed (support coefficient (by simp))
        (coefficientSum_supported (coefficient :: rest) allowed support)

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

/-- A round's stored interval is inside the chain's stored interval. -/
private theorem roundInterval_allowed {degree : Nat} (start : Nat)
    (rounds : Nat) (allowed : Nat → Prop)
    (stored : ∀ index, start ≤ index →
      index < start + 3 * degree * (rounds + 1) → allowed index) :
    ∀ index,
      NightstreamFPrime.Circuit.SupportRange.Extend allowed start
        (start + 3 * degree) index → allowed index := by
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · apply stored index lower
    rw [Nat.mul_succ]
    omega

private theorem laterInterval_allowed {degree : Nat} (start : Nat)
    (rounds : Nat) (allowed : Nat → Prop)
    (stored : ∀ index, start ≤ index →
      index < start + 3 * degree * (rounds + 1) → allowed index) :
    ∀ index, start + 3 * degree ≤ index →
      index < start + 3 * degree + 3 * degree * rounds → allowed index := by
  intro index lower upper
  apply stored index (by omega)
  rw [Nat.mul_succ]
  omega

private theorem roundOutput_supported {degree : Nat} (start : Nat)
    (round : FixedChain.Round degree) (rounds : Nat) (allowed : Nat → Prop)
    (roundSupport : RoundSupported round allowed)
    (stored : ∀ index, start ≤ index →
      index < start + 3 * degree * (rounds + 1) → allowed index) :
    KSupported (FixedChain.Owned.roundProgram start round).output allowed := by
  have supported := (NightstreamFPrime.Gadgets.Polynomial.Horner.compile_varsSatisfy
    start round.challenge round.coefficients allowed roundSupport.2
    (coefficients_supported round allowed roundSupport)).2
  rw [show (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start
      round.challenge round.coefficients).recipes.length = 3 * degree from
    FixedChain.Owned.roundProgram_recipes_length start round] at supported
  exact NightstreamFPrime.Gadgets.Polynomial.Horner.KSupported.mono supported
    (roundInterval_allowed start rounds allowed stored)

private theorem recipeRows_supported {degree : Nat} (start : Nat)
    (rounds : List (FixedChain.Round degree)) (allowed : Nat → Prop)
    (roundsSupport : ∀ round ∈ rounds, RoundSupported round allowed)
    (stored : ∀ index, start ≤ index →
      index < start + 3 * degree * rounds.length → allowed index) :
    ∀ expression ∈ recipeConstraints start
        (FixedChain.Owned.recipesFrom start rounds),
      expression.VarsSatisfy allowed := by
  induction rounds generalizing start with
  | nil =>
      intro expression member
      simp [FixedChain.Owned.recipesFrom, recipeConstraints] at member
  | cons round rounds inductionHypothesis =>
      have roundSupport := roundsSupport round (by simp)
      intro expression member
      rw [FixedChain.Owned.recipesFrom, recipeConstraints_append,
        FixedChain.Owned.roundProgram_recipes_length, List.mem_append] at member
      rcases member with headMember | tailMember
      · have supported :=
          NightstreamFPrime.Gadgets.Polynomial.Horner.compile_recipeConstraints_varsSatisfy
            start round.challenge round.coefficients allowed roundSupport.2
            (coefficients_supported round allowed roundSupport)
            expression headMember
        rw [show (NightstreamFPrime.Gadgets.Polynomial.Horner.compile start
            round.challenge round.coefficients).recipes.length = 3 * degree from
          FixedChain.Owned.roundProgram_recipes_length start round] at supported
        exact Expr.VarsSatisfy.mono expression supported
          (roundInterval_allowed start rounds.length allowed stored)
      · exact inductionHypothesis (start + 3 * degree)
          (fun later laterMember => roundsSupport later (by simp [laterMember]))
          (laterInterval_allowed start rounds.length allowed stored)
          expression tailMember

private theorem equalityRows_supported {degree : Nat} (start : Nat)
    (current : KExpr) (rounds : List (FixedChain.Round degree))
    (allowed : Nat → Prop) (currentSupport : KSupported current allowed)
    (roundsSupport : ∀ round ∈ rounds, RoundSupported round allowed)
    (stored : ∀ index, start ≤ index →
      index < start + 3 * degree * rounds.length → allowed index) :
    ∀ expression ∈ FixedChain.Owned.equalitiesFrom start current rounds,
      expression.VarsSatisfy allowed := by
  induction rounds generalizing start current with
  | nil =>
      intro expression member
      simp [FixedChain.Owned.equalitiesFrom] at member
  | cons round rounds inductionHypothesis =>
      have roundSupport := roundsSupport round (by simp)
      intro expression member
      rw [FixedChain.Owned.equalitiesFrom, List.mem_append] at member
      rcases member with headMember | tailMember
      · exact equalities_supported current _ allowed currentSupport
          (boundarySum_supported round.coefficients allowed
            (coefficients_supported round allowed roundSupport))
          expression headMember
      · exact inductionHypothesis (start + 3 * degree)
          (FixedChain.Owned.roundProgram start round).output
          (roundOutput_supported start round rounds.length allowed
            roundSupport stored)
          (fun later laterMember => roundsSupport later (by simp [laterMember]))
          (laterInterval_allowed start rounds.length allowed stored)
          expression tailMember

private theorem outputFrom_supported {degree : Nat} (start : Nat)
    (current : KExpr) (rounds : List (FixedChain.Round degree))
    (allowed : Nat → Prop) (currentSupport : KSupported current allowed)
    (roundsSupport : ∀ round ∈ rounds, RoundSupported round allowed)
    (stored : ∀ index, start ≤ index →
      index < start + 3 * degree * rounds.length → allowed index) :
    KSupported (FixedChain.Owned.outputFrom start current rounds) allowed := by
  induction rounds generalizing start current with
  | nil => exact currentSupport
  | cons round rounds inductionHypothesis =>
      exact inductionHypothesis (start + 3 * degree)
        (FixedChain.Owned.roundProgram start round).output
        (roundOutput_supported start round rounds.length allowed
          (roundsSupport round (by simp)) stored)
        (fun later member => roundsSupport later (by simp [member]))
        (laterInterval_allowed start rounds.length allowed stored)

private theorem coreRounds_supported {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (roundSupport : ∀ roundIndex,
      RoundSupported (interface.round offset roundIndex) allowed) :
    ∀ round ∈ (coreInterface interface offset).rounds,
      RoundSupported round allowed := by
  intro round member
  rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
  rcases member with ⟨roundIndex, rfl⟩
  exact roundSupport roundIndex

private theorem coreStored_allowed {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (storedSupport : ∀ index, offset ≤ index →
      index < offset + privateCount degree → allowed index) :
    ∀ index, offset ≤ index →
      index < offset + 3 * degree *
        (coreInterface interface offset).rounds.length → allowed index := by
  intro index lower upper
  apply storedSupport index lower
  simpa [privateCount, FixedChain.Owned.privateCount,
    FixedChain.Owned.Interface.rounds] using upper

/-- Exact support of the fixed PiCCS SumCheck-chain rows: the supported
input wires and the chain's stored interval. -/
theorem flatConstraints_varsSatisfy {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ roundIndex,
      RoundSupported (interface.round offset roundIndex) allowed)
    (storedSupport : ∀ index, offset ≤ index →
      index < offset + privateCount degree → allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  rw [circuit_ops, FixedChain.Owned.flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with recipeMember | equalityMember
  · exact recipeRows_supported offset (coreInterface interface offset).rounds
      allowed (coreRounds_supported interface offset allowed roundSupport)
      (coreStored_allowed interface offset allowed storedSupport)
      expression recipeMember
  · exact equalityRows_supported offset (coreInterface interface offset).initial
      (coreInterface interface offset).rounds allowed
      (by simpa [coreInterface] using initialSupport)
      (coreRounds_supported interface offset allowed roundSupport)
      (coreStored_allowed interface offset allowed storedSupport)
      expression equalityMember

/-- The stored final SumCheck claim has the support of the chain rows. -/
theorem output_varsSatisfy {degree : Nat}
    (interface : Interface degree) (offset : Nat) (allowed : Nat → Prop)
    (initialSupport : KSupported (interface.initial offset) allowed)
    (roundSupport : ∀ roundIndex,
      RoundSupported (interface.round offset roundIndex) allowed)
    (storedSupport : ∀ index, offset ≤ index →
      index < offset + privateCount degree → allowed index) :
    KSupported (output interface offset) allowed :=
  outputFrom_supported offset (coreInterface interface offset).initial
    (coreInterface interface offset).rounds allowed
    (by simpa [coreInterface] using initialSupport)
    (coreRounds_supported interface offset allowed roundSupport)
    (coreStored_allowed interface offset allowed storedSupport)

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain
