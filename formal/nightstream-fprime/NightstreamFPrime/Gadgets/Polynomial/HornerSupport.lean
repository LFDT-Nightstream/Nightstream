import NightstreamFPrime.Circuit.SupportRange
import NightstreamFPrime.Gadgets.Polynomial.Horner

/-!
Owns variable-support propagation for the generic quadratic-extension Horner
compiler. The structural compiler remains the semantic authority.
-/

namespace NightstreamFPrime.Gadgets.Polynomial.Horner

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Circuit.SupportRange

def KSupported (value : KExpr) (allowed : Nat → Prop) : Prop :=
  value.c0.VarsSatisfy allowed ∧ value.c1.VarsSatisfy allowed

theorem KSupported.mono {value : KExpr} {allowed larger : Nat → Prop}
    (support : KSupported value allowed)
    (includes : ∀ index, allowed index → larger index) :
    KSupported value larger :=
  ⟨Expr.VarsSatisfy.mono value.c0 support.1 includes,
    Expr.VarsSatisfy.mono value.c1 support.2 includes⟩

private theorem add_supported (left right : KExpr) (allowed : Nat → Prop)
    (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    KSupported (KExpr.add left right) allowed := by
  exact ⟨⟨leftSupport.1, rightSupport.1⟩,
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

theorem KSupported.zero (allowed : Nat → Prop) :
    KSupported KExpr.zero allowed :=
  ⟨trivial, trivial⟩

theorem KSupported.one (allowed : Nat → Prop) :
    KSupported KExpr.one allowed :=
  ⟨trivial, trivial⟩

theorem KSupported.add {left right : KExpr} {allowed : Nat → Prop}
    (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    KSupported (KExpr.add left right) allowed :=
  add_supported left right allowed leftSupport rightSupport

theorem KSupported.sub {left right : KExpr} {allowed : Nat → Prop}
    (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    KSupported (KExpr.sub left right) allowed := by
  change
    (left.c0.VarsSatisfy allowed ∧
      ((Expr.const (-1)).VarsSatisfy allowed ∧
        right.c0.VarsSatisfy allowed)) ∧
    (left.c1.VarsSatisfy allowed ∧
      ((Expr.const (-1)).VarsSatisfy allowed ∧
        right.c1.VarsSatisfy allowed))
  exact ⟨⟨leftSupport.1, ⟨trivial, rightSupport.1⟩⟩,
    ⟨leftSupport.2, ⟨trivial, rightSupport.2⟩⟩⟩

theorem KSupported.mul {left right : KExpr} {allowed : Nat → Prop}
    (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    KSupported (KExpr.mul left right) allowed :=
  mul_supported left right allowed leftSupport rightSupport

private theorem mulRecipes_supported (left right : KExpr)
    (allowed : Nat → Prop) (leftSupport : KSupported left allowed)
    (rightSupport : KSupported right allowed) :
    ∀ expression ∈ mulRecipes left right,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [mulRecipes, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact (mul_supported left right allowed leftSupport rightSupport).1
  · exact (mul_supported left right allowed leftSupport rightSupport).2

private theorem productAt_supported (allowed : Nat → Prop)
    (base productStart finish : Nat) (baseLeProduct : base ≤ productStart)
    (productEndLe : productStart + 2 ≤ finish) :
    KSupported (productAt productStart) (Extend allowed base finish) := by
  unfold productAt KSupported
  constructor
  · apply NightstreamFPrime.Circuit.SupportRange.interval <;> omega
  · apply NightstreamFPrime.Circuit.SupportRange.interval <;> omega

/-- Generic straight-line rows preserve one caller-selected support
predicate when every target and recipe already satisfies it. -/
theorem recipeConstraints_varsSatisfy
    (start : Nat) (recipes : List Expr) (allowed : Nat → Prop)
    (recipesSupported : ∀ recipe ∈ recipes,
      recipe.VarsSatisfy allowed)
    (targetsSupported : ∀ index, index < recipes.length →
      allowed (start + index)) :
    ∀ expression ∈ recipeConstraints start recipes,
      expression.VarsSatisfy allowed := by
  induction recipes generalizing start with
  | nil =>
      intro expression member
      simp [recipeConstraints] at member
  | cons recipe recipes inductionHypothesis =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · exact ⟨targetsSupported 0 (by simp),
          ⟨trivial, recipesSupported recipe (by simp)⟩⟩
      · apply inductionHypothesis (start := start + 1)
        · intro current currentMember
          exact recipesSupported current (by simp [currentMember])
        · intro index indexBound
          have target := targetsSupported (index + 1) (by simp; omega)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using target
        · exact member

/-- Every recipe and the final Horner output use only supported external
inputs or the exact recipe interval allocated by this compilation. -/
theorem compile_varsSatisfy (start : Nat) (point : KExpr)
    (coefficients : List KExpr) (allowed : Nat → Prop)
    (pointSupport : KSupported point allowed)
    (coefficientsSupport : ∀ coefficient ∈ coefficients,
      KSupported coefficient allowed) :
    (∀ expression ∈ (compile start point coefficients).recipes,
        expression.VarsSatisfy
          (Extend allowed start
            (start + (compile start point coefficients).recipes.length))) ∧
      KSupported (compile start point coefficients).output
        (Extend allowed start
          (start + (compile start point coefficients).recipes.length)) := by
  induction coefficients with
  | nil =>
      constructor
      · intro expression member
        simp [compile] at member
      · simp [compile, KSupported, KExpr.zero, Expr.VarsSatisfy]
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil =>
          constructor
          · intro expression member
            simp [compile] at member
          · apply KSupported.mono
              (coefficientsSupport coefficient (by simp))
            intro index support
            exact NightstreamFPrime.Circuit.SupportRange.base support
      | cons next rest =>
          let tail := compile start point (next :: rest)
          have tailCoefficientsSupport : ∀ current ∈ next :: rest,
              KSupported current allowed := by
            intro current member
            exact coefficientsSupport current (by simp [member])
          have tailProof :
              (∀ expression ∈ tail.recipes,
                  expression.VarsSatisfy
                    (Extend allowed start (start + tail.recipes.length))) ∧
                KSupported tail.output
                  (Extend allowed start (start + tail.recipes.length)) := by
            simpa [tail] using
              inductionHypothesis tailCoefficientsSupport
          let productStart := start + tail.recipes.length
          let finish := productStart + 2
          have tailFinishLe : start + tail.recipes.length ≤ finish := by
            unfold finish productStart
            omega
          have pointAtFinish : KSupported point
              (Extend allowed start finish) := by
            apply KSupported.mono pointSupport
            intro index support
            exact NightstreamFPrime.Circuit.SupportRange.base support
          have coefficientAtFinish : KSupported coefficient
              (Extend allowed start finish) := by
            apply KSupported.mono
              (coefficientsSupport coefficient (by simp))
            intro index support
            exact NightstreamFPrime.Circuit.SupportRange.base support
          have tailOutputAtFinish : KSupported tail.output
              (Extend allowed start finish) := by
            apply KSupported.mono tailProof.2
            intro index support
            exact NightstreamFPrime.Circuit.SupportRange.mono_finish support
              tailFinishLe
          have addedSupported : ∀ expression ∈ mulRecipes point tail.output,
              expression.VarsSatisfy (Extend allowed start finish) :=
            mulRecipes_supported point tail.output _ pointAtFinish
              tailOutputAtFinish
          have productSupport : KSupported (productAt productStart)
              (Extend allowed start finish) := by
            apply productAt_supported allowed start productStart finish
            · unfold productStart
              omega
            · unfold finish
              omega
          have finishEq :
              start +
                  (compile start point (coefficient :: next :: rest)).recipes.length =
                finish := by
            simp [compile, tail, finish, productStart, mulRecipes_length]
            omega
          rw [finishEq]
          constructor
          · intro expression member
            simp only [compile, List.mem_append] at member
            rcases member with tailMember | addedMember
            · apply Expr.VarsSatisfy.mono expression
                (tailProof.1 expression tailMember)
              intro index support
              exact NightstreamFPrime.Circuit.SupportRange.mono_finish support
                tailFinishLe
            · exact addedSupported expression addedMember
          · simpa only [compile, tail, productStart] using
              add_supported coefficient (productAt productStart)
                (Extend allowed start finish) coefficientAtFinish
                  productSupport

/-- The canonical Horner recipe rows inherit the same exact support union. -/
theorem compile_recipeConstraints_varsSatisfy
    (start : Nat) (point : KExpr) (coefficients : List KExpr)
    (allowed : Nat → Prop) (pointSupport : KSupported point allowed)
    (coefficientsSupport : ∀ coefficient ∈ coefficients,
      KSupported coefficient allowed) :
    ∀ expression ∈ recipeConstraints start
        (compile start point coefficients).recipes,
      expression.VarsSatisfy
        (Extend allowed start
          (start + (compile start point coefficients).recipes.length)) := by
  let finalAllowed := Extend allowed start
    (start + (compile start point coefficients).recipes.length)
  apply recipeConstraints_varsSatisfy start
    (compile start point coefficients).recipes finalAllowed
  · exact (compile_varsSatisfy start point coefficients allowed pointSupport
      coefficientsSupport).1
  · intro index indexBound
    apply NightstreamFPrime.Circuit.SupportRange.interval
    · omega
    · omega

namespace Owned

/-- The child-owned Horner rows use only supported external expressions and
the exact recipe interval allocated by this invocation. -/
theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (pointSupport : KSupported (interface.point offset) allowed)
    (coefficientsSupport : ∀ coefficient ∈ interface.coefficients offset,
      KSupported coefficient allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + (program interface offset).recipes.length →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have supported := compile_recipeConstraints_varsSatisfy offset
    (interface.point offset) (interface.coefficients offset) allowed
    pointSupport coefficientsSupport
  rw [circuit_ops, flatConstraints_opsAt]
  intro expression member
  apply Expr.VarsSatisfy.mono expression (supported expression member)
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · exact localSupport index lower (by
      simpa [program] using upper)

/-- The child-owned Horner output uses the same external support and exact
recipe interval as its rows. -/
theorem output_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (pointSupport : KSupported (interface.point offset) allowed)
    (coefficientsSupport : ∀ coefficient ∈ interface.coefficients offset,
      KSupported coefficient allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + (program interface offset).recipes.length →
      allowed index) :
    KSupported (output interface offset) allowed := by
  have supported := (compile_varsSatisfy offset (interface.point offset)
    (interface.coefficients offset) allowed pointSupport
      coefficientsSupport).2
  apply KSupported.mono supported
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · exact localSupport index lower (by
      simpa [program, output] using upper)

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Horner
