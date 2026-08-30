import NightstreamFPrime.Circuit.SupportRange
import NightstreamFPrime.Gadgets.Multilinear.PointEquality
import NightstreamFPrime.Gadgets.Polynomial.HornerSupport

/-!
Owns variable-support propagation for the child-owned multilinear
point-equality compiler. The structural compiler remains the semantic
authority.
-/

namespace NightstreamFPrime.Gadgets.Multilinear.PointEquality

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Circuit.SupportRange
open NightstreamFPrime.Gadgets.Polynomial

private theorem add_supported (left right : KExpr) (allowed : Nat → Prop)
    (leftSupport : Horner.KSupported left allowed)
    (rightSupport : Horner.KSupported right allowed) :
    Horner.KSupported (KExpr.add left right) allowed :=
  ⟨⟨leftSupport.1, rightSupport.1⟩,
    ⟨leftSupport.2, rightSupport.2⟩⟩

private theorem sub_supported (left right : KExpr) (allowed : Nat → Prop)
    (leftSupport : Horner.KSupported left allowed)
    (rightSupport : Horner.KSupported right allowed) :
    Horner.KSupported (KExpr.sub left right) allowed := by
  change
    (left.c0.VarsSatisfy allowed ∧
      ((Expr.const (-1)).VarsSatisfy allowed ∧
        right.c0.VarsSatisfy allowed)) ∧
    (left.c1.VarsSatisfy allowed ∧
      ((Expr.const (-1)).VarsSatisfy allowed ∧
        right.c1.VarsSatisfy allowed))
  exact ⟨⟨leftSupport.1, ⟨trivial, rightSupport.1⟩⟩,
    ⟨leftSupport.2, ⟨trivial, rightSupport.2⟩⟩⟩

private theorem mul_supported (left right : KExpr) (allowed : Nat → Prop)
    (leftSupport : Horner.KSupported left allowed)
    (rightSupport : Horner.KSupported right allowed) :
    Horner.KSupported (KExpr.mul left right) allowed := by
  unfold KExpr.mul Horner.KSupported
  simp only [Expr.VarsSatisfy]
  exact ⟨
    ⟨⟨leftSupport.1, rightSupport.1⟩,
      ⟨⟨trivial, leftSupport.2⟩, rightSupport.2⟩⟩,
    ⟨⟨leftSupport.1, rightSupport.2⟩,
      ⟨leftSupport.2, rightSupport.1⟩⟩⟩

private theorem factorExpr_supported (coordinate : CoordinateExpr)
    (allowed : Nat → Prop)
    (leftSupport : Horner.KSupported coordinate.left allowed)
    (rightSupport : Horner.KSupported coordinate.right allowed) :
    Horner.KSupported (factorExpr coordinate) allowed := by
  let oneMinusRight := KExpr.sub KExpr.one coordinate.right
  have oneSupport : Horner.KSupported KExpr.one allowed :=
    ⟨trivial, trivial⟩
  have oneMinusSupport := sub_supported KExpr.one coordinate.right allowed
    oneSupport rightSupport
  exact add_supported oneMinusRight
    (KExpr.mul coordinate.left
      (KExpr.sub coordinate.right oneMinusRight)) allowed
    oneMinusSupport
    (mul_supported coordinate.left
      (KExpr.sub coordinate.right oneMinusRight) allowed leftSupport
      (sub_supported coordinate.right oneMinusRight allowed rightSupport
        oneMinusSupport))

private theorem factorRecipes_supported (coordinate : CoordinateExpr)
    (allowed : Nat → Prop)
    (leftSupport : Horner.KSupported coordinate.left allowed)
    (rightSupport : Horner.KSupported coordinate.right allowed) :
    ∀ expression ∈ factorRecipes coordinate,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [factorRecipes, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact (factorExpr_supported coordinate allowed leftSupport
      rightSupport).1
  · exact (factorExpr_supported coordinate allowed leftSupport
      rightSupport).2

private theorem mulRecipes_supported (left right : KExpr)
    (allowed : Nat → Prop) (leftSupport : Horner.KSupported left allowed)
    (rightSupport : Horner.KSupported right allowed) :
    ∀ expression ∈ mulRecipes left right,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [mulRecipes, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact (mul_supported left right allowed leftSupport rightSupport).1
  · exact (mul_supported left right allowed leftSupport rightSupport).2

private theorem materializedAt_supported (allowed : Nat → Prop)
    (start position finish : Nat) (lower : start ≤ position)
    (upper : position + 2 ≤ finish) :
    Horner.KSupported (materializedAt position)
      (Extend allowed start finish) := by
  unfold materializedAt Horner.KSupported
  constructor
  · apply interval <;> omega
  · apply interval <;> omega

/-- Every point-equality recipe and the compiled output use only supported
coordinate inputs or the exact recipe interval. -/
theorem compile_varsSatisfy (start : Nat)
    (coordinates : List CoordinateExpr) (allowed : Nat → Prop)
    (coordinatesSupport : ∀ coordinate ∈ coordinates,
      Horner.KSupported coordinate.left allowed ∧
        Horner.KSupported coordinate.right allowed) :
    (∀ expression ∈ (compile start coordinates).recipes,
        expression.VarsSatisfy
          (Extend allowed start
            (start + (compile start coordinates).recipes.length))) ∧
      Horner.KSupported (compile start coordinates).output
        (Extend allowed start
          (start + (compile start coordinates).recipes.length)) := by
  induction coordinates with
  | nil =>
      constructor
      · intro expression member
        simp [compile] at member
      · simp [compile, Horner.KSupported, KExpr.one, Expr.VarsSatisfy]
  | cons coordinate coordinates inductionHypothesis =>
      cases coordinates with
      | nil =>
          have currentSupport := coordinatesSupport coordinate (by simp)
          constructor
          · intro expression member
            apply Expr.VarsSatisfy.mono expression
              (factorRecipes_supported coordinate allowed currentSupport.1
                currentSupport.2 expression (by simpa [compile] using member))
            intro index support
            exact base support
          · simpa [compile] using
              materializedAt_supported allowed start start (start + 2)
                (by omega) (by omega)
      | cons next rest =>
          let tail := compile start (next :: rest)
          let factorStart := start + tail.recipes.length
          let factor := materializedAt factorStart
          let productStart := factorStart + 2
          let finish := productStart + 2
          have tailCoordinatesSupport : ∀ current ∈ next :: rest,
              Horner.KSupported current.left allowed ∧
                Horner.KSupported current.right allowed := by
            intro current member
            exact coordinatesSupport current (by simp [member])
          have tailProof :
              (∀ expression ∈ tail.recipes,
                  expression.VarsSatisfy
                    (Extend allowed start factorStart)) ∧
                Horner.KSupported tail.output
                  (Extend allowed start factorStart) := by
            simpa [tail, factorStart] using
              inductionHypothesis tailCoordinatesSupport
          have currentSupport := coordinatesSupport coordinate (by simp)
          have factorSupport : Horner.KSupported factor
              (Extend allowed start finish) := by
            apply materializedAt_supported allowed start factorStart finish
            · unfold factorStart
              omega
            · unfold finish productStart
              omega
          have tailOutputSupport : Horner.KSupported tail.output
              (Extend allowed start finish) := by
            apply Horner.KSupported.mono tailProof.2
            intro index support
            exact mono_finish support (by
              unfold finish productStart
              omega)
          have currentLeftSupport : Horner.KSupported coordinate.left
              (Extend allowed start finish) := by
            apply Horner.KSupported.mono currentSupport.1
            intro index support
            exact base support
          have currentRightSupport : Horner.KSupported coordinate.right
              (Extend allowed start finish) := by
            apply Horner.KSupported.mono currentSupport.2
            intro index support
            exact base support
          have factorRecipeSupport : ∀ expression ∈ factorRecipes coordinate,
              expression.VarsSatisfy (Extend allowed start finish) :=
            factorRecipes_supported coordinate _ currentLeftSupport
              currentRightSupport
          have productRecipeSupport : ∀ expression ∈
              mulRecipes factor tail.output,
              expression.VarsSatisfy (Extend allowed start finish) :=
            mulRecipes_supported factor tail.output _ factorSupport
              tailOutputSupport
          have finishEq :
              start +
                  (compile start (coordinate :: next :: rest)).recipes.length =
                finish := by
            simp [compile, tail, factorStart, productStart, finish]
            omega
          rw [finishEq]
          constructor
          · intro expression member
            simp only [compile, List.mem_append] at member
            rcases member with tailOrFactor | productMember
            · rcases tailOrFactor with tailMember | factorMember
              · apply Expr.VarsSatisfy.mono expression
                    (tailProof.1 expression tailMember)
                intro index support
                exact mono_finish support (by
                  unfold finish productStart
                  omega)
              · exact factorRecipeSupport expression factorMember
            · exact productRecipeSupport expression productMember
          · simpa only [compile, tail, factorStart, factor, productStart]
              using materializedAt_supported allowed start productStart finish
                (by unfold productStart factorStart; omega) (by
                  unfold finish; omega)

namespace Owned

private theorem coordinateExprs_supported {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (allowed : Nat → Prop)
    (leftSupport : ∀ coordinate,
      Horner.KSupported (interface.left offset coordinate) allowed)
    (rightSupport : ∀ coordinate,
      Horner.KSupported (interface.right offset coordinate) allowed) :
    ∀ coordinate ∈ coordinateExprs interface offset,
      Horner.KSupported coordinate.left allowed ∧
        Horner.KSupported coordinate.right allowed := by
  intro coordinate member
  rw [coordinateExprs, List.mem_map] at member
  rcases member with ⟨index, _indexMember, rfl⟩
  exact ⟨leftSupport index, rightSupport index⟩

/-- Exact support propagation through the child-owned point-equality rows. -/
theorem flatConstraints_varsSatisfy {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (allowed : Nat → Prop)
    (leftSupport : ∀ coordinate,
      Horner.KSupported (interface.left offset coordinate) allowed)
    (rightSupport : ∀ coordinate,
      Horner.KSupported (interface.right offset coordinate) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + (program interface offset).recipes.length →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have compiled := PointEquality.compile_varsSatisfy offset
    (coordinateExprs interface offset) allowed
    (coordinateExprs_supported interface offset allowed leftSupport
      rightSupport)
  let finalAllowed := Extend allowed offset
    (offset + (program interface offset).recipes.length)
  have rowsSupported := Horner.recipeConstraints_varsSatisfy offset
    (program interface offset).recipes finalAllowed compiled.1 (by
      intro index indexBound
      apply interval <;> omega)
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsSatisfy allowed
  rw [flatConstraints_opsAt]
  intro expression member
  apply Expr.VarsSatisfy.mono expression (rowsSupported expression member)
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · exact localSupport index lower upper

/-- The child-owned point-equality output uses the same external support and
exact recipe interval as its rows. -/
theorem output_varsSatisfy {variableCount : Nat}
    (interface : Interface variableCount) (offset : Nat)
    (allowed : Nat → Prop)
    (leftSupport : ∀ coordinate,
      Horner.KSupported (interface.left offset coordinate) allowed)
    (rightSupport : ∀ coordinate,
      Horner.KSupported (interface.right offset coordinate) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + (program interface offset).recipes.length →
      allowed index) :
    Horner.KSupported (output interface offset) allowed := by
  have supported := (PointEquality.compile_varsSatisfy offset
    (coordinateExprs interface offset) allowed
      (coordinateExprs_supported interface offset allowed leftSupport
        rightSupport)).2
  apply Horner.KSupported.mono supported
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · exact localSupport index lower (by
      simpa [program, output] using upper)

end Owned

end NightstreamFPrime.Gadgets.Multilinear.PointEquality
