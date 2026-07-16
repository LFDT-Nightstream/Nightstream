import SuperNeo.FPrimeRecursiveVerifier.Cost
import Mathlib.Algebra.Ring.Defs

/-!
Owns: abstract sparse R1CS semantics and the refinement contract for modular
verifier blocks.

Does not own: concrete Rust rows, trace extraction, retained-column decoding,
or production witness compilation.

Emits constraints: no. It models and certifies supplied blocks.

Authority boundary: semantic authority comes from `BlockRefinement.sound` and
honest completeness from `PlanWitnessComplete`; dimensions alone prove neither.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Row semantics | `R1csConstraint.Holds`, `R1csBlock.Satisfied` | Defines sparse R1CS acceptance |
| Block bridge | `BlockRefinement` | Relates each block to one semantic check |
| Certified lowering | `CertifiedR1csPlan` | Combines exact plan, refinement, and witness completeness |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

open scoped BigOperators

universe u v w x

/-- A sparse linear combination over numbered local columns. -/
structure LinearCombination (R : Type u) where
  terms : List (Nat × R)
deriving Repr

namespace LinearCombination

/-- Evaluate a sparse linear combination against a local assignment. -/
def eval
    {R : Type u} [Semiring R]
    (combination : LinearCombination R)
    (assignment : Nat → R) : R :=
  (combination.terms.map (fun term => term.2 * assignment term.1)).sum

/-- Every referenced column lies inside the block's declared local width. -/
def WellFormed
    {R : Type u}
    (columns : Nat)
    (combination : LinearCombination R) : Prop :=
  ∀ term, term ∈ combination.terms → term.1 < columns

/-- Number of explicitly stored matrix entries. -/
def nonzeros {R : Type u} (combination : LinearCombination R) : Nat :=
  combination.terms.length

end LinearCombination

/-- One sparse R1CS row, interpreted as `A(z) * B(z) = C(z)`. -/
structure R1csConstraint (R : Type u) where
  a : LinearCombination R
  b : LinearCombination R
  c : LinearCombination R
deriving Repr

namespace R1csConstraint

/-- Standard R1CS row satisfaction. -/
def Holds
    {R : Type u} [Semiring R]
    (constraint : R1csConstraint R)
    (assignment : Nat → R) : Prop :=
  constraint.a.eval assignment * constraint.b.eval assignment =
    constraint.c.eval assignment

/-- All three sparse row vectors fit in the declared local width. -/
def WellFormed
    {R : Type u}
    (columns : Nat)
    (constraint : R1csConstraint R) : Prop :=
  constraint.a.WellFormed columns ∧
    constraint.b.WellFormed columns ∧
    constraint.c.WellFormed columns

/-- Number of explicitly stored entries across the three row vectors. -/
def nonzeros {R : Type u} (constraint : R1csConstraint R) : Nat :=
  constraint.a.nonzeros + constraint.b.nonzeros + constraint.c.nonzeros

end R1csConstraint

/-- An independently lowerable check block with a local witness namespace. -/
structure R1csBlock (R : Type u) where
  columns : Nat
  constraints : List (R1csConstraint R)
deriving Repr

namespace R1csBlock

/-- Every row of a block is satisfied by its local assignment. -/
def Satisfied
    {R : Type u} [Semiring R]
    (block : R1csBlock R)
    (assignment : Nat → R) : Prop :=
  ∀ constraint, constraint ∈ block.constraints →
    constraint.Holds assignment

/-- No sparse term addresses a column outside the block's local namespace. -/
def WellFormed
    {R : Type u}
    (block : R1csBlock R) : Prop :=
  ∀ constraint, constraint ∈ block.constraints →
    constraint.WellFormed block.columns

/-- Structural cost of a block; rows and nonzeros are computed, not asserted. -/
def cost {R : Type u} (block : R1csBlock R) : R1csCost :=
  { rows := block.constraints.length
    columns := block.columns
    nonzeros := (block.constraints.map R1csConstraint.nonzeros).sum }

end R1csBlock

/--
A modular lowering. Each check owns a block and a local assignment projection
from the common high-level input and compiler witness.
-/
structure ModularR1csEncoding
    (R : Type u) (Input : Type v) (Check : Type w) (Witness : Type x) where
  block : Check → R1csBlock R
  assignment : Check → Input → Witness → Nat → R

/-- Satisfaction of every selected block under one compiler witness. -/
def CompiledSatisfied
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (checks : Finset Check)
    (input : Input)
    (witness : Witness) : Prop :=
  ∀ check, check ∈ checks →
    (encoding.block check).Satisfied
      (encoding.assignment check input witness)

/-- A witness satisfying more blocks also satisfies every selected subset. -/
theorem compiledSatisfied_of_subset
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {small large : Finset Check} {input : Input} {witness : Witness}
    (hSubset : small ⊆ large)
    (hLarge : CompiledSatisfied encoding large input witness) :
    CompiledSatisfied encoding small input witness := by
  intro check hCheck
  exact hLarge check (hSubset hCheck)

/--
Per-block refinement certificate. `sound` is the critical lowering theorem:
no assignment may satisfy a block unless the corresponding semantic check is
true of the authoritative input.
-/
structure BlockRefinement
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (semantics : Check → Input → Prop) where
  wellFormed : ∀ check, (encoding.block check).WellFormed
  sound : ∀ check input witness,
    (encoding.block check).Satisfied
        (encoding.assignment check input witness) →
      semantics check input

/-- A witness compiler succeeds simultaneously for every selected check. -/
def PlanWitnessComplete
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (semantics : Check → Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, Accepts semantics checks input →
    ∃ witness, CompiledSatisfied encoding checks input witness

/-- Existential circuit acceptance for a fixed public/authoritative input. -/
def R1csAccepts
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (checks : Finset Check)
    (input : Input) : Prop :=
  ∃ witness, CompiledSatisfied encoding checks input witness

/-- The selected R1CS language has no false positives against the target. -/
def R1csSoundForTarget
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, R1csAccepts encoding checks input → target input

/-- The selected R1CS language has a witness for every target input. -/
def R1csCompleteForTarget
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, target input → R1csAccepts encoding checks input

/-- Exact equality of the existential R1CS and semantic target languages. -/
def R1csExactForTarget
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (target : Input → Prop)
    (checks : Finset Check) : Prop :=
  ∀ input, R1csAccepts encoding checks input ↔ target input

theorem compiledSatisfied_implies_accepts
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    (refinement : BlockRefinement encoding semantics)
    {checks : Finset Check} {input : Input} {witness : Witness}
    (hCompiled : CompiledSatisfied encoding checks input witness) :
    Accepts semantics checks input := by
  intro check hCheck
  exact refinement.sound check input witness (hCompiled check hCheck)

/-- Semantic plan soundness plus block refinement gives circuit soundness. -/
theorem r1csSound_of_plan
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    {target : Input → Prop} {checks : Finset Check}
    (refinement : BlockRefinement encoding semantics)
    (hSound : Sound semantics target checks) :
    R1csSoundForTarget encoding target checks := by
  rintro input ⟨witness, hCompiled⟩
  exact hSound input
    (compiledSatisfied_implies_accepts refinement hCompiled)

/-- Semantic plan completeness plus a witness compiler gives circuit completeness. -/
theorem r1csComplete_of_plan
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    {target : Input → Prop} {checks : Finset Check}
    (hComplete : Complete semantics target checks)
    (compilerComplete : PlanWitnessComplete encoding semantics checks) :
    R1csCompleteForTarget encoding target checks := by
  intro input hTarget
  exact compilerComplete input (hComplete input hTarget)

/-- A certified semantic plan and certified lowering establish exact R1CS semantics. -/
theorem r1csExact_of_certifiedPlan
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (plan : CertifiedPlan semantics target)
    (refinement : BlockRefinement encoding semantics)
    (compilerComplete :
      PlanWitnessComplete encoding semantics plan.checks) :
    R1csExactForTarget encoding target plan.checks := by
  intro input
  exact ⟨
    r1csSound_of_plan refinement plan.sound input,
    r1csComplete_of_plan plan.complete compilerComplete input⟩

/-- The exact structural cost of independently concatenating selected blocks. -/
def compiledCost
    {R : Type u}
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (checks : Finset Check) : R1csCost :=
  planCost (fun check => (encoding.block check).cost) checks

theorem compiledCost_eq_planCost
    {R : Type u}
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (checks : Finset Check) :
    compiledCost encoding checks =
      planCost (fun check => (encoding.block check).cost) checks := by
  rfl

/-- All certificates required before a modular R1CS candidate is trusted. -/
structure CertifiedR1csPlan
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    (encoding : ModularR1csEncoding R Input Check Witness)
    (semantics : Check → Input → Prop)
    (target : Input → Prop) where
  semanticPlan : CertifiedPlan semantics target
  refinement : BlockRefinement encoding semantics
  compilerComplete :
    PlanWitnessComplete encoding semantics semanticPlan.checks

namespace CertifiedR1csPlan

/-- The packaged candidate accepts exactly the intended target relation. -/
theorem exact
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (candidate : CertifiedR1csPlan encoding semantics target) :
    R1csExactForTarget
      encoding target candidate.semanticPlan.checks :=
  r1csExact_of_certifiedPlan
    candidate.semanticPlan candidate.refinement candidate.compilerComplete

/--
Remove a semantically redundant block while transporting all R1CS
certificates. The old witness compiler is reused and its output restricted to
the smaller block set.
-/
def eraseRedundant
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (candidate : CertifiedR1csPlan encoding semantics target)
    (check : Check)
    (hRedundant :
      Redundant semantics candidate.semanticPlan.checks check) :
    CertifiedR1csPlan encoding semantics target where
  semanticPlan :=
    candidate.semanticPlan.eraseRedundant check hRedundant
  refinement := candidate.refinement
  compilerComplete := by
    intro input hErased
    have hFull :
        Accepts semantics candidate.semanticPlan.checks input :=
      (accepts_erase_iff_of_redundant hRedundant input).mpr hErased
    rcases candidate.compilerComplete input hFull with
      ⟨witness, hCompiled⟩
    exact ⟨witness,
      compiledSatisfied_of_subset
        (Finset.erase_subset check candidate.semanticPlan.checks)
        hCompiled⟩

/-- Cost is exposed only after the semantic and lowering certificates coexist. -/
def cost
    {R : Type u} [Semiring R]
    {Input : Type v} {Check : Type w} [DecidableEq Check]
    {Witness : Type x}
    {encoding : ModularR1csEncoding R Input Check Witness}
    {semantics : Check → Input → Prop}
    {target : Input → Prop}
    (candidate : CertifiedR1csPlan encoding semantics target) : R1csCost :=
  compiledCost encoding candidate.semanticPlan.checks

end CertifiedR1csPlan

end SuperNeo.FPrimeRecursiveVerifier
