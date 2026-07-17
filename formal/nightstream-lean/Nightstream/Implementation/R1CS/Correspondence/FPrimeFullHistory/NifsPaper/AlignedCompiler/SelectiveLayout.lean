import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Contract: independent arithmetic model of the selective compiler's repaired
F' public-carrier layout.

Owns: the protocol -> compiler -> column-family partition derived from the
typed `FPrimeCarrier270` dimensions: logical values, public zero coordinates,
branch selectors, private alignment zeros, and branch-private start.

Does not own: a Rust trace, emitted sparse matrices, the selective CCS
polynomial, witness serialization, or a claim that a particular production
compile instantiated this model. Those require a raw compiler artifact.

Emits constraints: no.

Authority boundary: selectors start after the complete 270-coordinate public
carrier. The legacy placement is retained only as a counterexample showing
that putting selectors at column 257 leaks them into the public carrier.

| Stage path | Mathematical obligation | Authority class | Rust owner | Lean owner | Fixed-profile value | Removal status |
|---|---|---|---|---|---:|---|
| `f_prime.compiler.public.logical` | Preserve logical F' columns | direct dataflow | `prepare_selective_layout` | typed `FPrimeCarrier270` | `0..257` | retained |
| `f_prime.compiler.public.padding` | Complete the five 54-lane public blocks | checked | `prepare_selective_layout` | typed `dimensions_exact` | `257..270` | retained |
| `f_prime.compiler.selectors` | Keep branch authority outside public CE input | checked | `prepare_selective_layout` | `selectorColumn_bounds` | `270..273` | retained |
| `f_prime.compiler.private_alignment` | Restore residue 41 before branch data | computed | `prepare_selective_layout` | `exact_layout` | `273..311` | retained |
| `f_prime.compiler.branch_private` | Begin arm-owned data after shared layout | direct dataflow | `prepare_selective_layout` | `exact_layout` | `311` | retained |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.SelectiveLayout

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

/-- The production selective relation has base, bootstrap-recursive, and
steady-recursive arms. -/
def selectorCount : Nat := 3

/-- First verifier-fixed coefficient completing the public ring carrier. -/
def publicPaddingStart : Nat := legacyPublicWidth

/-- Branch selectors begin after all five public ring columns. -/
def selectorStart : Nat := alignedPublicWidth

/-- Selective-compiler column for one branch selector. -/
def selectorColumn (branch : Fin selectorCount) : Nat :=
  selectorStart + branch.val

def postSelectorCursor : Nat := selectorStart + selectorCount

/-- The low-norm public prefix originally ended at residue 41 modulo 54. -/
def privateAlignmentResidue : Nat := legacyPublicWidth % ringDegree

/-- Exact private zero width used to return the branch start to residue 41. -/
def privateAlignmentPaddingWidth : Nat :=
  (privateAlignmentResidue + ringDegree - postSelectorCursor % ringDegree) %
    ringDegree

def privateAlignmentStart : Nat := postSelectorCursor

def branchPrivateStart : Nat :=
  privateAlignmentStart + privateAlignmentPaddingWidth

/-- Pre-repair selector placement. This is diagnostic counterexample data,
not an accepted layout. -/
def legacySelectorColumn (branch : Fin selectorCount) : Nat :=
  legacyPublicWidth + branch.val

theorem exact_layout :
    publicPaddingStart = 257 ∧
      fixedPaddingWidth = 13 ∧
      selectorStart = 270 ∧
      selectorCount = 3 ∧
      postSelectorCursor = 273 ∧
      privateAlignmentResidue = 41 ∧
      privateAlignmentPaddingWidth = 38 ∧
      branchPrivateStart = 311 := by
  decide

theorem selectorColumn_bounds (branch : Fin selectorCount) :
    alignedPublicWidth ≤ selectorColumn branch ∧
      selectorColumn branch < postSelectorCursor := by
  simp [selectorColumn, selectorStart, postSelectorCursor, alignedPublicWidth,
    selectorCount] at branch ⊢

/-- The repaired layout makes every selector disjoint from the complete
public CE carrier. -/
theorem selector_not_in_public_carrier (branch : Fin selectorCount) :
    ¬ selectorColumn branch < alignedPublicWidth := by
  exact Nat.not_lt.mpr (selectorColumn_bounds branch).1

/-- Every selector in the pre-repair placement occupied a coefficient of the
five-block public CE carrier. This proves the placement change is semantic,
not cosmetic. -/
theorem legacy_selector_in_public_carrier (branch : Fin selectorCount) :
    legacySelectorColumn branch < alignedPublicWidth := by
  simp [legacySelectorColumn, legacyPublicWidth, alignedPublicWidth,
    publicRingColumns, ringDegree, selectorCount] at branch ⊢
  omega

theorem privateAlignment_range :
    privateAlignmentStart = 273 ∧
      branchPrivateStart - privateAlignmentStart = 38 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.SelectiveLayout
