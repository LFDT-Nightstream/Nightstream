import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Components

/-!
Contract: inclusion-necessity witnesses for the six chosen selective-polynomial
term families.

Owns: a precise "polynomial with one family omitted" relation and one closed
Goldilocks point per family that the weakened polynomial accepts while the
complete 66-term polynomial rejects.

Does not own: protocol-level necessity, uniqueness or gate-minimality of an
R1CS encoding, production row ownership, Rust conformance, or permission to
remove constraints. These witnesses only rule out deleting one whole chosen
term family while leaving the other five unchanged.

Emits constraints: no.

| Omitted family | Nonzero ports in the counterexample | Weakened result | Full result |
|---|---|---|---|
| boolean | `g=1, bit=2` | zero | nonzero |
| product | `g=1, a=b=1` | zero | nonzero |
| S-box | `g=1, s=1` | zero | nonzero |
| centered | `g=1, u=2` | zero | nonzero |
| evaluation | `e=1, bit=a=1` | zero | nonzero |
| canonical | `g=1, class2=class3=1` | zero | nonzero |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components

set_option maxRecDepth 10000

/-- Sum of the five retained component residuals after deleting exactly one
named family. -/
def residualWithout : Family -> (Fin 13 -> F) -> F
  | .boolean, point =>
      productResidual point + sboxResidual point + centeredResidual point +
        evaluationResidual point + canonicalResidual point
  | .product, point =>
      booleanResidual point + sboxResidual point + centeredResidual point +
        evaluationResidual point + canonicalResidual point
  | .sbox, point =>
      booleanResidual point + productResidual point + centeredResidual point +
        evaluationResidual point + canonicalResidual point
  | .centered, point =>
      booleanResidual point + productResidual point + sboxResidual point +
        evaluationResidual point + canonicalResidual point
  | .evaluation, point =>
      booleanResidual point + productResidual point + sboxResidual point +
        centeredResidual point + canonicalResidual point
  | .canonical, point =>
      booleanResidual point + productResidual point + sboxResidual point +
        centeredResidual point + evaluationResidual point

/-- Acceptance of the complete current selective polynomial. -/
def FullAccepts (point : Fin 13 -> F) : Prop := evaluate point = 0

/-- Acceptance after deleting exactly one chosen component family. -/
def AcceptsWithout (family : Family) (point : Fin 13 -> F) : Prop :=
  residualWithout family point = 0

instance (point : Fin 13 -> F) : Decidable (FullAccepts point) :=
  by
    unfold FullAccepts
    infer_instance

instance (family : Family) (point : Fin 13 -> F) :
    Decidable (AcceptsWithout family point) :=
  by
    unfold AcceptsWithout
    infer_instance

def booleanWitness : Fin 13 -> F := fun port =>
  if port = Role.bit.index then 2
  else if port = Role.generalSelector.index then 1
  else 0

def productWitness : Fin 13 -> F := fun port =>
  if port = Role.generalSelector.index then 1
  else if port = Role.a.index then 1
  else if port = Role.b.index then 1
  else 0

def sboxWitness : Fin 13 -> F := fun port =>
  if port = Role.generalSelector.index then 1
  else if port = Role.sboxInput.index then 1
  else 0

def centeredWitness : Fin 13 -> F := fun port =>
  if port = Role.generalSelector.index then 1
  else if port = Role.centeredUnit.index then 2
  else 0

def evaluationWitness : Fin 13 -> F := fun port =>
  if port = Role.bit.index then 1
  else if port = Role.a.index then 1
  else if port = Role.evalSelector.index then 1
  else 0

def canonicalWitness : Fin 13 -> F := fun port =>
  if port = Role.generalSelector.index then 1
  else if port = Role.canonicalNextBorrow.index then 1
  else if port = Role.canonicalBoundDigit.index then 1
  else 0

/-- Omitting the Boolean term family admits a non-Boolean value. -/
theorem boolean_necessary :
    AcceptsWithout .boolean booleanWitness ∧ ¬ FullAccepts booleanWitness := by
  decide

/-- Omitting the product term family admits `a*b ≠ c`. -/
theorem product_necessary :
    AcceptsWithout .product productWitness ∧ ¬ FullAccepts productWitness := by
  decide

/-- Omitting the seventh-power term family admits a nonzero S-box input with
zero output contribution. -/
theorem sbox_necessary :
    AcceptsWithout .sbox sboxWitness ∧ ¬ FullAccepts sboxWitness := by
  decide

/-- Omitting the centered-unit term family admits the value two. -/
theorem centered_necessary :
    AcceptsWithout .centered centeredWitness ∧
      ¬ FullAccepts centeredWitness := by
  decide

/-- Omitting the five-pair evaluation family admits a nonzero first pair. -/
theorem evaluation_necessary :
    AcceptsWithout .evaluation evaluationWitness ∧
      ¬ FullAccepts evaluationWitness := by
  decide

/-- Omitting the canonical transition family admits two active bound classes. -/
theorem canonical_necessary :
    AcceptsWithout .canonical canonicalWitness ∧
      ¬ FullAccepts canonicalWitness := by
  decide

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity
