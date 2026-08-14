import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Proof-free schema for the production-shaped strict-`PiDEC` canonical-X
receipt.

Owns: the compact canonical-to-actual coordinate map for radix two and four,
exact physical sparse rows, and one unique owner label per exported row.

Does not own: compiler semantics, row satisfaction, protocol acceptance,
commitment authority, or permission to remove constraints.

Emits constraints: no.

| Record | Payload | Authority before correspondence checking |
|---|---|---|
| `CoordinateColumns` | parent, children, sign, product, optional radix-four limbs | untrusted generated data |
| `RowOwner` | indexed equation owner | untrusted generated data |
| `PhysicalRow` | relative/physical index plus exact A/B/C row | untrusted generated data |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX

open Nightstream.Implementation.R1CS

structure CoordinateColumns where
  parent : Nat
  children : List Nat
  sign : Nat
  product : Nat
  limbs : List (List Nat)
deriving DecidableEq, Repr

inductive RowOwner where
  | recomposition (activeIndex : Nat)
  | signProduct (activeIndex : Nat)
  | signZero (activeIndex : Nat)
  | childDigit (activeIndex child : Nat)
  | radixFourLimb (activeIndex child limb : Nat)
  | radixFourReconstruction (activeIndex child : Nat)
deriving DecidableEq, Repr

structure PhysicalRow where
  relativeIndex : Nat
  physicalIndex : Nat
  owner : RowOwner
  row : Row
deriving DecidableEq, Repr

/-- One compact cross-language execution case. The two accepted cases are
read from raw Rust `WitnessMat` coordinates; rejected cases are mutations of
the same typed boundary. `rustAccepted` is evidence to compare, never semantic
authority. -/
structure DifferentialCase where
  caseId : Nat
  profileTag : Nat
  recursiveSelector : Nat
  publicColumn : Nat
  parent : Nat
  children : List Nat
  childEvaluationArities : List Nat
  rustAccepted : Bool
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
