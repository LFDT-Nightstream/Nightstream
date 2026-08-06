import Mathlib.Logic.Equiv.Fin.Basic
import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityNIVCCompatibility

/-!
Contract: concrete Construction 2 hash-instance encoding for the selected
`PaddedRowIdentity` public carrier.

Owns: the exact `[1 | 256 digest bits | 0^13]` field layout; its connection to
the sole fresh NIFS public input; all coordinate equations; and injectivity of
the complete 270-coordinate encoding.

Does not own: the Construction 2 state hash, the application step, generated
rows, Rust conformance, or a hash-collision assumption.

Emits constraints: no.

Assurance tier: model-level. This is the concrete `encHash` value required by
HyperNova Construction 2. It uses the same four little-endian 64-bit
Goldilocks lanes as the native F-prime boundary.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConstruction2Encoding

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova

abbrev Digest := Nightstream.Implementation.Encoding.FPrime.Digest
abbrev PublicInput := PaddedRowIdentityConcreteAlgebra.PublicInput
abbrev PublicFresh := PaddedRowIdentityHyperNova.PublicFresh

/-- The exact field coordinate for one lane-major, little-endian digest bit. -/
def digestColumn (lane : Fin 4) (bit : Fin 64) :
    Fin relationShape.publicWidth :=
  ⟨1 + (finProdFinEquiv (lane, bit)).val, by
    have positionBound := (finProdFinEquiv (lane, bit)).isLt
    simp only [relationShape_publicWidth] at positionBound ⊢
    omega⟩

/-- Convert one Boolean digest bit to its canonical base-field value. -/
def bitField (value : Bool) : F :=
  if value then 1 else 0

theorem bitField_injective : Function.Injective bitField := by
  have one_ne_zero : (1 : F) ≠ 0 := by decide
  intro left right equal
  cases left <;> cases right <;> simp_all [bitField]

/-- HyperNova's selected instance encoder: affine one, all 256 digest bits in
lane-major little-endian order, then the thirteen verifier-fixed zeros. -/
def encHash (digest : Digest) : PublicInput :=
  fun column =>
    if affine : column.val = 0 then
      1
    else if encoded : column.val < 257 then
      let position : Fin 256 := ⟨column.val - 1, by omega⟩
      let coordinate :=
        (finProdFinEquiv (m := 4) (n := 64)).symm position
      bitField ((encodeEncInst digest coordinate.1).getLsbD coordinate.2.val)
    else
      0

@[simp] theorem encHash_affine (digest : Digest) :
    encHash digest ⟨0, by simp [relationShape_publicWidth]⟩ = 1 := by
  simp [encHash]

@[simp] theorem digestColumn_val (lane : Fin 4) (bit : Fin 64) :
    (digestColumn lane bit).val =
      1 + bit.val + 64 * lane.val := by
  change 1 + (bit.val + 64 * lane.val) =
    1 + bit.val + 64 * lane.val
  omega

@[simp] theorem encHash_digestColumn
    (digest : Digest) (lane : Fin 4) (bit : Fin 64) :
    encHash digest (digestColumn lane bit) =
      bitField ((encodeEncInst digest lane).getLsbD bit.val) := by
  have nonzero : (digestColumn lane bit).val ≠ 0 := by
    simp [digestColumn_val]
  have beforePadding : (digestColumn lane bit).val < 257 := by
    have laneBound := lane.isLt
    have bitBound := bit.isLt
    simp only [digestColumn_val]
    omega
  simp only [encHash, dif_neg nonzero, dif_pos beforePadding]
  change
    bitField
        ((encodeEncInst digest
          ((finProdFinEquiv (m := 4) (n := 64)).symm
            ⟨(digestColumn lane bit).val - 1, by omega⟩).1).getLsbD
          ((finProdFinEquiv (m := 4) (n := 64)).symm
            ⟨(digestColumn lane bit).val - 1, by omega⟩).2.val) =
      bitField ((encodeEncInst digest lane).getLsbD bit.val)
  have positionEqual :
      (⟨(digestColumn lane bit).val - 1, by omega⟩ : Fin 256) =
        (finProdFinEquiv (m := 4) (n := 64)) (lane, bit) := by
    apply Fin.ext
    change (digestColumn lane bit).val - 1 =
      bit.val + 64 * lane.val
    rw [digestColumn_val]
    omega
  rw [positionEqual,
    (finProdFinEquiv (m := 4) (n := 64)).symm_apply_apply]

/-- Every coordinate after the 257-coordinate logical prefix is the unique
fresh zero padding required by the 270-coordinate paper carrier. -/
theorem encHash_padding
    (digest : Digest)
    (column : Fin relationShape.publicWidth)
    (padding : 257 ≤ column.val) :
    encHash digest column = 0 := by
  have nonzero : column.val ≠ 0 := by omega
  have notEncoded : ¬ column.val < 257 := Nat.not_lt.mpr padding
  simp [encHash, nonzero, notEncoded]

/-- The selected encoding binds all four digest lanes. No two digests have
the same complete public input. -/
theorem encHash_injective : Function.Injective encHash := by
  intro left right equal
  apply encInst_bits_injective
  intro lane bit bitLt
  let bitIndex : Fin 64 := ⟨bit, bitLt⟩
  have coordinateEqual := congrFun equal (digestColumn lane bitIndex)
  rw [encHash_digestColumn, encHash_digestColumn] at coordinateEqual
  have bitEqual := bitField_injective coordinateEqual
  simpa [bitIndex] using bitEqual

/-- The value read by Construction 2 from its sole fresh claim. -/
def freshPublic (fresh : PublicFresh) : PublicInput :=
  fresh.publicInputs freshIndex

/-- The paper recursive public-link equation on the selected concrete
carriers. -/
def FreshLinked (fresh : PublicFresh) (digest : Digest) : Prop :=
  freshPublic fresh = encHash digest

theorem freshLinked_iff
    (fresh : PublicFresh) (digest : Digest) :
    FreshLinked fresh digest ↔
      fresh.publicInputs freshIndex = encHash digest := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConstruction2Encoding
