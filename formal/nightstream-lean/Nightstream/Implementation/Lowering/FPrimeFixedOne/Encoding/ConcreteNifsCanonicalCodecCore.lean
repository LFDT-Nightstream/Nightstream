import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.Implementation.Lowering.Goldilocks.Codec

/-!
Contract: canonical field-coordinate codecs for the finite algebraic carriers
used by the selected ConcretePhi81 NIFS.

Owns: low/high order for `K`, coefficient order for `RingF` and `RingK`,
point-coordinate order, exact-size evaluation arrays, commitments, and public
inputs.

Does not own: application state or witness codecs, a relation structure,
transcript tags, prover acceptance, physical columns, Rust, or artifacts.

Every codec is derived from an explicit injective field encoding.  Classical
choice implements only the proof-side partial inverse; it does not select or
change any emitted coordinate.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Low limb followed by high limb. -/
def kPair (value : K) : Field × Field :=
  (value.c0, value.c1)

theorem kPair_injective : Function.Injective kPair := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

/-- Canonical two-coordinate quadratic-extension codec. -/
noncomputable def kCodec : Codec K :=
  Codec.pullback
    (Codec.product fieldCodec fieldCodec) kPair kPair_injective

@[simp] theorem kCodec_width :
    kCodec.width = 2 := by
  rfl

theorem kCodec_admissible (value : K) :
    kCodec.Admissible value := by
  exact ⟨True.intro, True.intro⟩

@[simp] theorem kCodec_encode (value : K) :
    kCodec.encode value = [value.c0, value.c1] := by
  rfl

/-- Ring coefficients in increasing degree order. -/
noncomputable def ringFCodec : Codec RingF :=
  Codec.finFunction ringDegree fieldCodec

/-- Each ring coefficient in increasing degree order, with each `K` value in
low/high limb order. -/
noncomputable def ringKCodec : Codec RingK :=
  Codec.finFunction ringDegree kCodec

@[simp] theorem ringFCodec_width :
    ringFCodec.width = ringDegree := by
  rfl

@[simp] theorem ringKCodec_width :
    ringKCodec.width = ringDegree * 2 := by
  rfl

theorem ringFCodec_admissible (value : RingF) :
    ringFCodec.Admissible value := by
  intro index
  trivial

theorem ringKCodec_admissible (value : RingK) :
    ringKCodec.Admissible value := by
  intro index
  exact kCodec_admissible (value index)

/-- A dimension-checked point is represented by its existing coordinate
list.  Its proof field contributes no coordinate. -/
def pointData
    {variables : Nat}
    (point : CubePoint K variables) : List K :=
  point.coordinates

theorem pointData_injective
    {variables : Nat} :
    Function.Injective
      (pointData (variables := variables)) := by
  intro left right coordinatesEqual
  cases left with
  | mk leftCoordinates leftDimension =>
      cases right with
      | mk rightCoordinates rightDimension =>
          cases coordinatesEqual
          rfl

noncomputable def pointCodec (variables : Nat) :
    Codec (CubePoint K variables) :=
  Codec.pullback
    (Codec.fixedList variables K.zero kCodec)
    pointData pointData_injective

@[simp] theorem pointCodec_width (variables : Nat) :
    (pointCodec variables).width = variables * 2 := by
  rfl

theorem pointCodec_admissible
    {variables : Nat}
    (point : CubePoint K variables) :
    (pointCodec variables).Admissible point := by
  constructor
  · exact point.dimension
  · intro index
    exact kCodec_admissible _

/-- Exact matrix-count array of complete ring evaluations. -/
noncomputable def evaluationsCodec (matrixCount : Nat) :
    Codec (Array RingK) :=
  Codec.fixedArray matrixCount ringKZero ringKCodec

@[simp] theorem evaluationsCodec_width (matrixCount : Nat) :
    (evaluationsCodec matrixCount).width =
      matrixCount * (ringDegree * 2) := by
  rfl

theorem evaluationsCodec_admissible
    {matrixCount : Nat}
    (values : Array RingK)
    (sizeExact : values.size = matrixCount) :
    (evaluationsCodec matrixCount).Admissible values := by
  constructor
  · exact sizeExact
  · intro index
    exact ringKCodec_admissible _

/-- One verifier commitment: row-major commitment rows, then ring
coefficients. -/
noncomputable def commitmentCodec (verifierRows : Nat) :
    Codec (Fin verifierRows → RingF) :=
  Codec.finFunction verifierRows ringFCodec

@[simp] theorem commitmentCodec_width (verifierRows : Nat) :
    (commitmentCodec verifierRows).width =
      verifierRows * ringDegree := by
  rfl

theorem commitmentCodec_admissible
    {verifierRows : Nat}
    (value : Fin verifierRows → RingF) :
    (commitmentCodec verifierRows).Admissible value := by
  intro row
  exact ringFCodec_admissible (value row)

/-- One aligned public input in increasing scalar-column order. -/
noncomputable def publicInputCodec (publicWidth : Nat) :
    Codec (Fin publicWidth → F) :=
  Codec.finFunction publicWidth fieldCodec

@[simp] theorem publicInputCodec_width (publicWidth : Nat) :
    (publicInputCodec publicWidth).width = publicWidth := by
  simp [publicInputCodec, Codec.finFunction,
    Codec.ofInjectiveEncoding, fieldCodec]

theorem publicInputCodec_admissible
    {publicWidth : Nat}
    (value : Fin publicWidth → F) :
    (publicInputCodec publicWidth).Admissible value := by
  intro column
  trivial

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
