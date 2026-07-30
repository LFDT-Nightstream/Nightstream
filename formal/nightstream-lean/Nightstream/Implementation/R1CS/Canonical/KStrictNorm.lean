import Nightstream.Implementation.R1CS.Canonical.KPointEquality

/-!
Contract: the strict-`b = 2` norm residual in the PiCCS terminal message.

Owns: the two extension multiplications computing
`(value + 1) * value * (value - 1)`, their exact allocation, and soundness to
the unchanged paper definition `ProtocolPolynomial.strictNormResidual`.

Addition and subtraction are row-free carried linear combinations.  The
program therefore emits exactly six rows and allocates exactly six auxiliary
columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KStrictNorm

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

structure Input where
  value : Carried
  frameBase : Nat

def firstFrame (input : Input) : Frame :=
  KFrames.frameAt input.frameBase 0

def secondFrame (input : Input) : Frame :=
  KFrames.frameAt input.frameBase 1

def firstOutput (input : Input) : Carried :=
  KMulChain.frameOutput (firstFrame input)

def output (input : Input) : Carried :=
  KMulChain.frameOutput (secondFrame input)

def rows (input : Input) : List Row :=
  KMul.rows (KLinear.addCarried input.value KLinear.oneCarried)
      input.value (firstFrame input) ++
    KMul.rows (firstOutput input)
      (KLinear.subCarried input.value KLinear.oneCarried) (secondFrame input)

def columns (input : Input) : List Nat :=
  KFrames.frameColumns input.frameBase 2

theorem rows_length (input : Input) : (rows input).length = 6 := by
  unfold rows
  rw [List.length_append, KMul.rows_length, KMul.rows_length]

theorem columns_length (input : Input) : (columns input).length = 6 :=
  KFrames.frameColumns_length _ _

theorem columns_nodup (input : Input) : (columns input).Nodup :=
  KFrames.frameColumns_nodup _ _

def decoded (assignment : Nat → Nat) (value : Carried) : ConcreteK :=
  KPointEquality.decoded assignment value

theorem ofConcrete_decoded (assignment : Nat → Nat) (value : Carried) :
    KConcreteBridge.ofConcrete (decoded assignment value) =
      carriedValue assignment value :=
  KPointEquality.ofConcrete_decoded assignment value

theorem rows_sound
    (input : Input) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (output input) =
      ProtocolPolynomial.strictNormResidual ConcreteCarrier.extensionOps
        (decoded assignment input.value) := by
  have firstSatisfied :
      Satisfies
        (KMul.rows (KLinear.addCarried input.value KLinear.oneCarried)
          input.value (firstFrame input)) assignment :=
    fun row member =>
      satisfied row (List.mem_append_left _ member)
  have secondSatisfied :
      Satisfies
        (KMul.rows (firstOutput input)
          (KLinear.subCarried input.value KLinear.oneCarried)
          (secondFrame input)) assignment :=
    fun row member =>
      satisfied row (List.mem_append_right _ member)
  have first :=
    KMulChain.frameOutput_sound assignment
      (KLinear.addCarried input.value KLinear.oneCarried)
      input.value (firstFrame input) firstSatisfied
  have second :=
    KMulChain.frameOutput_sound assignment (firstOutput input)
      (KLinear.subCarried input.value KLinear.oneCarried)
      (secondFrame input) secondSatisfied
  apply KConcreteBridge.ofConcrete_injective
  rw [ofConcrete_decoded]
  unfold output
  rw [second]
  unfold firstOutput
  rw [first, KLinear.carriedValue_add,
    KLinear.carriedValue_one assignment constantWire,
    KLinear.carriedValue_sub,
    KLinear.carriedValue_one assignment constantWire]
  unfold ProtocolPolynomial.strictNormResidual
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub,
    show ConcreteCarrier.extensionOps.one =
      Nightstream.SuperNeo.Concrete.K.one from rfl]
  change
    mulPair
        (mulPair
          (addPair (carriedValue assignment input.value) ⟨1, 0⟩)
          (carriedValue assignment input.value))
        (KPairLaws.subPair (carriedValue assignment input.value) ⟨1, 0⟩) =
      KConcreteBridge.ofConcrete
        (Nightstream.SuperNeo.Concrete.K.mul
          (Nightstream.SuperNeo.Concrete.K.mul
            (Nightstream.SuperNeo.Concrete.K.add
              (decoded assignment input.value)
              Nightstream.SuperNeo.Concrete.K.one)
            (decoded assignment input.value))
          (Nightstream.SuperNeo.Concrete.K.sub
            (decoded assignment input.value)
            Nightstream.SuperNeo.Concrete.K.one))
  rw [KConcreteBridge.ofConcrete_mul, KConcreteBridge.ofConcrete_mul,
    KConcreteBridge.ofConcrete_add, KConcreteBridge.ofConcrete_sub,
    ofConcrete_decoded]
  rfl

end Nightstream.Implementation.R1CS.Canonical.KStrictNorm
