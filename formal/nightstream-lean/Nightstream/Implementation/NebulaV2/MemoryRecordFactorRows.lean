import Nightstream.Implementation.R1CS.Canonical.KLinear
import Nightstream.Implementation.R1CS.Canonical.KMulChain
import Nightstream.Implementation.R1CS.Canonical.KMulHonest

/-!
Contract: exact R1CS rows for one Nebula V2 fingerprint-factor update.

Assurance tier: implementation model.

Owns the concrete factor `gamma2 - (packed + gamma1 * value)`, the optional
canonical pad gate, one running-product multiplication, exact row counts, and
soundness against an independently defined coordinate-pair expression.

Does not own record-bit decoding, inactive-field zero checks, operation or scan
slot order, product-chain endpoints, absolute columns, or the generated V2
artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryRecordFactorRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KLinear
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KMulChain
open Nightstream.Implementation.R1CS.Canonical.KPairLaws
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- Snapshot factors are always active. Operation factors use one canonical
pad bit, where zero is active and one is inactive. -/
inductive Activation where
  | always
  | padded (padColumn : Nat)
deriving DecidableEq, Repr

/-- One factor/update block. `gateFrame` is unused for an always-active
factor, but retaining it gives one stable generated layout type. -/
structure Layout where
  gamma1 : Carried
  gamma2 : Carried
  previous : Carried
  packed : LinComb
  value : LinComb
  activation : Activation
  scaleFrame : Frame
  gateFrame : Frame
  updateFrame : Frame

/-- Embed one base-field linear combination in the low extension coordinate. -/
def embedCarried (value : LinComb) : Carried :=
  ⟨value, []⟩

/-- Nonlinear `gamma1 * value` result. -/
def scaledValue (layout : Layout) : Carried :=
  frameOutput layout.scaleFrame

/-- Exact V2 factor as a row-free carried expression after the scale frame. -/
def factorCarried (layout : Layout) : Carried :=
  subCarried layout.gamma2
    (addCarried (embedCarried layout.packed) (scaledValue layout))

def padCarried (padColumn : Nat) : Carried :=
  embedCarried [(padColumn, 1)]

def activeCarried (padColumn : Nat) : Carried :=
  oneMinus (padCarried padColumn)

/-- `factor - 1`, used by `1 + active * (factor - 1)`. -/
def gateDelta (layout : Layout) : Carried :=
  subCarried (factorCarried layout) oneCarried

/-- A pad contributes exactly one. An active operation contributes the record
factor. The simplification to these two cases is proved below from the pad
value; it is not assumed by the row program. -/
def gateCarried (layout : Layout) : Carried :=
  match layout.activation with
  | .always => factorCarried layout
  | .padded _ => addCarried oneCarried (frameOutput layout.gateFrame)

def outputCarried (layout : Layout) : Carried :=
  frameOutput layout.updateFrame

def scaleRows (layout : Layout) : List Row :=
  KMul.rows layout.gamma1 (embedCarried layout.value) layout.scaleFrame

def gateRows (layout : Layout) : List Row :=
  match layout.activation with
  | .always => []
  | .padded padColumn =>
      [bitRow padColumn] ++
        KMul.rows (activeCarried padColumn) (gateDelta layout)
          layout.gateFrame

def updateRows (layout : Layout) : List Row :=
  KMul.rows layout.previous (gateCarried layout) layout.updateFrame

def rows (layout : Layout) : List Row :=
  scaleRows layout ++ gateRows layout ++ updateRows layout

theorem rows_length (layout : Layout) :
    (rows layout).length =
      match layout.activation with
      | .always => 6
      | .padded _ => 10 := by
  cases activation : layout.activation <;>
    simp [rows, scaleRows, gateRows, updateRows, activation,
      KMul.rows_length]

theorem carriedValue_embed (assignment : Nat → Nat) (value : LinComb) :
    carriedValue assignment (embedCarried value) =
      ⟨lcEval assignment value, 0⟩ := rfl

private theorem scale_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment) :
    Satisfies (scaleRows layout) assignment := by
  intro row member
  exact satisfies row (by simp [rows, member])

private theorem update_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment) :
    Satisfies (updateRows layout) assignment := by
  intro row member
  exact satisfies row (by simp [rows, member])

private theorem padded_gate_rows_hold
    {layout : Layout} {assignment : Nat → Nat} {padColumn : Nat}
    (activation : layout.activation = .padded padColumn)
    (satisfies : Satisfies (rows layout) assignment) :
    Satisfies
      (KMul.rows (activeCarried padColumn) (gateDelta layout)
        layout.gateFrame) assignment := by
  intro row member
  apply satisfies row
  simp [rows, gateRows, activation, member]

/-- Independent coordinate-pair meaning of the V2 record factor. -/
def semanticFactor (assignment : Nat → Nat) (layout : Layout) : Pair :=
  subPair (carriedValue assignment layout.gamma2)
    (addPair ⟨lcEval assignment layout.packed, 0⟩
      (mulPair (carriedValue assignment layout.gamma1)
        ⟨lcEval assignment layout.value, 0⟩))

theorem factor_sound
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment) :
    carriedValue assignment (factorCarried layout) =
      semanticFactor assignment layout := by
  have scaled := frameOutput_sound assignment layout.gamma1
    (embedCarried layout.value) layout.scaleFrame
    (by simpa [scaleRows] using scale_rows_hold satisfies)
  rw [factorCarried, carriedValue_sub, carriedValue_add,
    carriedValue_embed, scaledValue, scaled, carriedValue_embed]
  rfl

/-- Independent meaning of the always-active or padded gate. -/
def semanticGate (assignment : Nat → Nat) (layout : Layout) : Pair :=
  match layout.activation with
  | .always => semanticFactor assignment layout
  | .padded padColumn =>
      addPair ⟨1, 0⟩
        (mulPair
          (subPair ⟨1, 0⟩
            ⟨lcEval assignment [(padColumn, 1)], 0⟩)
          (subPair (semanticFactor assignment layout) ⟨1, 0⟩))

theorem gate_sound
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    carriedValue assignment (gateCarried layout) =
      semanticGate assignment layout := by
  cases activation : layout.activation with
  | always =>
      simp only [gateCarried, semanticGate, activation]
      exact factor_sound satisfies
  | padded padColumn =>
      have gated := frameOutput_sound assignment (activeCarried padColumn)
        (gateDelta layout) layout.gateFrame
        (by simpa using padded_gate_rows_hold activation satisfies)
      simp only [gateCarried, semanticGate, activation]
      rw [carriedValue_add, carriedValue_one assignment one, gated,
        activeCarried, carriedValue_oneMinus assignment _ one,
        padCarried, carriedValue_embed, gateDelta, carriedValue_sub,
        factor_sound satisfies, carriedValue_one assignment one]

theorem semanticFactor_canonical (assignment : Nat → Nat) (layout : Layout) :
    (semanticFactor assignment layout).low < goldilocksP ∧
      (semanticFactor assignment layout).high < goldilocksP := by
  exact subPair_canonical _ _

/-- An active operation slot contributes its exact factor. -/
theorem semanticGate_active
    {layout : Layout} {assignment : Nat → Nat} {padColumn : Nat}
    (activation : layout.activation = .padded padColumn)
    (active : assignment padColumn = 0) :
    semanticGate assignment layout = semanticFactor assignment layout := by
  have padValue : lcEval assignment [(padColumn, 1)] = 0 := by
    simp [lcEval, active]
  have factorCanonical := semanticFactor_canonical assignment layout
  have deltaCanonical :=
    subPair_canonical (semanticFactor assignment layout) ⟨1, 0⟩
  simp only [semanticGate, activation]
  rw [padValue,
    subPair_zero_right ⟨1, 0⟩ (by decide) (by decide),
    mulPair_one_left _ deltaCanonical.1 deltaCanonical.2,
    addPair_comm,
    addPair_subPair _ ⟨1, 0⟩ factorCanonical.1 factorCanonical.2]

/-- An inactive operation slot contributes the multiplicative identity. -/
theorem semanticGate_padded
    {layout : Layout} {assignment : Nat → Nat} {padColumn : Nat}
    (activation : layout.activation = .padded padColumn)
    (padded : assignment padColumn = 1) :
    semanticGate assignment layout = ⟨1, 0⟩ := by
  have padValue : lcEval assignment [(padColumn, 1)] = 1 := by
    simp [lcEval, padded, goldilocksP]
  simp only [semanticGate, activation]
  rw [padValue]
  simp [subPair, mulPair, addPair, goldilocksP]

theorem gate_sound_active
    {layout : Layout} {assignment : Nat → Nat} {padColumn : Nat}
    (activation : layout.activation = .padded padColumn)
    (active : assignment padColumn = 0)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    carriedValue assignment (gateCarried layout) =
      semanticFactor assignment layout := by
  rw [gate_sound one satisfies,
    semanticGate_active activation active]

theorem gate_sound_padded
    {layout : Layout} {assignment : Nat → Nat} {padColumn : Nat}
    (activation : layout.activation = .padded padColumn)
    (padded : assignment padColumn = 1)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    carriedValue assignment (gateCarried layout) = ⟨1, 0⟩ := by
  rw [gate_sound one satisfies,
    semanticGate_padded activation padded]

/-- The same factor in SuperNeo's executable concrete extension carrier. -/
def concreteFactor
    (gamma1 gamma2 : Nightstream.SuperNeo.Concrete.K)
    (packed value : Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.K :=
  Nightstream.SuperNeo.Concrete.K.sub gamma2
    (Nightstream.SuperNeo.Concrete.K.add
      (Nightstream.SuperNeo.Concrete.K.embed packed)
      (Nightstream.SuperNeo.Concrete.K.mul gamma1
        (Nightstream.SuperNeo.Concrete.K.embed value)))

theorem semanticFactor_eq_concrete
    {layout : Layout} {assignment : Nat → Nat}
    {gamma1 gamma2 : Nightstream.SuperNeo.Concrete.K}
    {packed value : Nightstream.SuperNeo.Concrete.F}
    (gamma1Placed : carriedValue assignment layout.gamma1 =
      Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete
        gamma1)
    (gamma2Placed : carriedValue assignment layout.gamma2 =
      Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete
        gamma2)
    (packedPlaced : lcEval assignment layout.packed = packed.val)
    (valuePlaced : lcEval assignment layout.value = value.val) :
    semanticFactor assignment layout =
      Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete
        (concreteFactor gamma1 gamma2 packed value) := by
  rw [semanticFactor, gamma1Placed, gamma2Placed, packedPlaced, valuePlaced]
  unfold concreteFactor
  rw [Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_sub,
    Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_add,
    Nightstream.Implementation.R1CS.Canonical.KConcreteBridge.ofConcrete_mul]
  rfl

/-- The exact product update derived from the emitted rows. -/
theorem update_sound
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    carriedValue assignment (outputCarried layout) =
      mulPair (carriedValue assignment layout.previous)
        (semanticGate assignment layout) := by
  have updated := frameOutput_sound assignment layout.previous
    (gateCarried layout) layout.updateFrame
    (by simpa [updateRows] using update_rows_hold satisfies)
  rw [outputCarried, updated, gate_sound one satisfies]

/-- The pad wire is Boolean as a conclusion of the padded row block. -/
theorem pad_le_one_of_rows
    {layout : Layout} {assignment : Nat → Nat} {padColumn : Nat}
    (activation : layout.activation = .padded padColumn)
    (prime : EuclidPrime goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    assignment padColumn ≤ 1 := by
  apply bitRow_le_one prime (canonical padColumn) one
  apply satisfies
  simp [rows, gateRows, activation]

end Nightstream.Implementation.NebulaV2.MemoryRecordFactorRows
