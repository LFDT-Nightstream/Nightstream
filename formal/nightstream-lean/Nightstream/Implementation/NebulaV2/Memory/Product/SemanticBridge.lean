import Nightstream.Implementation.NebulaV2.Core.ConcreteField
import Nightstream.Implementation.NebulaV2.Memory.Product.ChainRows
import Nightstream.Protocol.NebulaV2.ProductState

/-!
Contract: typed-record meaning of the exact Nebula V2 product-chain rows.

Assurance tier: implementation-to-semantics bridge.

Owns the proof that each always-active, active, or padded row entry contributes
exactly the protocol record factor or the multiplicative identity, and that a
satisfying complete chain fixes its declared endpoint to the left-to-right
concrete SuperNeo extension-field product of the represented records.

Does not own the operation and snapshot row decoders that establish
`GateRepresents`, claim-column placement, honest frame allocation, or the
generated V2 artifact.

Emits constraints: no. It gives meaning to existing emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge

open Nightstream.Implementation.NebulaV2.MemoryProductChainRows
open Nightstream.Implementation.NebulaV2.MemoryRecordFactorRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KLinear
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

/-- Exact concrete factor used by both the row bridge and the field-level
fingerprint proof. The subtype proof prevents reduction aliases. -/
def recordFactorK (challenge : ChallengePair K)
    (record : BoundedTuple) : K :=
  concreteFactor challenge.gamma1 challenge.gamma2
    ⟨packedNat record.1, by
      simpa [Fingerprint.goldilocksModulus,
        Nightstream.SuperNeo.Concrete.goldilocksModulus] using
        packedNat_lt_goldilocks record.2⟩
    ⟨record.1.value, by
      simpa [Fingerprint.goldilocksModulus,
        Nightstream.SuperNeo.Concrete.goldilocksModulus] using
        value_lt_goldilocks record.2⟩

def optionalFactorK (challenge : ChallengePair K) :
    Option BoundedTuple → K
  | none => K.one
  | some record => recordFactorK challenge record

/-- Left-to-right product order used by the generated running-product rows. -/
def foldOptionsK (challenge : ChallengePair K) :
    K → List (Option BoundedTuple) → K
  | initial, [] => initial
  | initial, record :: rest =>
      foldOptionsK challenge
        (K.mul initial (optionalFactorK challenge record)) rest

/-- Independent source-value relation for one product entry. It contains only
activation and source-coordinate facts. It does not contain a gate value,
running product, or final product equality. -/
inductive GateRepresents
    (assignment : Nat → Nat) : Entry → Option BoundedTuple → Prop
  | padded
      {entry : Entry} {padColumn : Nat}
      (activation : entry.activation = .padded padColumn)
      (pad : assignment padColumn = 1) :
      GateRepresents assignment entry none
  | active
      {entry : Entry} {padColumn : Nat} {record : BoundedTuple}
      (activation : entry.activation = .padded padColumn)
      (pad : assignment padColumn = 0)
      (packed : lcEval assignment entry.packed = packedNat record.1)
      (value : lcEval assignment entry.value = record.1.value) :
      GateRepresents assignment entry (some record)
  | always
      {entry : Entry} {record : BoundedTuple}
      (activation : entry.activation = .always)
      (packed : lcEval assignment entry.packed = packedNat record.1)
      (value : lcEval assignment entry.value = record.1.value) :
      GateRepresents assignment entry (some record)

/-- One represented entry has exactly the concrete factor selected by its
typed optional record. -/
theorem gateValue_eq_optionalFactor
    {assignment : Nat → Nat} {entry : Entry}
    {record : Option BoundedTuple}
    {gamma1 gamma2 : Carried} {challenge : ChallengePair K}
    (gamma1Placed : carriedValue assignment gamma1 =
      ofConcrete challenge.gamma1)
    (gamma2Placed : carriedValue assignment gamma2 =
      ofConcrete challenge.gamma2)
    (represents : GateRepresents assignment entry record) :
    entry.gateValue assignment gamma1 gamma2 =
      ofConcrete (optionalFactorK challenge record) := by
  cases represents with
  | padded activation pad =>
      rw [Entry.gateValue]
      rw [semanticGate_padded activation pad]
      rfl
  | active activation pad packed value =>
      rw [Entry.gateValue]
      rw [semanticGate_active activation pad]
      apply semanticFactor_eq_concrete gamma1Placed gamma2Placed
      · simpa [recordFactorK] using packed
      · simpa [recordFactorK] using value
  | always activation packed value =>
      rw [Entry.gateValue]
      have factorGate :
          semanticGate assignment
              (entry.layout gamma1 gamma2 zeroCarried) =
            semanticFactor assignment
              (entry.layout gamma1 gamma2 zeroCarried) := by
        simp [semanticGate, Entry.layout, activation]
      rw [factorGate]
      apply semanticFactor_eq_concrete gamma1Placed gamma2Placed
      · simpa [recordFactorK] using packed
      · simpa [recordFactorK] using value

/-- The independent product recursion equals the concrete product of the
represented optional records. -/
theorem productValue_eq_foldOptionsK
    {assignment : Nat → Nat} {gamma1 gamma2 : Carried}
    {challenge : ChallengePair K}
    (gamma1Placed : carriedValue assignment gamma1 =
      ofConcrete challenge.gamma1)
    (gamma2Placed : carriedValue assignment gamma2 =
      ofConcrete challenge.gamma2) :
    ∀ {entries : List Entry} {records : List (Option BoundedTuple)}
      (initial : K),
      List.Forall₂ (GateRepresents assignment) entries records →
      productValue assignment gamma1 gamma2 (ofConcrete initial) entries =
        ofConcrete (foldOptionsK challenge initial records)
  | [], [], _, .nil => rfl
  | entry :: entries, record :: records, initial,
      .cons represents rest => by
      simp only [productValue, foldOptionsK]
      rw [gateValue_eq_optionalFactor gamma1Placed gamma2Placed represents]
      rw [← ofConcrete_mul]
      exact productValue_eq_foldOptionsK gamma1Placed gamma2Placed
        (K.mul initial (optionalFactorK challenge record)) rest

/-- Satisfying chain rows derive the exact concrete product endpoint. The only
external premises describe the source records and the challenge/initial-value
placements; the conclusion is not present in any premise. -/
theorem final_eq_foldOptionsK
    {layout : MemoryProductChainRows.Layout}
    {assignment : Nat → Nat} {challenge : ChallengePair K}
    {initial : K} {records : List (Option BoundedTuple)}
    (one : assignment 0 = 1)
    (holds : Satisfies (MemoryProductChainRows.rows layout) assignment)
    (gamma1Placed : carriedValue assignment layout.gamma1 =
      ofConcrete challenge.gamma1)
    (gamma2Placed : carriedValue assignment layout.gamma2 =
      ofConcrete challenge.gamma2)
    (initialPlaced : carriedValue assignment layout.initial =
      ofConcrete initial)
    (represents :
      List.Forall₂ (GateRepresents assignment) layout.entries records) :
    carriedValue assignment layout.final =
      ofConcrete (foldOptionsK challenge initial records) := by
  rw [MemoryProductChainRows.final_sound one holds, initialPlaced]
  exact productValue_eq_foldOptionsK gamma1Placed gamma2Placed initial
    represents

/-- The mathematical-field view of one concrete challenge pair. -/
def fieldChallenge (challenge : ChallengePair K) :
    ChallengePair ChallengeField :=
  { gamma1 := superNeoEquiv challenge.gamma1
    gamma2 := superNeoEquiv challenge.gamma2 }

/-- Remove physical holes while preserving the order and multiplicity of all
active bounded records. -/
def activeRecords (records : List (Option BoundedTuple)) :
    List BoundedTuple :=
  records.filterMap id

def activeRecordMultiset (records : List (Option BoundedTuple)) :
    Multiset MemTuple :=
  ((activeRecords records).map Subtype.val : List MemTuple)

/-- One row-layer concrete factor is the exact field-level protocol factor. -/
theorem superNeoEquiv_recordFactorK
    (challenge : ChallengePair K) (record : BoundedTuple) :
    superNeoEquiv (recordFactorK challenge record) =
      ProductState.recordFactor encode (fieldChallenge challenge)
        record.1 := by
  simp only [recordFactorK, concreteFactor, ProductState.recordFactor,
    fieldChallenge, superNeoEquiv_sub, superNeoEquiv_add,
    superNeoEquiv_mul, superNeoEquiv_embed]
  rw [mul_comm]

/-- The exact left-to-right row product maps to the commutative multiset
product used by the independent protocol model. -/
theorem superNeoEquiv_foldOptionsK
    (challenge : ChallengePair K) :
    ∀ (initial : K) (records : List (Option BoundedTuple)),
      superNeoEquiv (foldOptionsK challenge initial records) =
        superNeoEquiv initial *
          ProductState.recordsProduct encode (fieldChallenge challenge)
            (activeRecordMultiset records)
  | initial, [] => by
      simp [foldOptionsK, activeRecordMultiset, activeRecords,
        ProductState.recordsProduct]
  | initial, none :: rest => by
      simp only [foldOptionsK, optionalFactorK]
      rw [superNeoEquiv_foldOptionsK challenge]
      rw [superNeoEquiv_mul, superNeoEquiv_one]
      simp [activeRecordMultiset, activeRecords,
        ProductState.recordsProduct]
  | initial, some record :: rest => by
      simp only [foldOptionsK, optionalFactorK]
      rw [superNeoEquiv_foldOptionsK challenge]
      rw [superNeoEquiv_mul, superNeoEquiv_recordFactorK]
      simp only [activeRecordMultiset, activeRecords, List.filterMap_cons,
        List.map_cons, List.coe_toFinset]
      simp [ProductState.recordsProduct, mul_assoc]

end Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge
