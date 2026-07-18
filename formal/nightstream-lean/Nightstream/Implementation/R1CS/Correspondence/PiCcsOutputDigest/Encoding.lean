import Nightstream.Implementation.R1CS.Core.SevenBytePacking
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Shared field encoding for the `Pi_CCS` output message bound before `Pi_RLC`.

Assurance tier: model-level representation semantics.

Owns: the two protocol domain tags; canonical conversion of packed tag words
to Goldilocks; `K` limb order; a fixed-width vector codec; and a generic
fixed-block family codec with exact lengths and injectivity.

Does not own: a source or matrix profile, Split-NC output authority, SIS,
Poseidon2, transcript placement, Rust/R1CS columns, costs, necessity, or row
removal.

Emits constraints: no.

Authority boundary: these functions are lossless pre-hash encodings. A digest
has authority only after a separate theorem binds its input to an accepted
typed message and proves the selected compression/hash execution.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.domain.outer` | exact outer v2 domain bytes and seven-byte packing | verifier-owned constant | `outputsDomainFields` |
| `nifs.pi_ccs.output_digest.domain.message` | exact per-source v2 domain bytes and seven-byte packing | verifier-owned constant | `outputMessageDomainFields` |
| `nifs.pi_ccs.output_digest.vector.limbs` | encode each `K` value as `(c0,c1)` | computed | `encodeK`, `encodeK_injective` |
| `nifs.pi_ccs.output_digest.vector.width` | bind the vector width before its ordered limbs | computed | `encodeKVector` |
| `nifs.pi_ccs.output_digest.vector.family` | concatenate equal-width vectors without ambiguity | computed | `encodeKVectorFamily_injective` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Encoding

open Nightstream.SuperNeo.Concrete

/-- Exact outer domain bytes, independent of Rust string storage. -/
def outputsDomainBytes : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97,
   110, 47, 112, 105, 95, 99, 99, 115, 95, 111, 117, 116, 112,
   117, 116, 115, 95, 100, 105, 103, 101, 115, 116, 47, 118, 50]

/-- Exact per-source domain bytes, independent of Rust string storage. -/
def outputMessageDomainBytes : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97,
   110, 47, 112, 105, 95, 99, 99, 115, 95, 111, 117, 116, 112,
   117, 116, 95, 109, 101, 115, 115, 97, 103, 101, 95, 100, 105,
   103, 101, 115, 116, 47, 118, 50]

/-- Canonical reduction into the production Goldilocks base field. -/
def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def packBytesAsFields (bytes : List Nat) : List F :=
  (Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats bytes).map
    fieldOfNat

/-- Closed checks of both independently written byte strings and packing. -/
theorem outputsDomainNats_eq :
    Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats
        outputsDomainBytes =
      [39, 30521782141150574, 31069335676202596,
       32478900775383087, 32780223149076319,
       32481117145948019, 846606196] := by
  decide

theorem outputMessageDomainNats_eq :
    Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats
        outputMessageDomainBytes =
      [46, 30521782141150574, 31069335676202596,
       32478900775383087, 32780223149076319,
       29099071086357855, 32481117145948005, 846606196] := by
  decide

def outputsDomainFields : List F :=
  packBytesAsFields outputsDomainBytes

def outputMessageDomainFields : List F :=
  packBytesAsFields outputMessageDomainBytes

@[simp] theorem outputsDomainFields_length :
    outputsDomainFields.length = 7 := by
  simp [outputsDomainFields, packBytesAsFields, outputsDomainNats_eq]

@[simp] theorem outputMessageDomainFields_length :
    outputMessageDomainFields.length = 8 := by
  simp [outputMessageDomainFields, packBytesAsFields,
    outputMessageDomainNats_eq]

private theorem getD_ofFn
    {Item : Type}
    {count : Nat}
    (items : Fin count -> Item)
    (index : Fin count)
    (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem ofFn_injective
    {Item : Type}
    {count : Nat} :
    Function.Injective (List.ofFn : (Fin count -> Item) -> List Item) := by
  intro left right same
  funext index
  have reads := congrArg
    (fun values => values.getD index.val (left index)) same
  simpa [getD_ofFn] using reads

namespace FixedBlocks

/-- Concatenate an ordered list of independently encoded blocks. -/
def encode {Item : Type} (encodeItem : Item -> List F) : List Item -> List F
  | [] => []
  | item :: items => encodeItem item ++ encode encodeItem items

@[simp] theorem length
    {Item : Type}
    (encodeItem : Item -> List F)
    (blockWidth : Nat)
    (blockLength : forall item, (encodeItem item).length = blockWidth)
    (items : List Item) :
    (encode encodeItem items).length = items.length * blockWidth := by
  induction items with
  | nil => simp [encode]
  | cons item items inductionHypothesis =>
      simp [encode, blockLength, inductionHypothesis, Nat.succ_mul,
        Nat.add_comm]

/-- Equal positive-width block streams have equal source lists whenever the
individual block encoder is injective. -/
theorem injective
    {Item : Type}
    (encodeItem : Item -> List F)
    (blockWidth : Nat)
    (blockWidthPositive : 0 < blockWidth)
    (blockLength : forall item, (encodeItem item).length = blockWidth)
    (blockInjective : Function.Injective encodeItem) :
    Function.Injective (encode encodeItem) := by
  intro left
  induction left with
  | nil =>
      intro right same
      cases right with
      | nil => rfl
      | cons item items =>
          have lengths := congrArg List.length same
          rw [length encodeItem blockWidth blockLength []] at lengths
          rw [length encodeItem blockWidth blockLength (item :: items)] at lengths
          simp only [List.length_nil, Nat.zero_mul, List.length_cons,
            Nat.succ_mul] at lengths
          omega
  | cons leftHead leftTail inductionHypothesis =>
      intro right same
      cases right with
      | nil =>
          have lengths := congrArg List.length same
          rw [length encodeItem blockWidth blockLength
            (leftHead :: leftTail)] at lengths
          rw [length encodeItem blockWidth blockLength []] at lengths
          simp only [List.length_cons, Nat.succ_mul, List.length_nil,
            Nat.zero_mul] at lengths
          omega
      | cons rightHead rightTail =>
          have heads := congrArg (List.take blockWidth) same
          have tails := congrArg (List.drop blockWidth) same
          simp only [encode] at heads tails
          have headFields : encodeItem leftHead = encodeItem rightHead := by
            simpa [blockLength] using heads
          have tailFields :
              encode encodeItem leftTail = encode encodeItem rightTail := by
            simpa [blockLength] using tails
          cases blockInjective headFields
          cases inductionHypothesis tailFields
          rfl

end FixedBlocks

/-- Encode a finite family in index order using a fixed-width item codec. -/
def encodeFamily
    {Item : Type}
    {count : Nat}
    (encodeItem : Item -> List F)
    (items : Fin count -> Item) : List F :=
  FixedBlocks.encode encodeItem (List.ofFn items)

@[simp] theorem encodeFamily_length
    {Item : Type}
    {count : Nat}
    (encodeItem : Item -> List F)
    (blockWidth : Nat)
    (blockLength : forall item, (encodeItem item).length = blockWidth)
    (items : Fin count -> Item) :
    (encodeFamily encodeItem items).length = count * blockWidth := by
  simp [encodeFamily, FixedBlocks.length encodeItem blockWidth blockLength]

theorem encodeFamily_injective
    {Item : Type}
    {count blockWidth : Nat}
    (encodeItem : Item -> List F)
    (blockWidthPositive : 0 < blockWidth)
    (blockLength : forall item, (encodeItem item).length = blockWidth)
    (blockInjective : Function.Injective encodeItem) :
    Function.Injective
      (encodeFamily encodeItem : (Fin count -> Item) -> List F) := by
  intro left right same
  apply ofFn_injective
  exact FixedBlocks.injective encodeItem blockWidth blockWidthPositive
    blockLength blockInjective same

/-- Quadratic-extension limb order used by every output vector. -/
def encodeK (value : K) : List F :=
  [value.c0, value.c1]

@[simp] theorem encodeK_length (value : K) :
    (encodeK value).length = 2 := by
  rfl

theorem encodeK_injective : Function.Injective encodeK := by
  intro left right same
  cases left with
  | mk leftC0 leftC1 =>
      cases right with
      | mk rightC0 rightC1 =>
          have fields : leftC0 = rightC0 /\ leftC1 = rightC1 := by
            simpa [encodeK] using same
          cases fields.1
          cases fields.2
          rfl

def kVectorFieldCount (width : Nat) : Nat :=
  1 + 2 * width

/-- Rust-compatible `K` vector: width followed by ordered `(c0,c1)` limbs. -/
def encodeKVector
    {width : Nat}
    (values : Fin width -> K) : List F :=
  fieldOfNat width :: encodeFamily encodeK values

@[simp] theorem encodeKVector_length
    {width : Nat}
    (values : Fin width -> K) :
    (encodeKVector values).length = kVectorFieldCount width := by
  simp [encodeKVector, kVectorFieldCount,
    encodeFamily_length encodeK 2 encodeK_length]
  omega

theorem encodeKVector_injective {width : Nat} :
    Function.Injective
      (encodeKVector : (Fin width -> K) -> List F) := by
  intro left right same
  apply encodeFamily_injective encodeK (by decide) encodeK_length
    encodeK_injective
  have tails := congrArg (List.drop 1) same
  simpa [encodeKVector] using tails

/-- Ordered family of equal-width `K` vectors. -/
def encodeKVectorFamily
    {count width : Nat}
    (values : Fin count -> Fin width -> K) : List F :=
  encodeFamily encodeKVector values

@[simp] theorem encodeKVectorFamily_length
    {count width : Nat}
    (values : Fin count -> Fin width -> K) :
    (encodeKVectorFamily values).length =
      count * kVectorFieldCount width := by
  simp [encodeKVectorFamily,
    encodeFamily_length encodeKVector (kVectorFieldCount width)
      encodeKVector_length]

theorem encodeKVectorFamily_injective {count width : Nat} :
    Function.Injective
      (encodeKVectorFamily :
        (Fin count -> Fin width -> K) -> List F) := by
  exact encodeFamily_injective encodeKVector (by
      simp [kVectorFieldCount]
      omega) encodeKVector_length
    encodeKVector_injective

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Encoding
