import NightstreamFPrime.Lifecycle.Types
import NightstreamFPrime.Spec.Poseidon2

/-!
Owns the verifier-context digest carried in every Stage 1 state preimage.
The production verifier recomputes it from authoritative static words. The
sealed package identity is a separate object and is not an input here.

The fixed descriptor commits to the Nightstream Goldilocks profile and the
digest-only PiCCS schedule. Four domain-separated component digests commit to
the logical relation, application, NIFS key material, and commitment key.
-/

namespace NightstreamFPrime.Lifecycle.VerifierContext

open NightstreamFPrime.Spec

/-- A four-field digest with no malformed-length state. -/
structure Digest4 where
  c0 : F
  c1 : F
  c2 : F
  c3 : F
deriving DecidableEq

def Digest4.toList (digest : Digest4) : List F :=
  [digest.c0, digest.c1, digest.c2, digest.c3]

@[simp] theorem Digest4.toList_length (digest : Digest4) :
    digest.toList.length = 4 := by
  rfl

theorem Digest4.toList_injective : Function.Injective Digest4.toList := by
  intro left right equal
  cases left
  cases right
  simp [Digest4.toList] at equal
  simp_all

def Digest4.ofList (words : List F) : Digest4 where
  c0 := words.getD 0 0
  c1 := words.getD 1 0
  c2 := words.getD 2 0
  c3 := words.getD 3 0

/-- Raw static words owned by the verifier or fixed setup process. Digests
compress these words; they do not replace their authority. -/
structure Authority where
  relationWords : List F
  applicationWords : List F
  nifsKeyWords : List F
  commitmentKeyWords : List F

/-- Each component uses a distinct domain before hashing its authoritative
word list. -/
def componentDomain (component : Nat) : List F :=
  [Poseidon2.ofNat 78, Poseidon2.ofNat 105, Poseidon2.ofNat 103,
    Poseidon2.ofNat 104, Poseidon2.ofNat 116, Poseidon2.ofNat 115,
    Poseidon2.ofNat 116, Poseidon2.ofNat 114, Poseidon2.ofNat 101,
    Poseidon2.ofNat 97, Poseidon2.ofNat 109, Poseidon2.ofNat 47,
    Poseidon2.ofNat 70, Poseidon2.ofNat 80, Poseidon2.ofNat 114,
    Poseidon2.ofNat 105, Poseidon2.ofNat 109, Poseidon2.ofNat 101,
    Poseidon2.ofNat 47, Poseidon2.ofNat 99, Poseidon2.ofNat 111,
    Poseidon2.ofNat 110, Poseidon2.ofNat 116, Poseidon2.ofNat 101,
    Poseidon2.ofNat 120, Poseidon2.ofNat 116, Poseidon2.ofNat 47,
    Poseidon2.ofNat 118, Poseidon2.ofNat 49, Poseidon2.ofNat 95,
    Poseidon2.ofNat 49, Poseidon2.ofNat component]

def framed (words : List F) : List F := Poseidon2.ofNat words.length :: words

def componentDigest (component : Nat) (words : List F) : Digest4 :=
  Digest4.ofList (Poseidon2.hash (componentDomain component ++ framed words))

/-- Fixed-length component identities used by the outer context hash. -/
structure Descriptor where
  relation : Digest4
  application : Digest4
  nifsKey : Digest4
  commitmentKey : Digest4
deriving DecidableEq

def descriptor (authority : Authority) : Descriptor where
  relation := componentDigest 1 authority.relationWords
  application := componentDigest 2 authority.applicationWords
  nifsKey := componentDigest 3 authority.nifsKeyWords
  commitmentKey := componentDigest 4 authority.commitmentKeyWords

/-- Fixed Nightstream Goldilocks profile, including the split modulus limbs,
`b = 2`, `k_rho = 16`, `B = 2^16`, and every Stage 1 PiCCS dimension. -/
def profileWords : List F :=
  ([4294967295, 1, 2, 16, 65536, 1, 16, 17, 16, 14, 25, 9, 54, 18] :
    List Nat).map Poseidon2.ofNat

/-- Fixed digest-only PiCCS schedule descriptor: state digest, fresh source,
25 causal rounds, complete output, then fail-closed PiRLC sampling. -/
def scheduleWords : List F :=
  ([1, 1, 1, 25, 10, 17, 14, 54, 16, 64] : List Nat).map Poseidon2.ofNat

def contextDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 118, 101, 114, 105, 102, 105, 101,
    114, 45, 99, 111, 110, 116, 101, 120, 116, 47, 118, 49, 95, 49] :
    List Nat).map Poseidon2.ofNat

def Descriptor.serialize (value : Descriptor) : List F :=
  contextDomain ++ framed profileWords ++ framed scheduleWords ++
    framed value.relation.toList ++ framed value.application.toList ++
    framed value.nifsKey.toList ++ framed value.commitmentKey.toList

def Descriptor.digest4 (value : Descriptor) : Digest4 :=
  Digest4.ofList (Poseidon2.hash value.serialize)

/-- Public four-word context digest used as `KeyDigest` in the lifecycle. -/
def digest (authority : Authority) : KeyDigest :=
  (descriptor authority).digest4.toList

@[simp] theorem digest_length (authority : Authority) :
    (digest authority).length = 4 := by
  rfl

/-- Recomputing the context from authoritative words is definitionally the
component-domain tree followed by the fixed outer descriptor. -/
theorem digest_recomputed (authority : Authority) :
    digest authority =
      (descriptor authority).digest4.toList := by
  rfl

/-- The outer descriptor has one fixed length for every authority. This
prevents one valid descriptor from being a nonempty trailing extension of
another valid descriptor. -/
theorem Descriptor.serialize_length (value : Descriptor) :
    value.serialize.length = 86 := by
  simp [Descriptor.serialize, contextDomain, framed, profileWords,
    scheduleWords, Digest4.toList]

/-- Distinct fixed verifier-context descriptors have distinct canonical
encodings. This theorem is deterministic; hash collisions are handled only
by the separate security boundary. -/
theorem Descriptor.serialize_injective :
    Function.Injective Descriptor.serialize := by
  intro left right equal
  have components :
      framed left.relation.toList ++ framed left.application.toList ++
          framed left.nifsKey.toList ++ framed left.commitmentKey.toList =
        framed right.relation.toList ++ framed right.application.toList ++
          framed right.nifsKey.toList ++ framed right.commitmentKey.toList := by
    apply List.append_cancel_left
      (as := contextDomain ++ framed profileWords ++ framed scheduleWords)
    simpa [Descriptor.serialize, List.append_assoc] using equal
  rcases left with
    ⟨⟨lr0, lr1, lr2, lr3⟩, ⟨la0, la1, la2, la3⟩,
      ⟨ln0, ln1, ln2, ln3⟩, ⟨lc0, lc1, lc2, lc3⟩⟩
  rcases right with
    ⟨⟨rr0, rr1, rr2, rr3⟩, ⟨ra0, ra1, ra2, ra3⟩,
      ⟨rn0, rn1, rn2, rn3⟩, ⟨rc0, rc1, rc2, rc3⟩⟩
  simp [framed, Digest4.toList] at components ⊢
  simp_all

theorem Descriptor.no_trailing_extension (left right : Descriptor)
    (suffix : List F) (nonempty : suffix ≠ []) :
    right.serialize ≠ left.serialize ++ suffix := by
  intro equal
  have lengths := congrArg List.length equal
  rw [right.serialize_length, List.length_append, left.serialize_length]
    at lengths
  have suffixZero : suffix.length = 0 := by omega
  exact nonempty (List.eq_nil_of_length_eq_zero suffixZero)

end NightstreamFPrime.Lifecycle.VerifierContext
