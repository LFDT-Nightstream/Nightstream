import Mathlib.Data.List.OfFn
import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.Protocol.Nebula.CompactChain
import Nightstream.Protocol.Nebula.Digest
import Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-!
Contract: exact canonical field frames for the V2 compact commitment chain.

Assurance tier: implementation model and cryptographic boundary.

Owns six numeric Poseidon2 domain tags, one frame version, the exact selected
profile fields, exact header/leaf/link field order, bounded link indexes,
canonical-field proofs, and lossless frame encoding.

Does not own Poseidon2 permutation rows, token computation rows, chain
iteration, collision resistance, generated absolute columns, or Rust
conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.CompactChainHashFrame

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.Lifecycle

def frameVersion : Nat := 1

def headerTag : Role → Nat
  | .operations => 0x4e484f32
  | .memory => 0x4e484d32

def leafTag : Role → Nat
  | .operations => 0x4e4c4f32
  | .memory => 0x4e4c4d32

def linkTag : Role → Nat
  | .operations => 0x4e4b4f32
  | .memory => 0x4e4b4d32

/-- These two tags select the verifier-key-owned Ajtai setup roles. They are
not extra message coefficients in either Ajtai map. -/
def tokenRoleTag : Role → Nat
  | .operations => 0x4e544f32
  | .memory => 0x4e544d32

theorem all_domain_tags_pairwise_distinct :
    List.Nodup
      [ headerTag .operations, headerTag .memory
      , leafTag .operations, leafTag .memory
      , linkTag .operations, linkTag .memory
      , tokenRoleTag .operations, tokenRoleTag .memory
      ] := by
  decide

def profileNameValue : Profile.Name -> Nat
  | .paddedRowIdentityMemoryV2 => 2
  | .paddedRowIdentityMemoryFieldNative => 3

def commitmentEncodingValue : Profile.CommitmentEncoding -> Nat
  | .shiftedTernary41V1 => 1

/-- Name, version, checked-step factor, and commitment encoding. -/
def profileFields (profile : Profile.Identity) : List Nat :=
  [profileNameValue profile.name, profile.version,
    profile.checkedStepsPerFreshClaim,
    commitmentEncodingValue profile.commitmentEncoding]

theorem profileFields_length (profile : Profile.Identity) :
    (profileFields profile).length = 4 := rfl

theorem profileFields_injective : Function.Injective profileFields := by
  intro left right equal
  cases left with
  | mk leftName leftVersion leftSteps leftEncoding =>
      cases right with
      | mk rightName rightVersion rightSteps rightEncoding =>
          cases leftName <;> cases rightName <;>
            cases leftEncoding <;> cases rightEncoding <;>
            simp [profileFields, profileNameValue, commitmentEncodingValue]
              at equal ⊢ <;> simp_all

theorem profileFields_canonical
    {profile : Profile.Identity}
    (supported : ProductionProfileCandidates.SupportedIdentity profile) :
    forall value, value ∈ profileFields profile ->
      value < Nightstream.Implementation.R1CS.goldilocksP := by
  rcases supported with rfl | ⟨candidate, rfl⟩
  · intro value member
    simp [profileFields, profileNameValue, commitmentEncodingValue,
      Profile.v2] at member
    rcases member with rfl | rfl | rfl | rfl <;> decide
  · cases candidate <;>
      intro value member <;>
      simp [profileFields, profileNameValue, commitmentEncodingValue,
        ProductionProfileCandidates.identity,
        ProductionProfileCandidates.version,
        ProductionProfileCandidates.checkedStepsPerFreshClaim] at member <;>
      rcases member with rfl | rfl | rfl | rfl <;> decide

/-- The profile is explicit in every header and leaf frame. The caller cannot
change it after the verifier-key manifest selects the frame. -/
inductive Input where
  | header (role : Role) (profile : Profile.Identity) (plan : Digest.Value)
  | leaf (role : Role) (profile : Profile.Identity) (plan : Digest.Value)
      (token : Token)
  | link
      (role : Role) (index : Fin claimsPerSegment)
      (prior leaf : Digest.Value)

def Input.Supported : Input -> Prop
  | .header _ profile _ | .leaf _ profile _ _ =>
      ProductionProfileCandidates.SupportedIdentity profile
  | .link _ _ _ _ => True

def digestFields (digest : Digest.Value) : List Nat :=
  List.ofFn fun lane => (digest.lanes lane).val

def tokenFields (token : Token) : List Nat :=
  List.ofFn fun coordinate => (token coordinate).val

theorem digestFields_length (digest : Digest.Value) :
    (digestFields digest).length = 4 := by
  simp [digestFields, Digest.laneCount]

theorem tokenFields_length (token : Token) :
    (tokenFields token).length = 54 := by
  simp [tokenFields, tokenFieldCount, shortRank, ringDegree]

theorem digestFields_injective : Function.Injective digestFields := by
  intro left right equal
  apply Digest.Value.ext
  funext lane
  apply Subtype.ext
  exact congrFun (List.ofFn_injective equal) lane

theorem tokenFields_injective : Function.Injective tokenFields := by
  intro left right equal
  funext coordinate
  apply Subtype.ext
  exact congrFun (List.ofFn_injective equal) coordinate

def encode : Input → List Nat
  | .header role profile plan =>
      [headerTag role, frameVersion] ++
        (profileFields profile ++ (digestFields plan ++ [claimsPerSegment]))
  | .leaf role profile plan token =>
      [leafTag role, frameVersion] ++
        (profileFields profile ++ (digestFields plan ++ tokenFields token))
  | .link role index prior leaf =>
      [linkTag role, frameVersion, index.val] ++
        (digestFields prior ++ digestFields leaf)

theorem header_length (role : Role) (profile : Profile.Identity)
    (plan : Digest.Value) :
    (encode (.header role profile plan)).length = 11 := by
  simp [encode, profileFields_length, digestFields_length]

theorem leaf_length (role : Role) (profile : Profile.Identity)
    (plan : Digest.Value) (token : Token) :
    (encode (.leaf role profile plan token)).length = 64 := by
  simp [encode, profileFields_length, digestFields_length, tokenFields_length]

theorem link_length (role : Role) (index : Fin claimsPerSegment)
    (prior leaf : Digest.Value) :
    (encode (.link role index prior leaf)).length = 11 := by
  simp [encode, digestFields_length]

private theorem headerTag_injective : Function.Injective headerTag := by
  intro left right equal
  cases left <;> cases right <;> simp_all [headerTag]

private theorem leafTag_injective : Function.Injective leafTag := by
  intro left right equal
  cases left <;> cases right <;> simp_all [leafTag]

private theorem linkTag_injective : Function.Injective linkTag := by
  intro left right equal
  cases left <;> cases right <;> simp_all [linkTag]

private theorem headerTag_ne_leafTag (left right : Role) :
    headerTag left ≠ leafTag right := by
  cases left <;> cases right <;> decide

private theorem headerTag_ne_linkTag (left right : Role) :
    headerTag left ≠ linkTag right := by
  cases left <;> cases right <;> decide

private theorem leafTag_ne_linkTag (left right : Role) :
    leafTag left ≠ linkTag right := by
  cases left <;> cases right <;> decide

/-- Different typed inputs have different complete field frames. This is a
deterministic encoding theorem, not a Poseidon2 assumption. -/
theorem encode_injective : Function.Injective encode := by
  intro left right equal
  cases left with
  | header leftRole leftProfile leftPlan =>
      cases right with
      | header rightRole rightProfile rightPlan =>
          have tagEqual : headerTag leftRole = headerTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          have roleEqual := headerTag_injective tagEqual
          cases roleEqual
          have profileEqual :
              profileFields leftProfile = profileFields rightProfile := by
            have middle :=
              congrArg (fun values : List Nat => (values.drop 2).take 4) equal
            simpa [encode, profileFields_length] using middle
          cases profileFields_injective profileEqual
          have planFieldsEqual : digestFields leftPlan = digestFields rightPlan := by
            have middle :=
              congrArg (fun values : List Nat => (values.drop 6).take 4) equal
            simpa [encode, profileFields_length] using middle
          cases digestFields_injective planFieldsEqual
          rfl
      | leaf rightRole rightProfile rightPlan rightToken =>
          have tagEqual : headerTag leftRole = leafTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          exact False.elim (headerTag_ne_leafTag leftRole rightRole tagEqual)
      | link rightRole rightIndex rightPrior rightLeaf =>
          have tagEqual : headerTag leftRole = linkTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          exact False.elim (headerTag_ne_linkTag leftRole rightRole tagEqual)
  | leaf leftRole leftProfile leftPlan leftToken =>
      cases right with
      | header rightRole rightProfile rightPlan =>
          have tagEqual : leafTag leftRole = headerTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          exact False.elim (headerTag_ne_leafTag rightRole leftRole tagEqual.symm)
      | leaf rightRole rightProfile rightPlan rightToken =>
          have tagEqual : leafTag leftRole = leafTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          have roleEqual := leafTag_injective tagEqual
          cases roleEqual
          have profileEqual :
              profileFields leftProfile = profileFields rightProfile := by
            have middle :=
              congrArg (fun values : List Nat => (values.drop 2).take 4) equal
            simpa [encode, profileFields_length] using middle
          cases profileFields_injective profileEqual
          have planFieldsEqual : digestFields leftPlan = digestFields rightPlan := by
            have middle :=
              congrArg (fun values : List Nat => (values.drop 6).take 4) equal
            simpa [encode, profileFields_length] using middle
          have tokenFieldsEqual : tokenFields leftToken = tokenFields rightToken := by
            have tails := congrArg (List.drop 10) equal
            simpa [encode, profileFields_length] using tails
          cases digestFields_injective planFieldsEqual
          cases tokenFields_injective tokenFieldsEqual
          rfl
      | link rightRole rightIndex rightPrior rightLeaf =>
          have tagEqual : leafTag leftRole = linkTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          exact False.elim (leafTag_ne_linkTag leftRole rightRole tagEqual)
  | link leftRole leftIndex leftPrior leftLeaf =>
      cases right with
      | header rightRole rightProfile rightPlan =>
          have tagEqual : linkTag leftRole = headerTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          exact False.elim (headerTag_ne_linkTag rightRole leftRole tagEqual.symm)
      | leaf rightRole rightProfile rightPlan rightToken =>
          have tagEqual : linkTag leftRole = leafTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          exact False.elim (leafTag_ne_linkTag rightRole leftRole tagEqual.symm)
      | link rightRole rightIndex rightPrior rightLeaf =>
          have tagEqual : linkTag leftRole = linkTag rightRole := by
            have atHead := congrArg List.head? equal
            simpa [encode] using atHead
          have roleEqual := linkTag_injective tagEqual
          cases roleEqual
          have indexEqual : leftIndex = rightIndex := by
            apply Fin.ext
            have third :=
              congrArg (fun values : List Nat => values.getD 2 0) equal
            simpa [encode] using third
          have priorFieldsEqual :
              digestFields leftPrior = digestFields rightPrior := by
            have middle :=
              congrArg (fun values : List Nat => (values.drop 3).take 4) equal
            simpa [encode] using middle
          have leafFieldsEqual :
              digestFields leftLeaf = digestFields rightLeaf := by
            have tails := congrArg (List.drop 7) equal
            simpa [encode] using tails
          cases indexEqual
          cases digestFields_injective priorFieldsEqual
          cases digestFields_injective leafFieldsEqual
          rfl

theorem encode_fields_canonical (input : Input) (supported : input.Supported) :
    ∀ value ∈ encode input,
      value < Nightstream.Implementation.R1CS.goldilocksP := by
  intro value member
  cases input with
  | header role profile plan =>
      simp only [encode, List.mem_append] at member
      rcases member with tagOrVersion | rest
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at tagOrVersion
        rcases tagOrVersion with rfl | rfl
        · cases role <;> decide
        · decide
      · rcases rest with profile | rest
        · exact profileFields_canonical supported value profile
        · rcases rest with digest | count
          · rcases List.mem_ofFn.mp digest with ⟨lane, equal⟩
            rw [← equal]
            exact (plan.lanes lane).property
          · rw [List.mem_singleton] at count
            subst value
            decide
  | leaf role profile plan token =>
      simp only [encode, List.mem_append] at member
      rcases member with tagOrVersion | rest
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at tagOrVersion
        rcases tagOrVersion with rfl | rfl
        · cases role <;> decide
        · decide
      · rcases rest with profile | rest
        · exact profileFields_canonical supported value profile
        · rcases rest with digest | coordinate
          · rcases List.mem_ofFn.mp digest with ⟨lane, equal⟩
            rw [← equal]
            exact (plan.lanes lane).property
          · rcases List.mem_ofFn.mp coordinate with ⟨index, equal⟩
            rw [← equal]
            exact (token index).property
  | link role index prior leaf =>
      simp only [encode, List.mem_append] at member
      rcases member with prefixMember | rest
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at prefixMember
        rcases prefixMember with rfl | rfl | rfl
        · cases role <;> decide
        · decide
        · exact index.isLt.trans_le (by decide)
      · rcases rest with priorField | leafField
        · rcases List.mem_ofFn.mp priorField with ⟨lane, equal⟩
          rw [← equal]
          exact (prior.lanes lane).property
        · rcases List.mem_ofFn.mp leafField with ⟨lane, equal⟩
          rw [← equal]
          exact (leaf.lanes lane).property

end Nightstream.Implementation.Nebula.CompactChainHashFrame
