import NightstreamFPrime.Export.BoundedDepth.StepPackage
import NightstreamFPrime.Export.StreamingIdentity

/-! The package identity binds the packages (`SECURITY_MODEL.md`, section Hashed encodings).

The identity hashes the preorder node words of a codec value (`Package.valuePreimage`,
`StreamingIdentity.canonicalWords`): four words per node, the tag and three limbs of its value,
where an array's value is its length. The node list is prefix-free (`nodes_append_inj`), and a
node's words fix the node when its value fits in 64 bits, the range of Rust `u64` (`Fits`). So
`valuePreimage` is injective on such values (`valuePreimage_injective`), and two pairs of
packages with one identity are equal or give two distinct hash inputs with one hash
(`identity_binds`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.IdentityBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.StreamingIdentity

/-- Every node value of the codec value, an atom or an array length, fits in 64 bits. -/
def Fits (value : Value) : Prop :=
  ∀ node ∈ nodes value, node.value < Package.limbBase * Package.limbBase

private theorem ofNat_injective {left right : Nat} (leftSmall : left < fieldModulus)
    (rightSmall : right < fieldModulus) (same : Poseidon2.ofNat left = Poseidon2.ofNat right) :
    left = right := by
  have values := congrArg Fin.val same
  simpa [Poseidon2.ofNat, Nat.mod_eq_of_lt leftSmall, Nat.mod_eq_of_lt rightSmall] using values

private theorem limb_small (value : Nat) : value % Package.limbBase < fieldModulus :=
  Nat.lt_of_lt_of_le (Nat.mod_lt _ (by decide)) (by decide)

theorem nodes_tag_le (value : Value) : ∀ node ∈ nodes value, node.tag ≤ 1 :=
  Value.rec
    (motive_1 := fun value => ∀ node ∈ nodes value, node.tag ≤ 1)
    (motive_2 := fun values => ∀ node ∈ values.flatMap nodes, node.tag ≤ 1)
    (fun _ node member => by
      simp only [nodes, List.mem_singleton] at member
      subst member
      exact Nat.zero_le 1)
    (fun _ children node member => by
      simp only [nodes, List.mem_cons] at member
      rcases member with head | rest
      · subst head
        exact Nat.le_refl 1
      · exact children node rest)
    (fun _ member => by simp at member)
    (fun _ _ head tail node member => by
      simp only [List.flatMap_cons, List.mem_append] at member
      rcases member with inHead | inTail
      · exact head node inHead
      · exact tail node (by simpa [List.flatMap] using inTail))
    value

/-- A node's words fix the node when its tag is at most 1 and its value fits in 64 bits. -/
theorem words_injective {node node' : Node} (tag : node.tag ≤ 1) (tag' : node'.tag ≤ 1)
    (fits : node.value < Package.limbBase * Package.limbBase)
    (fits' : node'.value < Package.limbBase * Package.limbBase)
    (same : node.words = node'.words) : node = node' := by
  obtain ⟨t, v⟩ := node
  obtain ⟨t', v'⟩ := node'
  simp only [Node.words, List.cons.injEq, and_true] at same tag tag' fits fits'
  obtain ⟨sameTag, low, middle, -⟩ := same
  have tags : t = t' :=
    ofNat_injective (by unfold fieldModulus; omega) (by unfold fieldModulus; omega) sameTag
  have lows := ofNat_injective (limb_small v) (limb_small v') low
  have middles := ofNat_injective (limb_small _) (limb_small _) middle
  have values : v = v' := by
    have split (w : Nat) (small : w < Package.limbBase * Package.limbBase) :
        w = w % Package.limbBase + Package.limbBase * ((w / Package.limbBase) % Package.limbBase) := by
      have high : w / Package.limbBase < Package.limbBase :=
        Nat.div_lt_of_lt_mul small
      rw [Nat.mod_eq_of_lt high]
      exact (Nat.mod_add_div w Package.limbBase).symm
    rw [split v fits, split v' fits', lows, middles]
  rw [tags, values]

/-- Node lists whose nodes have small tags and 64-bit values are fixed by their words. -/
theorem flatMap_words_injective : ∀ {nodes nodes' : List Node},
    (∀ node ∈ nodes, node.tag ≤ 1 ∧ node.value < Package.limbBase * Package.limbBase) →
    (∀ node ∈ nodes', node.tag ≤ 1 ∧ node.value < Package.limbBase * Package.limbBase) →
    nodes.flatMap Node.words = nodes'.flatMap Node.words → nodes = nodes'
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, _, same => by simp [Node.words] at same
  | _ :: _, [], _, _, same => by simp [Node.words] at same
  | node :: rest, node' :: rest', small, small', same => by
      simp only [List.flatMap_cons] at same
      obtain ⟨head, tail⟩ := List.append_inj same (by simp [Node.words])
      obtain ⟨tag, fits⟩ := small node List.mem_cons_self
      obtain ⟨tag', fits'⟩ := small' node' List.mem_cons_self
      rw [words_injective tag tag' fits fits' head,
        flatMap_words_injective (fun n member => small n (List.mem_cons_of_mem _ member))
          (fun n member => small' n (List.mem_cons_of_mem _ member)) tail]

/-- The preorder node list is prefix-free: a value's nodes followed by any tail fix the value
and the tail. -/
theorem nodes_append_inj (value : Value) :
    ∀ value' (rest rest' : List Node), nodes value ++ rest = nodes value' ++ rest' →
      value = value' ∧ rest = rest' :=
  Value.rec
    (motive_1 := fun value => ∀ value' (rest rest' : List Node),
      nodes value ++ rest = nodes value' ++ rest' → value = value' ∧ rest = rest')
    (motive_2 := fun values => ∀ (values' : List Value) (rest rest' : List Node),
      values.length = values'.length →
      values.flatMap nodes ++ rest = values'.flatMap nodes ++ rest' →
        values = values' ∧ rest = rest')
    (fun atom value' rest rest' same => by
      cases value' with
      | atom atom' =>
          simp only [nodes, List.singleton_append, List.cons.injEq, Node.mk.injEq,
            true_and] at same
          exact ⟨by rw [same.1], same.2⟩
      | array _ => simp [nodes] at same)
    (fun values children value' rest rest' same => by
      cases value' with
      | atom _ => simp [nodes] at same
      | array values' =>
          simp only [nodes, List.cons_append, List.cons.injEq, Node.mk.injEq, true_and] at same
          obtain ⟨equal, tail⟩ := children values' rest rest' same.1 same.2
          exact ⟨by rw [equal], tail⟩)
    (fun values' rest rest' lengths same => by
      cases values' with
      | nil => simpa using same
      | cons _ _ => simp at lengths)
    (fun head tail headCase tailCase values' rest rest' lengths same => by
      cases values' with
      | nil => simp at lengths
      | cons head' tail' =>
          simp only [List.flatMap_cons, List.append_assoc] at same
          obtain ⟨heads, rests⟩ := headCase head' _ _ same
          obtain ⟨tails, final⟩ := tailCase tail' rest rest' (by simpa using lengths) rests
          exact ⟨by rw [heads, tails], final⟩)
    value

/-- `valuePreimage` is injective on values whose nodes fit in 64 bits. -/
theorem valuePreimage_injective {value value' : Value} (fits : Fits value) (fits' : Fits value')
    (same : Package.valuePreimage value = Package.valuePreimage value') : value = value' := by
  rw [← canonicalWords_eq_valuePreimage, ← canonicalWords_eq_valuePreimage] at same
  have equal := flatMap_words_injective
    (fun node member => ⟨nodes_tag_le value node member, fits node member⟩)
    (fun node member => ⟨nodes_tag_le value' node member, fits' node member⟩) same
  simpa using nodes_append_inj value value' [] [] (by rw [equal])

open StepPackage in
/-- Two pairs of packages with one identity are equal, or their hash inputs are distinct and have
one Poseidon2 hash. `Spec.Sponge.collision_blocks` turns the second case into a sponge collision
under `Spec.Sponge.PaddingInjective`. -/
theorem identity_binds {light refresh light' refresh' : StepPackage.Package}
    (fits : Fits (identityValue light refresh)) (fits' : Fits (identityValue light' refresh'))
    (same : identity light refresh = identity light' refresh') :
    (light, refresh) = (light', refresh') ∨
      (boundedDepthDomain ++ Package.valuePreimage (identityValue light refresh) ≠
          boundedDepthDomain ++ Package.valuePreimage (identityValue light' refresh') ∧
        Poseidon2.hash (boundedDepthDomain ++ Package.valuePreimage (identityValue light refresh)) =
          Poseidon2.hash
            (boundedDepthDomain ++ Package.valuePreimage (identityValue light' refresh'))) := by
  by_cases inputs : boundedDepthDomain ++ Package.valuePreimage (identityValue light refresh) =
      boundedDepthDomain ++ Package.valuePreimage (identityValue light' refresh')
  · left
    have values := valuePreimage_injective fits fits' (List.append_cancel_left inputs)
    have decoded := congrArg (pair StepPackage.Package.format StepPackage.Package.format).decode values
    unfold identityValue at decoded
    rw [(pair _ _).decode_encode, (pair _ _).decode_encode] at decoded
    exact Except.ok.inj decoded
  · exact Or.inr ⟨inputs, same⟩

end NightstreamFPrime.Export.BoundedDepth.IdentityBinding
