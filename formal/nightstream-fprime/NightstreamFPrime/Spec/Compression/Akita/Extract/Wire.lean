import NightstreamFPrime.Spec.Compression.Akita.Setup
import NightstreamFPrime.Spec.Compression.Akita.Extract.PairComplete

/-! A finite type of wire messages for two packing folds of Akita (ePrint 2026/1983, revision of
2026-10-15, Figure 4: the messages of the prover), with the condition `Full` of
`packingPair_complete`.

- `Letter` has a claim of each level, a commitment and the digit vectors of the second level,
  and `width` elements of `E` for one message of the checks.
- `Letter.read` reads a message from these elements: for a type that the vectors of `width`
  elements map onto, `unpack` is one such map.
- `Letter.full` is the condition `Full`, and `Letter.finite` states that the type is finite, which
  `packingPair_extraction_le` needs. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

section Finite

variable {d n : ℕ} [NeZero d]

instance Payload.finite (chain : Option ChainShape) : Finite (Payload d n chain) := by
  cases chain <;> exact inferInstanceAs (Finite (Fin _ → Rq _))

instance Payload.nonempty (chain : Option ChainShape) : Nonempty (Payload d n chain) := by
  cases chain <;> exact ⟨fun _ => 0⟩

instance PathDigits.finite (chain : Option ChainShape) : Finite (PathDigits chain) := by
  cases chain with
  | none => exact inferInstanceAs (Finite PUnit)
  | some c =>
      exact Finite.of_injective (fun digits : ChainDigits c => (digits.first, digits.second))
        fun _ _ same => ChainDigits.ext (congrArg Prod.fst same) (congrArg Prod.snd same)

instance PathDigits.nonempty (chain : Option ChainShape) : Nonempty (PathDigits chain) := by
  cases chain with
  | none => exact ⟨PUnit.unit⟩
  | some c => exact ⟨(⟨fun _ => 0, fun _ => 0⟩ : ChainDigits c)⟩

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type}

instance Point.finite [Finite E] : Finite (Point O E) :=
  Finite.of_injective (fun point : Point O E =>
    (point.block, point.position, point.pack, point.tail)) fun left right same => by
      cases left
      cases right
      simp_all

instance Witness.finite : Finite (Witness O fold) :=
  Finite.of_injective (fun w : Witness O fold =>
    (w.inner, w.chain, w.partials, w.openChain, w.response)) fun left right same => by
      obtain ⟨⟨_, _, _, _⟩, _⟩ := left
      obtain ⟨⟨_, _, _, _⟩, _⟩ := right
      simp_all

instance Witness.nonempty : Nonempty (Witness O fold) :=
  ⟨⟨⟨fun _ => 0, Classical.arbitrary _, fun _ => 0, Classical.arbitrary _⟩, fun _ => 0⟩⟩

namespace Extract

instance Claim.finite [Finite E] : Finite (Claim O E) :=
  Finite.of_injective (fun claim : Claim O E =>
    (claim.payload, claim.openPayload, claim.point, claim.value)) fun left right same => by
      cases left
      cases right
      simp_all

instance Claim.nonempty [Zero E] : Nonempty (Claim O E) :=
  ⟨⟨Classical.arbitrary _, Classical.arbitrary _,
    ⟨fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩, 0⟩⟩

end Extract

end Finite

namespace Extract

variable {S : Shape} {S' : Shape}

/-- A wire message of two packing folds: the claim of the first level, the claim, the
commitment and the digit vectors of the second level, and `width` elements of `E` for one
message of the checks. -/
structure Letter (O : OpeningShape S) (O' : OpeningShape S') (fold' : Descriptor) (E : Type)
    (width : ℕ) where
  root : Claim O E
  next : Claim O' E
  payload : S'.Payload
  digits : Witness O' fold'
  message : Fin width → E

variable {O : OpeningShape S} {O' : OpeningShape S'} {fold' : Descriptor} {E : Type} {width : ℕ}

instance Letter.finite [Finite E] : Finite (Letter O O' fold' E width) :=
  Finite.of_injective (fun wire : Letter O O' fold' E width =>
    (wire.root, wire.next, wire.payload, wire.digits, wire.message)) fun left right same => by
      cases left
      cases right
      simp_all

instance Letter.nonempty [Zero E] : Nonempty (Letter O O' fold' E width) :=
  ⟨⟨Classical.arbitrary _, Classical.arbitrary _, Classical.arbitrary _, Classical.arbitrary _,
    fun _ => 0⟩⟩

open Classical in
/-- A map from the vectors of `width` elements of `E` onto the type `M`, when one exists. -/
noncomputable def unpack (width : ℕ) (E M : Type) [Nonempty M] : (Fin width → E) → M :=
  if onto : ∃ map : (Fin width → E) → M, Function.Surjective map then onto.choose
  else fun _ => Classical.arbitrary M

theorem unpack_surjective {M : Type} [Nonempty M]
    (onto : ∃ map : (Fin width → E) → M, Function.Surjective map) :
    Function.Surjective (unpack width E M) := by
  unfold unpack
  rw [dite_eq_left onto]
  exact onto.choose_spec

/-- The reader of the messages of the checks: each message is `unpack` of the elements of the
wire message. A wire message `Letter` has the nonce 0 of the response search. -/
noncomputable def Letter.read : Reader (Letter O O' fold' E width) where
  message M _ wire _ := unpack width E M wire.message
  nonce _ := 0

/-- The wire messages `Letter` satisfy `Full` for the message types that the vectors of `width`
elements of `E` map onto. -/
theorem Letter.full [Zero E] :
    Full (fun M => ∃ map : (Fin width → E) → M, Function.Surjective map)
      (Letter.root : Letter O O' fold' E width → Claim O E) Letter.next Letter.payload
      Letter.digits Letter.read where
  root value := ⟨{ Classical.arbitrary (Letter O O' fold' E width) with root := value }, rfl⟩
  message M _ message onto := by
    obtain ⟨vector, sent⟩ := unpack_surjective onto message
    exact ⟨{ Classical.arbitrary (Letter O O' fold' E width) with message := vector }, sent⟩
  commitment value M _ message onto := by
    obtain ⟨vector, sent⟩ := unpack_surjective onto message
    exact ⟨{ Classical.arbitrary (Letter O O' fold' E width) with
      payload := value, message := vector }, rfl, sent⟩
  next value M _ message onto := by
    obtain ⟨vector, sent⟩ := unpack_surjective onto message
    exact ⟨{ Classical.arbitrary (Letter O O' fold' E width) with
      next := value, message := vector }, rfl, sent⟩
  digits value M _ message onto := by
    obtain ⟨vector, sent⟩ := unpack_surjective onto message
    exact ⟨{ Classical.arbitrary (Letter O O' fold' E width) with
      digits := value, message := vector }, rfl, sent⟩
  tried _ := show 0 < responseTries by decide

end Extract

end NightstreamFPrime.Spec.Compression.Akita
