import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.SumCheck
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Transport
import Nightstream.SuperNeo.SumCheck.FixedPolynomial

/-!
Exact-width and exact-count message codec for the minimal mixed-width
`Pi_CCS` candidate SumCheck transcript.

Assurance tier: executable candidate-verifier semantics.

Owns: lossless transport of one statically sized semantic polynomial, exact
decoding of a verifier-owned number of messages, and the corresponding
left/right round-trip theorems.

Does not own: FE/NC polynomial semantics, honest message construction,
transcript replay, current Rust's loose `len <= d_sc + 1` language, Poseidon2,
R1CS rows, costs, or row removal.

Emits constraints: no.

Authority boundary: decoding rejects wrong widths and wrong counts. It never
pads, trims, canonicalizes away high zeros, or derives missing messages.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.sumcheck.message.exact_width` | one physical message has exactly `degree + 1` coefficients | checked | `decodeFixed` |
| `nifs.pi_ccs.sumcheck.message.transport` | semantic/concrete coefficient transport is lossless and ordered | derived | `decodeFixed_encodeFixed`, `encodeFixed_of_decodeFixed` |
| `nifs.pi_ccs.sumcheck.messages.exact_count` | decode exactly the verifier-owned number of messages | checked | `decodeExact` |
| `nifs.pi_ccs.sumcheck.messages.roundtrip` | exact typed lists and physical lists are inverse encodings | derived | `decodeExact_encode`, `encodeExact_of_decodeExact` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages

open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ConcreteRound :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.RoundMessage

/-- Canonical transport of one statically sized semantic polynomial. -/
def encodeFixed
    {degree : Nat}
    (round : FixedPolynomial K degree) : ConcreteRound where
  coefficients := round.coefficients.map toExtension

/-- Decode one concrete message only at the exact requested physical width. -/
def decodeFixed
    (degree : Nat)
    (round : ConcreteRound) : Option (FixedPolynomial K degree) :=
  if length : round.coefficients.length = degree + 1 then
    some {
      coefficients := round.coefficients.map toK
      coefficients_length := by simpa using length }
  else
    none

/-- Successful single-round decoding is exactly the requested width. -/
@[simp] theorem decodeFixed_isSome_iff
    (degree : Nat)
    (round : ConcreteRound) :
    (decodeFixed degree round).isSome = true <->
      round.coefficients.length = degree + 1 := by
  unfold decodeFixed
  split <;> simp_all

/-- Canonical encoding reparses to the original statically sized polynomial. -/
@[simp] theorem decodeFixed_encodeFixed
    {degree : Nat}
    (round : FixedPolynomial K degree) :
    decodeFixed degree (encodeFixed round) = some round := by
  cases round with
  | mk coefficients coefficientsLength =>
      simp [decodeFixed, encodeFixed, coefficientsLength, Function.comp_def]

/-- Re-encoding any successfully decoded concrete message is coefficient-for-
coefficient identical; no normalization occurs. -/
theorem encodeFixed_of_decodeFixed
    {degree : Nat}
    {concrete : ConcreteRound}
    {typed : FixedPolynomial K degree}
    (decoded : decodeFixed degree concrete = some typed) :
    encodeFixed typed = concrete := by
  unfold decodeFixed at decoded
  split at decoded
  · simp only [Option.some.injEq] at decoded
    subst typed
    cases concrete with
    | mk coefficients =>
        simp [encodeFixed, Function.comp_def]
  · simp at decoded

/-- A list whose verifier-owned physical count is carried by the type. -/
structure ExactRounds (Output : Type) (count : Nat) where
  values : List Output
  length_eq : values.length = count

/-- Canonical list encoding of an exact-count typed list. -/
def encodeExact
    {Input Output : Type}
    {count : Nat}
    (encode : Output -> Input)
    (rounds : ExactRounds Output count) : List Input :=
  rounds.values.map encode

/-- Decode exactly `count` elements. Too few, too many, or any failed element
rejects before an exact-count typed list exists. -/
def decodeExact
    {Input Output : Type} :
    (count : Nat) ->
    (Input -> Option Output) ->
    List Input ->
    Option (ExactRounds Output count)
  | 0, _, [] => some { values := [], length_eq := rfl }
  | 0, _, _ :: _ => none
  | _ + 1, _, [] => none
  | count + 1, decode, raw :: raws => do
      let value <- decode raw
      let tail <- decodeExact count decode raws
      pure {
        values := value :: tail.values
        length_eq := by simp [tail.length_eq] }

/-- Exact-list decoding succeeds exactly when the source has the requested
count and every source element is accepted by the element decoder. -/
@[simp] theorem decodeExact_isSome_iff
    {Input Output : Type}
    (count : Nat)
    (decode : Input -> Option Output)
    (raws : List Input) :
    (decodeExact count decode raws).isSome = true <->
      raws.length = count /\
        forall raw, raw ∈ raws -> (decode raw).isSome = true := by
  induction count generalizing raws with
  | zero =>
      cases raws with
      | nil => simp [decodeExact]
      | cons raw raws => simp [decodeExact]
  | succ count inductionHypothesis =>
      cases raws with
      | nil => simp [decodeExact]
      | cons raw raws =>
          cases decoded : decode raw with
          | none => simp [decodeExact, decoded]
          | some value =>
              have sourceIff :
                  ((raw :: raws).length = count + 1 /\
                      forall next, next ∈ raw :: raws ->
                        (decode next).isSome = true) <->
                    (raws.length = count /\
                      forall next, next ∈ raws ->
                        (decode next).isSome = true) := by
                simp [decoded]
              rw [sourceIff]
              rw [← inductionHypothesis raws]
              cases tailDecoded : decodeExact count decode raws with
              | none => simp [decodeExact, decoded, tailDecoded]
              | some tail => simp [decodeExact, decoded, tailDecoded]

/-- Element-level decode/encode correctness lifts to exact-count typed lists. -/
@[simp] theorem decodeExact_encode
    {Input Output : Type}
    {count : Nat}
    (decode : Input -> Option Output)
    (encode : Output -> Input)
    (decodeEncode : forall value, decode (encode value) = some value)
    (rounds : ExactRounds Output count) :
    decodeExact count decode (encodeExact encode rounds) = some rounds := by
  induction count with
  | zero =>
      cases rounds with
      | mk values length =>
          cases values with
          | nil =>
              change
                some
                    ({ values := [], length_eq := rfl } :
                      ExactRounds Output 0) =
                  some
                    ({ values := [], length_eq := length } :
                      ExactRounds Output 0)
              rfl
          | cons value values => cases length
  | succ count inductionHypothesis =>
      cases rounds with
      | mk values length =>
          cases values with
          | nil => cases length
          | cons value values =>
              have tailLength : values.length = count :=
                Nat.succ.inj length
              let tail : ExactRounds Output count := {
                values := values
                length_eq := tailLength }
              have tailDecoded := inductionHypothesis tail
              change
                decodeExact count decode (values.map encode) = some tail
                at tailDecoded
              simp only [encodeExact, List.map_cons, decodeExact]
              rw [decodeEncode value]
              rw [tailDecoded]
              apply congrArg some
              rfl

/-- Successful exact-count decoding certifies the physical source length. -/
theorem length_eq_of_decodeExact_eq_some
    {Input Output : Type}
    {count : Nat}
    (decode : Input -> Option Output)
    {raws : List Input}
    {rounds : ExactRounds Output count}
    (decoded : decodeExact count decode raws = some rounds) :
    raws.length = count := by
  induction count generalizing raws with
  | zero =>
      cases raws with
      | nil => rfl
      | cons raw raws => simp [decodeExact] at decoded
  | succ count inductionHypothesis =>
      cases raws with
      | nil => simp [decodeExact] at decoded
      | cons raw raws =>
          cases rawResult : decode raw with
          | none => simp [decodeExact, rawResult] at decoded
          | some value =>
              cases tailResult : decodeExact count decode raws with
              | none => simp [decodeExact, rawResult, tailResult] at decoded
              | some tail =>
                  have tailLength : raws.length = count :=
                    inductionHypothesis
                      (raws := raws)
                      (rounds := tail)
                      tailResult
                  simp [tailLength]

/-- If every successful element decode is lossless, re-encoding the exact
typed list recovers the entire physical source list. -/
theorem encodeExact_of_decodeExact
    {Input Output : Type}
    {count : Nat}
    (decode : Input -> Option Output)
    (encode : Output -> Input)
    (encodeDecode :
      forall raw value, decode raw = some value -> encode value = raw)
    {raws : List Input}
    {rounds : ExactRounds Output count}
    (decoded : decodeExact count decode raws = some rounds) :
    encodeExact encode rounds = raws := by
  induction count generalizing raws with
  | zero =>
      cases raws with
      | nil =>
          have roundsEq :
              ({ values := [], length_eq := rfl } :
                ExactRounds Output 0) = rounds := by
            simpa [decodeExact] using decoded
          rw [← roundsEq]
          rfl
      | cons raw raws => simp [decodeExact] at decoded
  | succ count inductionHypothesis =>
      cases raws with
      | nil => simp [decodeExact] at decoded
      | cons raw raws =>
          cases rawResult : decode raw with
          | none => simp [decodeExact, rawResult] at decoded
          | some value =>
              cases tailResult : decodeExact count decode raws with
              | none => simp [decodeExact, rawResult, tailResult] at decoded
              | some tail =>
                  have roundsEq :
                      ({ values := value :: tail.values
                         length_eq := by simp [tail.length_eq] } :
                        ExactRounds Output (count + 1)) = rounds := by
                    simpa [decodeExact, rawResult, tailResult] using decoded
                  rw [← roundsEq]
                  have headEq : encode value = raw :=
                    encodeDecode raw value rawResult
                  have tailEq : encodeExact encode tail = raws :=
                    inductionHypothesis
                      (raws := raws)
                      (rounds := tail)
                      tailResult
                  simp only [encodeExact, List.map_cons]
                  rw [headEq]
                  exact congrArg (List.cons raw) (by
                    simpa only [encodeExact] using tailEq)

/-- Exact-count encoding is injective whenever the element codec has the
proved decode-after-encode law. -/
theorem encodeExact_injective
    {Input Output : Type}
    {count : Nat}
    (decode : Input -> Option Output)
    (encode : Output -> Input)
    (decodeEncode : forall value, decode (encode value) = some value) :
    Function.Injective (encodeExact (count := count) encode) := by
  intro left right equal
  have decodedLeft := decodeExact_encode decode encode decodeEncode left
  have decodedRight := decodeExact_encode decode encode decodeEncode right
  rw [equal] at decodedLeft
  rw [decodedRight] at decodedLeft
  exact (Option.some.inj decodedLeft).symm

end Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages
