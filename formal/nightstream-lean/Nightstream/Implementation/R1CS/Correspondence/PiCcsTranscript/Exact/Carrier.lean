import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.ExactMessages
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Interface
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Interface
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol

/-!
Exact typed carrier for the minimal mixed-width Split-NC `Pi_CCS` SumCheck
messages.

Assurance tier: executable candidate-verifier semantics.

Owns: the exact FE row/lane and NC physical message language, lossless
encoding/decoding, verifier-owned FE-initial binding, and adapters into the
typed FE/NC checker interfaces.

Does not own: transcript replay or scheduling, FE/NC polynomial semantics,
honest certificate construction, source assignments, SumCheck soundness,
Poseidon2 execution, equality with the current uniform-width Rust/R1CS
encoding, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: `decode` receives the expected FE initial claim from the
verifier. The prover's raw `feInitial` must equal its lossless concrete
transport. Every round count and coefficient width is checked exactly; this
module never pads, trims, or synthesizes a message.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.exact.fe.initial` | raw `feInitial = toExtension expectedFeInitial` | verifier-owned binding | `ExactLanguage`, `decode` |
| `nifs.pi_ccs.exact.fe.row` | `shape.rowVariables` messages of width `Drow(input)+1` | checked | `feRowLanguage` |
| `nifs.pi_ccs.exact.fe.lane` | `domain.laneVariables` messages of width three | checked | `feLaneLanguage` |
| `nifs.pi_ccs.exact.fe.total` | the two exact phase counts imply the complete FE count | derived | `feRounds_length_of_exactLanguage` |
| `nifs.pi_ccs.exact.nc` | `Transcript.Nc.roundCount domain` messages of width five | checked | `ncLanguage` |
| `nifs.pi_ccs.exact.codec` | typed and raw carriers are lossless inverses | derived | `decode_encode`, `encode_of_decode` |
| `nifs.pi_ccs.exact.adapters` | exact lists enter the FE/NC typed checker carriers without changing order | direct dataflow | `toFeCertificate`, `toNcCertificate` |
| `nifs.pi_ccs.exact.prover_projection` | semantic protocol certificates enter the exact carrier and round-trip | direct dataflow | `ofProtocolCertificate`, `toFeCertificate_ofProtocolCertificate`, `toNcCertificate_ofProtocolCertificate` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Exact

open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages

private abbrev RawRound :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.RoundMessage
private abbrev RawMessages :=
  Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.Messages
private abbrev FeRowMessage
    {shape : SemanticShape}
    (input : PublicInput shape) :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.RowMessage
    input
private abbrev FeLaneMessage :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.LaneMessage
private abbrev NcRoundMessage :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.RoundMessage
private abbrev FeDrow
    {shape : SemanticShape}
    (input : PublicInput shape) : Nat :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Drow input
private abbrev LaneDegree : Nat :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.laneSumcheckDegreeBound
private abbrev NcDegree : Nat :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.ncSumcheckDegreeBound
private abbrev NcRoundCount (domain : FlatNcDomain) : Nat :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.roundCount
    domain
private abbrev FeCertificate
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain) :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
    input domain
private abbrev NcCertificate (domain : FlatNcDomain) :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.Certificate
    domain

/-- A raw list obeys one exact fixed-degree phase language. -/
def FixedRoundsLanguage
    (degree count : Nat)
    (rounds : List RawRound) : Prop :=
  rounds.length = count /\
    forall round, round ∈ rounds ->
      round.coefficients.length = degree + 1

/-- Exact typed data carried by the three physical `Pi_CCS` SumCheck phases.

The FE initial claim is deliberately absent: it is verifier-owned and is an
argument to `encode` and `decode`, not prover certificate data. -/
structure Carrier
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain) where
  feRowRounds :
    ExactRounds (FeRowMessage input) shape.rowVariables
  feLaneRounds :
    ExactRounds FeLaneMessage domain.laneVariables
  ncRounds :
    ExactRounds NcRoundMessage (NcRoundCount domain)

/-- Exact raw language for the complete Split-NC SumCheck message carrier.

The FE list is partitioned at the verifier-owned row count. Expressing the
row prefix and lane suffix separately makes both physical widths auditable
and rejects short, long, or phase-shifted lists. -/
def ExactLanguage
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages) : Prop :=
  messages.feInitial = toExtension expectedFeInitial /\
    FixedRoundsLanguage (FeDrow input) shape.rowVariables
      (messages.feRounds.take shape.rowVariables) /\
    FixedRoundsLanguage LaneDegree domain.laneVariables
      (messages.feRounds.drop shape.rowVariables) /\
    FixedRoundsLanguage NcDegree (NcRoundCount domain)
      messages.ncRounds

namespace ExactRoundProjection

/-- Interpret an exact-count list as its canonically indexed finite
function. -/
def toFunction
    {Element : Type}
    {count : Nat}
    (rounds : ExactRounds Element count) :
    Fin count -> Element :=
  fun index => rounds.values.get (Fin.cast rounds.length_eq.symm index)

/-- Canonically enumerate one finite function as an exact-count list. -/
def ofFunction
    {Element : Type}
    {count : Nat}
    (values : Fin count -> Element) :
    ExactRounds Element count where
  values := List.ofFn values
  length_eq := by simp

/-- Returning through the exact-list view preserves every finite-function
coordinate. -/
@[simp] theorem toFunction_ofFunction
    {Element : Type}
    {count : Nat}
    (values : Fin count -> Element) :
    toFunction (ofFunction values) = values := by
  funext index
  simp [toFunction, ofFunction]

/-- Enumerating the canonical finite-function view recovers the original
exact-count list in the same order. -/
@[simp] theorem ofFn_toFunction
    {Element : Type}
    {count : Nat}
    (rounds : ExactRounds Element count) :
    List.ofFn (toFunction rounds) = rounds.values := by
  unfold toFunction
  rcases rounds with ⟨values, length⟩
  subst count
  simp

end ExactRoundProjection

/-- Enter the independently typed FE checker with the exact decoded row and
lane messages. -/
def Carrier.toFeCertificate
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (carrier : Carrier input domain) :
    FeCertificate input domain where
  rowRounds := ExactRoundProjection.toFunction carrier.feRowRounds
  laneRounds := ExactRoundProjection.toFunction carrier.feLaneRounds

/-- Enter the transcript-owned exact-count NC checker carrier without
changing round order or width. -/
def Carrier.toNcCertificate
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (carrier : Carrier input domain) :
    NcCertificate domain where
  rounds := ExactRoundProjection.toFunction carrier.ncRounds

/-- Canonical exact physical carrier underlying one semantic two-phase
protocol certificate. The output message is intentionally not copied: it is
not part of the SumCheck transcript carrier. -/
def Carrier.ofProtocolCertificate
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Protocol.Certificate input domain) :
    Carrier input domain where
  feRowRounds :=
    ExactRoundProjection.ofFunction certificate.fe.rowRounds
  feLaneRounds :=
    ExactRoundProjection.ofFunction certificate.fe.laneRounds
  ncRounds :=
    ExactRoundProjection.ofFunction certificate.nc.rounds

/-- The canonical exact carrier returns the original typed FE certificate. -/
@[simp] theorem Carrier.toFeCertificate_ofProtocolCertificate
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Protocol.Certificate input domain) :
    (Carrier.ofProtocolCertificate certificate).toFeCertificate =
      certificate.fe := by
  cases certificate with
  | mk fe nc output =>
      cases fe
      simp [Carrier.ofProtocolCertificate, Carrier.toFeCertificate]

/-- The canonical exact carrier returns the original typed NC certificate. -/
@[simp] theorem Carrier.toNcCertificate_ofProtocolCertificate
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate : Protocol.Certificate input domain) :
    (Carrier.ofProtocolCertificate certificate).toNcCertificate =
      certificate.nc := by
  cases certificate with
  | mk fe nc output =>
      cases nc
      simp [Carrier.ofProtocolCertificate, Carrier.toNcCertificate]

private def encodeRowRounds
    {shape : SemanticShape}
    {input : PublicInput shape}
    (rounds :
      ExactRounds (FeRowMessage input) shape.rowVariables) :
    List RawRound :=
  encodeExact encodeFixed rounds

private def encodeLaneRounds
    {domain : FlatNcDomain}
    (rounds :
      ExactRounds FeLaneMessage domain.laneVariables) :
    List RawRound :=
  encodeExact encodeFixed rounds

private def encodeNcRounds
    {domain : FlatNcDomain}
    (rounds :
      ExactRounds NcRoundMessage (NcRoundCount domain)) :
    List RawRound :=
  encodeExact encodeFixed rounds

/-- Canonical raw encoding. The verifier-owned FE initial value is inserted
directly; the carrier cannot override it. -/
def encode
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) : RawMessages where
  feInitial := toExtension expectedFeInitial
  feRounds :=
    encodeRowRounds carrier.feRowRounds ++
      encodeLaneRounds carrier.feLaneRounds
  ncRounds := encodeNcRounds carrier.ncRounds

/-- FE encoding is exactly the row list followed by the lane list, with no
intermediate normalization or phase marker. -/
theorem encode_feRounds_values
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).feRounds =
      carrier.feRowRounds.values.map encodeFixed ++
        carrier.feLaneRounds.values.map encodeFixed :=
  rfl

/-- NC encoding preserves the carrier's exact list and order. -/
theorem encode_ncRounds_values
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).ncRounds =
      carrier.ncRounds.values.map encodeFixed :=
  rfl

/-- Exact raw decoder. FE is split at the verifier-owned row count before
the two independently sized phases are parsed. -/
def decode
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages) : Option (Carrier input domain) :=
  if _initial :
      messages.feInitial = toExtension expectedFeInitial then
    match
        decodeExact shape.rowVariables
          (decodeFixed (FeDrow input))
          (messages.feRounds.take shape.rowVariables) with
    | none => none
    | some feRowRounds =>
        match
            decodeExact domain.laneVariables
              (decodeFixed LaneDegree)
              (messages.feRounds.drop shape.rowVariables) with
        | none => none
        | some feLaneRounds =>
            match
                decodeExact (NcRoundCount domain)
                  (decodeFixed NcDegree)
                  messages.ncRounds with
            | none => none
            | some ncRounds =>
                some { feRowRounds, feLaneRounds, ncRounds }
  else
    none

private theorem encodedRounds_length
    {Output : Type}
    {count : Nat}
    (encodeElement : Output -> RawRound)
    (rounds : ExactRounds Output count) :
    (encodeExact encodeElement rounds).length = count := by
  simp [encodeExact, rounds.length_eq]

@[simp] theorem encode_feInitial
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).feInitial =
      toExtension expectedFeInitial :=
  rfl

@[simp] theorem encode_feRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).feRounds.length =
      shape.rowVariables + domain.laneVariables := by
  simp only [encode, List.length_append]
  have rowLength :
      (encodeRowRounds carrier.feRowRounds).length =
        shape.rowVariables := by
    exact encodedRounds_length _ _
  have laneLength :
      (encodeLaneRounds carrier.feLaneRounds).length =
        domain.laneVariables := by
    exact encodedRounds_length _ _
  rw [rowLength, laneLength]

@[simp] theorem encode_ncRounds_length
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    (encode expectedFeInitial carrier).ncRounds.length =
      NcRoundCount domain := by
  simp [encode, encodeNcRounds, encodedRounds_length]

private theorem decodeFixedExact_isSome_iff_language
    (degree count : Nat)
    (rounds : List RawRound) :
    (decodeExact count (decodeFixed degree) rounds).isSome = true <->
      FixedRoundsLanguage degree count rounds := by
  simp [decodeExact_isSome_iff, decodeFixed_isSome_iff,
    FixedRoundsLanguage]

private theorem feSplitCounts_iff
    (rowCount laneCount : Nat)
    (rounds : List RawRound) :
    ((rounds.take rowCount).length = rowCount /\
        (rounds.drop rowCount).length = laneCount) <->
      rounds.length = rowCount + laneCount := by
  constructor
  · intro counts
    have combined :
        (rounds.take rowCount).length +
            (rounds.drop rowCount).length =
          rounds.length := by
      rw [← List.length_append, List.take_append_drop]
    omega
  · intro total
    have rowCountLe : rowCount <= rounds.length := by
      omega
    constructor
    · rw [List.length_take, Nat.min_eq_left rowCountLe]
    · rw [List.length_drop, total]
      omega

/-- The executable decoder accepts exactly the named component-wise physical
language. This is not a semantic SumCheck acceptance theorem. -/
theorem decode_isSome_iff_exactLanguage
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages) :
    (decode input domain expectedFeInitial messages).isSome = true <->
      ExactLanguage input domain expectedFeInitial messages := by
  unfold decode
  split
  · rename_i initial
    have rowIff :=
      decodeFixedExact_isSome_iff_language
        (FeDrow input) shape.rowVariables
        (messages.feRounds.take shape.rowVariables)
    have laneIff :=
      decodeFixedExact_isSome_iff_language
        LaneDegree domain.laneVariables
        (messages.feRounds.drop shape.rowVariables)
    have ncIff :=
      decodeFixedExact_isSome_iff_language
        NcDegree (NcRoundCount domain) messages.ncRounds
    cases rowResult :
        decodeExact shape.rowVariables
          (decodeFixed (FeDrow input))
          (messages.feRounds.take shape.rowVariables) with
    | none =>
        have rowBad :
            ¬ FixedRoundsLanguage
              (FeDrow input) shape.rowVariables
              (messages.feRounds.take shape.rowVariables) := by
          intro rowGood
          have accepted := rowIff.mpr rowGood
          simp [rowResult] at accepted
        constructor
        · intro accepted
          simp at accepted
        · intro language
          exact (rowBad language.2.1).elim
    | some rowRounds =>
        have rowGood :
            FixedRoundsLanguage
              (FeDrow input) shape.rowVariables
              (messages.feRounds.take shape.rowVariables) :=
          rowIff.mp (by simp [rowResult])
        cases laneResult :
            decodeExact domain.laneVariables
              (decodeFixed LaneDegree)
              (messages.feRounds.drop shape.rowVariables) with
        | none =>
            have laneBad :
                ¬ FixedRoundsLanguage
                  LaneDegree domain.laneVariables
                  (messages.feRounds.drop shape.rowVariables) := by
              intro laneGood
              have accepted := laneIff.mpr laneGood
              simp [laneResult] at accepted
            constructor
            · intro accepted
              simp at accepted
            · intro language
              exact (laneBad language.2.2.1).elim
        | some laneRounds =>
            have laneGood :
                FixedRoundsLanguage
                  LaneDegree domain.laneVariables
                  (messages.feRounds.drop shape.rowVariables) :=
              laneIff.mp (by simp [laneResult])
            cases ncResult :
                decodeExact (NcRoundCount domain)
                  (decodeFixed NcDegree) messages.ncRounds with
            | none =>
                have ncBad :
                    ¬ FixedRoundsLanguage
                      NcDegree (NcRoundCount domain)
                      messages.ncRounds := by
                  intro ncGood
                  have accepted := ncIff.mpr ncGood
                  simp [ncResult] at accepted
                constructor
                · intro accepted
                  simp at accepted
                · intro language
                  exact (ncBad language.2.2.2).elim
            | some ncRounds =>
                have ncGood :
                    FixedRoundsLanguage
                      NcDegree (NcRoundCount domain)
                      messages.ncRounds :=
                  ncIff.mp (by simp [ncResult])
                constructor
                · intro _
                  exact ⟨initial, rowGood, laneGood, ncGood⟩
                · intro _
                  simp
  · rename_i initial
    constructor
    · intro accepted
      simp at accepted
    · intro language
      exact (initial language.1).elim

/-- The complete FE count is derived from the two independently checked
physical phase counts; it is not a duplicated retained obligation. -/
theorem feRounds_length_of_exactLanguage
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    {expectedFeInitial : K}
    {messages : RawMessages}
    (language :
      ExactLanguage input domain expectedFeInitial messages) :
    messages.feRounds.length =
      shape.rowVariables + domain.laneVariables :=
  (feSplitCounts_iff
    shape.rowVariables domain.laneVariables messages.feRounds).mp
      ⟨language.2.1.1, language.2.2.1.1⟩

/-- Canonical encoding always belongs to the exact physical language. -/
theorem exactLanguage_encode
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    ExactLanguage input domain expectedFeInitial
      (encode expectedFeInitial carrier) := by
  rw [← decode_isSome_iff_exactLanguage]
  simp [decode, encode, encodeRowRounds, encodeLaneRounds, encodeNcRounds,
    encodedRounds_length]

/-- Canonical typed data decodes without loss. -/
@[simp] theorem decode_encode
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K)
    (carrier : Carrier input domain) :
    decode input domain expectedFeInitial
      (encode expectedFeInitial carrier) = some carrier := by
  unfold decode encode
  have rowLength :
      (encodeRowRounds carrier.feRowRounds).length =
        shape.rowVariables :=
    encodedRounds_length _ _
  have takeEq :
      (encodeRowRounds carrier.feRowRounds ++
          encodeLaneRounds carrier.feLaneRounds).take
          shape.rowVariables =
        encodeRowRounds carrier.feRowRounds := by
    calc
      _ =
          (encodeRowRounds carrier.feRowRounds ++
            encodeLaneRounds carrier.feLaneRounds).take
            (encodeRowRounds carrier.feRowRounds).length := by
        rw [rowLength]
      _ = encodeRowRounds carrier.feRowRounds :=
        List.take_append_length
  have dropEq :
      (encodeRowRounds carrier.feRowRounds ++
          encodeLaneRounds carrier.feLaneRounds).drop
          shape.rowVariables =
        encodeLaneRounds carrier.feLaneRounds := by
    calc
      _ =
          (encodeRowRounds carrier.feRowRounds ++
            encodeLaneRounds carrier.feLaneRounds).drop
            (encodeRowRounds carrier.feRowRounds).length := by
        rw [rowLength]
      _ = encodeLaneRounds carrier.feLaneRounds :=
        List.drop_append_length
  rw [takeEq, dropEq]
  simp [encodeRowRounds, encodeLaneRounds, encodeNcRounds]

/-- Every successfully decoded raw value re-encodes byte-for-byte at the
message-carrier level. No high-zero normalization or phase reshaping occurs. -/
theorem encode_of_decode
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    {expectedFeInitial : K}
    {messages : RawMessages}
    {carrier : Carrier input domain}
    (decoded :
      decode input domain expectedFeInitial messages = some carrier) :
    encode expectedFeInitial carrier = messages := by
  unfold decode at decoded
  split at decoded
  · rename_i initial
    cases rowResult :
        decodeExact shape.rowVariables
          (decodeFixed (FeDrow input))
          (messages.feRounds.take shape.rowVariables) with
    | none => simp [rowResult] at decoded
    | some rowRounds =>
        cases laneResult :
            decodeExact domain.laneVariables
              (decodeFixed LaneDegree)
              (messages.feRounds.drop shape.rowVariables) with
        | none => simp [rowResult, laneResult] at decoded
        | some laneRounds =>
            cases ncResult :
                decodeExact (NcRoundCount domain)
                  (decodeFixed NcDegree)
                  messages.ncRounds with
            | none => simp [rowResult, laneResult, ncResult] at decoded
            | some ncRounds =>
                have carrierEq :
                    ({ feRowRounds := rowRounds
                       feLaneRounds := laneRounds
                       ncRounds := ncRounds } :
                      Carrier input domain) = carrier := by
                  simpa [rowResult, laneResult, ncResult] using decoded
                rw [← carrierEq]
                have rowEncoded :=
                  encodeExact_of_decodeExact
                    (decodeFixed (FeDrow input))
                    encodeFixed
                    (fun raw typed decoded =>
                      encodeFixed_of_decodeFixed decoded)
                    rowResult
                have laneEncoded :=
                  encodeExact_of_decodeExact
                    (decodeFixed LaneDegree)
                    encodeFixed
                    (fun raw typed decoded =>
                      encodeFixed_of_decodeFixed decoded)
                    laneResult
                have ncEncoded :=
                  encodeExact_of_decodeExact
                    (decodeFixed NcDegree)
                    encodeFixed
                    (fun raw typed decoded =>
                      encodeFixed_of_decodeFixed decoded)
                    ncResult
                cases messages with
                | mk feInitial feRounds ncRounds =>
                    simp only at initial rowEncoded laneEncoded ncEncoded
                    simp only [encode, encodeRowRounds, encodeLaneRounds,
                      encodeNcRounds]
                    have feRoundsEq :
                        encodeExact encodeFixed rowRounds ++
                            encodeExact encodeFixed laneRounds =
                          feRounds := by
                      rw [rowEncoded, laneEncoded]
                      exact List.take_append_drop shape.rowVariables feRounds
                    cases initial.symm
                    cases feRoundsEq
                    cases ncEncoded
                    rfl
  · simp at decoded

/-- The canonical raw encoding is injective for a fixed verifier-owned FE
initial claim. -/
theorem encode_injective
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expectedFeInitial : K) :
    Function.Injective
      (encode (input := input) (domain := domain) expectedFeInitial) := by
  intro left right equal
  have leftDecoded := decode_encode expectedFeInitial left
  have rightDecoded := decode_encode expectedFeInitial right
  rw [equal] at leftDecoded
  rw [rightDecoded] at leftDecoded
  exact (Option.some.inj leftDecoded).symm

/-- A prover-selected FE initial value outside the verifier-owned binding is
rejected before any round is decoded. -/
theorem decode_none_of_feInitial_ne
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages)
    (wrong :
      messages.feInitial ≠ toExtension expectedFeInitial) :
    decode input domain expectedFeInitial messages = none := by
  unfold decode
  simp [wrong]

/-- Any short or long complete FE list is rejected. -/
theorem decode_none_of_feRoundCount_ne
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages)
    (wrong :
      messages.feRounds.length ≠
        shape.rowVariables + domain.laneVariables) :
    decode input domain expectedFeInitial messages = none := by
  cases result : decode input domain expectedFeInitial messages with
  | none => rfl
  | some carrier =>
      exfalso
      have encoded := encode_of_decode result
      apply wrong
      rw [← encoded]
      exact encode_feRounds_length expectedFeInitial carrier

/-- Any short or long NC list is rejected. -/
theorem decode_none_of_ncRoundCount_ne
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages)
    (wrong :
      messages.ncRounds.length ≠ NcRoundCount domain) :
    decode input domain expectedFeInitial messages = none := by
  cases result : decode input domain expectedFeInitial messages with
  | none => rfl
  | some carrier =>
      exfalso
      have encoded := encode_of_decode result
      apply wrong
      rw [← encoded]
      exact encode_ncRounds_length expectedFeInitial carrier

/-- A malformed FE row-prefix width is rejected. -/
theorem decode_none_of_feRowWidth_ne
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages)
    (round : RawRound)
    (member : round ∈ messages.feRounds.take shape.rowVariables)
    (wrong :
      round.coefficients.length ≠ FeDrow input + 1) :
    decode input domain expectedFeInitial messages = none := by
  cases result : decode input domain expectedFeInitial messages with
  | none => rfl
  | some carrier =>
      exfalso
      have isSome :
          (decode input domain expectedFeInitial messages).isSome = true := by
        simp [result]
      have language :=
        (decode_isSome_iff_exactLanguage
          input domain expectedFeInitial messages).mp isSome
      exact wrong (language.2.1.2 round member)

/-- A malformed FE lane-suffix width is rejected. -/
theorem decode_none_of_feLaneWidth_ne
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages)
    (round : RawRound)
    (member : round ∈ messages.feRounds.drop shape.rowVariables)
    (wrong : round.coefficients.length ≠ 3) :
    decode input domain expectedFeInitial messages = none := by
  cases result : decode input domain expectedFeInitial messages with
  | none => rfl
  | some carrier =>
      exfalso
      have isSome :
          (decode input domain expectedFeInitial messages).isSome = true := by
        simp [result]
      have language :=
        (decode_isSome_iff_exactLanguage
          input domain expectedFeInitial messages).mp isSome
      have exactWidth :=
        language.2.2.1.2 round member
      exact wrong (by simpa [LaneDegree] using exactWidth)

/-- A malformed NC round width is rejected. -/
theorem decode_none_of_ncWidth_ne
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain)
    (expectedFeInitial : K)
    (messages : RawMessages)
    (round : RawRound)
    (member : round ∈ messages.ncRounds)
    (wrong : round.coefficients.length ≠ 5) :
    decode input domain expectedFeInitial messages = none := by
  cases result : decode input domain expectedFeInitial messages with
  | none => rfl
  | some carrier =>
      exfalso
      have isSome :
          (decode input domain expectedFeInitial messages).isSome = true := by
        simp [result]
      have language :=
        (decode_isSome_iff_exactLanguage
          input domain expectedFeInitial messages).mp isSome
      have exactWidth :=
        language.2.2.2.2 round member
      exact wrong (by simpa [NcDegree] using exactWidth)

end Nightstream.Implementation.R1CS.PiCcsTranscript.Exact
