import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.ExactMessages

/-!
Focused regressions for the exact-width and exact-count `Pi_CCS` message
codec.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.sumcheck.message.exact_width` | degree-two/four messages use exactly three/five slots | padding/trimming or loose width acceptance |
| `nifs.pi_ccs.sumcheck.message.transport` | concrete/semantic extension transport round-trips | limb order or modulus drift |
| `nifs.pi_ccs.sumcheck.messages.exact_count` | finite vectors reject short and long physical lists | ignored or synthesized rounds |
| `nifs.pi_ccs.sumcheck.messages.roundtrip` | exact lists re-encode identically | message reordering or normalization |
-/

namespace NightstreamTests.PiCcsTranscriptExactMessages

open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck.Finite

example (round : FixedPolynomial K 2) :
    decodeFixed 2 (encodeFixed round) = some round := by
  simp

example (round : FixedPolynomial K 4) :
    decodeFixed 4 (encodeFixed round) = some round := by
  simp

example
    (round : SumCheck.RoundMessage)
    (short : round.coefficients.length = 2) :
    decodeFixed 2 round = none := by
  unfold decodeFixed
  split
  · omega
  · rfl

example
    (round : SumCheck.RoundMessage)
    (long : round.coefficients.length = 4) :
    decodeFixed 2 round = none := by
  unfold decodeFixed
  split
  · omega
  · rfl

example
    (round : SumCheck.RoundMessage)
    (short : round.coefficients.length = 4) :
    decodeFixed 4 round = none := by
  unfold decodeFixed
  split
  · omega
  · rfl

example
    (round : SumCheck.RoundMessage)
    (long : round.coefficients.length = 6) :
    decodeFixed 4 round = none := by
  unfold decodeFixed
  split
  · omega
  · rfl

example
    (rounds : ExactRounds (FixedPolynomial K 2) 3) :
    decodeExact 3 (decodeFixed 2) (encodeExact encodeFixed rounds) =
      some rounds := by
  exact decodeExact_encode (decodeFixed 2) encodeFixed
    decodeFixed_encodeFixed rounds

example :
    decodeExact 2 (fun value : Nat => some value) [11] = none := by
  rfl

example :
    decodeExact 2 (fun value : Nat => some value) [11, 13, 17] = none := by
  rfl

example :
    Function.Injective
      (encodeExact (count := 3)
        (encodeFixed : FixedPolynomial K 2 -> SumCheck.RoundMessage)) := by
  exact encodeExact_injective (decodeFixed 2) encodeFixed
    decodeFixed_encodeFixed

end NightstreamTests.PiCcsTranscriptExactMessages
