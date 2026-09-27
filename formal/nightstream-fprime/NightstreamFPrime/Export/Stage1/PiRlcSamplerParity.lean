import Lean
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript

/-! Native-decoder parity at reduction boundaries and across 17 chained draws.
Adapted from tests/WideSamplerParity.lean at e1a7c96a617237005859fd6a01ae3c21ba0727da. -/

namespace NightstreamFPrime.Export.Stage1.PiRlcSamplerParity

open NightstreamFPrime.Spec
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler

private def coefficients (draw : Draw) : List Int :=
  List.ofFn fun position => Int.ofNat (sample draw position).val - 2

private def boundary (integer : Nat) : Lean.Json :=
  let draw := drawIndex.symm ⟨integer % drawCount, Nat.mod_lt _ (by decide)⟩
  Lean.Json.mkObj [
    ("integer", Lean.toJson (toString integer)),
    ("draw", Lean.toJson (List.ofFn fun lane => (draw lane).val)),
    ("coefficients", Lean.toJson (coefficients draw))]

private def transcript (initial : Poseidon2.State) : Lean.Json :=
  Lean.Json.mkObj [
    ("initial", Lean.toJson (initial.map Fin.val)),
    ("steps", Lean.toJson ((List.range 17).map fun source =>
      let entered := Transcript.enter (Transcript.stateAt initial source) source
      Lean.Json.mkObj [
        ("source", Lean.toJson source),
        ("entered", Lean.toJson (entered.map Fin.val)),
        ("draw", Lean.toJson (List.ofFn fun lane => (Transcript.block entered lane).val)),
        ("coefficients", Lean.toJson (coefficients (Transcript.block entered))),
        ("outgoing", Lean.toJson ((Transcript.stateAt initial (source + 1)).map Fin.val))])),
    ("final", Lean.toJson ((Transcript.stateAt initial 17).map Fin.val))]

def fixture : Lean.Json :=
  let boundaries := [0, 1, scalarCount - 1, scalarCount, scalarCount + 1,
    goldilocksModulus - 1, goldilocksModulus, goldilocksModulus + 1,
    drawCount - scalarCount, drawCount - 2, drawCount - 1]
  let states := [List.replicate 8 (Poseidon2.ofNat 0),
    (List.range 8).map Poseidon2.ofNat,
    List.replicate 8 (Poseidon2.ofNat (goldilocksModulus - 1))]
  Lean.Json.mkObj [
    ("schema", Lean.toJson (1 : Nat)),
    ("modulus", Lean.toJson goldilocksModulus),
    ("degree", Lean.toJson ringDegree),
    ("boundaries", Lean.toJson (boundaries.map boundary)),
    ("transcripts", Lean.toJson (states.map transcript))]

end NightstreamFPrime.Export.Stage1.PiRlcSamplerParity
