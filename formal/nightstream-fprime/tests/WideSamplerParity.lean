import Lean
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript

/-! Candidate native-decoder parity vectors. These do not select a package. -/

namespace NightstreamFPrime.Tests.WideSamplerParity

open NightstreamFPrime.Spec
open Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

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

def run (args : List String) : IO Unit := do
  match args with
  | [path] => IO.FS.writeFile path (fixture.compress ++ "\n")
  | _ => throw (IO.userError "usage: emitWideSamplerParity OUTPUT.json")

end NightstreamFPrime.Tests.WideSamplerParity

def main (args : List String) : IO Unit := NightstreamFPrime.Tests.WideSamplerParity.run args
