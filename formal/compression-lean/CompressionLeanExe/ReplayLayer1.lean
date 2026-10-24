import CompressionLean.Ports.Replay.Stages

/-!
The `replayLayer1` executable: replay every variant of a Rust export and
print, per variant, its name, every draw in order (`e c0 c1 c2` for an
extension draw, `s w` for a word split into bits) and the verdict (`accept`,
or `reject` and the first failed check). It does not read Rust's draws or
verdicts; the comparison is the Rust test's.
-/

open Lean CompressionLean.Replay

def extLine (value : Ext) : String := s!"e {value.c0.toNat} {value.c1.toNat} {value.c2.toNat}"

/-- One line per sample, as Rust logs them: a pair is two extension draws and
the setup query words are one sample each. -/
def drawLines : Draw → List String
  | .ext value => [extLine value]
  | .pair first second => [extLine first, extLine second]
  | .words values => values.toList.map fun word => s!"s {word.toNat}"
  | .sample word => [s!"s {word.toNat}"]

def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: replayLayer1 <export.json>"; return 2
  let json ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let data : Export ← IO.ofExcept (fromJson? json)
  let out ← IO.getStdout
  for variant in data.variants do
    let (verdict, transcript) :=
      (verify data.relation variant.seed variant.statement variant.proof).run.run ⟨Duplex.new, #[]⟩
    out.putStrLn s!"variant {variant.name}"
    for draw in transcript.draws do
      for line in drawLines draw do
        out.putStrLn line
    out.putStrLn (match verdict with | .ok _ => "accept" | .error what => s!"reject {what}")
  return 0
