import CompressionLean.Replay.Stages

open Lean CompressionLean.Replay

def extLine (value : Ext) : String := s!"e {value.c0.toNat} {value.c1.toNat} {value.c2.toNat}"

/-- Rust's draw-log format. -/
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
