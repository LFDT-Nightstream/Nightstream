import CompressionLean.Spec.OptionBCover
import CompressionLeanExe.Parse
import CompressionLeanExe.PackageJson

/-! What a slot's layer-1 key covers (CW-10.6): `Spec.OptionB.coverFailure` on a step package and
the layer-1 key of its circuit, which `option_b_layer1_matches_lean` (`neo-spartan`) writes. -/

open CompressionLean in
def main (args : List String) : IO UInt32 := do
  let [package, layer1] := args | IO.eprintln "usage: coverCheck <package.json> <layer1.json>"; return 2
  let package ← IO.ofExcept (← CompressionLeanExe.PackageJson.readPackage package)
  let key ← IO.ofExcept (← CompressionLeanExe.Parse.readJson Circuit.Layer1Key layer1)
  let width := NightstreamFPrime.Layout.ProductionAssignment.logicalWidth (Spec.OptionB.slotsOf package)
  let blocks := key.relation.key.blocks
  IO.println s!"package logical width {width}; key blocks {blocks} ({NightstreamFPrime.Spec.ringDegree * blocks} \
    columns); key norm bound {key.relation.normBound}; slot bound {Spec.OptionB.slotBound}"
  match Spec.OptionB.coverFailure package key with
  | none => IO.println "the key covers the package's honest witnesses"; return 0
  | some fact => IO.println s!"cover fact fails: {fact}"; return 1
