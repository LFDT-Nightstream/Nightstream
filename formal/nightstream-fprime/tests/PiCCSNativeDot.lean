import NightstreamFPrime.Export.Stage1.PiCCSNativeDot

/-! Executable checks of the native dot loop against the original arithmetic.
The reference is expanded here so the dot-product compiler rewrite cannot
replace both sides of the comparison. Cases cover word boundaries and all
real/imaginary boundary pairs at empty, single-lane and production lengths. -/

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Export.Stage1

def main : IO Unit := do
  let words : Array Nat := #[0, 1, 7, 2^32 - 1, 2^32, 2^63 - 1, 2^63,
    goldilocksModulus - 2, goldilocksModulus - 1]
  let fields := words.map Poseidon2.ofNat
  let inputs : Array K := fields.flatMap fun real => fields.map fun imaginary => ⟨real, imaginary⟩
  let mut checked := 0
  for lanes in [0, 1, ringDegree] do
    for left in [:inputs.size] do
      for right in [:inputs.size] do
        let prepared := Vector.ofFn fun index : Fin lanes =>
          inputs[(left + index.val) % inputs.size]?.getD K.zero
        let source := fun index : Fin lanes => inputs[(right + index.val) % inputs.size]?.getD K.zero
        let expected := FiniteSumAlgebra.sumMap extensionOps (canonicalFinIndices lanes)
          fun index => extensionOps.mul (prepared.get index) (source index)
        let actual := PiCCSNativeDot.dotK prepared source
        unless actual == expected do
          throw (IO.userError s!"native dot mismatch: lanes={lanes} left={left} right={right}")
        checked := checked + 1
  IO.println s!"native dot: {checked} complete boundary comparisons passed"
