import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution
import NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

open NightstreamFPrime
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

/-- Check every candidate at both the direct-plan and emitted-matrix boundary.
The comparison includes physical columns and coefficients. -/
def checkSamplerCandidateWiring : IO Unit := do
  let program := Lifecycle.Stage1.Poseidon2HashChainV1.program
  let width := PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth program
  let geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program width := ⟨Nat.le_refl _⟩
  let inputs := PiRLCRetainedInputs.first54Inputs
    (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry)
  let substitution := PiRLCSamplerOrdinaryMatrixSubstitution.substitution program
  let started ← IO.monoMsNow
  let mut entries := 0
  for slot in List.finRange PiRLCSamplerOrdinaryRetainedBlocks.logicalSlotCount do
    let (descriptor, position) := PiRLCSamplerOrdinaryRetainedBlocks.logicalDescriptor slot
    let location := PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor position
    let some form := substitution.form? width (Spartan.sourceToSpartan location.sourceColumn)
      | throw <| IO.userError s!"missing logical slot {slot.val}"
    entries := entries + form.entries.length
  let elapsed := (← IO.monoMsNow) - started
  IO.println s!"Sampler logical lookup: {elapsed} ms; {entries} entries."
  for source in List.finRange PiRLCFirst54DirectSchedule.sourceCount do
    for round in List.finRange PiRLCFirst54DirectSchedule.roundCount do
      let candidate : PiRLCFirst54DirectSchedule.Candidate := ⟨source, round⟩
      let descriptor := PiRLCSamplerDirectSemantics.candidateDescriptor candidate
      for (name, position, expected) in
          [("reject", PiRLCSamplerDirectSemantics.candidateRejectPosition candidate,
            inputs.reject candidate),
           ("symbol", PiRLCSamplerDirectSemantics.candidateSymbolPosition candidate,
            inputs.symbol candidate)] do
        let location := PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor position
        unless location.form geometry == expected do
          throw <| IO.userError s!"direct {name} differs at source {source.val}, candidate {round.val}"
        unless substitution.form? width (Spartan.sourceToSpartan location.sourceColumn) ==
            some expected do
          throw <| IO.userError s!"matrix {name} differs at source {source.val}, candidate {round.val}"
  IO.println "PiRLC candidate wiring: all 1088 reject bits and 1088 symbols match."

#eval checkSamplerCandidateWiring
