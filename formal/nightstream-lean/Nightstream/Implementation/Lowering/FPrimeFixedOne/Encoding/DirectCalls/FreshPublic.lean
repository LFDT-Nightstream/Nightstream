import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Footprints

/-!
Contract: exact `freshPublic` call recipe.

The recipe is the affine coordinate map stored in `DirectProfile`.  It emits
one row per encoded output coordinate, allocates no temporaries, and proves
exact `Vocabulary.callEval` correspondence.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

private def frameRecipe
    (parameters : Parameters)
    (profile : DirectProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference :
      Ref (typeSystem parameters) context (.data .fresh)}
    (frame :
      CallFrame profile.family .freshPublic
        (Refs.cons reference .nil)) :
    AffineMapRecipe profile.freshPublicMap
      reference.port.layout
      (Ports.auxiliaryEncoded parameters).layout where
  owner := frame.owner
  firstOrdinal := 0
  one := frame.one
  active := frame.active
  source := unaryOperand frame.operands
  output := unaryOutput frame.outputs
  sourceWidth := by
    simpa [DirectProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using
        frame.operandWidthsAgree.1
  targetWidth := by
    have width :=
      frame.outputWidthsAgree
        (Ports.auxiliaryEncoded parameters) (by simp)
    simpa [DirectProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using width

private theorem footprint_exact
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    (signature parameters).callFootprint .freshPublic =
      affineFootprint profile.codecs.encoded.width := by
  simpa [signature, callFootprint] using profile.freshPublicFootprint

/-- Certified physical recipe for the exact direct `freshPublic` call. -/
def recipe
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    CallRecipe (signature parameters) profile.family .freshPublic := by
  rw [footprint_exact parameters profile]
  refine
    { rows := ?_
      rowCount := ?_
      rowsOwned := ?_
      rowIdsNodup := ?_
      rowsSupported := ?_
      activeSoundness := ?_
      activeHonestCompleteness := ?_
      inactiveSatisfiable := ?_ }
  · intro context references frame
    cases references with
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).rows
  · intro context references frame
    cases references with
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).row_count
  · intro context references frame row member
    cases references with
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).rows_owned row member
  · intro context references frame
    cases references with
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).row_ids_nodup
  · intro context references frame row member column columnMember
    cases references with
    | cons reference tail =>
        cases tail
        rcases
            (frameRecipe parameters profile frame).rows_supported
              row member column columnMember with
          one | active | source | output
        · subst column
          simp [CallFrame.visibleIds]
        · subst column
          simp [CallFrame.visibleIds]
        · have operandMember : column ∈ frame.operands.ids := by
            simpa [frameRecipe, unaryOperand, RefBundles.ids_cons] using source
          have contextMember :=
            RefBundles.fromSchema_ids_subset
              (Refs.cons reference .nil) frame.contextBundles
              column operandMember
          simp [CallFrame.visibleIds, contextMember]
        · have outputMember : column ∈ frame.outputs.ids := by
            simpa [frameRecipe, unaryOutput, SchemaBundles.ids,
              SchemaBundles.columns, SchemaBundles.portColumns,
              ColumnBundle.ids] using output
          simp [CallFrame.visibleIds, outputMember]
  · intro context references frame assignment inputs
      constantOne activeOne decoded holds
    cases references with
    | cons reference tail =>
        cases tail
        cases inputs with
        | cons fresh inputs =>
            cases inputs
            have sourceDecoded :
                profile.codecs.fresh.decode
                    ((unaryOperand frame.operands).values assignment) =
                  some fresh := by
              simpa [DirectProfile.family, Encoding.Profile.family,
                DataCodecs.family, Family.codecFor] using decoded.1
            have outputDecoded :=
              (frameRecipe parameters profile frame).active_sound
                assignment fresh constantOne activeOne sourceDecoded holds
            refine ⟨.cons (parameters.machine.freshPublic fresh) .nil,
              rfl, ?_⟩
            exact ⟨by
              simpa [DirectProfile.family, Encoding.Profile.family,
                DataCodecs.family, Family.codecFor] using outputDecoded,
              trivial⟩
  · intro context references frame assignment inputs outputs
      constantOne activeOne inputsEncoded outputsEncoded evaluated
    cases references with
    | cons reference tail =>
        cases tail
        cases inputs with
        | cons fresh inputs =>
            cases inputs
            cases outputs with
            | cons output outputs =>
                cases outputs
                have outputEqual :
                    output = parameters.machine.freshPublic fresh := by
                  simpa [callEval] using Option.some.inj evaluated.symm
                subst output
                have sourceCoordinates :
                    (unaryOperand frame.operands).values assignment =
                      profile.codecs.fresh.encode fresh := by
                  simpa [DirectProfile.family, Encoding.Profile.family,
                    DataCodecs.family, Family.codecFor] using
                      inputsEncoded.1.2
                have outputCoordinates :
                    (unaryOutput frame.outputs).values assignment =
                      profile.codecs.encoded.encode
                        (parameters.machine.freshPublic fresh) := by
                  simpa [DirectProfile.family, Encoding.Profile.family,
                    DataCodecs.family, Family.codecFor] using
                      outputsEncoded.1.2
                have sourceAdmissible :
                    profile.codecs.fresh.Admissible fresh := by
                  simpa [DirectProfile.family, Encoding.Profile.family,
                    DataCodecs.family, Family.codecFor] using
                      inputsEncoded.1.1
                refine ⟨assignment, ?_, ?_, ?_⟩
                · intro id member
                  rfl
                · intro id member
                  rfl
                · exact
                    (frameRecipe parameters profile frame).active_complete
                      assignment fresh constantOne activeOne
                      sourceCoordinates outputCoordinates sourceAdmissible
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons reference tail =>
        cases tail
        refine ⟨assignment, ?_, ?_, ?_⟩
        · intro id member
          rfl
        · intro id member
          rfl
        · exact
            (frameRecipe parameters profile frame).inactive_complete
              assignment activeZero

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
