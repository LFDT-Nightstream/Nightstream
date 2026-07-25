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
      CallFrame (signature := signature parameters)
        profile.family Call.freshPublic
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
        (Ports.auxiliaryEncoded parameters) (by
          change Ports.auxiliaryEncoded parameters ∈
            callOutputs parameters Call.freshPublic
          exact List.mem_cons_self)
    unfold PortWidthAgrees at width
    simpa [DirectProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor, Ports.auxiliaryEncoded,
      dataPort, auxiliaryLayout, ownedLayout,
      profile.toProfile.encoded_width_eq_codec parameters] using width

private theorem footprint_exact
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    (signature parameters).callFootprint Call.freshPublic =
      affineFootprint profile.codecs.encoded.width := by
  simpa [signature, callFootprint] using profile.freshPublicFootprint

/-- Certified physical recipe for the exact direct `freshPublic` call. -/
def freshPublicRecipe
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    CallRecipe (signature parameters) profile.family Call.freshPublic := by
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
        rw [footprint_exact parameters profile]
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
          simp [frameRecipe, CallFrame.visibleIds]
        · subst column
          simp [frameRecipe, CallFrame.visibleIds]
        · have operandMember : column ∈ frame.operands.ids := by
            simpa only [frameRecipe, unaryOperand_ids] using source
          have contextMember :=
            RefBundles.fromSchema_ids_subset
              (Refs.cons reference .nil) frame.contextBundles
              column operandMember
          simp [CallFrame.visibleIds, contextMember]
        · have outputMember : column ∈ frame.outputs.ids := by
            simpa only [frameRecipe, unaryOutput_ids] using output
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
              have headDecoded :=
                (unaryOperand_decodes_iff
                  profile.family assignment frame.operands fresh).mp decoded
              simpa [DirectProfile.family, Encoding.Profile.family,
                DataCodecs.family, Family.codecFor] using headDecoded
            have outputDecoded :=
              (frameRecipe parameters profile frame).active_sound
                assignment fresh constantOne activeOne sourceDecoded holds
            refine ⟨.cons (parameters.machine.freshPublic fresh) .nil,
              rfl, ?_⟩
            apply (unaryOutput_decodes_iff
              profile.family assignment frame.outputs
              (parameters.machine.freshPublic fresh)).mpr
            simpa [DirectProfile.family, Encoding.Profile.family,
              DataCodecs.family, Family.codecFor] using outputDecoded
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
                  exact congrArg HVec.head
                    (Option.some.inj evaluated.symm)
                subst output
                have sourceCoordinates :
                    (unaryOperand frame.operands).values assignment =
                      profile.codecs.fresh.encode fresh := by
                  have headEncoded :=
                    (unaryOperand_encodes_iff
                      profile.family assignment frame.operands fresh).mp
                      inputsEncoded
                  simpa [DirectProfile.family, Encoding.Profile.family,
                    DataCodecs.family, Family.codecFor] using
                      headEncoded.2
                have outputCoordinates :
                    (unaryOutput frame.outputs).values assignment =
                      profile.codecs.encoded.encode
                        (parameters.machine.freshPublic fresh) := by
                  have headEncoded :=
                    (unaryOutput_encodes_iff
                      profile.family assignment frame.outputs
                      (parameters.machine.freshPublic fresh)).mp
                      outputsEncoded
                  simpa [DirectProfile.family, Encoding.Profile.family,
                    DataCodecs.family, Family.codecFor] using
                      headEncoded.2
                have sourceAdmissible :
                    profile.codecs.fresh.Admissible fresh := by
                  have headEncoded :=
                    (unaryOperand_encodes_iff
                      profile.family assignment frame.operands fresh).mp
                      inputsEncoded
                  simpa [DirectProfile.family, Encoding.Profile.family,
                    DataCodecs.family, Family.codecFor] using
                      headEncoded.1
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
