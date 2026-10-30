import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
import CompressionLean.Terminal.PiCCS
import CompressionLean.Terminal.PiRLC
import CompressionLean.Terminal.Outputs

namespace CompressionLean.Terminal

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem priorSpec (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    PriorStateHash.SpecHolds PilotProduction.priorInterface
      PilotProduction.witnessOffset
      (PilotSpartan.pullback (pilotEnv env)) := by
  have priorHolds := priorChain_holds env holds
  have priorHash := NightstreamFPrime.Export.Pilot.canonicalChainDigest_eq_hash
    Data.priorChain env (by rfl) priorHolds
  have priorInputs := NightstreamFPrime.Export.Stage1.Package.chainInputValues_lift
    PilotData.priorChain env (by
      norm_num [PilotData.priorChain,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount])
  have priorState := NightstreamFPrime.Export.Stage1.Package.chainOutputState_lift
    PilotData.priorChain env (by
      norm_num [PilotData.priorChain, PilotData.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
        NightstreamFPrime.Layout.PilotValues.stateHashWords,
        NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount]) (by
      norm_num [PilotData.priorChain, PilotData.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.priorWitnessStart,
        NightstreamFPrime.Layout.PilotValues.witnessPrivateStart,
        NightstreamFPrime.Layout.PilotValues.absorbCount,
        NightstreamFPrime.Layout.PilotValues.stateHashWords,
        NightstreamFPrime.Layout.PilotValues.stateHashBaseWords,
        NightstreamFPrime.Spec.Poseidon2.rate,
        NightstreamFPrime.Layout.Stage1.Spartan.pilotPrivateColumnCount])
  have pilotPriorHash :
      List.ofFn (fun lane : Fin 4 =>
        NightstreamFPrime.Export.Pilot.chainOutputState
          PilotData.priorChain PilotData.priorChain.absorbCount
          (pilotEnv env)
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
        NightstreamFPrime.Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.priorChain (pilotEnv env)) := by
    calc
      _ = List.ofFn (fun lane : Fin 4 =>
          NightstreamFPrime.Export.Pilot.chainOutputState
            (Data.liftPilotChain PilotData.priorChain)
            PilotData.priorChain.absorbCount env
            ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        rw [← priorState]
        rfl
      _ = NightstreamFPrime.Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            (Data.liftPilotChain PilotData.priorChain) env) := by
        simpa [Data.priorChain] using! priorHash
      _ = NightstreamFPrime.Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.priorChain (pilotEnv env)) := by
        rw [← priorInputs]
        rfl
  have rawSpec := NightstreamFPrime.Export.Pilot.priorRawSpec_of_chainHash
    (pilotEnv env) pilotPriorHash
  have physical := priorExtraRows_hold env holds
  rw [PilotData.priorExtraRows, PilotData.remapCompiledRows_toR1CS,
    Rows.compileRowsTR_toR1CS] at physical
  have sourceRows := (PilotSpartan.remapRows_hold (pilotEnv env)
    (Rows.lowerConstraintsTR (PilotData.priorExtraConstraints ())
      PilotValues.logicalColumnCount).rows).mp physical
  rw [Rows.lowerConstraintsTR_eq] at sourceRows
  have extraLogical := R1CS.lowerConstraints_sound
    (PilotSpartan.pullback (pilotEnv env))
    (PilotData.priorExtraConstraints ()) PilotValues.logicalColumnCount
    sourceRows
  have postHashRows : holdsFlat (PilotSpartan.pullback (pilotEnv env))
      (PriorStateHash.wordOps PilotProduction.priorInterface
          PilotProduction.witnessOffset ++
        PriorStateHash.bindingAssertions PilotProduction.priorInterface
          PilotProduction.witnessOffset) := by
    rw [PilotData.priorExtraConstraints,
      PilotProduction.fastPriorWordOps_eq] at extraLogical
    simpa only [holdsFlat] using extraLogical
  exact PriorStateHash.soundness_of_hash_and_postHash
    PilotProduction.priorInterface (PilotSpartan.pullback (pilotEnv env))
    PilotProduction.witnessOffset
    (PilotProduction.layoutAssumptions
      (PilotSpartan.pullback (pilotEnv env))).1
    rawSpec postHashRows

theorem samplerInitialState_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsPhase : NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.PhaseHolds
      relation ajtai
      (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.piCcsInterface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset env
      (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.proof relation env)) :
    PiRLC.v1_2.SamplerChain.evalInitialState
        (PiRLC.v1_2.Formal.samplerInterface
          (PiRLC.v1_2.Formal.atOffset
            (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
              (logicalWidth := logicalWidth) (publicFits := publicFits))
            NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset))
        (PiRLC.v1_2.Formal.samplerOffset
          NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset) env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.running
          logicalWidth publicFits env)
        (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.fresh
          logicalWidth publicFits env)
        (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.proof
          relation env)).outgoingState := by
  calc
    _ = NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.evalState
        env (NightstreamFPrime.Layout.Stage1.PiRLCInputs.piCcsOutputState
          (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
          rfl
    _ = NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.evalState
        env
        (NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.outputBindingFinalState
          relation
          (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.piCcsInterface
            logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) := by
          rw [NightstreamFPrime.Layout.Stage1.PiRLCInputs.piCcsOutputState_eq_parent
            relation]
          rfl
    _ = _ := piCcsPhase.outgoingState

theorem circuitPackage_implies_terminal
    {lw : Nat} {pf : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : AjtaiKey
      (logicalWidth := lw) (publicFits := pf))
    (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    let pulled := NightstreamFPrime.Layout.Stage1.Spartan.pullback env
    let running := NightstreamFPrime.Layout.Stage1.AccumulatorInputs.running
      lw pf pulled
    let fresh := NightstreamFPrime.Layout.Stage1.AccumulatorInputs.fresh
      lw pf pulled
    let proof := NightstreamFPrime.Layout.Stage1.AccumulatorInputs.proof
      relation pulled
    Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai)
        running fresh proof = true ∧
      (ProductionKey.key relation ajtai).parent running fresh proof =
        some (PiRLC.v1_2.Semantics.evalOutput relation
          (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
            (logicalWidth := lw) (publicFits := pf))
          NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset pulled) ∧
      PriorStateHash.SpecHolds PilotProduction.priorInterface
        PilotProduction.witnessOffset
        (PilotSpartan.pullback (pilotEnv env)) ∧
      PiRLC.v1_2.Sampler.evalState pulled samplerState =
        Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt
          ((ProductionKey.key relation ajtai).piCcsExecution running fresh
            proof).outgoingState PiRLC.v1_2.SamplerChain.sourceCount := by
  intro pulled running fresh proof
  have piCcsPhase :
      NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.PhaseHolds relation ajtai
        (NightstreamFPrime.Layout.Stage1.AccumulatorInputs.piCcsInterface
          lw pf)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset pulled proof := by
    simpa [PiCCSInvocations.parentInterface,
      NightstreamFPrime.Layout.Stage1.AccumulatorInputs.piCcsInterface] using
      piCcsPhaseHolds relation ajtai proof env holds
  have piRlcPhase := piRlcPhaseHolds relation ajtai env holds
  have challenges :=
    NightstreamFPrime.Layout.Stage1.AccumulatorSemantics.piRlcChallenges_eq_key
      relation ajtai pulled piCcsPhase piRlcPhase
  have output :=
    NightstreamFPrime.Layout.Stage1.AccumulatorSemantics.piRlcOutput_eq_keyParentForChallenges
      relation ajtai pulled piCcsPhase piRlcPhase
  have initialState := samplerInitialState_eq relation ajtai pulled piCcsPhase
  refine ⟨piCcsPhase.accepted, ?_, priorSpec env holds, ?_⟩
  · unfold Nifs.PaperNonInteractive.Key.parent
    rw [challenges, output]
    rfl
  · rw [← initialState]
    exact piRlcPhase.sampler.state

end CompressionLean.Terminal
