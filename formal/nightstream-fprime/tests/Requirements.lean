import NightstreamFPrime.Claims

/-! Requirements of `SPEC.md` as Lean types. Each example states a rule of `SPEC.md` with the
public definitions of the package, and it elaborates only against the claim or the definition
that it names. -/

namespace Requirements

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Spec.Compression
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth
open scoped NightstreamFPrime.Spec.Compression.Production

variable {logicalWidth : ℕ}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

-- SPEC.md Algorithm 10, step 5, in the light mode, with Algorithm 9, steps 2 and 3, and
-- Algorithm 11, step 1.
example (setup : FinalVerifier.Setup .light logicalWidth publicFits)
    (claims : FinalVerifier.Claims logicalWidth publicFits) :
    setup.Gate claims ↔
      (Rust.schedules.find? fun s =>
          decide ((setup.rows 0).matrices.columns / ringDegree ≤ s.blocks)).map
          Rust.Schedule.blocks = some setup.keyBlocks ∧
        (∃ l, Rust.layoutNew
            [⟨(setup.rows 0).matrices.columns / ringDegree,
                some (productionGlobalParams.accumulatorBound (claims.running 0).2)⟩,
              ⟨(setup.rows (claims.pc - 1)).matrices.columns / ringDegree, none⟩]
            setup.keyBlocks = .ok l ∧
          setup.minimum ≤ Rust.runBits (l.counts 320 4)) ∧
        320 ≤ ringDegree * ((setup.rows 0).matrices.columns / ringDegree) ∧
        320 ≤ ringDegree * ((setup.rows (claims.pc - 1)).matrices.columns / ringDegree) := by
  refine and_congr Iff.rfl (and_congr Iff.rfl ⟨fun cells => ⟨cells 0, cells 1⟩, ?_⟩)
  rintro ⟨first, fresh⟩ ⟨_ | _ | _, below⟩
  exacts [first, fresh, by simp only [FinalVerifier.Mode.slots] at below; omega]

-- SPEC.md Algorithm 10, step 5, with both slots.
example (setup : FinalVerifier.Setup .both logicalWidth publicFits)
    (claims : FinalVerifier.Claims logicalWidth publicFits) :
    setup.Gate claims ↔
      (Rust.schedules.find? fun s =>
          decide (max ((setup.rows 0).matrices.columns / ringDegree)
            ((setup.rows 1).matrices.columns / ringDegree) ≤ s.blocks)).map
          Rust.Schedule.blocks = some setup.keyBlocks ∧
        (∃ l, Rust.layoutNew
            [⟨(setup.rows 0).matrices.columns / ringDegree,
                some (productionGlobalParams.accumulatorBound (claims.running 0).2)⟩,
              ⟨(setup.rows 1).matrices.columns / ringDegree,
                some (productionGlobalParams.accumulatorBound (claims.running 1).2)⟩,
              ⟨(setup.rows (claims.pc - 1)).matrices.columns / ringDegree, none⟩]
            setup.keyBlocks = .ok l ∧
          setup.minimum ≤ Rust.runBits (l.counts 320 4)) ∧
        320 ≤ ringDegree * ((setup.rows 0).matrices.columns / ringDegree) ∧
        320 ≤ ringDegree * ((setup.rows 1).matrices.columns / ringDegree) ∧
        320 ≤ ringDegree * ((setup.rows (claims.pc - 1)).matrices.columns / ringDegree) := by
  refine and_congr Iff.rfl
    (and_congr Iff.rfl ⟨fun cells => ⟨cells 0, cells 1, cells 2⟩, ?_⟩)
  rintro ⟨first, second, fresh⟩ ⟨_ | _ | _ | _, below⟩
  exacts [first, second, fresh, by simp only [FinalVerifier.Mode.slots] at below; omega]

-- SPEC.md section 3 and Algorithm 10, step 5: in the light mode, `error_bits` is 119 at each
-- depth up to the cap.
example (blocks depth keyBlocks : ℕ) (l : Rust.Placed)
    (ok : Rust.layoutNew
      [⟨blocks, some (productionGlobalParams.accumulatorBound depth)⟩, ⟨blocks, none⟩]
      keyBlocks = .ok l)
    (legal : depth ≤ 64) :
    Rust.runBits (l.counts 320 4) = 119 :=
  @Claims.Compression.light_error_bits blocks depth keyBlocks l ok legal

-- SPEC.md Algorithm 10, steps 1 to 5, in the light mode: the verifier takes the key of
-- Algorithm 11, which has `2^19` blocks, and its security minimum is at most 119 bits.
example (light : FinalVerifier.Setup .light logicalWidth publicFits) (packageWidths : Fin 2 → ℕ)
    (step : AppState → AppWitness → AppState) (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    (key : (Rust.schedules.find? fun s =>
        decide ((light.rows 0).matrices.columns / ringDegree ≤ s.blocks)).map
        Rust.Schedule.blocks = some light.keyBlocks)
    (shipped : light.keyBlocks = 2 ^ 19) (minimum : light.minimum ≤ 119)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (verifies : RunCompleteness.Verifies light.ajtai light.circuitKeys step packageWidths cap
      control statement (.recursive payload))
    (pcOne : payload.pc = 1) (slotOne : payload.running 1 = (defaultSlot, 0))
    (commitment : PaperAlgebra.Commitment) :
    FinalVerifier.Checks .light statement ⟨payload.pc, payload.running, commitment⟩ ∧
      light.Gate ⟨payload.pc, payload.running, commitment⟩ :=
  @Claims.Final.light_gate_complete logicalWidth publicFits light packageWidths step cap control
    key shipped minimum statement payload verifies pcOne slotOne commitment

-- SPEC.md Algorithm 6 and section 1, at the shipped packages: the verifier accepts a statement
-- that no valid run of the program reaches with probability at most
-- `Σ_{j < depth} (finderChance(stage j) + error(stage j))`.
example {ajtai : Shipped.Ajtai} (facts : Shipped.Facts Shipped.program ajtai)
    (identity : List F) (relations : Fin 2 → Shipped.Relation)
    (lightKey : relations 0 = Shipped.lightRelation facts)
    (refreshKey : relations 1 = Shipped.refreshRelation facts)
    {Admitted : NifsAdversary
      (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {StageAdmitted : Stage (IvcWalk.Visit StepRelation.keyWidth StepRelation.keyPublicFits)
      (IvcWalk.Answer StepRelation.keyWidth StepRelation.keyPublicFits) → Prop}
    {Efficient : (adversary : NifsAdversary
        (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits)) →
      NifsExtractor (IvcWalk.Answer StepRelation.keyWidth StepRelation.keyPublicFits) adversary →
        Prop}
    {error : NifsAdversary
      (IvcWalk.WalkAttempt StepRelation.keyWidth StepRelation.keyPublicFits) → ℝ}
    (assumption : Assumption1 (IvcWalk.WalkSucceeds ajtai) (IvcWalk.WalkExtracted ajtai) Admitted
      Efficient error)
    (closed : Closed
      (IvcWalk.attemptOf ajtai (Shipped.keys identity relations)
        (Shipped.exports facts identity relations lightKey refreshKey))
      (IvcWalk.back ajtai (Shipped.keys identity relations)
        (Shipped.exports facts identity relations lightKey refreshKey))
      Admitted StageAdmitted Efficient)
    (adversary : IvcAdversary
      (IvcSecurity.Output StepRelation.keyWidth StepRelation.keyPublicFits))
    (admitted : StageAdmitted (Stage.start (IvcSecurity.walkAdversary adversary)))
    (packageWidths : Fin 2 → ℕ) (control : AppState → AppWitness → Fin 2) (cap depth : ℕ)
    (depthBound : ∀ tape ∈ adversary.tape.support,
      (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure {tape |
        RunCompleteness.Verifies ajtai (Shipped.keys identity relations) Shipped.program.step
            packageWidths cap control (adversary.output tape).1 (adversary.output tape).2 ∧
          ¬ IvcWalk.Reaches Shipped.program (adversary.output tape).1}).toReal ≤
      ∑ j : Fin depth,
        (IvcSecurity.finderChance ajtai (Shipped.keys identity relations)
            (Shipped.exports facts identity relations lightKey refreshKey)
            (reverseStages assumption closed (IvcSecurity.walkAdversary adversary) admitted j).1 +
          error ((reverseStages assumption closed (IvcSecurity.walkAdversary adversary) admitted
            j).1.nifs
            (IvcWalk.attemptOf ajtai (Shipped.keys identity relations)
              (Shipped.exports facts identity relations lightKey refreshKey))
            (IvcWalk.back ajtai (Shipped.keys identity relations)
              (Shipped.exports facts identity relations lightKey refreshKey)))) :=
  @Claims.Shipped.verify_sound ajtai facts identity relations lightKey refreshKey Admitted
    StageAdmitted Efficient error assumption closed adversary admitted packageWidths control cap
    depth depthBound

-- SPEC.md Algorithm 10, Source: the final verifier accepts a statement and the witnesses that
-- the reverse extractor returns from its first `depth` stages are not a valid history of it with
-- probability at most `q · (2^−119 + 38/q²) + Σ_{j < depth} (finderChance(stage j) +
-- error(stage j))`, with `q = 2^64 − 59` in `38/q²`.
example {mode : FinalVerifier.Mode} (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {step : AppState → AppWitness → AppState}
    {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : Stage (IvcWalk.Visit logicalWidth publicFits)
      (IvcWalk.Answer logicalWidth publicFits) → Prop}
    {Efficient : (adversary : NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
      NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
      (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
      (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (s : Game.Strategy (Option (Option (FinalVerifier.Outcome logicalWidth publicFits))))
    (q : ℕ)
    (admitted : StageAdmitted
      (Stage.start (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))))
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2,
      (setup.rows circuit).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key : Fin 2, (setup.circuitKeys key).digest.length = 4)
    (depth : ℕ)
    (depthBound : ∀ o : FinalVerifier.Outcome logicalWidth publicFits,
      some (some o) ∈ (Game.law s (setup.game.restore q [])).support →
        o.1.iteration ≤ depth) :
    ((reverseStages assumption closed (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q))
        admitted depth).1.tape.toOuterMeasure
        {tape | FinalHistory.AdviceFailure setup exports assumption closed s q admitted depth
          tape}).toReal ≤
      q * (1 / 2 ^ 119 + 38 / (((2 ^ 64 - 59 : ℕ) : ℝ)) ^ 2) +
        ∑ j : Fin depth,
          (IvcSecurity.finderChance setup.ajtai setup.circuitKeys exports
              (reverseStages assumption closed
                (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q)) admitted j).1 +
            error ((reverseStages assumption closed
                (IvcSecurity.walkAdversary (FinalHistory.adversary setup s q)) admitted j).1.nifs
              (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
              (IvcWalk.back setup.ajtai setup.circuitKeys exports))) :=
  @Claims.Final.history_knowledge logicalWidth publicFits mode setup relation step cubeFits exports
    Admitted StageAdmitted Efficient error assumption closed s q admitted complete rows keysShaped
    depth depthBound

end Requirements
