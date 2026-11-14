import NightstreamFPrime.Export.BoundedDepth.FinalVerifier
import NightstreamFPrime.Export.BoundedDepth.IvcSecurity
import NightstreamFPrime.Spec.Compression.Game.Law

/-! The final verifier against a false history (SPEC.md, Algorithm 10; `SECURITY_MODEL.md`,
adversaries 4 and 6). A false history is a result of the state-restoration experiment in which the
final verifier accepts `(i, z0, zi)` and no `i` valid steps of the application, each with a
witness of the application's word count, lead from `z0` to `zi` (`FalseHistory`).

A prover of the final verifier with `q` queries is an IVC adversary (`adversary`). Its tape is the
experiment's result under the law of the experiment's draws (`Spec.Compression.Game.law`). Its
output is the accepted statement with the proof from `W` (`FinalVerifier.Setup.proofOf`). A false
history is a result in which `Verifies` rejects that proof, or a false acceptance of this
adversary. So its probability is at most `q · finalError` plus the bound of HyperNova Lemma 17 for
this adversary (`soundness`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalHistory

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Spec.Compression.Game (Strategy prob restore law prob_eq_law prob_mono
  restore_sound)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.IvcWalk
open NightstreamFPrime.Export.BoundedDepth.IvcSecurity
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  (setup : Setup logicalWidth publicFits)

/-- The IVC adversary's output for a result: an accepted statement with the proof from `W`, and
otherwise the bottom proof of `(0, [], [])`. -/
def outputOf : Option (Option (Outcome logicalWidth publicFits)) → Output logicalWidth publicFits
  | some (some o) => (o.1, setup.proofOf o)
  | _ => (⟨0, [], []⟩, .bottom)

/-- The prover `s` of the final verifier in the state-restoration experiment with `q` queries, as
an IVC adversary. -/
noncomputable def adversary (s : Strategy (Option (Option (Outcome logicalWidth publicFits))))
    (q : ℕ) : IvcAdversary (Output logicalWidth publicFits) :=
  ⟨_, law s (restore setup.game q []), outputOf setup⟩

/-- The final verifier accepts `(i, z0, zi)`, and no `i` valid steps of `program`, each with a
witness of the program's word count, lead from `z0` to `zi`. -/
def FalseHistory (program : Stage1.Application.Program)
    (r : Option (Option (Outcome logicalWidth publicFits))) : Prop :=
  ∃ o, r = some (some o) ∧ ¬ Reaches program o.1

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation setup.ajtai setup.circuitKeys step cubeFits)
  {Admitted : NifsAdversary (WalkAttempt logicalWidth publicFits) → Prop}
  {StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary : NifsAdversary (WalkAttempt logicalWidth publicFits)) →
    NifsExtractor (Answer logicalWidth publicFits) adversary → Prop}
  {error : NifsAdversary (WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : Assumption1 (WalkSucceeds setup.ajtai) (WalkExtracted setup.ajtai) Admitted
    Efficient error)
  (closed : Closed (attemptOf setup.ajtai setup.circuitKeys exports)
    (back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
  (s : Strategy (Option (Option (Outcome logicalWidth publicFits)))) (q : ℕ)
  (admitted : StageAdmitted (Stage.start (walkAdversary (adversary setup s q))))

local notation "stages" => reverseStages assumption closed (walkAdversary (adversary setup s q))
  admitted

/-- In the state-restoration experiment with `q` queries, the probability of a false history is at
most `q · (1/2^119 + 38/q²)` plus, for each of the first `depth` stages of the reverse extractor
of the prover's IVC adversary, the probability of a collision at the stage's visit and the stage's
error. -/
theorem soundness {packageWidths : Fin 2 → ℕ} (premises : Premises setup packageWidths)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4) (depth : ℕ)
    (depthBound : ∀ o, some (some o) ∈ (law s (restore setup.game q [])).support →
      o.1.iteration ≤ depth) :
    prob s (restore setup.game q []) (FalseHistory exports.program) ≤
      q * finalError + ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | WalkCollision setup.ajtai setup.circuitKeys exports
              (visitAfter (back setup.ajtai setup.circuitKeys exports)
                ((stages j.val).1.input tape) ((stages j.val).1.results tape))}).toReal +
          error ((stages j.val).1.nifs (attemptOf setup.ajtai setup.circuitKeys exports)
            (back setup.ajtai setup.circuitKeys exports))) := by
  classical
  let Rejected : Option (Option (Outcome logicalWidth publicFits)) → Prop := fun r =>
    ¬ ∀ a, r = some a → ∀ o, a = some o →
      Verifies setup.ajtai setup.circuitKeys step packageWidths 0 (fun _ _ => 0) o.1
        (setup.proofOf o)
  let Accepted : Option (Option (Outcome logicalWidth publicFits)) → Prop := fun r =>
    FalseAcceptance setup.ajtai setup.circuitKeys exports (fun _ _ => 0) 0 (outputOf setup r)
  have cover (r : Option (Option (Outcome logicalWidth publicFits)))
      (falseHistory : FalseHistory exports.program r) : Rejected r ∨ Accepted r := by
    obtain ⟨o, rfl, unreached⟩ := falseHistory
    by_cases verifies : Verifies setup.ajtai setup.circuitKeys step packageWidths 0 (fun _ _ => 0)
      o.1 (setup.proofOf o)
    · exact Or.inr ⟨holds_of_verifies verifies, unreached⟩
    · exact Or.inl fun all => verifies (all _ rfl o rfl)
  have rejected : prob s (restore setup.game q []) Rejected ≤ q * finalError :=
    restore_sound _ s _ _ (game_avoids setup step 0 (fun _ _ => 0) premises)
      (game_densityLe setup premises) (by unfold finalError; positivity) q
  have accepted := probability_bound setup.ajtai setup.circuitKeys exports assumption closed
    (adversary setup s q) admitted keysShaped (fun _ _ => 0) 0 depth (by
      rintro (_ | _ | o) inside
      · exact Nat.zero_le depth
      · exact Nat.zero_le depth
      · exact depthBound o inside)
  calc prob s (restore setup.game q []) (FalseHistory exports.program)
      ≤ prob s (restore setup.game q []) (fun r => Rejected r ∨ Accepted r) :=
        prob_mono s _ cover
    _ ≤ prob s (restore setup.game q []) Rejected + prob s (restore setup.game q []) Accepted := by
        rw [prob_eq_law, prob_eq_law, prob_eq_law, Set.ofPred_or,
          ← ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _)]
        exact ENNReal.toReal_mono (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩)
          (MeasureTheory.measure_union_le _ _)
    _ ≤ _ := add_le_add rejected (by rw [prob_eq_law]; exact accepted)

/-- With Akita openings, the probability `history` of a false history is at most
`q · (1/2^119 + 38/q²) + εAkita` plus the sum of `soundness`. `akita` is Akita's knowledge
soundness (`FinalVerifier.AkitaKnowledge`) for the event `FalseHistory`: `s` is the prover of the
game with ideal openings that it gives for the prover with Akita openings. -/
theorem real_soundness {packageWidths : Fin 2 → ℕ} (premises : Premises setup packageWidths)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4) (depth : ℕ)
    (depthBound : ∀ o, some (some o) ∈ (law s (restore setup.game q [])).support →
      o.1.iteration ≤ depth) (history εAkita : ℝ)
    (akita : history ≤ prob s (restore setup.game q []) (FalseHistory exports.program) + εAkita) :
    history ≤ q * finalError + εAkita + ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | WalkCollision setup.ajtai setup.circuitKeys exports
              (visitAfter (back setup.ajtai setup.circuitKeys exports)
                ((stages j.val).1.input tape) ((stages j.val).1.results tape))}).toReal +
          error ((stages j.val).1.nifs (attemptOf setup.ajtai setup.circuitKeys exports)
            (back setup.ajtai setup.circuitKeys exports))) := by
  linarith [soundness setup exports assumption closed s q admitted premises keysShaped depth
    depthBound]

end NightstreamFPrime.Export.BoundedDepth.FinalHistory
