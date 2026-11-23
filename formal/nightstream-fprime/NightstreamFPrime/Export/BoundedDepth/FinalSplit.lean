import NightstreamFPrime.Export.BoundedDepth.FinalVerifier
import NightstreamFPrime.Spec.Compression.Akita.Extract.FreshGame
import NightstreamFPrime.Spec.Compression.Akita.Extract.SplitGame

/-! The game of the final verifier in two parts, with the Akita openings as data (SPEC.md,
Algorithm 10; Akita, ePrint 2026/1983, revision of 2026-11-23, Definition 10.16 (vii)).

`Setup.game` tests the three openings against the tables that the prover names. Here the game
after `W` is the stages of two checkpoints, and the test of the openings is outside them.

- Checkpoint 1 has the candidate `W`. Its stages are Π_CCS on the fresh instance and the draw of
  the lookup challenges `τ`: `Verifier.led` on `Setup.freshRun`. `Setup.lead` is the run after
  the checks of steps 1 to 5, for a first message that has the claims.
- Checkpoint 2 has the candidate `(W, V)`. Its stages are `Verifier.back` for the output claim
  of the run. `Setup.rest` is the message `V`, these stages and the test of the openings
  (`Setup.verdict`).
- `afterW_eq` and `game_eq`: `Setup.game` is these parts (`Setup.split`).
- `front_avoids`: for a table `W` whose proof `Setup.envelope` the verifier of Algorithm 6
  rejects, on each path of checkpoint 1 that avoids the bad sets no table `V` is without error.
  `back_avoids`: on each path of checkpoint 2 that avoids the bad sets, true opening claims give
  no error for `(W, V)`.
- `lead_staged`, `front_staged`: each draw has its bound of `Fresh.costs` and `tauError`.
  `costs_sum`: with the bounds of `Verifier.backCosts` they add to `Setup.runCost`, the sum of
  the bounds of the draws of one run. `lead_bare`, `front_bare`: the verifier does not read `W`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalVerifier

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.Compression.Verifier NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Game NightstreamFPrime.Spec.Compression.Claim
open NightstreamFPrime.Spec.Compression.Akita.Extract (Fair StagedTo Staged bare)
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth.CompressionSlot
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness NightstreamFPrime.Export.BoundedDepth.StepExtend

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode}

namespace Setup

variable (setup : Setup mode logicalWidth publicFits)

/-- The recursive proof with witnesses that `W` gives for the claims: the proof of
`Setup.proofOf`. -/
def envelope (statement : TerminalStatement AppState) (claims : Claims logicalWidth publicFits)
    (W : ℕ → F) :
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  .recursive ⟨claims.running, setup.runningWitness claims W, setup.fresh statement claims,
    setup.witness claims W (Fin.last _), claims.pc⟩

/-- Π_CCS on the fresh instance after the checks of steps 1 to 5 (Algorithm 10): the stages of
the final verifier before the lookup challenges, for a first message with the claims. -/
noncomputable def lead (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) :
    Compression.Game (Option (Probe K CompressionFresh.oneShape)) := by
  classical
  exact if Checks mode statement claims ∧ setup.Gate claims then setup.freshRun statement claims W
    else .pure none

/-- The result of the final verifier for a result of `Verifier.back`: the test of the ideal
openings. -/
noncomputable def verdict (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W V : ℕ → F)
    (probe : Probe K CompressionFresh.oneShape) (result : Option (Finals K)) :
    Option (Outcome logicalWidth publicFits) := by
  classical
  exact result.bind fun opening =>
    if Opened (keyCells (setup.params claims)) (fun x => algebraMap F K (W x))
        (fun x => algebraMap F K (V x)) opening then some (statement, some (claims, W, probe))
    else none

/-- The final verifier after the lookup challenges: the message `V`, `Verifier.back` for the
output claim of the run, and the test of the openings. -/
noncomputable def rest (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) :
    Option (Probe K CompressionFresh.oneShape × (Fin (setup.params claims).lookups → F)) →
      Compression.Game (Option (Outcome logicalWidth publicFits))
  | none => .pure none
  | some (probe, τ) => .recv (ℕ → F) fun V =>
      Compression.Game.bind
        (back (setup.verifier claims) (setup.claimsOf statement claims probe) W V τ)
        fun result => .pure (setup.verdict statement claims W V probe result)

/-- The game after `W` is the stages of checkpoint 1 and then `rest`. -/
theorem afterW_eq (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) :
    setup.afterW statement claims W =
      Compression.Game.bind (led (setup.params claims) (setup.freshRun statement claims W) W)
        (setup.rest statement claims W) := by
  unfold Setup.afterW led
  rw [Compression.Game.bind_assoc]
  refine congrArg (Compression.Game.bind _) (funext fun out => ?_)
  rcases out with _ | probe
  · rfl
  · show Compression.Game.bind (afterTables _ _ W) _ = _
    rw [afterTables_eq]
    refine congrArg (Compression.Game.draw _ _) (funext fun τ =>
      congrArg (Compression.Game.recv _) (funext fun V => ?_))
    show Compression.Game.bind (Compression.Game.bind _ _) _ = Compression.Game.bind _ _
    rw [Compression.Game.bind_assoc]
    refine congrArg (Compression.Game.bind _) (funext fun result => ?_)
    show Compression.Game.pure _ = Compression.Game.pure _
    rcases result with _ | opening
    · rfl
    · by_cases opened : Opened (keyCells (setup.params claims)) (fun x => algebraMap F K (W x))
          (fun x => algebraMap F K (V x)) opening
      · simp [Compression.Verifier.verdict, Setup.verdict, opened]
      · simp [Compression.Verifier.verdict, Setup.verdict, opened]

/-- The final verifier with the game after `W` in two parts. -/
noncomputable def split : Compression.Game (Option (Outcome logicalWidth publicFits)) := by
  classical
  exact .recv (TerminalStatement AppState × Option (Claims logicalWidth publicFits)) fun input =>
    match input.2 with
    | none => .pure (if Terminal.StatementValid input.1 ∧ input.1.iteration = 0 ∧
        input.1.zi = input.1.z0 then some (input.1, none) else none)
    | some claims =>
        if Checks mode input.1 claims ∧ setup.Gate claims then
          .recv (ℕ → F) fun W =>
            Compression.Game.bind
              (led (setup.params claims) (setup.freshRun input.1 claims W) W)
              (setup.rest input.1 claims W)
        else .pure none

theorem game_eq : setup.game = setup.split := by
  unfold Setup.game Setup.split
  refine congrArg (Compression.Game.recv _) (funext fun input => ?_)
  rcases input with ⟨statement, _ | claims⟩
  · rfl
  · simp only [afterW_eq]

/-- The sum of the bounds of the draws of one run of the final verifier for the claims: Π_CCS on
the fresh instance and the argument. -/
noncomputable def runCost (claims : Claims logicalWidth publicFits) : ℝ :=
  (Compression.Fresh.costs CompressionFresh.oneShape 8).sum +
    Compression.Verifier.runCost (setup.verifier claims)

theorem costs_sum (claims : Claims logicalWidth publicFits) :
    (Compression.Fresh.costs CompressionFresh.oneShape 8 ++ [tauError (setup.params claims)]).sum +
      (backCosts (setup.verifier claims)).sum = setup.runCost claims := by
  rw [List.sum_append, List.sum_singleton, add_assoc, backCosts_sum]
  rfl

end Setup

variable (setup : Setup mode logicalWidth publicFits) {packageWidths : Fin 2 → ℕ}
  (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)

/-- On each path of the run that avoids its bad sets, an accepted run whose output claims hold
gives the fresh instance under `Spec.CCS.Holds`. -/
theorem freshRun_avoids (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) :
    (setup.freshRun statement claims W).Avoids fun out => ∀ probe, out = some probe →
      AmbientOutputHolds extensionOps K.embed (openingMaps setup.ajtai) productionGlobalParams
          (setup.piCCS statement claims) probe
          (CompressionFresh.witnessOf (setup.witness claims W (Fin.last _))) →
        CCS.Holds (semantics setup.ajtai) productionGlobalParams
          (freshStatement (setup.circuitKeys (freshIndex claims)).relation
            (setup.fresh statement claims)) (setup.witness claims W (Fin.last _)) := by
  unfold Setup.freshRun
  exact CompressionFresh.game_avoids setup.ajtai (setup.circuitKeys (freshIndex claims)).relation
    (setup.fresh statement claims) (setup.witness claims W (Fin.last _))

/-- For a table `W` whose proof the verifier of Algorithm 6 rejects, on each path of `lead` that
avoids the bad sets `W` does not satisfy `Relation` for the output claim of the run. -/
theorem lead_sound (premises : Premises setup packageWidths)
    {statement : TerminalStatement AppState} {claims : Claims logicalWidth publicFits} {W : ℕ → F}
    (refuted : ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control statement
      (setup.envelope statement claims W)) :
    (setup.lead statement claims W).Avoids fun result => ∀ probe, result = some probe →
      ¬ Relation (setup.params claims) (setup.claimsOf statement claims probe) W := by
  unfold Setup.lead
  split_ifs with accepted
  · exact (freshRun_avoids setup statement claims W).mono fun _ fresh probe equal holds =>
      refuted (verifies_of_run setup step cap control premises accepted.1 accepted.2.laid holds
        (fresh probe equal))
  · exact fun _ equal => nomatch equal

/-- The facts of laid claims that the stages of the argument use: the hypotheses of
`Verifier.back_avoids` and `Verifier.front_avoids`. -/
theorem laid_facts {claims : Claims logicalWidth publicFits} (laid : setup.Laid claims) :
    WellFormed (setup.params claims) ∧ CountsNonzero (setup.params claims) ∧
      MatrixSpec (setup.verifier claims) ∧
      (setup.params claims).lookups ≤ 2 ^ (setup.verifier claims).columnVars ∧
      (setup.params claims).rows ≤ 2 ^ (setup.params claims).rowVars ∧
      ∀ c, ((setup.params claims).shape c).witness.vars - (setup.params claims).laneVars ≤
        (setup.params claims).keyBlockVars := by
  obtain ⟨wf, small, key_fits, -⟩ := setup.layout_facts laid
  exact ⟨wf, params_countsNonzero _ _ _ _ fun c _ => small c,
    verifierData_matrixSpec _ _ _ _ _ _, le_of_eq rfl, Nat.le_of_ble_eq_true rfl, key_fits⟩

/-- Akita Definition 10.16 (vii)(a) for checkpoint 1: for a table `W` whose proof the verifier
of Algorithm 6 rejects, on each path of Π_CCS and the draw of `τ` that avoids the bad sets, no
table `V` is without error for the output claim of the run. -/
theorem front_avoids (premises : Premises setup packageWidths)
    {statement : TerminalStatement AppState} {claims : Claims logicalWidth publicFits}
    (laid : setup.Laid claims) {W : ℕ → F}
    (refuted : ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control statement
      (setup.envelope statement claims W)) :
    (led (setup.params claims) (setup.lead statement claims W) W).Avoids fun result =>
      ∀ probe τ, result = some (probe, τ) → ¬ (Apart (setup.params claims) τ ∧
        ∃ V, ¬ Fails (setup.params claims) (setup.claimsOf statement claims probe) W V τ) :=
  led_avoids (laid_facts setup laid).2.1 _ W (lead_sound setup step cap control premises refuted)

/-- Akita Definition 10.16 (vii)(a) for checkpoint 2: on each path of `Verifier.back` that
avoids the bad sets, opening claims that are true for the key polynomial, `W` and `V` give
lookup challenges that are no table values and no error for `(W, V)`. -/
theorem back_avoids {claims : Claims logicalWidth publicFits} (laid : setup.Laid claims)
    (st : Compression.Verifier.Statement (setup.params claims)) (W V : ℕ → F)
    (τ : Fin (setup.params claims).lookups → F) :
    (back (setup.verifier claims) st W V τ).Avoids fun result => ∀ opening,
      result = some opening →
      Opened (keyCells (setup.params claims)) (fun x => algebraMap F K (W x))
        (fun x => algebraMap F K (V x)) opening →
      Apart (setup.params claims) τ ∧ ¬ Fails (setup.params claims) st W V τ :=
  Compression.Verifier.back_avoids _ (laid_facts setup laid).1 (laid_facts setup laid).2.2.1
    (laid_facts setup laid).2.2.2.1 (laid_facts setup laid).2.2.2.2.1
    (laid_facts setup laid).2.2.2.2.2 st W V τ

section Stages

variable {Reply : Type} [Fintype Reply]

/-- Each draw of `lead` has its bound of `Fresh.costs`, and a path that ends in a probe has all
the draws: the bounds `more` of later draws remain. `vector` and `scalar`: an answer of the
oracle gives a uniform `α` and a uniform element of `K`. -/
theorem lead_staged
    (vector : ∃ decode : Reply → Fin CompressionFresh.oneShape.cubeVariables → K, Fair decode)
    (scalar : ∃ decode : Reply → K, Fair decode) (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) (more : List ℝ) :
    StagedTo Reply (setup.lead statement claims W)
      (Compression.Fresh.costs CompressionFresh.oneShape 8 ++ more) fun result rest =>
        ∀ probe, result = some probe → rest = more := by
  unfold Setup.lead
  split_ifs
  · unfold Setup.freshRun
    exact (Compression.Fresh.game_staged _ _ 8 vector scalar (CompressionFresh.degree_le _ _ _)
      more).mono fun _ _ same _ _ => same
  · exact fun _ equal => nomatch equal

/-- Each draw of checkpoint 1 has its bound of `Fresh.costs` and `tauError`. `lookups`: an
answer of the oracle gives uniform lookup challenges. -/
theorem front_staged
    (vector : ∃ decode : Reply → Fin CompressionFresh.oneShape.cubeVariables → K, Fair decode)
    (scalar : ∃ decode : Reply → K, Fair decode) (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits)
    (lookups : ∃ decode : Reply → Fin (setup.params claims).lookups → F, Fair decode)
    (W : ℕ → F) :
    Staged Reply (led (setup.params claims) (setup.lead statement claims W) W)
      (Compression.Fresh.costs CompressionFresh.oneShape 8 ++ [tauError (setup.params claims)]) :=
  led_staged lookups W (lead_staged setup vector scalar statement claims W _)

end Stages

/-- The verifier of `lead` is the same for each table `W`. -/
theorem lead_bare (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W other : ℕ → F) :
    bare (setup.lead statement claims W) = bare (setup.lead statement claims other) := by
  unfold Setup.lead
  split_ifs
  · unfold Setup.freshRun
    exact Compression.Fresh.game_bare _ _ _ _
  · rfl

/-- The verifier of checkpoint 1 is the same for each table `W`. -/
theorem front_bare (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W other : ℕ → F) :
    bare (led (setup.params claims) (setup.lead statement claims W) W) =
      bare (led (setup.params claims) (setup.lead statement claims other) other) :=
  led_bare (lead_bare setup statement claims W other) W other

end NightstreamFPrime.Export.BoundedDepth.FinalVerifier
