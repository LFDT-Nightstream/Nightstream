import NightstreamFPrime.Export.BoundedDepth.RunCompleteness
import NightstreamFPrime.Lifecycle.BoundedDepth.CompressionFresh

/-! The final verifier as one public-coin game (SPEC.md, Algorithm 10), and its soundness against
the verifier of Algorithm 6 (`Export.BoundedDepth.RunCompleteness.Verifies`).

The prover names the statement `(i, z0, zi)` and the values of the final proof that come before
`W`: no steps, or `pc`, each slot's claim and depth and the fresh commitment (`Claims`). The
verifier runs the checks of steps 1 to 4 (`Checks`) and computes the fresh public input
`x = encHash(stateHash(…))` from these values. The prover names `W`. Π_CCS on the fresh instance
runs (`Spec.Compression.Fresh.game`), and then the argument on the slot claims and the output
claim of the run (`Spec.Compression.Verifier.afterTables`). The Akita openings are ideal: the
prover names the tables.

On each path that avoids the bad sets, `Verifies` accepts an accepted statement with the proof
from `W` (`Setup.proofOf`, `game_avoids`), and each bad set has density at most
`1/2^119 + 38/q²` (`game_densityLe`). So in the state-restoration experiment with `q` queries, the
verifier accepts a statement without such a proof with probability at most
`q · (1/2^119 + 38/q²)` (`soundness`). `AkitaKnowledge` names Akita's knowledge soundness with the
error `εAkita` as an assumption, and `real_soundness` adds `εAkita` for the verifier with Akita
openings. -/

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
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth.CompressionSlot
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness NightstreamFPrime.Export.BoundedDepth.StepExtend
open NightstreamFPrime.Spec.Folding.Schedule (productionLightFoldCap)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- The values of a final proof that the prover names before `W` (SPEC.md section 4.5). -/
structure Claims (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  pc : ℕ
  running : Fin 2 → SlotData logicalWidth publicFits
  commitment : PaperAlgebra.Commitment

/-- What the verifier holds: the Ajtai key that the compression key commits, the circuit keys,
the dense rows of each circuit's relation (Rust `FactoredRows`, by circuit), the layout of Rust
`Layout::new`, which Lean does not model, and the zeros of the opening point of `V`. -/
structure Setup (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)
  circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits
  rows : ℕ → Factored F
  layout : Claims logicalWidth publicFits → Layout
  zeros : ℕ

/-- The relation of each claim: slot 0, slot 1, then circuit `pc − 1`. -/
def which (claims : Claims logicalWidth publicFits) : Fin 3 → ℕ := ![0, 1, claims.pc - 1]

/-- The circuit of the fresh instance. -/
def freshIndex (claims : Claims logicalWidth publicFits) : Fin 2 := if claims.pc = 2 then 1 else 0

theorem avoids_bind_of {α β : Type} {g : Compression.Game α} {f : α → Compression.Game β}
    {Q : α → Prop} {R : β → Prop} (h : g.Avoids Q) (hf : ∀ a, Q a → (f a).Avoids R) :
    (Compression.Game.bind g f).Avoids R :=
  (avoids_bind g f R).mpr (h.mono hf)

/-- The checks of Algorithm 10, steps 1 to 4. -/
def Checks (statement : TerminalStatement AppState) (claims : Claims logicalWidth publicFits) :
    Prop :=
  Terminal.StatementValid statement ∧ 0 < statement.iteration ∧ InRange 2 claims.pc ∧
    ∀ s, (claims.running s).2 ≤ productionLightFoldCap

abbrev Outcome (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) :=
  TerminalStatement AppState ×
    Option (Claims logicalWidth publicFits × (ℕ → F) × Probe K CompressionFresh.oneShape)

instance : Nonempty (TerminalStatement AppState) := ⟨⟨0, [], []⟩⟩

namespace Setup

variable (setup : Setup logicalWidth publicFits)

/-- The parameters of the argument for `claims`. -/
noncomputable abbrev params (claims : Claims logicalWidth publicFits) :=
  Compression.Production.params (keyOf setup.ajtai) (fun c => (setup.rows (which claims c)).matrices)
    (fun s => (claims.running s).2) (setup.layout claims)

/-- The verifier of the argument for `claims`, with the one pass of Rust `matrix_part`. -/
noncomputable def verifier (claims : Claims logicalWidth publicFits) :
    VerifierData (setup.params claims) :=
  verifierData (keyOf setup.ajtai) setup.rows (which claims) (fun s => (claims.running s).2)
    (setup.layout claims) setup.zeros

/-- `x`, the fresh public input that the verifier computes (Algorithm 10, step 6). -/
def publicInput (statement : TerminalStatement AppState) (claims : Claims logicalWidth publicFits) :
    PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  encHash (publicFits := publicFits) (stateHash (toState {
    verifierKeys := setup.circuitKeys
    iteration := statement.iteration
    z0 := statement.z0
    current := statement.zi
    running := claims.running
    pc := claims.pc }))

/-- The fresh claim: the proof's commitment and `x`. -/
def fresh (statement : TerminalStatement AppState) (claims : Claims logicalWidth publicFits) :
    StepFresh logicalWidth publicFits :=
  ⟨fun _ => claims.commitment, fun _ => setup.publicInput statement claims⟩

/-- The witness slice of claim `c`. -/
def witness (claims : Claims logicalWidth publicFits) (W : ℕ → F) (c : Fin 3) :
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  slotWitness (fun c => (setup.rows (which claims c)).matrices) (setup.layout claims) W c

/-- The Π_CCS statement of the fresh instance. -/
def piCCS (statement : TerminalStatement AppState) (claims : Claims logicalWidth publicFits) :=
  CompressionFresh.statement (setup.circuitKeys (freshIndex claims)).relation
    ((setup.fresh statement claims).commitments ⟨0, by decide⟩)
    ((setup.fresh statement claims).publicInputs ⟨0, by decide⟩)

/-- The claims of the argument: the two slots, then the output claim of the run. -/
noncomputable def claimsOf (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (probe : Probe K CompressionFresh.oneShape) :
    Compression.Verifier.Statement (setup.params claims) := fun c =>
  if h : (c : ℕ) < 2 then claimValues (claims.running ⟨c, h⟩).1
  else claimValues (logicalWidth := logicalWidth) (publicFits := publicFits)
    ⟨claims.commitment, setup.publicInput statement claims, probe.coins.roundPoint,
      ⟨probe.response.fullOutput.padCoordinate ⟨0, by decide⟩,
        probe.response.fullOutput.matrixCoordinate ⟨0, by decide⟩⟩⟩

/-- Π_CCS on the fresh instance, with the fresh witness slice of `W`. It is irreducible, so that
no check of a definitional equality runs its `ℓ` rounds. -/
@[irreducible] noncomputable def freshRun (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) :
    Compression.Game (Option (Probe K CompressionFresh.oneShape)) :=
  Compression.Fresh.game (setup.piCCS statement claims)
    (CompressionFresh.witnessOf (setup.witness claims W 2)) 8

/-- After the prover names `W`: Π_CCS on the fresh instance, then the argument. -/
noncomputable def afterW (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (W : ℕ → F) :
    Compression.Game (Option (Outcome logicalWidth publicFits)) := by
  classical
  exact Compression.Game.bind (setup.freshRun statement claims W) fun out =>
    match out with
    | none => .pure none
    | some probe =>
        Compression.Game.bind (afterTables (setup.verifier claims) (setup.claimsOf statement claims probe) W)
          fun r => .pure (r.bind fun o =>
            if o = (setup.claimsOf statement claims probe, W) then
              some (statement, some (claims, W, probe))
            else none)

/-- The final verifier against an adaptive prover. -/
noncomputable def game : Compression.Game (Option (Outcome logicalWidth publicFits)) := by
  classical
  exact .recv (TerminalStatement AppState × Option (Claims logicalWidth publicFits)) fun input =>
    match input.2 with
    | none => .pure (if Terminal.StatementValid input.1 ∧ input.1.iteration = 0 ∧
        input.1.zi = input.1.z0 then some (input.1, none) else none)
    | some claims =>
        if Checks input.1 claims then .recv (ℕ → F) fun W => setup.afterW input.1 claims W
        else .pure none

/-- The proof with witnesses of an outcome: the bottom proof, or the slots' claims and witness
slices, the fresh claim, the fresh witness slice and `pc`, all from `W`. -/
def proofOf (o : Outcome logicalWidth publicFits) :
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  match o.2 with
  | none => .bottom
  | some (claims, W, _) => .recursive ⟨claims.running, fun s => setup.witness claims W s.castSucc,
      setup.fresh o.1 claims, setup.witness claims W 2, claims.pc⟩

end Setup

/-- The premises of the theorem on the verifier's setup. `represents`: the dense rows of each
circuit are its relation on the cells of its blocks (`Spec.Compression.Claim.Represents`).
`support`: a claim has at most the blocks of its circuit's package. `layout`: Rust `Layout::new`
gives a well-formed layout with the production bounds for each statement that passes the checks. -/
structure Premises (setup : Setup logicalWidth publicFits) (packageWidths : Fin 2 → ℕ) : Prop where
  represents : ∀ j : Fin 2,
    Represents (setup.rows j).matrices (setup.circuitKeys j).relation.matrices
  support : ∀ j : Fin 2, (setup.rows j).matrices.columns / ringDegree ≤
    ConcreteRingColumnLayout.blockCount (packageWidths j)
  layout : ∀ statement claims, Checks statement claims →
    WellFormed (setup.params claims) ∧
      (∀ c : Fin 3, (c : ℕ) < 2 → ((setup.layout claims).witness c).vars ≤ 25) ∧
      (∀ c : Fin 3, ((setup.layout claims).witness c).vars - 6 ≤
        (setup.layout claims).keyBlockVars) ∧
      (setup.layout claims).spanVars ≤ 27

theorem which_slot (claims : Claims logicalWidth publicFits) (s : Fin 2) :
    which claims s.castSucc = s := by
  fin_cases s <;> rfl

theorem freshIndex_eq {claims : Claims logicalWidth publicFits} (valid : InRange 2 claims.pc) :
    freshIndex claims = selectedIndex valid := by
  apply Fin.ext
  simp only [freshIndex, selectedIndex]
  obtain ⟨lower, upper⟩ := valid
  split <;> simp <;> omega

theorem which_fresh {claims : Claims logicalWidth publicFits} (valid : InRange 2 claims.pc) :
    which claims 2 = freshIndex claims := by
  rw [freshIndex_eq valid]
  rfl

theorem supported_slice {W : ℕ → F} {offset blocks width : ℕ}
    (le : blocks ≤ ConcreteRingColumnLayout.blockCount width) :
    BlockSupport.Supported (logicalWidth := logicalWidth) width
      (sliceAssignment W offset (ringDegree * blocks)) := by
  intro column far
  have beyond : ringDegree * blocks ≤ column := by
    simp only [ConcreteRingColumnLayout.blockCount, ringDegree] at le far ⊢
    omega
  simp only [sliceAssignment, ite_eq_right (not_lt.mpr beyond)]

variable (setup : Setup logicalWidth publicFits) {packageWidths : Fin 2 → ℕ}
  (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)

/-- An accepted path of the argument and of Π_CCS on the fresh instance gives a proof with
witnesses that the verifier of Algorithm 6 accepts. -/
theorem verifies_of_run (premises : Premises setup packageWidths)
    {statement : TerminalStatement AppState} {claims : Claims logicalWidth publicFits}
    (checks : Checks statement claims) {W : ℕ → F} {probe : Probe K CompressionFresh.oneShape}
    (holds : Relation (setup.params claims) (setup.claimsOf statement claims probe) W)
    (ccs : AmbientOutputHolds extensionOps K.embed (openingMaps setup.ajtai)
        productionGlobalParams (setup.piCCS statement claims) probe
        (CompressionFresh.witnessOf (setup.witness claims W 2)) →
      CCS.Holds (semantics setup.ajtai) productionGlobalParams
        (freshStatement (setup.circuitKeys (freshIndex claims)).relation
          (setup.fresh statement claims)) (setup.witness claims W 2)) :
    Verifies setup.ajtai setup.circuitKeys step packageWidths cap control statement
      (.recursive ⟨claims.running, fun s => setup.witness claims W s.castSucc,
        setup.fresh statement claims, setup.witness claims W 2, claims.pc⟩) := by
  obtain ⟨valid, positive, pcValid, depth_le⟩ := checks
  obtain ⟨wf, -, -, -⟩ := premises.layout statement claims ⟨valid, positive, pcValid, depth_le⟩
  have slotRows (s : Fin 2) : Represents (setup.rows (which claims s.castSucc)).matrices
      (setup.circuitKeys s).relation.matrices := by
    rw [which_slot]
    exact premises.represents s
  have freshRows : Represents (setup.rows (which claims 2)).matrices
      (setup.circuitKeys (freshIndex claims)).relation.matrices := by
    rw [which_fresh pcValid]
    exact premises.represents _
  have ambient := CompressionFresh.fresh_ambient holds wf setup.ajtai
    (setup.circuitKeys (freshIndex claims)).relation rfl freshRows claims.commitment
    (setup.publicInput statement claims) probe (by simp [Setup.claimsOf])
  have freshHolds := ccs ambient
  rw [freshIndex_eq pcValid] at freshHolds
  refine (verifies_recursive_iff _ _ _ _ _).mpr ⟨(Terminal.holdsFor_recursive_iff _ _ _ _ _ _ _).mpr
    ⟨valid, pcValid, positive, rfl, fun s => ?_, freshHolds⟩,
    ⟨depth_le, fun s => slot_normBounded_slice holds wf s (depth_le s), fun s => ?_,
      fun valid' => ?_⟩⟩
  · exact slot_holds holds setup.ajtai (setup.circuitKeys s) (claims.running s).1 wf s
      (depth_le s) rfl (slotRows s) (by
        simp only [Setup.claimsOf, Fin.val_castSucc, s.isLt, ↓reduceDIte, Fin.eta])
  · refine supported_slice ?_
    change (setup.rows (which claims s.castSucc)).matrices.columns / ringDegree ≤ _
    rw [which_slot]
    exact premises.support s
  · refine supported_slice ?_
    change (setup.rows (which claims 2)).matrices.columns / ringDegree ≤ _
    rw [which_fresh pcValid, freshIndex_eq valid']
    exact premises.support _

theorem verifies_of_bottom {statement : TerminalStatement AppState}
    (bottom : Terminal.StatementValid statement ∧ statement.iteration = 0 ∧
      statement.zi = statement.z0) :
    Verifies setup.ajtai setup.circuitKeys step packageWidths cap control statement .bottom :=
  (verifies_bottom_iff _ _ _ _).mpr bottom

/-- On each path that avoids the bad sets, the verifier of Algorithm 6 accepts each accepted
statement with the proof that its outcome gives (`Setup.proofOf`). -/
theorem game_avoids (premises : Premises setup packageWidths) :
    setup.game.Avoids fun a => ∀ o, a = some o →
      Verifies setup.ajtai setup.circuitKeys step packageWidths cap control o.1
        (setup.proofOf o) := by
  intro input
  obtain ⟨statement, claims?⟩ := input
  cases claims? with
  | none =>
      intro o equal
      dsimp only at equal
      split_ifs at equal with bottom
      cases equal
      exact verifies_of_bottom setup step cap control bottom
  | some claims =>
      dsimp only
      split_ifs with checks
      · intro W
        obtain ⟨wf, slots_le, key_fits, -⟩ := premises.layout statement claims checks
        dsimp only
        unfold Setup.afterW
        have freshAvoids : (setup.freshRun statement claims W).Avoids fun out =>
            ∀ probe, out = some probe →
              AmbientOutputHolds extensionOps K.embed (openingMaps setup.ajtai)
                  productionGlobalParams (setup.piCCS statement claims) probe
                  (CompressionFresh.witnessOf (setup.witness claims W 2)) →
                CCS.Holds (semantics setup.ajtai) productionGlobalParams
                  (freshStatement (setup.circuitKeys (freshIndex claims)).relation
                    (setup.fresh statement claims)) (setup.witness claims W 2) := by
          unfold Setup.freshRun
          exact CompressionFresh.game_avoids setup.ajtai
            (setup.circuitKeys (freshIndex claims)).relation (setup.fresh statement claims)
            (setup.witness claims W 2)
        refine avoids_bind_of freshAvoids ?_
        intro out freshGood
        cases out with
        | none => exact fun _ equal => nomatch equal
        | some probe =>
            dsimp only
            rw [avoids_bind]
            refine (Compression.Verifier.game_avoids (setup.verifier claims) wf
              (params_countsNonzero _ _ _ _ slots_le) (verifierData_matrixSpec _ _ _ _ _ _)
              (le_of_eq rfl) (Nat.le_of_ble_eq_true rfl) key_fits
              (setup.claimsOf statement claims probe) W).mono fun r relationGood => ?_
            intro o equal
            cases r with
            | none => exact nomatch equal
            | some found =>
                dsimp only [Option.bind_some] at equal
                split_ifs at equal with same
                cases equal
                have holds := relationGood found rfl
                rw [same] at holds
                exact verifies_of_run setup step cap control premises checks holds
                  (freshGood probe rfl)
      · exact fun _ equal => nomatch equal

/-- The bound on the density of each bad set: the argument's and Π_CCS's. -/
noncomputable def finalError : ℝ := 1 / 2 ^ 119 + 38 / (fieldModulus : ℝ) ^ 2

theorem game_densityLe (premises : Premises setup packageWidths) :
    setup.game.DensityLe finalError := by
  have ccsLe : (38 / (fieldModulus : ℝ) ^ 2) ≤ finalError := by
    unfold finalError
    linarith [show (0 : ℝ) ≤ 1 / 2 ^ 119 by positivity]
  intro input
  obtain ⟨statement, claims?⟩ := input
  cases claims? with
  | none => exact trivial
  | some claims =>
      dsimp only
      split_ifs with checks
      · intro W
        obtain ⟨wf, slots_le, -, span_le⟩ := premises.layout statement claims checks
        have errorLe : error (setup.verifier claims) ≤ finalError := by
          have := params_error_lt _ _ _ _ (setup.verifier claims) rfl checks.2.2.2 slots_le span_le
          unfold finalError
          linarith [show (0 : ℝ) ≤ 38 / (fieldModulus : ℝ) ^ 2 by positivity]
        unfold Setup.afterW
        have freshDensity : (setup.freshRun statement claims W).DensityLe finalError := by
          unfold Setup.freshRun
          exact (CompressionFresh.game_densityLe _ _ _ _).mono ccsLe
        refine densityLe_bind (P := fun _ => True) freshDensity
          (Ends.of_forall (fun _ => trivial) _) fun out _ => ?_
        cases out with
        | none => exact trivial
        | some probe =>
            exact densityLe_bind (P := fun _ => True)
              (((Compression.Verifier.game_densityLe (setup.verifier claims) wf)
                (setup.claimsOf statement claims probe) W).mono errorLe)
              (Ends.of_forall (fun _ => trivial) _) fun _ _ => trivial
      · exact trivial

/-- The final verifier accepts a statement for which no proof with witnesses satisfies
`Verifies`. -/
def Wins (setup : Setup logicalWidth publicFits) (packageWidths : Fin 2 → ℕ)
    (step : AppState → AppWitness → AppState) (cap : ℕ)
    (control : AppState → AppWitness → Fin 2) (r : Option (Option (Outcome logicalWidth publicFits))) :
    Prop :=
  ¬ ∀ a, r = some a → ∀ o, a = some o →
    ∃ proof, Verifies setup.ajtai setup.circuitKeys step packageWidths cap control o.1 proof

/-- In the state-restoration experiment with `q` queries, the final verifier accepts a statement
for which no proof with witnesses satisfies `Verifies` with probability at most
`q · (1/2^119 + 38/q²)`. -/
theorem soundness (premises : Premises setup packageWidths)
    (s : Strategy (Option (Option (Outcome logicalWidth publicFits)))) (q : ℕ) :
    prob s (restore setup.game q []) (Wins setup packageWidths step cap control) ≤
      q * finalError :=
  restore_sound _ s _ _ ((game_avoids setup step cap control premises).mono fun _ verifies o
      equal => ⟨_, verifies o equal⟩)
    (game_densityLe setup premises) (by unfold finalError; positivity) q

/-- Akita's knowledge soundness with the error `εAkita`, as an assumption (SPEC.md, Algorithm 10;
`SECURITY_MODEL.md`, trust item 7). For each prover `real` of the final verifier with Akita
openings, whose wins have the probability `winsOf real`, there is a prover of the game with ideal
openings and `q` queries that wins at most `εAkita` less often. -/
def AkitaKnowledge {Real : Type} (winsOf : Real → ℝ) (setup : Setup logicalWidth publicFits)
    (packageWidths : Fin 2 → ℕ) (step : AppState → AppWitness → AppState) (cap : ℕ)
    (control : AppState → AppWitness → Fin 2) (q : ℕ) (εAkita : ℝ) : Prop :=
  ∀ real, ∃ s, winsOf real ≤
    prob s (restore setup.game q []) (Wins setup packageWidths step cap control) + εAkita

/-- With Akita openings, the final verifier accepts a statement for which no proof with
witnesses satisfies `Verifies` with probability at most `q · (1/2^119 + 38/q²) + εAkita`. -/
theorem real_soundness (premises : Premises setup packageWidths) {Real : Type}
    (winsOf : Real → ℝ) (q : ℕ) (εAkita : ℝ)
    (akita : AkitaKnowledge winsOf setup packageWidths step cap control q εAkita) (real : Real) :
    winsOf real ≤ q * finalError + εAkita := by
  obtain ⟨s, bound⟩ := akita real
  linarith [soundness setup step cap control premises s q]

end NightstreamFPrime.Export.BoundedDepth.FinalVerifier
