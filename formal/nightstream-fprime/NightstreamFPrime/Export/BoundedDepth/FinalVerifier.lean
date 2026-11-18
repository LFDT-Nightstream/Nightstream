import NightstreamFPrime.Export.BoundedDepth.RunCompleteness
import NightstreamFPrime.Lifecycle.BoundedDepth.CompressionFresh
import NightstreamFPrime.Spec.Compression.Rust.Soundness

/-! The final verifier as one public-coin game (SPEC.md, Algorithm 10), and its soundness against
the verifier of Algorithm 6 (`Export.BoundedDepth.RunCompleteness.Verifies`).

The verifier's circuit fixes the mode (`Mode`): the argument has both slot claims, or only the
claim of slot 0 (`Mode.light`, SPEC.md Algorithm 9). The prover names the statement `(i, z0, zi)`
and the values of the final proof that come before `W`: no steps, or `pc`, each slot's claim and
depth and the fresh commitment (`Claims`). The verifier runs the checks of steps 1 to 4
(`Checks`). In the light mode they require `pc = 1` and slot 1 at the default slot of depth 0,
which the verifier builds from its own constants. It then runs the layout of step 5
(`Setup.Laid`): the Lean model of Rust `Layout::new` accepts the claims of the mode
(`Spec.Compression.Rust.layoutNew`), and each claim has its 320 public cells. The layout of the
argument is that layout (`Setup.layout`). The verifier computes the fresh public input
`x = encHash(stateHash(…))` from these values. The prover names `W`. Π_CCS on the fresh instance
runs (`Spec.Compression.Fresh.game`), and then the argument on the slot claims of the mode and
the output claim of the run (`Spec.Compression.Verifier.afterTables`). The Akita openings are
ideal: the prover names the tables. A slot outside the argument has the zero witness, which
opens the default slot (`StepExtend.defaultSlot_holds`).

On each path that avoids the bad sets, `Verifies` accepts an accepted statement with the proof
from `W` (`Setup.proofOf`, `game_avoids`), and each bad set has density at most
`1/2^119 + 38/fieldModulus²` (`game_densityLe`). So in the state-restoration experiment with `q` queries, the
verifier accepts a statement without such a proof with probability at most
`q · (1/2^119 + 38/fieldModulus²)` (`soundness`). `AkitaKnowledge` names Akita's knowledge soundness with the
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

/-- The slots of the argument of a final proof (SPEC.md, Algorithm 9): both slots, or slot 0
alone when no run of the circuit reaches a refresh fold. -/
inductive Mode where
  | both
  | light
  deriving DecidableEq

/-- The slot claims of the argument. -/
def Mode.slots : Mode → ℕ
  | .both => 2
  | .light => 1

theorem Mode.slots_le (mode : Mode) : mode.slots ≤ 2 := by
  cases mode <;> decide

/-- The slot of slot claim `s` of the argument. -/
def Mode.slot (mode : Mode) (s : Fin mode.slots) : Fin 2 :=
  ⟨s, lt_of_lt_of_le s.isLt mode.slots_le⟩

/-- The values of a final proof that the prover names before `W` (SPEC.md section 4.5). -/
structure Claims (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  pc : ℕ
  running : Fin 2 → SlotData logicalWidth publicFits
  commitment : PaperAlgebra.Commitment

/-- What the verifier holds: the Ajtai key that the compression key commits, the circuit keys,
the dense rows of each circuit's relation (Rust `FactoredRows`, by circuit), the block count of
the compression key, under which the Lean model of Rust `Layout::new` lays out the claims
(`Setup.placed`), and the zeros of the opening point of `V`. -/
structure Setup (mode : Mode) (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)
  circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits
  rows : ℕ → Factored F
  keyBlocks : ℕ
  zeros : ℕ

variable {mode : Mode}

/-- The relation of each claim: the slots of the mode, then circuit `pc − 1`. -/
def which (mode : Mode) (claims : Claims logicalWidth publicFits) : Fin (mode.slots + 1) → ℕ :=
  fun c => if (c : ℕ) < mode.slots then c else claims.pc - 1

/-- The circuit of the fresh instance. -/
def freshIndex (claims : Claims logicalWidth publicFits) : Fin 2 := if claims.pc = 2 then 1 else 0

theorem avoids_bind_of {α β : Type} {g : Compression.Game α} {f : α → Compression.Game β}
    {Q : α → Prop} {R : β → Prop} (h : g.Avoids Q) (hf : ∀ a, Q a → (f a).Avoids R) :
    (Compression.Game.bind g f).Avoids R :=
  (avoids_bind g f R).mpr (h.mono hf)

/-- The checks that the mode adds: in the light mode, the circuit that ran last is the light
circuit, and slot 1 is the default slot at depth 0. -/
def ModeChecks : Mode → Claims logicalWidth publicFits → Prop
  | .both, _ => True
  | .light, claims => claims.pc = 1 ∧ claims.running 1 = (defaultSlot, 0)

/-- The checks of Algorithm 10, steps 1 to 4. -/
def Checks (mode : Mode) (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) : Prop :=
  Terminal.StatementValid statement ∧ 0 < statement.iteration ∧ InRange 2 claims.pc ∧
    (∀ s, (claims.running s).2 ≤ productionLightFoldCap) ∧ ModeChecks mode claims

/-- A slot outside the argument is the default slot at depth 0. -/
theorem running_default {claims : Claims logicalWidth publicFits} (modeChecks : ModeChecks mode claims)
    (s : Fin 2) (outside : ¬ (s : ℕ) < mode.slots) : claims.running s = (defaultSlot, 0) := by
  cases mode with
  | both => exact absurd s.isLt outside
  | light =>
      have one : s = 1 := Fin.ext (by have := s.isLt; simp only [Mode.slots] at outside; omega)
      rw [one]
      exact modeChecks.2

abbrev Outcome (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) :=
  TerminalStatement AppState ×
    Option (Claims logicalWidth publicFits × (ℕ → F) × Probe K CompressionFresh.oneShape)

instance : Nonempty (TerminalStatement AppState) := ⟨⟨0, [], []⟩⟩

/-- The depths of the slots of the argument. -/
def slotDepths (mode : Mode) (claims : Claims logicalWidth publicFits) (s : Fin mode.slots) : ℕ :=
  (claims.running (mode.slot s)).2

namespace Setup

variable (setup : Setup mode logicalWidth publicFits)

/-- Rust `Layout::new` on the claims of the mode (Algorithm 9, step 2), under the key polynomial
of `keyBlocks` blocks (Algorithm 10, step 5). -/
def placed (claims : Claims logicalWidth publicFits) :
    Except Compression.Rust.LayoutError Compression.Rust.Placed :=
  Compression.Rust.layoutNew (Compression.Rust.claimsOf
    (fun c => (setup.rows (which mode claims c)).matrices) (slotDepths mode claims))
    setup.keyBlocks

/-- The layout of the argument: the layout of `placed`. The game rejects claims without one
(`Laid`). -/
def layout (claims : Claims logicalWidth publicFits) : Layout (mode.slots + 1) :=
  match setup.placed claims with
  | .ok l => l.production
  | .error _ => (⟨[], 0, 0, 0, 0, 0, 0⟩ : Compression.Rust.Placed).production

/-- Algorithm 10, step 5: Rust `Layout::new` accepts the claims, and each claim has its 320
public cells (the shape check of Rust `layout`). -/
def Laid (claims : Claims logicalWidth publicFits) : Prop :=
  (∃ l, setup.placed claims = .ok l) ∧
    ∀ c, 320 ≤ ringDegree * ((setup.rows (which mode claims c)).matrices.columns / ringDegree)

/-- The parameters of the argument for `claims`. -/
noncomputable abbrev params (claims : Claims logicalWidth publicFits) :=
  Compression.Production.params (keyOf setup.ajtai)
    (fun c => (setup.rows (which mode claims c)).matrices) (slotDepths mode claims)
    (setup.layout claims)

/-- The verifier of the argument for `claims`, with the one pass of Rust `matrix_part`. -/
noncomputable def verifier (claims : Claims logicalWidth publicFits) :
    VerifierData (setup.params claims) :=
  verifierData (keyOf setup.ajtai) setup.rows (which mode claims) (slotDepths mode claims)
    (setup.layout claims) setup.zeros

/-- The layout of laid claims is well formed, the key polynomial holds each claim, each witness
slice has at most `2^25` cells, and the span has at most 26 variables
(`Spec.Compression.Rust.layoutNew_production`, `Spec.Compression.Rust.layoutNew_span`). -/
theorem layout_facts {claims : Claims logicalWidth publicFits} (laid : setup.Laid claims) :
    WellFormed (setup.params claims) ∧
      (∀ c : Fin (mode.slots + 1), ((setup.layout claims).witness c).vars ≤ 25) ∧
      (∀ c : Fin (mode.slots + 1), ((setup.layout claims).witness c).vars - 6 ≤
        (setup.layout claims).keyBlockVars) ∧
      (setup.layout claims).spanVars ≤ 26 := by
  obtain ⟨⟨l, ok⟩, cells⟩ := laid
  have same : setup.layout claims = l.production := by simp only [Setup.layout, ok]
  obtain ⟨wf, keyFits, small⟩ := Compression.Rust.layoutNew_production (keyOf setup.ajtai)
    (fun c => (setup.rows (which mode claims c)).matrices) (slotDepths mode claims) ok cells
  simp only [Setup.params, same]
  exact ⟨wf, small, keyFits, Compression.Rust.layoutNew_span ok⟩

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
def witness (claims : Claims logicalWidth publicFits) (W : ℕ → F) (c : Fin (mode.slots + 1)) :
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  slotWitness (fun c => (setup.rows (which mode claims c)).matrices) (setup.layout claims) W c

/-- The witness of slot `s`: its slice when the argument has the slot, and the zero witness of
the default slot otherwise. -/
def runningWitness (claims : Claims logicalWidth publicFits) (W : ℕ → F) (s : Fin 2) :
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  if h : (s : ℕ) < mode.slots then setup.witness claims W (⟨s, h⟩ : Fin mode.slots).castSucc
  else RingRelation.EvaluationHomomorphism.BaseLinear.assignmentZero

/-- The Π_CCS statement of the fresh instance. -/
def piCCS (statement : TerminalStatement AppState) (claims : Claims logicalWidth publicFits) :=
  CompressionFresh.statement (setup.circuitKeys (freshIndex claims)).relation
    ((setup.fresh statement claims).commitments ⟨0, by decide⟩)
    ((setup.fresh statement claims).publicInputs ⟨0, by decide⟩)

/-- The claims of the argument: the slots of the mode, then the output claim of the run. -/
noncomputable def claimsOf (statement : TerminalStatement AppState)
    (claims : Claims logicalWidth publicFits) (probe : Probe K CompressionFresh.oneShape) :
    Compression.Verifier.Statement (setup.params claims) := fun c =>
  if h : (c : ℕ) < mode.slots then claimValues (claims.running (mode.slot ⟨c, h⟩)).1
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
    (CompressionFresh.witnessOf (setup.witness claims W (Fin.last _))) 8

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
        if Checks mode input.1 claims ∧ setup.Laid claims then
          .recv (ℕ → F) fun W => setup.afterW input.1 claims W
        else .pure none

/-- The proof with witnesses of an outcome: the bottom proof, or the slots' claims and witness
slices, the fresh claim, the fresh witness slice and `pc`, all from `W`. -/
def proofOf (o : Outcome logicalWidth publicFits) :
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  match o.2 with
  | none => .bottom
  | some (claims, W, _) => .recursive ⟨claims.running, setup.runningWitness claims W,
      setup.fresh o.1 claims, setup.witness claims W (Fin.last _), claims.pc⟩

end Setup

/-- The premises of the theorem on the verifier's setup. `represents`: the dense rows of each
circuit are its relation on the cells of its blocks (`Spec.Compression.Claim.Represents`).
`support`: a claim has at most the blocks of its circuit's package. -/
structure Premises (setup : Setup mode logicalWidth publicFits) (packageWidths : Fin 2 → ℕ) :
    Prop where
  represents : ∀ j : Fin 2,
    Represents (setup.rows j).matrices (setup.circuitKeys j).relation.matrices
  support : ∀ j : Fin 2, (setup.rows j).matrices.columns / ringDegree ≤
    ConcreteRingColumnLayout.blockCount (packageWidths j)

theorem slot_val (s : Fin mode.slots) : ((mode.slot s : Fin 2) : ℕ) = s := rfl

theorem which_slot (claims : Claims logicalWidth publicFits) (s : Fin mode.slots) :
    which mode claims s.castSucc = mode.slot s := by
  simp only [which, Fin.val_castSucc, s.isLt, ite_true, slot_val]

theorem freshIndex_eq {claims : Claims logicalWidth publicFits} (valid : InRange 2 claims.pc) :
    freshIndex claims = selectedIndex valid := by
  apply Fin.ext
  simp only [freshIndex, selectedIndex]
  obtain ⟨lower, upper⟩ := valid
  split <;> simp <;> omega

theorem which_fresh {claims : Claims logicalWidth publicFits} (valid : InRange 2 claims.pc) :
    which mode claims (Fin.last _) = freshIndex claims := by
  rw [freshIndex_eq valid]
  simp [which, selectedIndex]

theorem supported_slice {W : ℕ → F} {offset blocks width : ℕ}
    (le : blocks ≤ ConcreteRingColumnLayout.blockCount width) :
    BlockSupport.Supported (logicalWidth := logicalWidth) width
      (sliceAssignment W offset (ringDegree * blocks)) := by
  intro column far
  have beyond : ringDegree * blocks ≤ column := by
    simp only [ConcreteRingColumnLayout.blockCount, ringDegree] at le far ⊢
    omega
  simp only [sliceAssignment, ite_eq_right (not_lt.mpr beyond)]

variable (setup : Setup mode logicalWidth publicFits) {packageWidths : Fin 2 → ℕ}
  (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)

theorem runningWitness_slot (claims : Claims logicalWidth publicFits) (W : ℕ → F)
    (s : Fin mode.slots) :
    setup.runningWitness claims W (mode.slot s) = setup.witness claims W s.castSucc := by
  simp only [Setup.runningWitness, slot_val, s.isLt, ↓reduceDIte, Fin.eta]

theorem runningWitness_default (claims : Claims logicalWidth publicFits) (W : ℕ → F) (s : Fin 2)
    (outside : ¬ (s : ℕ) < mode.slots) :
    setup.runningWitness claims W s = RingRelation.EvaluationHomomorphism.BaseLinear.assignmentZero :=
  by simp only [Setup.runningWitness, outside, ↓reduceDIte]

/-- Each slot is a slot of the argument or outside it. -/
theorem slot_or_outside (s : Fin 2) :
    (∃ s' : Fin mode.slots, mode.slot s' = s) ∨ ¬ (s : ℕ) < mode.slots :=
  if h : (s : ℕ) < mode.slots then Or.inl ⟨⟨s, h⟩, Fin.ext rfl⟩ else Or.inr h

/-- An accepted path of the argument and of Π_CCS on the fresh instance gives a proof with
witnesses that the verifier of Algorithm 6 accepts. -/
theorem verifies_of_run (premises : Premises setup packageWidths)
    {statement : TerminalStatement AppState} {claims : Claims logicalWidth publicFits}
    (checks : Checks mode statement claims) (laid : setup.Laid claims) {W : ℕ → F}
    {probe : Probe K CompressionFresh.oneShape}
    (holds : Relation (setup.params claims) (setup.claimsOf statement claims probe) W)
    (ccs : AmbientOutputHolds extensionOps K.embed (openingMaps setup.ajtai)
        productionGlobalParams (setup.piCCS statement claims) probe
        (CompressionFresh.witnessOf (setup.witness claims W (Fin.last _))) →
      CCS.Holds (semantics setup.ajtai) productionGlobalParams
        (freshStatement (setup.circuitKeys (freshIndex claims)).relation
          (setup.fresh statement claims)) (setup.witness claims W (Fin.last _))) :
    Verifies setup.ajtai setup.circuitKeys step packageWidths cap control statement
      (.recursive ⟨claims.running, setup.runningWitness claims W,
        setup.fresh statement claims, setup.witness claims W (Fin.last _), claims.pc⟩) := by
  obtain ⟨valid, positive, pcValid, depth_le, modeChecks⟩ := checks
  obtain ⟨wf, -, -, -⟩ := setup.layout_facts laid
  have slotRows (s : Fin mode.slots) : Represents
      (setup.rows (which mode claims s.castSucc)).matrices
      (setup.circuitKeys (mode.slot s)).relation.matrices := by
    rw [which_slot]
    exact premises.represents _
  have freshRows : Represents (setup.rows (which mode claims (Fin.last _))).matrices
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
    ⟨depth_le, fun s => ?_, fun s => ?_, fun valid' => ?_⟩⟩
  · dsimp only
    rcases slot_or_outside (mode := mode) s with ⟨s, rfl⟩ | outside
    · rw [runningWitness_slot]
      exact slot_holds holds setup.ajtai (setup.circuitKeys (mode.slot s))
        (claims.running (mode.slot s)).1 wf s (depth_le _) rfl (slotRows s) (by
          simp only [Setup.claimsOf, Fin.val_castSucc, s.isLt, ↓reduceDIte, Fin.eta])
    · rw [running_default modeChecks s outside, runningWitness_default _ _ _ s outside]
      exact defaultSlot_holds (setup.circuitKeys s)
  · dsimp only
    rcases slot_or_outside (mode := mode) s with ⟨s, rfl⟩ | outside
    · rw [runningWitness_slot]
      exact slot_normBounded_slice holds wf s (depth_le _)
    · rw [running_default modeChecks s outside, runningWitness_default _ _ _ s outside]
      exact (defaultSlot_honest (ajtai := setup.ajtai) 0).1
  · dsimp only
    rcases slot_or_outside (mode := mode) s with ⟨s, rfl⟩ | outside
    · rw [runningWitness_slot]
      refine supported_slice ?_
      change (setup.rows (which mode claims s.castSucc)).matrices.columns / ringDegree ≤ _
      rw [which_slot]
      exact premises.support _
    · rw [runningWitness_default _ _ _ s outside]
      exact (defaultSlot_honest (ajtai := setup.ajtai) _).2
  · refine supported_slice ?_
    change (setup.rows (which mode claims (Fin.last _))).matrices.columns / ringDegree ≤ _
    rw [which_fresh pcValid, freshIndex_eq valid']
    exact premises.support _

theorem verifies_of_bottom {statement : TerminalStatement AppState}
    (bottom : Terminal.StatementValid statement ∧ statement.iteration = 0 ∧
      statement.zi = statement.z0) :
    Verifies setup.ajtai setup.circuitKeys step packageWidths cap control statement .bottom :=
  (verifies_bottom_iff _ _ _ _).mpr bottom

/-- The honest proof of `RunCompleteness.light_run_complete` passes the checks of the light mode,
with any commitment of the fresh instance. -/
theorem checks_light {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits)} {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
    {statement : TerminalStatement AppState}
    {payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2}
    (holds : Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload))
    (honest : Honest (ajtai := ajtai) packageWidths payload) (pcOne : payload.pc = 1)
    (slotOne : payload.running 1 = (defaultSlot, 0)) (commitment : PaperAlgebra.Commitment) :
    Checks .light statement ⟨payload.pc, payload.running, commitment⟩ := by
  obtain ⟨statementValid, pcValid, positive, -⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).mp holds
  exact ⟨statementValid, positive, pcValid, honest.depth, pcOne, slotOne⟩

/-- In the light mode, the claims of a statement that passes the checks of steps 1 to 4 pass the
layout of step 5 (Algorithm 10) under the key of `2^19` blocks, when the relation of the light
circuit has at most `2^19` blocks and its 320 public cells. With `pc = 1`, both claims are on the
rows of the light circuit (`Spec.Compression.Rust.layoutNew_light`). -/
theorem laid_light (light : Setup .light logicalWidth publicFits)
    (keyBlocks : light.keyBlocks = 2 ^ 19)
    (positive : 0 < (light.rows 0).matrices.columns / ringDegree)
    (fits : (light.rows 0).matrices.columns / ringDegree ≤ 2 ^ 19)
    (cells : 320 ≤ ringDegree * ((light.rows 0).matrices.columns / ringDegree))
    {statement : TerminalStatement AppState} {claims : Claims logicalWidth publicFits}
    (checks : Checks .light statement claims) : light.Laid claims := by
  obtain ⟨-, -, -, depth_le, pcOne, -⟩ := checks
  have rowsZero : ∀ c : Fin (Mode.light.slots + 1), which .light claims c = 0 := by
    intro c
    unfold which
    split_ifs with slot
    · change (c : ℕ) < 1 at slot
      omega
    · omega
  refine ⟨?_, fun c => by rw [rowsZero c]; exact cells⟩
  have same : Compression.Rust.claimsOf (fun c => (light.rows (which .light claims c)).matrices)
      (slotDepths .light claims) =
      Compression.Rust.lightClaims ((light.rows 0).matrices.columns / ringDegree)
        (claims.running 0).2 := by
    rw [Compression.Rust.lightClaims_eq]
    simp only [Compression.Rust.claimsOf, Compression.Rust.paramsClaims, List.ofFn_succ,
      List.ofFn_zero, slotDepths, Mode.slot, bound, Mode.slots]
    simp
    exact ⟨congrArg (fun j => (light.rows j).matrices.columns / ringDegree) (rowsZero _),
      congrArg (fun j => (light.rows j).matrices.columns / ringDegree) (rowsZero _)⟩
  unfold Setup.placed
  rw [same, keyBlocks]
  exact Compression.Rust.layoutNew_light positive fits (depth_le 0)

/-- The honest proof of `RunCompleteness.light_run_complete` passes the checks of steps 1 to 4 of
Algorithm 10 in the light mode and the layout of step 5, under the hypotheses of `laid_light`
(`checks_light`, `laid_light`). Step 5 also requires the block count of Algorithm 11, which is the
hypothesis `keyBlocks` here, and `error_bits` at least the security minimum, which the game does
not model. -/
theorem accepts_light (light : Setup .light logicalWidth publicFits)
    (keyBlocks : light.keyBlocks = 2 ^ 19)
    (positive : 0 < (light.rows 0).matrices.columns / ringDegree)
    (fits : (light.rows 0).matrices.columns / ringDegree ≤ 2 ^ 19)
    (cells : 320 ≤ ringDegree * ((light.rows 0).matrices.columns / ringDegree))
    {statement : TerminalStatement AppState}
    {payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2}
    (holds : Terminal.HoldsFor cap light.ajtai light.circuitKeys step control statement
      (.recursive payload))
    (honest : Honest (ajtai := light.ajtai) packageWidths payload) (pcOne : payload.pc = 1)
    (slotOne : payload.running 1 = (defaultSlot, 0)) (commitment : PaperAlgebra.Commitment) :
    Checks .light statement ⟨payload.pc, payload.running, commitment⟩ ∧
      light.Laid ⟨payload.pc, payload.running, commitment⟩ :=
  have checks := checks_light step cap control holds honest pcOne slotOne commitment
  ⟨checks, laid_light light keyBlocks positive fits cells checks⟩

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
      split_ifs with accepted
      · obtain ⟨checks, laid⟩ := accepted
        intro W
        obtain ⟨wf, small, key_fits, -⟩ := setup.layout_facts laid
        dsimp only
        unfold Setup.afterW
        have freshAvoids : (setup.freshRun statement claims W).Avoids fun out =>
            ∀ probe, out = some probe →
              AmbientOutputHolds extensionOps K.embed (openingMaps setup.ajtai)
                  productionGlobalParams (setup.piCCS statement claims) probe
                  (CompressionFresh.witnessOf (setup.witness claims W (Fin.last _))) →
                CCS.Holds (semantics setup.ajtai) productionGlobalParams
                  (freshStatement (setup.circuitKeys (freshIndex claims)).relation
                    (setup.fresh statement claims)) (setup.witness claims W (Fin.last _)) := by
          unfold Setup.freshRun
          exact CompressionFresh.game_avoids setup.ajtai
            (setup.circuitKeys (freshIndex claims)).relation (setup.fresh statement claims)
            (setup.witness claims W (Fin.last _))
        refine avoids_bind_of freshAvoids ?_
        intro out freshGood
        cases out with
        | none => exact fun _ equal => nomatch equal
        | some probe =>
            dsimp only
            rw [avoids_bind]
            refine (Compression.Verifier.game_avoids (setup.verifier claims) wf
              (params_countsNonzero _ _ _ _ fun c _ => small c) (verifierData_matrixSpec _ _ _ _ _ _)
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
                exact verifies_of_run setup step cap control premises checks laid holds
                  (freshGood probe rfl)
      · exact fun _ equal => nomatch equal

/-- The bound on the density of each bad set: the argument's and Π_CCS's. -/
noncomputable def finalError : ℝ := 1 / 2 ^ 119 + 38 / (fieldModulus : ℝ) ^ 2

theorem game_densityLe : setup.game.DensityLe finalError := by
  have ccsLe : (38 / (fieldModulus : ℝ) ^ 2) ≤ finalError := by
    unfold finalError
    linarith [show (0 : ℝ) ≤ 1 / 2 ^ 119 by positivity]
  intro input
  obtain ⟨statement, claims?⟩ := input
  cases claims? with
  | none => exact trivial
  | some claims =>
      dsimp only
      split_ifs with accepted
      · obtain ⟨checks, laid⟩ := accepted
        intro W
        obtain ⟨wf, small, -, span⟩ := setup.layout_facts laid
        have errorLe : error (setup.verifier claims) ≤ finalError := by
          have := params_error_lt _ _ _ _ (setup.verifier claims) mode.slots_le rfl
            (fun _ => checks.2.2.2.1 _) (fun c _ => small c) (span.trans (by decide))
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
def Wins (setup : Setup mode logicalWidth publicFits) (packageWidths : Fin 2 → ℕ)
    (step : AppState → AppWitness → AppState) (cap : ℕ)
    (control : AppState → AppWitness → Fin 2) (r : Option (Option (Outcome logicalWidth publicFits))) :
    Prop :=
  ¬ ∀ a, r = some a → ∀ o, a = some o →
    ∃ proof, Verifies setup.ajtai setup.circuitKeys step packageWidths cap control o.1 proof

/-- In the state-restoration experiment with `q` queries, the final verifier accepts a statement
for which no proof with witnesses satisfies `Verifies` with probability at most
`q · (1/2^119 + 38/fieldModulus²)`. -/
theorem soundness (premises : Premises setup packageWidths)
    (s : Strategy (Option (Option (Outcome logicalWidth publicFits)))) (q : ℕ) :
    prob s (restore setup.game q []) (Wins setup packageWidths step cap control) ≤
      q * finalError :=
  restore_sound _ s _ _ ((game_avoids setup step cap control premises).mono fun _ verifies o
      equal => ⟨_, verifies o equal⟩)
    (game_densityLe setup) (by unfold finalError; positivity) q

/-- Akita's knowledge soundness with the error `εAkita`, as an assumption (SPEC.md, Algorithm 10;
`SECURITY_MODEL.md`, trust item 7). For each prover `real` of the final verifier with Akita
openings, whose wins have the probability `winsOf real`, there is a prover of the game with ideal
openings and `q` queries that wins at most `εAkita` less often. -/
def AkitaKnowledge {Real : Type} (winsOf : Real → ℝ) (setup : Setup mode logicalWidth publicFits)
    (packageWidths : Fin 2 → ℕ) (step : AppState → AppWitness → AppState) (cap : ℕ)
    (control : AppState → AppWitness → Fin 2) (q : ℕ) (εAkita : ℝ) : Prop :=
  ∀ real, ∃ s, winsOf real ≤
    prob s (restore setup.game q []) (Wins setup packageWidths step cap control) + εAkita

/-- With Akita openings, the final verifier accepts a statement for which no proof with
witnesses satisfies `Verifies` with probability at most
`q · (1/2^119 + 38/fieldModulus²) + εAkita`. -/
theorem real_soundness (premises : Premises setup packageWidths) {Real : Type}
    (winsOf : Real → ℝ) (q : ℕ) (εAkita : ℝ)
    (akita : AkitaKnowledge winsOf setup packageWidths step cap control q εAkita) (real : Real) :
    winsOf real ≤ q * finalError + εAkita := by
  obtain ⟨s, bound⟩ := akita real
  linarith [soundness setup step cap control premises s q]

end NightstreamFPrime.Export.BoundedDepth.FinalVerifier
