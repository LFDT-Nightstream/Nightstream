import Nightstream.Implementation.Nebula.FPrime.State.OutputPoseidonBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeXOutLifecycleBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigestRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullPhaseSemanticRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullXOutContextRowSound

/-!
Contract: one-assignment terminal XOut authority for the exact Goldilocks
streaming profile.

Owns the composition of the full-layout 24 context rows, 330,401
phase-semantic rows, and 19,353 Nebula-state-digest rows. An explicit
outer-hash binding recovers the complete typed 32-field terminal frame or the
existing named outer Poseidon2 collision. Exact typed phase and Nebula
encoders then recover both compressed preimages or named Poseidon2 recipe
collisions.

The encoder compatibility fields are deterministic implementation claims.
They are not collision-resistance assumptions. This module does not own the
generated outer-hash/public-input link, the selected phase semantics, the
typed Nebula encoder, or collision resistance.

Assurance tier: security-reduced conditional adapter for Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutLifecycleBridge

open Nightstream.Implementation.Nebula.StateOutputAuthorityRows
open Nightstream.Implementation.Nebula.StateOutputPoseidonBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen
  uRunningWitness uFreshWitness

private abbrev contextArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext.rawArtifact

private abbrev phaseArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic.rawArtifact

private abbrev nebulaArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest.rawArtifact

abbrev EncodedDigest := Fin 4 → Nat

def assignmentFrame (assignment : Nat → Nat) : List Nat :=
  contextArtifact.xOutColumns.map assignment

def recipeDigest (recipe : VariableHashRecipe)
    (values : List Nat) : EncodedDigest :=
  fun lane => runValueRounds recipe.trace.rounds values (fun _ => 0) lane.val

def phaseInput (assignment : Nat → Nat) : List Nat :=
  phaseArtifact.hashRecipe.inputColumns.map assignment

inductive LaneBranch where
  | absent
  | present
deriving DecidableEq, Repr

def laneRecipe : LaneBranch → VariableHashRecipe
  | .absent => nebulaArtifact.absentRecipe
  | .present => nebulaArtifact.presentRecipe

def assignmentLaneBranch (assignment : Nat → Nat) : LaneBranch :=
  if assignment nebulaArtifact.openColumn = 1 then .present else .absent

def laneInput (assignment : Nat → Nat) : List Nat :=
  (laneRecipe (assignmentLaneBranch assignment)).inputColumns.map assignment

/-- Exact field encoding of one delayed typed phase payload. -/
structure PhasePayloadEncoding (Fresh : Type uFresh) where
  fields : List Fresh → List Nat
  length_exact : ∀ latest,
    (fields latest).length =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact.payloadFields
  canonical : ∀ latest value, value ∈ fields latest → value < goldilocksP

def phasePreimage
    {Fresh : Type uFresh}
    (encoding : PhasePayloadEncoding Fresh)
    (phaseState :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Digest)
    (latest : List Fresh) : List Nat :=
  phaseArtifact.constantValues ++
    List.ofFn
      (Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        phaseState) ++
    encoding.fields latest

/-- Deterministic agreement between the abstract lifecycle phase digest and
the exact Rust-emitted Poseidon2 recipe. -/
structure PhasePoseidon2Compatible
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Configuration
        Params StructureDigest Running Fresh NifsProof Nebula NebulaOpen
        armCount)
    (encoding : PhasePayloadEncoding Fresh) : Prop where
  exact : ∀ phaseState latest,
    recipeDigest phaseArtifact.hashRecipe
        (phasePreimage encoding phaseState latest) =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        (configuration.phaseEnvelopeDigest phaseState latest)

/-- Exact typed branch and field preimage for one Nebula lane. -/
structure NebulaEncoding (Nebula : Type) where
  branch : Nebula → LaneBranch
  preimage : Nebula → List Nat
  length_exact : ∀ nebula,
    (preimage nebula).length =
      (laneRecipe (branch nebula)).inputColumns.length
  canonical : ∀ nebula value, value ∈ preimage nebula → value < goldilocksP

/-- Deterministic agreement between the lifecycle Nebula digest and the exact
selected Rust absent/present Poseidon2 recipe. -/
structure NebulaPoseidon2Compatible
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Configuration
        Params StructureDigest Running Fresh NifsProof Nebula NebulaOpen
        armCount)
    (encoding : NebulaEncoding Nebula) : Prop where
  exact : ∀ nebula,
    recipeDigest (laneRecipe (encoding.branch nebula))
        (encoding.preimage nebula) =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        (configuration.hashSemantics.nebulaDigest nebula)

/-- Collision witness for two exact variable-length Poseidon2 applications.
The recipes fix the respective absorb schedules; both messages are canonical
and have the exact declared lengths. -/
structure RecipeCollisionWitness
    (leftRecipe rightRecipe : VariableHashRecipe) where
  left : List Nat
  right : List Nat
  leftLength : left.length = leftRecipe.inputColumns.length
  rightLength : right.length = rightRecipe.inputColumns.length
  leftCanonical : ∀ value ∈ left, value < goldilocksP
  rightCanonical : ∀ value ∈ right, value < goldilocksP
  different : left ≠ right
  digestEqual : recipeDigest leftRecipe left = recipeDigest rightRecipe right

def RecipeCollision
    (leftRecipe rightRecipe : VariableHashRecipe) : Prop :=
  Nonempty (RecipeCollisionWitness leftRecipe rightRecipe)

inductive Failure : Prop where
  | outer (collision : OuterCollision)
  | phase (collision : RecipeCollision
      phaseArtifact.hashRecipe phaseArtifact.hashRecipe)
  | nebula (supplied authoritative : LaneBranch)
      (collision : RecipeCollision
        (laneRecipe supplied) (laneRecipe authoritative))

private theorem inputs_eq_or_collision
    (leftRecipe rightRecipe : VariableHashRecipe)
    (left right : List Nat)
    (leftLength : left.length = leftRecipe.inputColumns.length)
    (rightLength : right.length = rightRecipe.inputColumns.length)
    (leftCanonical : ∀ value ∈ left, value < goldilocksP)
    (rightCanonical : ∀ value ∈ right, value < goldilocksP)
    (digestEqual : recipeDigest leftRecipe left =
      recipeDigest rightRecipe right) :
    left = right ∨ RecipeCollision leftRecipe rightRecipe := by
  by_cases same : left = right
  · exact Or.inl same
  · exact Or.inr ⟨{
      left := left
      right := right
      leftLength := leftLength
      rightLength := rightLength
      leftCanonical := leftCanonical
      rightCanonical := rightCanonical
      different := same
      digestEqual := digestEqual }⟩

theorem assignmentFrame_length (assignment : Nat → Nat) :
    (assignmentFrame assignment).length = 32 := by
  rfl

private theorem assignmentFrame_canonical
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ value ∈ assignmentFrame assignment, value < goldilocksP := by
  intro value member
  rw [assignmentFrame] at member
  rcases List.mem_map.mp member with ⟨column, _, rfl⟩
  exact canonical column

private theorem phaseInput_length (assignment : Nat → Nat) :
    (phaseInput assignment).length =
      phaseArtifact.hashRecipe.inputColumns.length := by
  simp [phaseInput]

private theorem laneInput_length (assignment : Nat → Nat) :
    (laneInput assignment).length =
      (laneRecipe (assignmentLaneBranch assignment)).inputColumns.length := by
  simp [laneInput]

private theorem mapped_input_canonical
    (recipe : VariableHashRecipe)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ value ∈ recipe.inputColumns.map assignment, value < goldilocksP := by
  intro value member
  rcases List.mem_map.mp member with ⟨column, _, rfl⟩
  exact canonical column

private theorem phase_constants_canonical :
    ∀ value ∈ phaseArtifact.constantValues, value < goldilocksP := by
  exact
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic.rawArtifact_valid.constantsCanonical

private theorem phasePreimage_length
    {Fresh : Type uFresh}
    (encoding : PhasePayloadEncoding Fresh)
    (phaseState :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Digest)
    (latest : List Fresh) :
    (phasePreimage encoding phaseState latest).length =
      phaseArtifact.hashRecipe.inputColumns.length := by
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic.input_length]
  have constantLength :
      phaseArtifact.constantValues.length =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact.constantFields := by
    rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic.rawArtifact_valid.constants]
    rfl
  simp [phasePreimage, constantLength, encoding.length_exact,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact.hashInputFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact.digestFields]
  omega

private theorem phasePreimage_canonical
    {Fresh : Type uFresh}
    (encoding : PhasePayloadEncoding Fresh)
    (phaseState :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Digest)
    (latest : List Fresh) :
    ∀ value ∈ phasePreimage encoding phaseState latest,
      value < goldilocksP := by
  intro value member
  simp only [phasePreimage, List.mem_append] at member
  rcases member with (constantOrDigest | payload)
  · rcases constantOrDigest with (constant | digest)
    · exact phase_constants_canonical value constant
    · simp only [List.mem_ofFn] at digest
      rcases digest with ⟨lane, rfl⟩
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues_canonical
          phaseState lane
  · exact encoding.canonical latest value payload

private theorem recipeDigest_mapped_eq_computedDigest
    (recipe : VariableHashRecipe)
    (assignment : Nat → Nat) :
    recipeDigest recipe (recipe.inputColumns.map assignment) =
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.computedDigest
        recipe assignment := by
  rfl

private theorem selected_recipe_digest
    (assignment : Nat → Nat) :
    recipeDigest (laneRecipe (assignmentLaneBranch assignment))
        (laneInput assignment) =
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.computedSelectedDigestFor
        nebulaArtifact assignment := by
  by_cases present : assignment nebulaArtifact.openColumn = 1
  · simpa [assignmentLaneBranch, laneInput, laneRecipe, present,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.computedSelectedDigestFor]
      using recipeDigest_mapped_eq_computedDigest
        nebulaArtifact.presentRecipe assignment
  · simpa [assignmentLaneBranch, laneInput, laneRecipe, present,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.computedSelectedDigestFor]
      using recipeDigest_mapped_eq_computedDigest
        nebulaArtifact.absentRecipe assignment

private theorem semantic_digest_of_frame_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Configuration
        Params StructureDigest Running Fresh NifsProof Nebula NebulaOpen
        armCount)
    (state :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.OuterState
        Running Fresh Nebula)
    (nebula : Nebula)
    (assignment : Nat → Nat)
    (frameExact : assignmentFrame assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
        configuration state nebula) :
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.xOutSemanticDigestFor
        phaseArtifact assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        state.semanticState := by
  funext lane
  fin_cases lane
  · have selected := congrArg (fun values => values.getD 19 0) frameExact
    norm_num [assignmentFrame, contextArtifact, phaseArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.xOutSemanticDigestFor]
      at selected ⊢
    exact selected
  · have selected := congrArg (fun values => values.getD 20 0) frameExact
    norm_num [assignmentFrame, contextArtifact, phaseArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.xOutSemanticDigestFor]
      at selected ⊢
    exact selected
  · have selected := congrArg (fun values => values.getD 21 0) frameExact
    norm_num [assignmentFrame, contextArtifact, phaseArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.xOutSemanticDigestFor]
      at selected ⊢
    exact selected
  · have selected := congrArg (fun values => values.getD 22 0) frameExact
    norm_num [assignmentFrame, contextArtifact, phaseArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.xOutSemanticDigestFor]
      at selected ⊢
    exact selected

private theorem nebula_digest_of_frame_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Configuration
        Params StructureDigest Running Fresh NifsProof Nebula NebulaOpen
        armCount)
    (state :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.OuterState
        Running Fresh Nebula)
    (nebula : Nebula)
    (assignment : Nat → Nat)
    (frameExact : assignmentFrame assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
        configuration state nebula) :
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.xOutStateDigestFor
        nebulaArtifact assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        (configuration.hashSemantics.nebulaDigest nebula) := by
  funext lane
  fin_cases lane
  · have selected := congrArg (fun values => values.getD 28 0) frameExact
    norm_num [assignmentFrame, contextArtifact, nebulaArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.xOutStateDigestFor]
      at selected ⊢
    exact selected
  · have selected := congrArg (fun values => values.getD 29 0) frameExact
    norm_num [assignmentFrame, contextArtifact, nebulaArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.xOutStateDigestFor]
      at selected ⊢
    exact selected
  · have selected := congrArg (fun values => values.getD 30 0) frameExact
    norm_num [assignmentFrame, contextArtifact, nebulaArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.xOutStateDigestFor]
      at selected ⊢
    exact selected
  · have selected := congrArg (fun values => values.getD 31 0) frameExact
    norm_num [assignmentFrame, contextArtifact, nebulaArtifact,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload,
      fullFrame, payloadFields, u64Halves,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.xOutStateDigestFor]
      at selected ⊢
    exact selected

private theorem branch_eq_of_input_eq
    {Nebula : Type}
    (encoding : NebulaEncoding Nebula)
    (assignment : Nat → Nat)
    (nebula : Nebula)
    (inputExact : laneInput assignment = encoding.preimage nebula) :
    assignmentLaneBranch assignment = encoding.branch nebula := by
  have lengths :
      (laneRecipe (assignmentLaneBranch assignment)).inputColumns.length =
        (laneRecipe (encoding.branch nebula)).inputColumns.length := by
    calc
      (laneRecipe (assignmentLaneBranch assignment)).inputColumns.length =
          (laneInput assignment).length := (laneInput_length assignment).symm
      _ = (encoding.preimage nebula).length := congrArg List.length inputExact
      _ = (laneRecipe (encoding.branch nebula)).inputColumns.length :=
        encoding.length_exact nebula
  cases supplied : assignmentLaneBranch assignment <;>
    cases authoritative : encoding.branch nebula
  · rfl
  · exfalso
    rw [supplied, authoritative] at lengths
    norm_num [laneRecipe,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest.absent_input_length,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest.present_input_length,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact.absentInputFields,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact.presentInputFields]
      at lengths
  · exfalso
    rw [supplied, authoritative] at lengths
    norm_num [laneRecipe,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest.absent_input_length,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest.present_input_length,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact.absentInputFields,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact.presentInputFields]
      at lengths
  · rfl

/-- The three exact terminal families share one assignment. With the explicit
outer-hash/public-input link, they recover the complete typed terminal frame,
the exact phase preimage, and the exact selected Nebula preimage. Every
remaining ambiguity is one named Poseidon2 collision. -/
theorem rows_bind_terminal_frame_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Configuration
        Params StructureDigest Running Fresh NifsProof Nebula NebulaOpen
        armCount}
    {authority :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.TerminalAuthority
        Running Fresh RunningWitness FreshWitness Nebula}
    (terminal :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Terminal
        configuration authority)
    (phaseEncoding : PhasePayloadEncoding Fresh)
    (nebulaEncoding : NebulaEncoding Nebula)
    (outerCompatible :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOutLifecycleBridge.Poseidon2Compatible
        configuration)
    (phaseCompatible : PhasePoseidon2Compatible configuration phaseEncoding)
    (nebulaCompatible : NebulaPoseidon2Compatible configuration nebulaEncoding)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (contextSatisfied :
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext.Satisfied
        assignment)
    (phaseSatisfied : phaseArtifact.Satisfied assignment)
    (nebulaSatisfied : nebulaArtifact.Satisfied assignment)
    (outerHashBound : outerHash (assignmentFrame assignment) =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        terminal.publicXOut)
    (authoritativeFrameCanonical :
      ∀ value ∈
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
          configuration terminal.state terminal.nebula,
        value < goldilocksP) :
    (Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullXOutContextRowSound.Sound
        assignment ∧
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullPhaseSemanticRowSound.Sound
        assignment ∧
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigestRowSound.Sound
        assignment) ∧
      ((assignmentFrame assignment =
            Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
              configuration terminal.state terminal.nebula ∧
          phaseInput assignment =
            phasePreimage phaseEncoding terminal.phaseState terminal.latest ∧
          assignmentLaneBranch assignment =
            nebulaEncoding.branch terminal.nebula ∧
          laneInput assignment = nebulaEncoding.preimage terminal.nebula) ∨
        Failure) := by
  have contextSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullXOutContextRowSound.rows_sound
      assignment canonical one contextSatisfied
  have phaseSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullPhaseSemanticRowSound.rows_sound
      assignment canonical one phaseSatisfied
  have nebulaSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigestRowSound.rows_sound
      assignment canonical one nebulaSatisfied
  refine ⟨⟨contextSound, phaseSound, nebulaSound⟩, ?_⟩
  let suppliedFrame : CanonicalFrame :=
    ⟨assignmentFrame assignment, assignmentFrame_length assignment,
      assignmentFrame_canonical assignment canonical⟩
  let authoritativeFrame : CanonicalFrame :=
    ⟨Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
        configuration terminal.state terminal.nebula,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame_length
        configuration terminal.state terminal.nebula,
      authoritativeFrameCanonical⟩
  have authoritativeHash :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
          terminal.publicXOut =
        outerHash
          (Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
            configuration terminal.state terminal.nebula) := by
    calc
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
          terminal.publicXOut =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
          (XOut.compute configuration.hashSemantics .stateful
            configuration.context terminal.state) :=
        congrArg
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
          terminal.publicExact
      _ = outerHash
          (Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.frame
            configuration terminal.state terminal.nebula) :=
        outerCompatible.stateOutput terminal.state terminal.nebula
          terminal.nebulaExact
  have outerEqual : digest suppliedFrame = digest authoritativeFrame := by
    simpa [digest, suppliedFrame, authoritativeFrame] using
      outerHashBound.trans authoritativeHash
  rcases frame_values_eq_or_outer_collision suppliedFrame authoritativeFrame
      outerEqual with frameExact | outerCollision
  · have semanticExact := semantic_digest_of_frame_exact configuration
      terminal.state terminal.nebula assignment frameExact
    have phaseHashEqual :
        recipeDigest phaseArtifact.hashRecipe (phaseInput assignment) =
          recipeDigest phaseArtifact.hashRecipe
            (phasePreimage phaseEncoding terminal.phaseState terminal.latest) := by
      calc
        recipeDigest phaseArtifact.hashRecipe (phaseInput assignment) =
            Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.computedDigestFor
              phaseArtifact assignment := rfl
        _ = Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.assignedDigestFor
              phaseArtifact assignment := phaseSound.hash.symm
        _ = Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound.xOutSemanticDigestFor
              phaseArtifact assignment := phaseSound.xOutLink.symm
        _ = Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
              terminal.state.semanticState := semanticExact
        _ = Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
              (configuration.phaseEnvelopeDigest terminal.phaseState
                terminal.latest) := congrArg
              Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
              terminal.semanticEnvelopeExact
        _ = recipeDigest phaseArtifact.hashRecipe
              (phasePreimage phaseEncoding terminal.phaseState terminal.latest) :=
            (phaseCompatible.exact terminal.phaseState terminal.latest).symm
    rcases inputs_eq_or_collision phaseArtifact.hashRecipe
        phaseArtifact.hashRecipe (phaseInput assignment)
        (phasePreimage phaseEncoding terminal.phaseState terminal.latest)
        (phaseInput_length assignment)
        (phasePreimage_length phaseEncoding terminal.phaseState terminal.latest)
        (mapped_input_canonical phaseArtifact.hashRecipe assignment canonical)
        (phasePreimage_canonical phaseEncoding terminal.phaseState
          terminal.latest)
        phaseHashEqual with phaseExact | phaseCollision
    · have nebulaFrameExact := nebula_digest_of_frame_exact configuration
        terminal.state terminal.nebula assignment frameExact
      have nebulaHashEqual :
          recipeDigest (laneRecipe (assignmentLaneBranch assignment))
              (laneInput assignment) =
            recipeDigest
              (laneRecipe (nebulaEncoding.branch terminal.nebula))
              (nebulaEncoding.preimage terminal.nebula) := by
        calc
          recipeDigest (laneRecipe (assignmentLaneBranch assignment))
              (laneInput assignment) =
            Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.computedSelectedDigestFor
              nebulaArtifact assignment := selected_recipe_digest assignment
          _ = Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.selectedDigestFor
              nebulaArtifact assignment := nebulaSound.selectedHash.symm
          _ = Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound.xOutStateDigestFor
              nebulaArtifact assignment := nebulaSound.xOutLink.symm
          _ = Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
              (configuration.hashSemantics.nebulaDigest terminal.nebula) :=
            nebulaFrameExact
          _ = recipeDigest
              (laneRecipe (nebulaEncoding.branch terminal.nebula))
              (nebulaEncoding.preimage terminal.nebula) :=
            (nebulaCompatible.exact terminal.nebula).symm
      rcases inputs_eq_or_collision
          (laneRecipe (assignmentLaneBranch assignment))
          (laneRecipe (nebulaEncoding.branch terminal.nebula))
          (laneInput assignment) (nebulaEncoding.preimage terminal.nebula)
          (laneInput_length assignment)
          (nebulaEncoding.length_exact terminal.nebula)
          (mapped_input_canonical
            (laneRecipe (assignmentLaneBranch assignment)) assignment canonical)
          (nebulaEncoding.canonical terminal.nebula) nebulaHashEqual with
        laneExact | nebulaCollision
      · exact Or.inl ⟨frameExact, phaseExact,
          branch_eq_of_input_eq nebulaEncoding assignment terminal.nebula
            laneExact,
          laneExact⟩
      · exact Or.inr (.nebula (assignmentLaneBranch assignment)
          (nebulaEncoding.branch terminal.nebula) nebulaCollision)
    · exact Or.inr (.phase phaseCollision)
  · exact Or.inr (.outer outerCollision)

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutLifecycleBridge
