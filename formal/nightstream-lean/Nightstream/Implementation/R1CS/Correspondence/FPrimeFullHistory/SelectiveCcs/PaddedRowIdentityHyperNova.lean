import Nightstream.HyperNova.Construction2.Default
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteNifs
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs

/-!
Contract: concrete HyperNova Construction 2 integration for
`PaddedRowIdentity`.

Owns: the deterministic committed-zero running pair, its validity for every
selected padded application structure, the replicated default running vector,
the selected one-joint-SumCheck NIFS setup, and the exact terminal CCS/CE
relations.

Does not own: the application compiler, the compact decider, Rust, generated
R1CS rows, cryptographic assumptions, or costs.

Emits constraints: no.

Assurance tier: concrete model-level integration. The setup uses one paper
SuperNeo NIFS call on the recursive branch. Rectangular support comes only
from zero row padding and `M_0 = [I; 0]`; it does not add a SumCheck.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova

open Nightstream.HyperNova.Construction2
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity

abbrev Assignment := PaddedRowIdentityConcreteAlgebra.Assignment
abbrev Commitment := PaddedRowIdentityConcreteAlgebra.Commitment
abbrev PublicInput := PaddedRowIdentityConcreteAlgebra.PublicInput
abbrev Point := PaddedRowIdentityConcreteAlgebra.Point
abbrev Evaluation := PaddedRowIdentityConcreteAlgebra.Evaluation
abbrev AjtaiKey := PaddedRowIdentityConcreteAlgebra.AjtaiKey
abbrev Structure := ApplicationMatrices
abbrev StatementId := PaddedRowIdentityConcreteNifs.Poseidon2.StatementId

abbrev VerifierKey :=
  Key K Commitment PublicInput RingF
    PaddedRowIdentityConcreteNifs.Poseidon2.State shape
    assignmentColumns (Phi81ColumnLayout.blockCount assignmentColumns) 9

abbrev PublicRunning := Running K Commitment PublicInput shape
abbrev PublicFresh := Fresh Commitment PublicInput shape
abbrev NifsProof := Proof K Commitment shape 9

/-- The unique fresh coordinate in the selected Construction 2 profile. -/
def freshIndex : Fin shape.freshCount := ⟨0, by decide⟩

/-- One structure-free running claim. The verifier supplies the common padded
matrix structure. -/
structure RunningClaim where
  commitment : Commitment
  publicInput : PublicInput
  point : Point
  evaluation : Evaluation

/-- One structure-free fresh CCS claim. The verifier supplies the common
padded matrix structure. -/
structure FreshClaim where
  commitment : Commitment
  publicInput : PublicInput

/-- Convert a structure-free claim to the exact fresh-stage CE statement used
by the paper NIFS running product. -/
def runningStatement (system : Structure) (claim : RunningClaim) :
    CE.Instance PaddedRowIdentityConcreteAlgebra.Structure PublicInput Point
      Evaluation Commitment where
  constraintSystem := matrixSource system
  commitment := claim.commitment
  publicInput := claim.publicInput
  point := claim.point
  evaluations := #[claim.evaluation]
  stage := .fresh

/-- Concrete running relation selected by one verifier-owned Ajtai key. -/
def RunningHolds (ajtaiKey : AjtaiKey)
    (system : Structure) (claim : RunningClaim) (assignment : Assignment) : Prop :=
  CE.Holds (PaddedRowIdentityConcreteAlgebra.semantics ajtaiKey)
    productionGlobalParams
    (runningStatement system claim) assignment

/-- Convert a structure-free fresh claim to the exact CCS statement used by
the paper NIFS. -/
def freshStatement (system : Structure) (claim : FreshClaim) :
    CCS.Instance PaddedRowIdentityConcreteAlgebra.Structure PublicInput
      Commitment where
  constraintSystem := matrixSource system
  commitment := claim.commitment
  publicInput := claim.publicInput
  stage := .fresh

/-- Concrete fresh relation selected by one verifier-owned Ajtai key. -/
def FreshHolds (ajtaiKey : AjtaiKey)
    (system : Structure) (claim : FreshClaim) (assignment : Assignment) : Prop :=
  CCS.Holds (PaddedRowIdentityConcreteAlgebra.semantics ajtaiKey)
    productionGlobalParams (freshStatement system claim) assignment

/-- Canonical all-zero complete Phi81 assignment. -/
def zeroAssignment : Assignment := BaseLinear.assignmentZero

/-- Canonical point in the selected 24-variable row cube. -/
def zeroPoint : Point where
  coordinates := List.replicate rowVariables K.zero
  dimension := by simp

/-- Parameter-independent public zero claim. Its commitment remains correct
for every Ajtai key because the committed assignment is zero. -/
def zeroClaim : RunningClaim where
  commitment := PiRLCAlgebra.Commitment.commitmentZero
  publicInput := PiRLCAlgebra.PublicInput.publicZero
  point := zeroPoint
  evaluation := PaddedRowIdentityConcreteAlgebra.evaluationZero

/-- The deterministic zero pair satisfies the running CE relation for every
Ajtai key and every selected padded application structure. -/
theorem zeroClaim_holds (ajtaiKey : AjtaiKey) (system : Structure) :
    RunningHolds ajtaiKey system zeroClaim zeroAssignment := by
  refine ⟨?_, True.intro, ?_⟩
  · refine ⟨PiRLCAlgebra.Commitment.commit_zero ajtaiKey, ?_, ?_⟩
    · exact PiRLCAlgebra.PublicInput.projectPublicInput_zero
    · intro column
      exact Nat.zero_lt_succ 1
  · change
      #[PaddedRowIdentityConcreteAlgebra.evaluationFamily
          (matrixSource system) zeroAssignment zeroPoint] =
        #[PaddedRowIdentityConcreteAlgebra.evaluationZero]
    apply congrArg (fun evaluation => #[evaluation])
    funext matrix
    exact BaseLinear.matrixEvaluation_zero
      (PaddedRowIdentityConcreteAlgebra.canonicalStructure
        (matrixSource system)) zeroPoint matrix

/-- HyperNova's exact universal default-pair obligation for the selected
running relation. -/
def paperDefault (ajtaiKey : AjtaiKey) :
    Default.DefaultPair Structure RunningClaim Assignment
      (RunningHolds ajtaiKey) where
  claim := zeroClaim
  witness := zeroAssignment
  satisfies := zeroClaim_holds ajtaiKey

/-- The deterministic default running product installed in every
Construction 2 slot. -/
def defaultRunning : PublicRunning where
  point := zeroPoint
  commitments := fun _ => zeroClaim.commitment
  publicInputs := fun _ => zeroClaim.publicInput
  evaluations := fun _ => zeroClaim.evaluation

/-- Canonical unused fresh value in the base-step advice encoding. -/
def baseDummyFresh : PublicFresh where
  commitments := fun _ => zeroClaim.commitment
  publicInputs := fun _ => zeroClaim.publicInput

/-- Canonical unused NIFS message in the base-step advice encoding. Its
contents have no protocol authority because the base branch performs no fold. -/
def baseDummyNifsProof : NifsProof where
  piCcsRounds := fun _ => {
    coefficients := List.replicate 10 K.zero
    coefficients_length := by simp
  }
  piCcsOutput := { coordinate := fun _ _ _ => K.zero }
  piDecCommitments := fun _ => zeroClaim.commitment
  piDecEvaluations := fun _ => zeroClaim.evaluation

/-- Prover openings for the deterministic default running product. -/
def defaultRunningWitness : Fin shape.runningCount -> Assignment :=
  fun _ => zeroAssignment

/-- Extract one structure-free claim from the public running product. -/
def claimAt (running : PublicRunning) (index : Fin shape.runningCount) :
    RunningClaim where
  commitment := running.commitments index
  publicInput := running.publicInputs index
  point := running.point
  evaluation := running.evaluations index

@[simp] theorem claimAt_defaultRunning (index : Fin shape.runningCount) :
    claimAt defaultRunning index = zeroClaim := by
  rfl

/-- Every coordinate of the installed running product is the paper default
pair and has its canonical zero opening. -/
theorem defaultRunning_holds
    (ajtaiKey : AjtaiKey) (system : Structure)
    (index : Fin shape.runningCount) :
    RunningHolds ajtaiKey system (claimAt defaultRunning index)
      (defaultRunningWitness index) := by
  exact zeroClaim_holds ajtaiKey system

/-- Concrete Construction 2 setup. Each slot owns its Ajtai key and padded
application matrices. Every selected NIFS verifier uses the same one-joint
paper protocol. -/
noncomputable def setup {slotCount : Nat}
    (statementIds : Fin slotCount -> StatementId)
    (ajtaiKeys : Fin slotCount -> AjtaiKey)
    (systems : Fin slotCount -> Structure) :
    Paper.Setup VerifierKey PublicRunning PublicFresh NifsProof slotCount where
  verifierKeys := fun slot => PaddedRowIdentityConcreteNifs.key
    (statementIds slot) (ajtaiKeys slot) (systems slot)
  nifs := { verify := PaddedRowIdentityConcreteNifs.verify }
  defaultRunning := defaultRunning

@[simp] theorem setup_defaultRunning {slotCount : Nat}
    (statementIds : Fin slotCount -> StatementId)
    (ajtaiKeys : Fin slotCount -> AjtaiKey)
    (systems : Fin slotCount -> Structure) :
    (setup statementIds ajtaiKeys systems).defaultRunning = defaultRunning := by
  rfl

@[simp] theorem setup_verifierKey {slotCount : Nat}
    (statementIds : Fin slotCount -> StatementId)
    (ajtaiKeys : Fin slotCount -> AjtaiKey)
    (systems : Fin slotCount -> Structure)
    (slot : Fin slotCount) :
    (setup statementIds ajtaiKeys systems).verifierKeys slot =
      PaddedRowIdentityConcreteNifs.key
        (statementIds slot) (ajtaiKeys slot) (systems slot) := by
  rfl

/-- Prover-only openings for every running coordinate at one terminal slot. -/
abbrev RunningWitness := Fin shape.runningCount -> Assignment

/-- Exact terminal CE relation for all running coordinates. -/
def TerminalRunningHolds
    (key : VerifierKey) (running : PublicRunning)
    (witness : RunningWitness) : Prop :=
  forall index,
    CE.Holds key.piRlcSemantics key.params {
      constraintSystem := key.matrixSource
      commitment := running.commitments index
      publicInput := running.publicInputs index
      point := running.point
      evaluations := #[running.evaluations index]
      stage := .fresh
    } (witness index)

/-- Exact terminal CCS relation for Construction 2's sole fresh coordinate. -/
def TerminalFreshHolds
    (key : VerifierKey) (fresh : PublicFresh) (witness : Assignment) : Prop :=
  CCS.Holds key.piRlcSemantics key.params {
    constraintSystem := key.matrixSource
    commitment := fresh.commitments freshIndex
    publicInput := fresh.publicInputs freshIndex
    stage := .fresh
  } witness

/-- Terminal relation package used by the outer Construction 2 verifier. It
checks all running CE openings and the one selected fresh CCS opening. -/
def terminalRelations {slotCount : Nat} :
    Paper.TerminalRelations VerifierKey PublicRunning RunningWitness
      PublicFresh Assignment slotCount where
  runningHolds := fun _ => TerminalRunningHolds
  freshHolds := fun _ => TerminalFreshHolds

/-- Exact Boolean terminal membership checkers for the selected CE and CCS
relations. The checks evaluate the same propositions stored in
`terminalRelations`; they do not accept a caller-supplied validity bit. -/
noncomputable def terminalChecks {slotCount : Nat} :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      (terminalRelations (slotCount := slotCount)) := by
  classical
  exact {
    runningCheck := fun _ key running witness =>
      decide (TerminalRunningHolds key running witness)
    freshCheck := fun _ key fresh witness =>
      decide (TerminalFreshHolds key fresh witness)
    runningCheck_iff := by
      intro slot key running witness
      exact decide_eq_true_iff
    freshCheck_iff := by
      intro slot key fresh witness
      exact decide_eq_true_iff
  }

/-- The selected setup's default running vector satisfies every terminal
running relation at every slot. -/
theorem setup_defaultRunning_terminal
    {slotCount : Nat}
    (statementIds : Fin slotCount -> StatementId)
    (ajtaiKeys : Fin slotCount -> AjtaiKey)
    (systems : Fin slotCount -> Structure)
    (slot : Fin slotCount) :
    TerminalRunningHolds
      ((setup statementIds ajtaiKeys systems).verifierKeys slot)
      (setup statementIds ajtaiKeys systems).defaultRunning
      defaultRunningWitness := by
  intro index
  exact zeroClaim_holds (ajtaiKeys slot) (systems slot)

/-- Concrete outer Construction 2 verifier theorem. The base branch accepts
only the bottom constructor and checks the initial endpoint. The recursive
branch checks the prior public link, all running CE relations, and the selected
fresh CCS relation. It performs no NIFS call and adds no SumCheck. -/
theorem terminalHolds_iff_transition
    {Digest State Witness Encoded : Type}
    {slotCount : Nat}
    (statementIds : Fin slotCount -> StatementId)
    (ajtaiKeys : Fin slotCount -> AjtaiKey)
    (systems : Fin slotCount -> Structure)
    (machine : Paper.Machine VerifierKey Digest State Witness PublicRunning
      PublicFresh Encoded slotCount)
    (statement : Paper.TerminalStatement State)
    (proof : Paper.OuterTerminalProof PublicRunning RunningWitness PublicFresh
      Assignment slotCount) :
    Paper.OuterTerminalHolds (setup statementIds ajtaiKeys systems) machine
        terminalRelations
        statement proof <->
      Paper.OuterTerminalTransition (setup statementIds ajtaiKeys systems) machine
        terminalRelations statement proof := by
  exact Paper.outerTerminalHolds_iff_transition
    (setup statementIds ajtaiKeys systems) machine terminalRelations
      statement proof

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova
