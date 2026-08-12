import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityCompilerDescription
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityHyperNova
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier

/-!
Contract: corrected HyperNova Definition 12 boundary for the selected
`PaddedRowIdentity` SuperNeo NIFS.

Owns: distinct running CE and fresh CCS carriers, canonical protocol codecs,
the canonical sparse compiler description, deterministic expansion to the
exact thirteen relation matrices, the universal committed-zero running pair,
the fixed rectangular compiler capacity, a Poseidon2 statement identifier,
and a compact verifier projection for one fixed augmented circuit.

Does not own: the selected application circuit compiler. Until that compiler
is closed, the final theorem keeps its NP-completeness and inverse proofs as
explicit inputs.

Emits constraints: no.

Assurance tier: concrete protocol integration over a model-level application
compiler contract.

The row capacity is `2^24`; the assignment width is `11,437,038`. The latter
fits the former, so the selected SuperNeo proof uses `M_0 = [I; 0]` and one
24-round joint SumCheck. No column SumCheck exists in this module.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility

open Nightstream.HyperNova.NIVCCompatibility
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova

abbrev DenseStructure := PaddedRowIdentity.ApplicationMatrices
abbrev Structure := PaddedRowIdentityCompilerDescription.Description
abbrev Assignment := PaddedRowIdentityHyperNova.Assignment
abbrev AjtaiKey := PaddedRowIdentityHyperNova.AjtaiKey
abbrev RunningClaim := PaddedRowIdentityHyperNova.RunningClaim
abbrev FreshClaim := PaddedRowIdentityHyperNova.FreshClaim
abbrev PublicRunning := PaddedRowIdentityHyperNova.PublicRunning
abbrev PublicFresh := PaddedRowIdentityHyperNova.PublicFresh
abbrev NifsProof := PaddedRowIdentityHyperNova.NifsProof

/-- Public parameters read by the selected relation. Fixed profile dimensions
and Poseidon2 constants are module constants; the Ajtai key is the sole
parameter value. -/
structure Parameters where
  ajtaiKey : AjtaiKey

/-! ## Total fixed-width protocol codecs -/

noncomputable def ajtaiCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec AjtaiKey :=
  Codec.finFunction PaddedRowIdentityConcreteAlgebra.verifierRows
    (Codec.finFunction
      (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount
        assignmentColumns) ringFCodec)

theorem ajtaiCodec_admissible (key : AjtaiKey) :
    ajtaiCodec.Admissible key := by
  intro row block
  exact ringFCodec_admissible (key row block)

def parametersData (parameters : Parameters) : AjtaiKey :=
  parameters.ajtaiKey

theorem parametersData_injective : Function.Injective parametersData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def parametersGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec Parameters :=
  Codec.pullback ajtaiCodec parametersData parametersData_injective

theorem parametersGoldCodec_admissible (parameters : Parameters) :
    parametersGoldCodec.Admissible parameters :=
  ajtaiCodec_admissible parameters.ajtaiKey

noncomputable def assignmentGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec Assignment :=
  Codec.finFunction assignmentColumns fieldCodec

theorem assignmentGoldCodec_admissible (assignment : Assignment) :
    assignmentGoldCodec.Admissible assignment := by
  intro column
  trivial

def runningClaimData (claim : RunningClaim) :=
  (claim.commitment,
    (claim.publicInput, (claim.point, claim.evaluation)))

theorem runningClaimData_injective : Function.Injective runningClaimData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def runningClaimGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec RunningClaim :=
  Codec.pullback
    (Codec.product
      (commitmentCodec PaddedRowIdentityConcreteAlgebra.verifierRows)
      (Codec.product
        (publicInputCodec
          PaddedRowIdentityConcreteAlgebra.relationShape.publicWidth)
        (Codec.product (pointCodec rowVariables)
          PaddedRowIdentityCodec.evaluationCodec)))
    runningClaimData runningClaimData_injective

theorem runningClaimGoldCodec_admissible (claim : RunningClaim) :
    runningClaimGoldCodec.Admissible claim := by
  exact ⟨commitmentCodec_admissible claim.commitment,
    publicInputCodec_admissible claim.publicInput,
    pointCodec_admissible claim.point,
    PaddedRowIdentityCodec.evaluationCodec_admissible claim.evaluation⟩

def freshClaimData (claim : FreshClaim) :=
  (claim.commitment, claim.publicInput)

theorem freshClaimData_injective : Function.Injective freshClaimData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def freshClaimGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec FreshClaim :=
  Codec.pullback
    (Codec.product
      (commitmentCodec PaddedRowIdentityConcreteAlgebra.verifierRows)
      (publicInputCodec
        PaddedRowIdentityConcreteAlgebra.relationShape.publicWidth))
    freshClaimData freshClaimData_injective

theorem freshClaimGoldCodec_admissible (claim : FreshClaim) :
    freshClaimGoldCodec.Admissible claim := by
  exact ⟨commitmentCodec_admissible claim.commitment,
    publicInputCodec_admissible claim.publicInput⟩

noncomputable def parametersCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec Parameters F :=
  toTotalNivcCodec parametersGoldCodec

noncomputable def structureCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec Structure F :=
  PaddedRowIdentityCompilerDescription.codec

noncomputable def assignmentCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec Assignment F :=
  toTotalNivcCodec assignmentGoldCodec

noncomputable def runningClaimCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec RunningClaim F :=
  toTotalNivcCodec runningClaimGoldCodec

noncomputable def freshClaimCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec FreshClaim F :=
  toTotalNivcCodec freshClaimGoldCodec

theorem parametersCodec_canonical : parametersCodec.Canonical :=
  toTotalNivcCodec_canonical parametersGoldCodec
    parametersGoldCodec_admissible

theorem structureCodec_canonical : structureCodec.Canonical :=
  PaddedRowIdentityCompilerDescription.codec_canonical

theorem assignmentCodec_canonical : assignmentCodec.Canonical :=
  toTotalNivcCodec_canonical assignmentGoldCodec
    assignmentGoldCodec_admissible

theorem runningClaimCodec_canonical : runningClaimCodec.Canonical :=
  toTotalNivcCodec_canonical runningClaimGoldCodec
    runningClaimGoldCodec_admissible

theorem freshClaimCodec_canonical : freshClaimCodec.Canonical :=
  toTotalNivcCodec_canonical freshClaimGoldCodec
    freshClaimGoldCodec_admissible

/-! ## Selected relation semantics and universal default -/

/-- Definition 12 semantics specialized to the selected CE/CCS pair. -/
def semantics
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat) :
    Semantics Parameters Circuit Input Advice Output
      Structure Structure RunningClaim Assignment FreshClaim Assignment where
  execute := execute
  runningHolds := fun parameters system claim witness =>
    RunningHolds parameters.ajtaiKey
      (PaddedRowIdentityCompilerDescription.matrices system) claim witness
  freshUnderlyingHolds := fun parameters system claim witness =>
    FreshHolds parameters.ajtaiKey
      (PaddedRowIdentityCompilerDescription.matrices system) claim witness
  runningStructureAdmissible := fun _ _ => True
  structuresCompatible := Eq
  circuitSize := circuitSize
  structureSize := fun _ => 2 ^ rowVariables
  structureRows := fun _ => 2 ^ rowVariables
  structureColumns := fun _ => assignmentColumns

/-- Exact canonical zero-padding condition for the selected rectangular
relation. Every coefficient outside the finite logical-row prefix is zero in
each of the thirteen matrices used by CCS. -/
def CanonicalZeroPadding (system : Structure) : Prop :=
  forall matrix vertex column,
    logicalRows <=
        Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowIndex
          vertex ->
      (PaddedRowIdentity.applicationSystem
          (PaddedRowIdentityCompilerDescription.matrices system)).matrices
        matrix vertex column = 0

theorem canonicalZeroPadding (system : Structure) :
    CanonicalZeroPadding system := by
  intro matrix vertex column padding
  change
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding.padRows
        ((PaddedRowIdentityCompilerDescription.matrices system).matrixAt matrix)
        vertex column = 0
  exact
    Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding.padRows_atPadding
      _ vertex column padding

/-- The deterministic committed-zero pair from HyperNova's base case. -/
def defaultAlgorithm : DefaultAlgorithm Parameters RunningClaim Assignment where
  choose := fun _ => (zeroClaim, zeroAssignment)

theorem defaultAlgorithm_holds
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat) :
    defaultAlgorithm.Holds (semantics execute circuitSize) := by
  intro parameters system _
  exact zeroClaim_holds parameters.ajtaiKey
    (PaddedRowIdentityCompilerDescription.matrices system)

/-! ## Rectangular compiler capacity -/

def compilerLayout
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat)
    (encoding : Encoding Circuit Input Advice Output
      Structure Structure FreshClaim Assignment) :
    CompilerLayout (semantics execute circuitSize) encoding where
  Fits := fun _ _ => True
  rowCapacity := fun _ => 2 ^ rowVariables
  columnCapacity := fun _ => assignmentColumns
  columnsFitRows := fun _ => assignmentColumns_covered
  paddedCanonical := fun _ system => CanonicalZeroPadding system

theorem compilerLayout_holds
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat)
    (encoding : Encoding Circuit Input Advice Output
      Structure Structure FreshClaim Assignment) :
    (compilerLayout execute circuitSize encoding).Holds := by
  intro parameters circuit _
  exact ⟨rfl, rfl,
    canonicalZeroPadding (encoding.encodeStructures circuit).2⟩

theorem monotone
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat)
    (encoding : Encoding Circuit Input Advice Output
      Structure Structure FreshClaim Assignment) :
    Monotone (semantics execute circuitSize) encoding := by
  intro left right _
  exact Nat.le_refl _

/-! ## Poseidon2 statement identifier and fixed compact verifier -/

abbrev VerifierData := Unit
abbrev VerifierProjection := Unit
abbrev StatementId := PaddedRowIdentityPoseidon2.StatementId
abbrev VerifierInput := PublicRunning × PublicFresh × NifsProof
abbrev VerifierOutput := Option PublicRunning
abbrev FullStatement :=
  Nightstream.HyperNova.NIVCCompatibility.FullStatement
    Parameters Structure Structure VerifierData

noncomputable def unitGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec Unit :=
  Codec.ofInjectiveEncoding 0 (fun _ => True) (fun _ => [])
    (fun _ => rfl) (by intro left right _ _ _; cases left; cases right; rfl)

theorem unitGoldCodec_admissible (value : Unit) :
    unitGoldCodec.Admissible value := by
  trivial

noncomputable def unitCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec Unit F :=
  toTotalNivcCodec unitGoldCodec

theorem unitCodec_canonical : unitCodec.Canonical :=
  toTotalNivcCodec_canonical unitGoldCodec unitGoldCodec_admissible

def fullStatementData (statement : FullStatement) :=
  (statement.parameters,
    (statement.runningStructure,
      (statement.freshStructure, statement.verifierKey)))

theorem fullStatementData_injective :
    Function.Injective fullStatementData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def fullStatementProductCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec
      (Parameters × (Structure × (Structure × Unit))) F :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product parametersCodec
    (Nightstream.HyperNova.NIVCCompatibility.Codec.product structureCodec
      (Nightstream.HyperNova.NIVCCompatibility.Codec.product structureCodec
        unitCodec))

theorem fullStatementProductCodec_canonical :
    fullStatementProductCodec.Canonical := by
  apply Nightstream.HyperNova.NIVCCompatibility.Codec.product_canonical
  · exact parametersCodec_canonical
  · apply Nightstream.HyperNova.NIVCCompatibility.Codec.product_canonical
    · exact structureCodec_canonical
    · apply Nightstream.HyperNova.NIVCCompatibility.Codec.product_canonical
      · exact structureCodec_canonical
      · exact unitCodec_canonical

noncomputable def fullStatementBaseCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec FullStatement F :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.pullback
    fullStatementProductCodec fullStatementData

theorem fullStatementBaseCodec_canonical :
    fullStatementBaseCodec.Canonical :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.pullback_canonical
    fullStatementProductCodec fullStatementData
    fullStatementProductCodec_canonical fullStatementData_injective

/-- Exact Construction 3 statement-identifier prefix: `dst_MF`, the literal
`statement-id` label, and the complete fixed public event schedule. -/
def statementIdentifierPrefix : List F :=
  PaddedRowIdentityPoseidon2.statementIdentifierPrefixFields.map
    PaddedRowIdentityCodec.fieldOfNat

@[simp] theorem statementIdentifierPrefix_length :
    statementIdentifierPrefix.length = 353 := by
  simp [statementIdentifierPrefix]

noncomputable def taggedStatementFields (statement : FullStatement) : List F :=
  statementIdentifierPrefix ++ fullStatementBaseCodec.encode statement

noncomputable def statementCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec FullStatement F :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.withClassicalDecoder
    taggedStatementFields

noncomputable def identifierGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec StatementId :=
  Codec.finFunction 4 fieldCodec

noncomputable def identifierCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec StatementId F :=
  toTotalNivcCodec identifierGoldCodec

theorem statementCodec_canonical : statementCodec.Canonical :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.injectivePrefixFree_canonical
    taggedStatementFields
    (by
      intro left right equal
      apply Nightstream.HyperNova.NIVCCompatibility.Codec.encode_injective
        fullStatementBaseCodec fullStatementBaseCodec_canonical
      exact (List.append_right_inj statementIdentifierPrefix).mp equal)
    (by
      intro left right suffix prefixed
      have bodyPrefixed :
          fullStatementBaseCodec.encode right =
            fullStatementBaseCodec.encode left ++ suffix := by
        apply (List.append_right_inj statementIdentifierPrefix).mp
        simpa only [taggedStatementFields, List.append_assoc] using prefixed
      exact fullStatementBaseCodec_canonical.2.2 left right suffix
        bodyPrefixed)

/-- After the exact Construction 3 prefix, the statement transcript contains
only the parameter coordinates and two canonical sparse compiler streams. No
dense matrix coordinate appears. -/
theorem statementCodec_encode_exact (statement : FullStatement) :
    statementCodec.encode statement =
      statementIdentifierPrefix ++
        (parametersGoldCodec.encode statement.parameters ++
          (PaddedRowIdentityCompilerDescription.fields
              statement.runningStructure ++
            (PaddedRowIdentityCompilerDescription.fields
              statement.freshStructure ++ []))) := by
  rfl

/-- Exact logical statement size after removal of dense matrix tables. -/
theorem statementCodec_encode_length (statement : FullStatement) :
    (statementCodec.encode statement).length =
      353 + parametersGoldCodec.width +
        (962 + 3 * statement.runningStructure.entryCount) +
        (962 + 3 * statement.freshStructure.entryCount) := by
  rw [statementCodec_encode_exact]
  simp [parametersGoldCodec.encode_length,
    PaddedRowIdentityCompilerDescription.fields_length]
  omega

theorem identifierCodec_canonical : identifierCodec.Canonical :=
  toTotalNivcCodec_canonical identifierGoldCodec (by
    intro _ _
    trivial)

/-- Fixed-length Poseidon2 compression of the complete canonical statement.
One gated permutation supplies all four digest lanes. -/
def poseidon2Identifier (fields : List F) : StatementId :=
  let digestState := Poseidon2Duplex.gate PaddedRowIdentityPoseidon2.constants
    (Poseidon2Duplex.absorbList PaddedRowIdentityPoseidon2.constants
      (fields.map fun field => field.val) Poseidon2Duplex.empty)
  fun lane => PaddedRowIdentityCodec.fieldOfNat
    (digestState.lanes
      ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩)

noncomputable def statementIdentifier :
    StatementIdentifierScheme FullStatement StatementId F where
  statementCodec := statementCodec
  domainLabel := statementIdentifierPrefix
  hash := poseidon2Identifier
  identifierCodec := identifierCodec
  identifierWidth := 4

theorem statementIdentifier_holds : statementIdentifier.Holds := by
  refine ⟨statementCodec_canonical, identifierCodec_canonical, ?_, ?_, ?_⟩
  · change statementIdentifierPrefix ≠ []
    intro empty
    have sameLength := congrArg List.length empty
    simp at sameLength
  · intro statement
    exact ⟨fullStatementBaseCodec.encode statement, rfl⟩
  · intro statementId
    exact identifierGoldCodec.encode_length statementId

/-- Equal Poseidon2 statement identifiers bind the exact thirteen-matrix
families used by both relation positions, or expose the named complete-
statement collision event. No digest equality is used as matrix authority. -/
theorem statementIdentifier_matrices_eq_or_collision
    (left right : FullStatement)
    (sameIdentifier :
      statementIdentifier.identifier left =
        statementIdentifier.identifier right) :
    (PaddedRowIdentityCompilerDescription.matrices left.runningStructure =
        PaddedRowIdentityCompilerDescription.matrices right.runningStructure /\
      PaddedRowIdentityCompilerDescription.matrices left.freshStructure =
        PaddedRowIdentityCompilerDescription.matrices right.freshStructure) \/
      statementIdentifier.Collision := by
  rcases statementIdentifier.eq_or_collision left right sameIdentifier with
    sameStatement | collision
  · subst right
    exact Or.inl ⟨rfl, rfl⟩
  · exact Or.inr collision

noncomputable def projectionCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec VerifierProjection F :=
  toTotalNivcCodec unitGoldCodec

theorem projectionCodec_canonical : projectionCodec.Canonical :=
  toTotalNivcCodec_canonical unitGoldCodec unitGoldCodec_admissible

def fixedStatement (parameters : Parameters) (system : Structure) :
    FullStatement where
  parameters := parameters
  runningStructure := system
  freshStructure := system
  verifierKey := ()

noncomputable def verifyFull
    (statement : FullStatement) (input : VerifierInput) : VerifierOutput :=
  PaddedRowIdentityConcreteNifs.verify
    (PaddedRowIdentityConcreteNifs.key
      (statementIdentifier.identifier statement)
      statement.parameters.ajtaiKey
        (PaddedRowIdentityCompilerDescription.matrices
          statement.freshStructure))
    input.1 input.2.1 input.2.2

noncomputable def verifyRecursive
    (key : RecursiveVerifierKey VerifierProjection StatementId)
    (input : VerifierInput) : VerifierOutput :=
  PaddedRowIdentityConcreteNifs.verify
    (PaddedRowIdentityConcreteNifs.compactKey key.statementId)
    input.1 input.2.1 input.2.2

noncomputable def compactVerifier
    (parameters : Parameters) (system : Structure) :
    CompactVerifierInterface F Parameters Structure Structure VerifierData
      VerifierProjection StatementId VerifierInput VerifierOutput where
  fixedStatement := fixedStatement parameters system
  declaredSizeBound := fun _ => rowVariables
  project := fun _ _ => ()
  projectionCodec := projectionCodec
  projectionWidth := fun _ => 0
  statementIdentifier := statementIdentifier
  verifyFull := verifyFull
  verifyRecursive := verifyRecursive

theorem compactVerifier_holds
    (parameters : Parameters) (system : Structure) :
    (compactVerifier parameters system).Holds := by
  refine ⟨projectionCodec_canonical, statementIdentifier_holds, ?_, ?_⟩
  · intro _ _
    rfl
  · intro input
    simpa only [compactVerifier, CompactVerifierInterface.recursiveKey,
      verifyRecursive, verifyFull] using
      (PaddedRowIdentityConcreteNifs.verify_eq_compact
        (statementIdentifier.identifier (fixedStatement parameters system))
        parameters.ajtaiKey
        (PaddedRowIdentityCompilerDescription.matrices system)
        input.1 input.2.1 input.2.2).symm

/-! ## Construction 2 setup with Definition 12 statement binding -/

/-- The exact fixed-length statement identifier for one selected augmented
circuit. It binds the public parameters, both structures, and verifier data. -/
noncomputable def statementId
    (parameters : Parameters) (system : Structure) : StatementId :=
  statementIdentifier.identifier (fixedStatement parameters system)

/-- Construction 2 setup whose every NIFS transcript starts with the exact
Definition 12 statement identifier for that slot. -/
noncomputable def construction2Setup {slotCount : Nat}
    (parameters : Fin slotCount -> Parameters)
    (systems : Fin slotCount -> Structure) :=
  PaddedRowIdentityHyperNova.setup
    (fun slot => statementId (parameters slot) (systems slot))
    (fun slot => (parameters slot).ajtaiKey)
    (fun slot =>
      PaddedRowIdentityCompilerDescription.matrices (systems slot))

@[simp] theorem construction2Setup_verifierKey
    {slotCount : Nat}
    (parameters : Fin slotCount -> Parameters)
    (systems : Fin slotCount -> Structure)
    (slot : Fin slotCount) :
    (construction2Setup parameters systems).verifierKeys slot =
      PaddedRowIdentityConcreteNifs.key
        (statementId (parameters slot) (systems slot))
        (parameters slot).ajtaiKey
        (PaddedRowIdentityCompilerDescription.matrices (systems slot)) := by
  rfl

/-- The selected verifier key starts its Poseidon2 transcript with the exact
full-statement identifier. -/
theorem construction2Setup_initialTranscriptState
    {slotCount : Nat}
    (parameters : Fin slotCount -> Parameters)
    (systems : Fin slotCount -> Structure)
    (slot : Fin slotCount) :
    ((construction2Setup parameters systems).verifierKeys slot).initialTranscriptState =
      PaddedRowIdentityPoseidon2.initialStateForStatement
        (statementId (parameters slot) (systems slot)) := by
  rw [construction2Setup_verifierKey,
    PaddedRowIdentityConcreteNifs.key_initialTranscriptState]

/-! ## Application-owned compiler boundary and full closure -/

/-- The only canonical codecs not fixed by the protocol are the source
circuit tuple and its public input/output pair. -/
structure ApplicationCodecs
    (Circuit Input Advice Output : Type) where
  sourceTuple :
    Nightstream.HyperNova.NIVCCompatibility.Codec
      (Circuit × Input × Advice × Output) F
  inputOutput :
    Nightstream.HyperNova.NIVCCompatibility.Codec (Input × Output) F
  sourceCanonical : sourceTuple.Canonical
  inputOutputCanonical : inputOutput.Canonical

noncomputable def structuresCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec
      (Structure × Structure) F :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product
    structureCodec structureCodec

theorem structuresCodec_canonical : structuresCodec.Canonical :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product_canonical
    structureCodec structureCodec structureCodec_canonical
      structureCodec_canonical

noncomputable def freshAssignmentCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec
      (FreshClaim × Assignment) F :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product
    freshClaimCodec assignmentCodec

theorem freshAssignmentCodec_canonical : freshAssignmentCodec.Canonical :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product_canonical
    freshClaimCodec assignmentCodec freshClaimCodec_canonical
      assignmentCodec_canonical

noncomputable def encodedTupleCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec
      (Structure × FreshClaim × Assignment) F :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product
    structureCodec freshAssignmentCodec

theorem encodedTupleCodec_canonical : encodedTupleCodec.Canonical :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.product_canonical
    structureCodec freshAssignmentCodec structureCodec_canonical
      freshAssignmentCodec_canonical

noncomputable def canonicalLayouts
    {Circuit Input Advice Output : Type}
    (codecs : ApplicationCodecs Circuit Input Advice Output) :
    CanonicalLayouts F Parameters Circuit Input Advice Output
      Structure Structure RunningClaim Assignment FreshClaim Assignment where
  parameters := parametersCodec
  sourceTuple := codecs.sourceTuple
  structures := structuresCodec
  inputOutput := codecs.inputOutput
  runningInstance := runningClaimCodec
  runningWitness := assignmentCodec
  freshInstance := freshClaimCodec
  encodedTuple := encodedTupleCodec

theorem canonicalLayouts_holds
    {Circuit Input Advice Output : Type}
    (codecs : ApplicationCodecs Circuit Input Advice Output) :
    (canonicalLayouts codecs).Holds := by
  refine ⟨parametersCodec_canonical, codecs.sourceCanonical, ?_,
    codecs.inputOutputCanonical, runningClaimCodec_canonical,
    assignmentCodec_canonical, freshClaimCodec_canonical, ?_⟩
  · exact structuresCodec_canonical
  · exact encodedTupleCodec_canonical

/-- Application compiler evidence that the protocol cannot derive. Every
field is an exact corrected Definition 12 compiler law. -/
structure ApplicationCompiler
    (Circuit Input Advice Output : Type)
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat) where
  encoding : Encoding Circuit Input Advice Output
    Structure Structure FreshClaim Assignment
  codecs : ApplicationCodecs Circuit Input Advice Output
  npComplete : NPComplete (semantics execute circuitSize) encoding
  partialFunctions :
    PartialFunctions encoding (semantics execute circuitSize).structuresCompatible

/-- Full corrected Definition 12 closure for one fixed selected augmented
circuit. The application compiler remains an explicit, proof-carrying input;
all SuperNeo-specific obligations are discharged here. -/
theorem definition12_holds
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat)
    (compiler : ApplicationCompiler Circuit Input Advice Output
      execute circuitSize)
    (recursiveSize : RecursiveSizeClosure F Parameters Advice)
    (recursiveSizeHolds :
      recursiveSize.Holds
        (compilerLayout execute circuitSize compiler.encoding))
    (parameters : Parameters)
    (system : Structure) :
    Nightstream.HyperNova.NIVCCompatibility.Holds
      (semantics execute circuitSize) compiler.encoding
      (canonicalLayouts compiler.codecs)
      (compilerLayout execute circuitSize compiler.encoding)
      recursiveSize defaultAlgorithm (compactVerifier parameters system) := by
  exact ⟨compiler.npComplete, compiler.partialFunctions,
    monotone execute circuitSize compiler.encoding,
    canonicalLayouts_holds compiler.codecs,
    compilerLayout_holds execute circuitSize compiler.encoding,
    recursiveSizeHolds,
    defaultAlgorithm_holds execute circuitSize,
    compactVerifier_holds parameters system⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility
