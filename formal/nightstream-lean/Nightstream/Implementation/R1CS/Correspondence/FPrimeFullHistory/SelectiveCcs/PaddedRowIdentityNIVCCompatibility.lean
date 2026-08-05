import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityHyperNova
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Verifier

/-!
Contract: corrected HyperNova Definition 12 boundary for the selected
`PaddedRowIdentity` SuperNeo NIFS.

Owns: distinct running CE and fresh CCS carriers, canonical protocol codecs,
the universal committed-zero running pair, the fixed rectangular compiler
capacity, a Poseidon2 statement identifier, and a compact verifier projection
for one fixed augmented circuit.

Does not own: an application circuit compiler. The final theorem takes the
application encoder plus its NP-completeness and inverse proofs as explicit
inputs. It does not turn a missing compiler into a protocol theorem.

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
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova

abbrev Structure := PaddedRowIdentity.ApplicationMatrices
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

def matrixData (matrices : Structure) :
    Fin applicationMatrixCount ->
      Fin logicalRows -> Fin assignmentColumns -> F :=
  fun matrix => matrices.matrixAt matrix

theorem matrixData_injective : Function.Injective matrixData := by
  intro left right equal
  cases left with
  | mk leftMatrices =>
      cases right with
      | mk rightMatrices =>
          congr
          funext role row column
          have coordinate :=
            congrFun (congrFun (congrFun equal role.index) row) column
          have leftCoordinate := congrFun (congrFun
            (RelationProfile.FiniteRelation.matrixAt_role
              ({ matrices := leftMatrices } : Structure) role) row) column
          have rightCoordinate := congrFun (congrFun
            (RelationProfile.FiniteRelation.matrixAt_role
              ({ matrices := rightMatrices } : Structure) role) row) column
          exact leftCoordinate.symm.trans (coordinate.trans rightCoordinate)

noncomputable def structureGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec Structure :=
  Codec.pullback
    (Codec.finFunction applicationMatrixCount
      (Codec.finFunction logicalRows
        (Codec.finFunction assignmentColumns fieldCodec)))
    matrixData matrixData_injective

theorem structureGoldCodec_admissible (matrices : Structure) :
    structureGoldCodec.Admissible matrices := by
  intro matrix row column
  trivial

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
  toTotalNivcCodec structureGoldCodec

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
  toTotalNivcCodec_canonical structureGoldCodec
    structureGoldCodec_admissible

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
    RunningHolds parameters.ajtaiKey system claim witness
  freshUnderlyingHolds := fun parameters system claim witness =>
    FreshHolds parameters.ajtaiKey system claim witness
  runningStructureAdmissible := fun _ _ => True
  structuresCompatible := Eq
  circuitSize := circuitSize
  structureSize := fun _ => 2 ^ rowVariables
  structureRows := fun _ => 2 ^ rowVariables
  structureColumns := fun _ => assignmentColumns

/-- The deterministic committed-zero pair from HyperNova's base case. -/
def defaultAlgorithm : DefaultAlgorithm Parameters RunningClaim Assignment where
  choose := fun _ => (zeroClaim, zeroAssignment)

theorem defaultAlgorithm_holds
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat) :
    defaultAlgorithm.Holds (semantics execute circuitSize) := by
  intro parameters system _
  exact zeroClaim_holds parameters.ajtaiKey system

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
  paddedCanonical := fun _ _ => True

theorem compilerLayout_holds
    {Circuit Input Advice Output : Type}
    (execute : Circuit -> Input -> Advice -> Output)
    (circuitSize : Circuit -> Nat)
    (encoding : Encoding Circuit Input Advice Output
      Structure Structure FreshClaim Assignment) :
    (compilerLayout execute circuitSize encoding).Holds := by
  intro parameters circuit _
  exact ⟨rfl, rfl, True.intro⟩

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
abbrev StatementId := F
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

noncomputable def fullStatementBaseCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec FullStatement :=
  Codec.pullback
    (Codec.product parametersGoldCodec
      (Codec.product structureGoldCodec
        (Codec.product structureGoldCodec unitGoldCodec)))
    fullStatementData fullStatementData_injective

theorem fullStatementBaseCodec_admissible (statement : FullStatement) :
    fullStatementBaseCodec.Admissible statement := by
  exact ⟨parametersGoldCodec_admissible statement.parameters,
    structureGoldCodec_admissible statement.runningStructure,
    structureGoldCodec_admissible statement.freshStructure,
    unitGoldCodec_admissible statement.verifierKey⟩

def statementDomain : F := PaddedRowIdentityCodec.fieldOfNat 1201

noncomputable def taggedStatementGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec FullStatement :=
  Codec.ofInjectiveEncoding (fullStatementBaseCodec.width + 1)
    (fun _ => True)
    (fun statement => statementDomain :: fullStatementBaseCodec.encode statement)
    (by
      intro statement
      simp [fullStatementBaseCodec.encode_length])
    (by
      intro left right _ _ equal
      have tail :
          fullStatementBaseCodec.encode left =
            fullStatementBaseCodec.encode right :=
        (List.cons.inj equal).2
      exact fullStatementBaseCodec.encode_injective_of_admissible
        (fullStatementBaseCodec_admissible left)
        (fullStatementBaseCodec_admissible right) tail)

theorem taggedStatementGoldCodec_admissible (statement : FullStatement) :
    taggedStatementGoldCodec.Admissible statement := by
  trivial

noncomputable def statementCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec FullStatement F :=
  toTotalNivcCodec taggedStatementGoldCodec

noncomputable def identifierCodec :
    Nightstream.HyperNova.NIVCCompatibility.Codec F F :=
  toTotalNivcCodec fieldCodec

theorem statementCodec_canonical : statementCodec.Canonical :=
  toTotalNivcCodec_canonical taggedStatementGoldCodec
    taggedStatementGoldCodec_admissible

theorem identifierCodec_canonical : identifierCodec.Canonical :=
  toTotalNivcCodec_canonical fieldCodec (fun _ => True.intro)

/-- Fixed-length Poseidon2 compression of the complete canonical statement. -/
def poseidon2Identifier (fields : List F) : F :=
  PaddedRowIdentityCodec.fieldOfNat
    (Poseidon2Duplex.challengeField PaddedRowIdentityPoseidon2.constants
      (Poseidon2Duplex.absorbList PaddedRowIdentityPoseidon2.constants
        (fields.map fun field => field.val) Poseidon2Duplex.empty)).1

noncomputable def statementIdentifier :
    StatementIdentifierScheme FullStatement StatementId F where
  statementCodec := statementCodec
  domainLabel := [statementDomain]
  hash := poseidon2Identifier
  identifierCodec := identifierCodec
  identifierWidth := 1

theorem statementIdentifier_holds : statementIdentifier.Holds := by
  refine ⟨statementCodec_canonical, identifierCodec_canonical, ?_, ?_, ?_⟩
  · change ([statementDomain] : List F) ≠ []
    simp
  · intro statement
    exact ⟨fullStatementBaseCodec.encode statement, rfl⟩
  · intro statementId
    exact fieldCodec.encode_length statementId

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
  verify
    (PaddedRowIdentityConcreteNifs.key
      (statementIdentifier.identifier statement)
      statement.parameters.ajtaiKey statement.freshStructure)
    input.1 input.2.1 input.2.2

noncomputable def verifyRecursive
    (key : RecursiveVerifierKey VerifierProjection StatementId)
    (input : VerifierInput) : VerifierOutput :=
  verify
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
  projectionCompact := fun _ _ => True
  statementIdentifier := statementIdentifier
  verifyFull := verifyFull
  verifyRecursive := verifyRecursive

theorem compactVerifier_holds
    (parameters : Parameters) (system : Structure) :
    (compactVerifier parameters system).Holds := by
  refine ⟨projectionCodec_canonical, statementIdentifier_holds, ?_, ?_⟩
  · exact ⟨rfl, True.intro⟩
  · intro input
    simpa only [compactVerifier, CompactVerifierInterface.recursiveKey,
      verifyRecursive, verifyFull] using
      (PaddedRowIdentityConcreteNifs.verify_eq_compact
        (statementIdentifier.identifier (fixedStatement parameters system))
        parameters.ajtaiKey system input.1 input.2.1 input.2.2).symm

/-! ## Construction 2 setup with Definition 12 statement binding -/

/-- The exact fixed-length statement identifier for one selected augmented
circuit. It binds the public parameters, both structures, and verifier data. -/
noncomputable def statementId (parameters : Parameters) (system : Structure) : F :=
  statementIdentifier.identifier (fixedStatement parameters system)

/-- Construction 2 setup whose every NIFS transcript starts with the exact
Definition 12 statement identifier for that slot. -/
noncomputable def construction2Setup {slotCount : Nat}
    (parameters : Fin slotCount -> Parameters)
    (systems : Fin slotCount -> Structure) :=
  PaddedRowIdentityHyperNova.setup
    (fun slot => statementId (parameters slot) (systems slot))
    (fun slot => (parameters slot).ajtaiKey)
    systems

@[simp] theorem construction2Setup_verifierKey
    {slotCount : Nat}
    (parameters : Fin slotCount -> Parameters)
    (systems : Fin slotCount -> Structure)
    (slot : Fin slotCount) :
    (construction2Setup parameters systems).verifierKeys slot =
      PaddedRowIdentityConcreteNifs.key
        (statementId (parameters slot) (systems slot))
        (parameters slot).ajtaiKey (systems slot) := by
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

noncomputable def encodedTupleGoldCodec :
    Nightstream.Implementation.Lowering.Goldilocks.Codec
      (Structure × FreshClaim × Assignment) :=
  Codec.product structureGoldCodec
    (Codec.product freshClaimGoldCodec assignmentGoldCodec)

theorem encodedTupleGoldCodec_admissible
    (value : Structure × FreshClaim × Assignment) :
    encodedTupleGoldCodec.Admissible value := by
  exact ⟨structureGoldCodec_admissible value.1,
    freshClaimGoldCodec_admissible value.2.1,
    assignmentGoldCodec_admissible value.2.2⟩

noncomputable def canonicalLayouts
    {Circuit Input Advice Output : Type}
    (codecs : ApplicationCodecs Circuit Input Advice Output) :
    CanonicalLayouts F Parameters Circuit Input Advice Output
      Structure Structure RunningClaim Assignment FreshClaim Assignment where
  parameters := parametersCodec
  sourceTuple := codecs.sourceTuple
  structures := toTotalNivcCodec
    (Codec.product structureGoldCodec structureGoldCodec)
  inputOutput := codecs.inputOutput
  runningInstance := runningClaimCodec
  runningWitness := assignmentCodec
  freshInstance := freshClaimCodec
  encodedTuple := toTotalNivcCodec encodedTupleGoldCodec

theorem canonicalLayouts_holds
    {Circuit Input Advice Output : Type}
    (codecs : ApplicationCodecs Circuit Input Advice Output) :
    (canonicalLayouts codecs).Holds := by
  refine ⟨parametersCodec_canonical, codecs.sourceCanonical, ?_,
    codecs.inputOutputCanonical, runningClaimCodec_canonical,
    assignmentCodec_canonical, freshClaimCodec_canonical, ?_⟩
  · exact toTotalNivcCodec_canonical
      (Codec.product structureGoldCodec structureGoldCodec)
      (fun value => ⟨structureGoldCodec_admissible value.1,
        structureGoldCodec_admissible value.2⟩)
  · exact toTotalNivcCodec_canonical encodedTupleGoldCodec
      encodedTupleGoldCodec_admissible

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
    (parameters : Parameters)
    (system : Structure) :
    Nightstream.HyperNova.NIVCCompatibility.Holds
      (semantics execute circuitSize) compiler.encoding
      (canonicalLayouts compiler.codecs)
      (compilerLayout execute circuitSize compiler.encoding)
      defaultAlgorithm (compactVerifier parameters system) := by
  exact ⟨compiler.npComplete, compiler.partialFunctions,
    monotone execute circuitSize compiler.encoding,
    canonicalLayouts_holds compiler.codecs,
    compilerLayout_holds execute circuitSize compiler.encoding,
    defaultAlgorithm_holds execute circuitSize,
    compactVerifier_holds parameters system⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility
