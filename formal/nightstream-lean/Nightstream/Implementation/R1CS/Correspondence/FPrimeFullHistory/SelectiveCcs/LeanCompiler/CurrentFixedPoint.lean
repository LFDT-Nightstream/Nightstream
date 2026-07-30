import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentM4

/-!
Contract: construct the recursive same-system fixed point for one
system-parametric, Lean-owned fixed-one deployment family.

Assurance tier: model-level.

Owns: the fixed selective constraint polynomial, a zero-matrix seed at the
selected relation shape, the exact row/domain/column shape equations, and the
compiler interface required to replace the seed matrices with the matrices
compiled from the complete Step rows.

Does not own: a WASM application compiler, one application deployment family,
an Ajtai key, selection of shipping program bytes, Rust emission, or equality
with Rust.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs

universe u

private abbrev TranscriptState := Poseidon2Duplex.State

/-- The independent thirteen-port polynomial, transported to the selected
dimension record. -/
def fixedPolynomial
    (dimensions : Dimensions)
    (matrixCountExact : dimensions.matrixCount = RelationProfile.matrixCount) :
    CCSResidualTable.ConstraintPolynomial F dimensions.shape.matrixCount := by
  rw [Dimensions.shape]
  rw [matrixCountExact]
  exact Polynomial.Semantics.polynomial

/-- Shape-only seed used to discover the complete Step encoding. Its matrices
carry no authority and are replaced by the exact compiler output. -/
def seedSystem
    (dimensions : Dimensions)
    (matrixCountExact : dimensions.matrixCount = RelationProfile.matrixCount) :
    Structure dimensions.shape where
  matrices := fun _ _ _ => 0
  constraintPolynomial := fixedPolynomial dimensions matrixCountExact

/-- A candidate recursive system keeps the selected SuperNeo constraint
polynomial fixed while its matrix payload changes.

The fixed-point compiler may replace relation matrices. It may not range over
an unrelated constraint polynomial because that polynomial changes the
selected verifier program itself. -/
def UsesFixedPolynomial
    (dimensions : Dimensions)
    (matrixCountExact : dimensions.matrixCount = RelationProfile.matrixCount)
    (system : Structure dimensions.shape) : Prop :=
  system.constraintPolynomial =
    fixedPolynomial dimensions matrixCountExact

@[simp] theorem seedSystem_usesFixedPolynomial
    (dimensions : Dimensions)
    (matrixCountExact :
      dimensions.matrixCount = RelationProfile.matrixCount) :
    UsesFixedPolynomial dimensions matrixCountExact
      (seedSystem dimensions matrixCountExact) :=
  rfl

/-- Exact dimension equations required before the compiler output can replace
the seed relation without changing any recursive carrier type. -/
structure ShapeFixedPoint
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (dimensions : Dimensions)
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length) : Prop where
  rowVariablesExact :
    dimensions.rowVariables =
      (Profile.ofEncoding encoding publicWidth).rowVariables
  logicalWidthExact :
    dimensions.alignedLogicalWidth = encoding.columnIds.length
  matrixCountExact :
    dimensions.matrixCount = RelationProfile.matrixCount

private theorem shape_extensionality
    (left right : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape)
    (rowVariables : left.rowVariables = right.rowVariables)
    (logicalWidth : left.logicalWidth = right.logicalWidth)
    (matrixCount : left.matrixCount = right.matrixCount)
    (publicRingColumns :
      left.publicRingColumns = right.publicRingColumns) :
    left = right := by
  cases left with
  | mk leftRows leftWidth leftMatrices leftPublic leftFits =>
    cases right with
    | mk rightRows rightWidth rightMatrices rightPublic rightFits =>
      simp only at rowVariables logicalWidth matrixCount publicRingColumns
      subst rightRows
      subst rightWidth
      subst rightMatrices
      subst rightPublic
      rfl

/-- A shape fixed point identifies the selected recursive relation shape with
the exact shape inferred from the emitted rows and columns. -/
theorem shape_eq_compiled
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {dimensions : Dimensions}
    {encoding : Encoding source}
    {publicWidth : 270 ≤ encoding.columnIds.length}
    (fixed : ShapeFixedPoint dimensions encoding publicWidth) :
    dimensions.shape =
      RelationProfile.Profile.shape
        (Profile.ofEncoding encoding publicWidth) := by
  apply shape_extensionality
  · exact fixed.rowVariablesExact
  · exact fixed.logicalWidthExact
  · exact fixed.matrixCountExact
  · rfl

/-- Setup fields that are independent of the relation matrices. -/
structure SetupTemplate
    (dimensions : Dimensions)
    (verifierRows : Nat) where
  verifierKey :
    VerifierKey
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns (publicFits dimensions) verifierRows
  domainCovers :
    PiCcsDomains.production.nc.Covers
      (ConcreteNifsPlain270Profile.Shape dimensions)
  rowNonempty :
    0 < (ConcreteNifsPlain270Profile.Shape dimensions).rowVariables

namespace SetupTemplate

/-- Install one relation system without changing any other setup authority. -/
def withSystem
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (system : Structure dimensions.shape) :
    RelationSetup dimensions verifierRows where
  verifierKey := template.verifierKey
  system := system
  domainCovers := template.domainCovers
  rowNonempty := template.rowNonempty

@[simp] theorem withSystem_system
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (system : Structure dimensions.shape) :
    HEq (template.withSystem system).system system := by
  rfl

end SetupTemplate

section DeploymentFamily

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable
  (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

private abbrev DeploymentFor
    (template : SetupTemplate dimensions verifierRows)
    (system : Structure dimensions.shape) :=
  ConcreteNifsCanonicalCertification.Deployment
    (template.withSystem system) defaultRunning machine terminalRelations
      terminalChecks widths footprints

/-- Complete Step encoding emitted for one matrix candidate. -/
noncomputable def stepEncoding
    (template : SetupTemplate dimensions verifierRows)
    (system : Structure dimensions.shape)
    (deployment :
      DeploymentFor defaultRunning machine terminalRelations terminalChecks
        widths footprints template system) :=
  (ConcreteNifsCanonicalCertification.complete
    (template.withSystem system) defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment
    ).canonicalStep.program.toEncoding

/-- The protocol-owned 270-coordinate prefix is present in every complete
Step encoding in the family. -/
def stepPublicWidth
    (template : SetupTemplate dimensions verifierRows)
    (system : Structure dimensions.shape)
    (deployment :
      DeploymentFor defaultRunning machine terminalRelations terminalChecks
        widths footprints template system) :
    270 ≤
      (stepEncoding defaultRunning machine terminalRelations terminalChecks
        widths footprints template system deployment).columnIds.length :=
  CurrentDeployment.deployment_step_columns_ge_270
    (template.withSystem system) defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment

/-- A proof-carrying deployment compiler whose physical Step relation is
independent of the seed matrix coefficients while the selected SuperNeo
constraint polynomial stays fixed.

`physicalStable` is not a caller-supplied relation equality. A concrete
application compiler must prove that its exact physical columns, rows, and
constant-one column do not change with the seed matrix coefficients. The
generic compiler transports those facts to compiled-system equality. -/
structure Family
    (template : SetupTemplate dimensions verifierRows) where
  matrixCountExact :
    dimensions.matrixCount = RelationProfile.matrixCount
  deployment :
    ∀ (system : Structure dimensions.shape),
      UsesFixedPolynomial dimensions matrixCountExact system ->
      DeploymentFor defaultRunning machine terminalRelations terminalChecks
        widths footprints template system
  shapeFixed :
    ∀ (system : Structure dimensions.shape)
      (polynomialFixed :
        UsesFixedPolynomial dimensions matrixCountExact system),
      let selected := deployment system polynomialFixed
      ShapeFixedPoint dimensions
        (stepEncoding defaultRunning machine terminalRelations terminalChecks
          widths footprints template system selected)
        (stepPublicWidth defaultRunning machine terminalRelations
          terminalChecks widths footprints template system selected)
  physicalStable :
    ∀ system : Structure dimensions.shape,
      ∀ polynomialFixed :
        UsesFixedPolynomial dimensions matrixCountExact system,
      CurrentCompiler.PhysicalEncoding.ofEncoding
          (stepEncoding defaultRunning machine terminalRelations terminalChecks
            widths footprints template
            (seedSystem dimensions matrixCountExact)
            (deployment
              (seedSystem dimensions matrixCountExact)
              (seedSystem_usesFixedPolynomial dimensions matrixCountExact))) =
        CurrentCompiler.PhysicalEncoding.ofEncoding
          (stepEncoding defaultRunning machine terminalRelations terminalChecks
            widths footprints template system
            (deployment system polynomialFixed))

private theorem castStructure_constraintPolynomial
    {left right : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (same : left = right)
    (system : Structure left) :
    (Eq.mp (congrArg Structure same) system).constraintPolynomial =
      Eq.mp
        (congrArg
          (fun shape =>
            CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
          same)
        system.constraintPolynomial := by
  cases same
  rfl

namespace Family

/-- Physical application stability is transported once through the generic
compiler. A deployment family does not prove equality of dependent relation
records directly. -/
theorem compilationStable
    {template : SetupTemplate dimensions verifierRows}
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    let seed := seedSystem dimensions family.matrixCountExact
    let seedDeployment :=
      family.deployment seed
        (seedSystem_usesFixedPolynomial dimensions family.matrixCountExact)
    let seedEncoding :=
      stepEncoding defaultRunning machine terminalRelations terminalChecks
        widths footprints template seed seedDeployment
    let seedPublic :=
      stepPublicWidth defaultRunning machine terminalRelations terminalChecks
        widths footprints template seed seedDeployment
    ∀ system : Structure dimensions.shape,
      ∀ polynomialFixed :
        UsesFixedPolynomial dimensions family.matrixCountExact system,
      let selected := family.deployment system polynomialFixed
      let encoding :=
        stepEncoding defaultRunning machine terminalRelations terminalChecks
          widths footprints template system selected
      let publicWidth :=
        stepPublicWidth defaultRunning machine terminalRelations terminalChecks
          widths footprints template system selected
      HEq (CurrentCompiler.compiledSystem seedEncoding seedPublic)
        (CurrentCompiler.compiledSystem encoding publicWidth) := by
  dsimp only
  intro system polynomialFixed
  have stable := family.physicalStable system polynomialFixed
  exact
    CurrentCompiler.compiledSystem_heq_of_physicalEncoding_eq
      (stepEncoding defaultRunning machine terminalRelations terminalChecks
        widths footprints template
        (seedSystem dimensions family.matrixCountExact)
        (family.deployment
          (seedSystem dimensions family.matrixCountExact)
          (seedSystem_usesFixedPolynomial dimensions family.matrixCountExact)))
      (stepEncoding defaultRunning machine terminalRelations terminalChecks
        widths footprints template system
        (family.deployment system polynomialFixed))
      (stepPublicWidth defaultRunning machine terminalRelations terminalChecks
        widths footprints template
        (seedSystem dimensions family.matrixCountExact)
        (family.deployment
          (seedSystem dimensions family.matrixCountExact)
          (seedSystem_usesFixedPolynomial dimensions family.matrixCountExact)))
      (stepPublicWidth defaultRunning machine terminalRelations terminalChecks
        widths footprints template system
        (family.deployment system polynomialFixed))
      stable

/-- The seed deployment used only to compute the final matrix payload. -/
noncomputable def seedDeployment
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :=
  family.deployment
    (seedSystem dimensions family.matrixCountExact)
    (seedSystem_usesFixedPolynomial dimensions family.matrixCountExact)

/-- Exact complete Step encoding produced from the shape-only seed. -/
noncomputable def seedEncoding
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :=
  stepEncoding defaultRunning machine terminalRelations terminalChecks
    widths footprints template
    (seedSystem dimensions family.matrixCountExact)
    (family.seedDeployment defaultRunning machine terminalRelations
      terminalChecks widths footprints template)

/-- The selected public-prefix proof for the seed encoding. -/
noncomputable def seedPublicWidth
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    270 ≤
      (family.seedEncoding defaultRunning machine terminalRelations
        terminalChecks widths footprints template).columnIds.length :=
  stepPublicWidth defaultRunning machine terminalRelations terminalChecks
    widths footprints template
    (seedSystem dimensions family.matrixCountExact)
    (family.seedDeployment defaultRunning machine terminalRelations
      terminalChecks widths footprints template)

/-- Equality between the recursive carrier shape and the seed compiler
output shape. -/
theorem seedShape_eq
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    dimensions.shape =
      RelationProfile.Profile.shape
        (Profile.ofEncoding
          (family.seedEncoding defaultRunning machine terminalRelations
            terminalChecks widths footprints template)
          (family.seedPublicWidth defaultRunning machine terminalRelations
            terminalChecks widths footprints template)) :=
  by
    simpa [seedEncoding, seedPublicWidth, seedDeployment] using
      shape_eq_compiled
        (family.shapeFixed
          (seedSystem dimensions family.matrixCountExact)
          (seedSystem_usesFixedPolynomial dimensions family.matrixCountExact))

/-- Replace the seed matrices with the matrices compiled from the exact seed
Step encoding, transported back to the recursive carrier shape. -/
noncomputable def finalSystem
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    Structure dimensions.shape :=
  Eq.mp
    (congrArg Structure
      (family.seedShape_eq defaultRunning machine terminalRelations
        terminalChecks widths footprints template).symm)
    (CurrentCompiler.compiledSystem
      (family.seedEncoding defaultRunning machine terminalRelations
        terminalChecks widths footprints template)
      (family.seedPublicWidth defaultRunning machine terminalRelations
        terminalChecks widths footprints template))

/-- The transported compiler output retains the selected SuperNeo constraint
polynomial. -/
theorem finalSystem_usesFixedPolynomial
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    UsesFixedPolynomial dimensions family.matrixCountExact
      (family.finalSystem defaultRunning machine terminalRelations
        terminalChecks widths footprints template) := by
  let encoding :=
    family.seedEncoding defaultRunning machine terminalRelations
      terminalChecks widths footprints template
  let publicWidth :=
    family.seedPublicWidth defaultRunning machine terminalRelations
      terminalChecks widths footprints template
  let same :=
    family.seedShape_eq defaultRunning machine terminalRelations
      terminalChecks widths footprints template
  change
    (Eq.mp (congrArg Structure same.symm)
      (CurrentCompiler.compiledSystem encoding publicWidth)
    ).constraintPolynomial =
      fixedPolynomial dimensions family.matrixCountExact
  rw [castStructure_constraintPolynomial same.symm
    (CurrentCompiler.compiledSystem encoding publicWidth)]
  rw [CurrentCompiler.compiledSystem_constraintPolynomial]
  unfold fixedPolynomial
  apply eq_of_heq
  exact HEq.trans (cast_heq _ _) (cast_heq _ _).symm

/-- The final setup is constructed from the compiled system. -/
noncomputable def finalSetup
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    RelationSetup dimensions verifierRows :=
  template.withSystem
    (family.finalSystem defaultRunning machine terminalRelations
      terminalChecks widths footprints template)

/-- Rebuild the deployment at the exact compiled relation. -/
noncomputable def finalDeployment
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    ConcreteNifsCanonicalCertification.Deployment
      (family.finalSetup defaultRunning machine terminalRelations
        terminalChecks widths footprints template)
      defaultRunning machine terminalRelations terminalChecks
        widths footprints :=
  family.deployment
    (family.finalSystem defaultRunning machine terminalRelations
      terminalChecks widths footprints template)
    (family.finalSystem_usesFixedPolynomial defaultRunning machine
      terminalRelations terminalChecks widths footprints template)

/-- Transporting the seed compiler output back to the recursive carrier shape
does not change the system value. -/
theorem finalSystem_heq_seedCompiled
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    HEq
      (family.finalSystem defaultRunning machine terminalRelations
        terminalChecks widths footprints template)
      (CurrentCompiler.compiledSystem
        (family.seedEncoding defaultRunning machine terminalRelations
          terminalChecks widths footprints template)
        (family.seedPublicWidth defaultRunning machine terminalRelations
          terminalChecks widths footprints template)) := by
  unfold finalSystem
  exact cast_heq _ _

/-- Recompiling the final deployment returns the system installed in its
selected NIFS key. -/
theorem finalSystem_heq_recompiled
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    let system :=
      family.finalSystem defaultRunning machine terminalRelations
        terminalChecks widths footprints template
    let deployment :=
      family.finalDeployment defaultRunning machine terminalRelations
        terminalChecks widths footprints template
    let encoding :=
      stepEncoding defaultRunning machine terminalRelations terminalChecks
        widths footprints template system deployment
    let publicWidth :=
      stepPublicWidth defaultRunning machine terminalRelations terminalChecks
        widths footprints template system deployment
    HEq system
      (CurrentCompiler.compiledSystem encoding publicWidth) := by
  dsimp only
  exact HEq.trans
    (family.finalSystem_heq_seedCompiled defaultRunning machine
      terminalRelations terminalChecks widths footprints template)
    (compilationStable defaultRunning machine terminalRelations
      terminalChecks widths footprints family
      (family.finalSystem defaultRunning machine terminalRelations
        terminalChecks widths footprints template)
      (family.finalSystem_usesFixedPolynomial defaultRunning machine
        terminalRelations terminalChecks widths footprints template))

/-- The fixed-point compiler discharges the recursive same-system condition
required by current M4. -/
theorem recursiveSystemCoherence
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    CurrentM4.RecursiveSystemCoherence
      (family.finalSetup defaultRunning machine terminalRelations
        terminalChecks widths footprints template)
      defaultRunning machine terminalRelations terminalChecks
        widths footprints
      (family.finalDeployment defaultRunning machine terminalRelations
        terminalChecks widths footprints template) := by
  let system :=
    family.finalSystem defaultRunning machine terminalRelations
      terminalChecks widths footprints template
  let deployment :=
    family.finalDeployment defaultRunning machine terminalRelations
      terminalChecks widths footprints template
  have fixed :=
    family.shapeFixed system
      (family.finalSystem_usesFixedPolynomial defaultRunning machine
        terminalRelations terminalChecks widths footprints template)
  unfold CurrentM4.RecursiveSystemCoherence
  change
    dimensions.rowVariables =
        (Profile.ofEncoding
          (stepEncoding defaultRunning machine terminalRelations
            terminalChecks widths footprints template system deployment)
          (stepPublicWidth defaultRunning machine terminalRelations
            terminalChecks widths footprints template system deployment)
        ).rowVariables ∧
      dimensions.alignedLogicalWidth =
        (stepEncoding defaultRunning machine terminalRelations
          terminalChecks widths footprints template system deployment
        ).columnIds.length ∧
      dimensions.matrixCount = 13 ∧
      HEq system
        (CurrentCompiler.compiledSystem
          (stepEncoding defaultRunning machine terminalRelations
            terminalChecks widths footprints template system deployment)
          (stepPublicWidth defaultRunning machine terminalRelations
            terminalChecks widths footprints template system deployment))
  refine ⟨fixed.rowVariablesExact, fixed.logicalWidthExact, ?_, ?_⟩
  · simpa [RelationProfile.matrixCount,
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.portCount]
      using fixed.matrixCountExact
  · exact
      family.finalSystem_heq_recompiled defaultRunning machine
        terminalRelations terminalChecks widths footprints template

/-- Current recursive M4 evidence constructed from the exact fixed point. -/
noncomputable def m4
    (template : SetupTemplate dimensions verifierRows)
    (family :
      Family defaultRunning machine terminalRelations terminalChecks
        widths footprints template) :
    CurrentM4.Evidence
      (family.finalSetup defaultRunning machine terminalRelations
        terminalChecks widths footprints template)
      defaultRunning machine terminalRelations terminalChecks
        widths footprints
      (family.finalDeployment defaultRunning machine terminalRelations
        terminalChecks widths footprints template) :=
  CurrentM4.deployment_m4
    (family.finalSetup defaultRunning machine terminalRelations
      terminalChecks widths footprints template)
    defaultRunning machine terminalRelations terminalChecks widths footprints
    (family.finalDeployment defaultRunning machine terminalRelations
      terminalChecks widths footprints template)
    (family.recursiveSystemCoherence defaultRunning machine
      terminalRelations terminalChecks widths footprints template)

end Family

end DeploymentFamily

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
