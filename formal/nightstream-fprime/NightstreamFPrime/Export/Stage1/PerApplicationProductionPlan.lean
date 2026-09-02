import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Owns the compact Lean instruction program for one per-application 14-matrix
relation. Each opcode denotes one proved direct-plan constructor. The ordered
program is the transport authority; consumers may interpret it but may not
select block order, row counts, geometry, or application rows.

This module defines the in-Lean interpreter and proves that the canonical
program expands to the self-derived structural plan. Package encoding and the
Rust interpreter are separate consumers of this result.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationProductionPlan

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev ProgramApplication := Lifecycle.Stage1.Application.Program

abbrev FitsTwoPow28 (application : ProgramApplication) :=
  PerApplicationFixedPoint.FitsTwoPow28 application

/-- Fixed production-plan instruction vocabulary. The order is supplied by
`canonical`; enum order has no semantic effect. -/
inductive BlockKind where
  | pilotPoseidon
  | piCcsPoseidon
  | piCcsOrdinary
  | pilotOrdinary
  | pilotDigestBinding
  | piCcsEndpoint
  | samplerPoseidon
  | samplerOrdinary
  | piRlc
  | piDec
  | runningTransition
  | application
  | nextPreimage
  | recursivePublicOutput
deriving Repr, DecidableEq

def BlockKind.format : Format BlockKind where
  encode
    | .pilotPoseidon => .atom 0
    | .piCcsPoseidon => .atom 1
    | .piCcsOrdinary => .atom 2
    | .pilotOrdinary => .atom 3
    | .pilotDigestBinding => .atom 4
    | .piCcsEndpoint => .atom 5
    | .samplerPoseidon => .atom 6
    | .samplerOrdinary => .atom 7
    | .piRlc => .atom 8
    | .piDec => .atom 9
    | .runningTransition => .atom 10
    | .application => .atom 11
    | .nextPreimage => .atom 12
    | .recursivePublicOutput => .atom 13
  decode
    | .atom 0 => .ok .pilotPoseidon
    | .atom 1 => .ok .piCcsPoseidon
    | .atom 2 => .ok .piCcsOrdinary
    | .atom 3 => .ok .pilotOrdinary
    | .atom 4 => .ok .pilotDigestBinding
    | .atom 5 => .ok .piCcsEndpoint
    | .atom 6 => .ok .samplerPoseidon
    | .atom 7 => .ok .samplerOrdinary
    | .atom 8 => .ok .piRlc
    | .atom 9 => .ok .piDec
    | .atom 10 => .ok .runningTransition
    | .atom 11 => .ok .application
    | .atom 12 => .ok .nextPreimage
    | .atom 13 => .ok .recursivePublicOutput
    | _ => .error "invalid production matrix block kind"
  decode_encode := by
    intro kind
    cases kind <;> rfl

/-- Small tree language for ordered plan concatenation. -/
inductive Program where
  | leaf (kind : BlockKind)
  | append (left right : Program)
deriving Repr, DecidableEq

def applicationGeometry (application : ProgramApplication) :=
  PerApplicationFixedPoint.geometry application

def samplerGeometry (application : ProgramApplication) :=
  DirectApplicationPrefixPlan.prefixGeometry (applicationGeometry application)

def piDecGeometry (application : ProgramApplication) :=
  DirectPiRLCSamplerCompletePrefixPlan.piDecGeometry
    (samplerGeometry application)

def relation (application : ProgramApplication)
    (fits : FitsTwoPow28 application) :=
  PerApplicationFixedPoint.seedRelation application fits

/-- Exact direct plan selected by one opcode. -/
def BlockKind.plan (application : ProgramApplication)
    (fits : FitsTwoPow28 application) : BlockKind →
    ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth application)
  | .pilotPoseidon => DirectPiDECPrefixPlan.pilotPlan
      (piDecGeometry application)
  | .piCcsPoseidon => DirectPiDECPrefixPlan.piCcsPoseidonPlan
      (piDecGeometry application)
  | .piCcsOrdinary => DirectPiDECPrefixPlan.piCcsOrdinaryPlan
      (relation application fits) (piDecGeometry application)
  | .pilotOrdinary => DirectPiDECPrefixPlan.pilotOrdinaryPlan
      (piDecGeometry application)
  | .pilotDigestBinding => DirectPiDECPrefixPlan.pilotBindingPlan
      (piDecGeometry application)
  | .piCcsEndpoint => DirectPiDECPrefixPlan.piCcsEndpointPlan
      (piDecGeometry application)
  | .samplerPoseidon => DirectPiDECPrefixPlan.samplerPlan
      (piDecGeometry application)
  | .samplerOrdinary =>
      DirectPiRLCSamplerCompletePrefixPlan.samplerOrdinaryPlan
        (relation application fits) (samplerGeometry application)
  | .piRlc => DirectPiRLCSamplerCompletePrefixPlan.piRlcPlan
      (samplerGeometry application)
  | .piDec => DirectPiRLCSamplerCompletePrefixPlan.piDecPlan
      (relation application fits) (samplerGeometry application)
  | .runningTransition =>
      DirectPiRLCSamplerCompletePrefixPlan.transitionPlan
        (relation application fits) (samplerGeometry application)
  | .application => DirectApplicationPrefixPlan.applicationPlan fits.package
      (applicationGeometry application)
  | .nextPreimage => DirectApplicationPrefixPlan.nextPreimagePlan
      (applicationGeometry application)
  | .recursivePublicOutput => DirectApplicationPrefixPlan.publicOutputPlan
      (applicationGeometry application)

/-- Direct wire-facing live-row count for one production block. -/
def BlockKind.rowCount (application : ProgramApplication) : BlockKind → Nat
  | .pilotPoseidon => 2321800
  | .piCcsPoseidon => 729984
  | .piCcsOrdinary => 811669
  | .pilotOrdinary => 1330
  | .pilotDigestBinding => 8
  | .piCcsEndpoint => 32
  | .samplerPoseidon => 14382
  | .samplerOrdinary => 220881
  | .piRlc => 1898781
  | .piDec => 25488
  | .runningTransition => 345495
  | .application => (PerApplicationPackage.applicationPlan application).rowCount
  | .nextPreimage => 5
  | .recursivePublicOutput => 4

/-- The direct block count is exactly the count of the selected semantic
plan. -/
theorem BlockKind.plan_rowCount (application : ProgramApplication)
    (fits : FitsTwoPow28 application) (kind : BlockKind) :
    (kind.plan application fits).rowCount = kind.rowCount application := by
  cases kind <;>
    simp [BlockKind.plan, BlockKind.rowCount,
      DirectPiDECPrefixPlan.piCcsOrdinaryPlan,
      DirectPiRLCSamplerCompletePrefixPlan.samplerOrdinaryPlan,
      DirectPiRLCSamplerCompletePrefixPlan.piDecPlan,
      DirectPiDECPrefixPlan.piDecPlan,
      DirectPiRLCSamplerCompletePrefixPlan.transitionPlan,
      DirectPiDECPrefixPlan.transitionPlan,
      DirectApplicationPrefixPlan.applicationPlan,
      DirectApplicationPrefixPlan.nextPreimagePlan,
      DirectApplicationPrefixPlan.publicOutputPlan]

/-- Interpreter for the compact tree. Every concatenation checks the final
row bound before it constructs a plan. -/
def Program.compile (application : ProgramApplication)
    (fits : FitsTwoPow28 application) : Program →
    Option (ProductionRelation.Plan
      (PerApplicationFixedPoint.logicalWidth application))
  | .leaf kind => some (kind.plan application fits)
  | .append left right => do
      let leftPlan ← left.compile application fits
      let rightPlan ← right.compile application fits
      if bounded : leftPlan.rowCount + rightPlan.rowCount ≤
          2 ^ Lifecycle.cubeVariables then
        some (ProductionRelation.Plan.append leftPlan rightPlan bounded)
      else
        none

def piCcsPoseidonPrefixProgram : Program :=
  .append (.leaf .pilotPoseidon) (.leaf .piCcsPoseidon)

def piCcsCoreProgram : Program :=
  .append piCcsPoseidonPrefixProgram (.leaf .piCcsOrdinary)

def pilotOrdinaryPrefixProgram : Program :=
  .append piCcsCoreProgram (.leaf .pilotOrdinary)

def pilotBindingPrefixProgram : Program :=
  .append pilotOrdinaryPrefixProgram (.leaf .pilotDigestBinding)

def piCcsCompleteProgram : Program :=
  .append pilotBindingPrefixProgram (.leaf .piCcsEndpoint)

def samplerPrefixProgram : Program :=
  .append piCcsCompleteProgram (.leaf .samplerPoseidon)

def samplerCompleteProgram : Program :=
  .append samplerPrefixProgram (.leaf .samplerOrdinary)

def piRlcCompleteProgram : Program :=
  .append samplerCompleteProgram (.leaf .piRlc)

def piDecCompleteProgram : Program :=
  .append piRlcCompleteProgram (.leaf .piDec)

def runningCompleteProgram : Program :=
  .append piDecCompleteProgram (.leaf .runningTransition)

def applicationCompleteProgram : Program :=
  .append runningCompleteProgram (.leaf .application)

def throughNextPreimageProgram : Program :=
  .append applicationCompleteProgram (.leaf .nextPreimage)

def canonical : Program :=
  .append throughNextPreimageProgram (.leaf .recursivePublicOutput)

def Program.kinds : Program → List BlockKind
  | .leaf kind => [kind]
  | .append left right => left.kinds ++ right.kinds

/-- Exact flat transport order of the canonical tree. -/
def canonicalKinds : List BlockKind :=
  [.pilotPoseidon, .piCcsPoseidon, .piCcsOrdinary, .pilotOrdinary,
    .pilotDigestBinding, .piCcsEndpoint, .samplerPoseidon,
    .samplerOrdinary, .piRlc, .piDec, .runningTransition, .application,
    .nextPreimage, .recursivePublicOutput]

@[simp] theorem canonical_kinds : canonical.kinds = canonicalKinds := by
  rfl

/-- The flat wire schedule has the exact live-row count of the semantic
structural plan. -/
theorem canonicalKinds_rowCount (application : ProgramApplication)
    (fits : FitsTwoPow28 application) :
    (canonicalKinds.map fun kind => kind.rowCount application).sum =
      (PerApplicationFixedPoint.structuralPlan application fits).rowCount := by
  rw [PerApplicationFixedPoint.structuralPlan_rowCount]
  norm_num [canonicalKinds, BlockKind.rowCount]
  omega

/-- One identity-bound ordered production block header. -/
structure BlockHeader where
  kind : BlockKind
  rowStart : Nat
  rowCount : Nat
deriving Repr, DecidableEq

def BlockHeader.format : Format BlockHeader where
  encode := fun header => .array [
    BlockKind.format.encode header.kind,
    .atom header.rowStart,
    .atom header.rowCount]
  decode
    | .array [kind, .atom rowStart, .atom rowCount] => do
      pure ⟨← BlockKind.format.decode kind, rowStart, rowCount⟩
    | _ => .error "invalid production matrix block header"
  decode_encode := by
    intro header
    cases header
    simp [BlockKind.format.decode_encode]

def headersFrom (application : ProgramApplication) :
    Nat → List BlockKind → List BlockHeader
  | _, [] => []
  | rowStart, kind :: rest =>
      { kind := kind
        rowStart := rowStart
        rowCount := kind.rowCount application } ::
      headersFrom application (rowStart + kind.rowCount application) rest

/-- Exact identity-bound header stream for the canonical production tree. -/
def canonicalHeaders (application : ProgramApplication) : List BlockHeader :=
  headersFrom application 0 canonicalKinds

theorem headersFrom_kinds (application : ProgramApplication)
    (rowStart : Nat) (kinds : List BlockKind) :
    (headersFrom application rowStart kinds).map BlockHeader.kind = kinds := by
  induction kinds generalizing rowStart with
  | nil => rfl
  | cons kind rest inductionHypothesis =>
      simp [headersFrom, inductionHypothesis]

@[simp] theorem canonicalHeaders_kinds (application : ProgramApplication) :
    (canonicalHeaders application).map BlockHeader.kind = canonicalKinds := by
  exact headersFrom_kinds application 0 canonicalKinds

theorem headersFrom_rowCount (application : ProgramApplication)
    (rowStart : Nat) (kinds : List BlockKind) :
    ((headersFrom application rowStart kinds).map BlockHeader.rowCount).sum =
      (kinds.map fun kind => kind.rowCount application).sum := by
  induction kinds generalizing rowStart with
  | nil => rfl
  | cons kind rest inductionHypothesis =>
      simp [headersFrom, inductionHypothesis]

theorem canonicalHeaders_rowCount (application : ProgramApplication)
    (fits : FitsTwoPow28 application) :
    ((canonicalHeaders application).map BlockHeader.rowCount).sum =
      (PerApplicationFixedPoint.structuralPlan application fits).rowCount := by
  rw [canonicalHeaders, headersFrom_rowCount]
  exact canonicalKinds_rowCount application fits

private theorem compile_append
    (application : ProgramApplication) (fits : FitsTwoPow28 application)
    {left right : Program}
    {leftPlan rightPlan : ProductionRelation.Plan
      (PerApplicationFixedPoint.logicalWidth application)}
    (leftCompiled : left.compile application fits = some leftPlan)
    (rightCompiled : right.compile application fits = some rightPlan)
    (bounded : leftPlan.rowCount + rightPlan.rowCount ≤
      2 ^ Lifecycle.cubeVariables) :
    (Program.append left right).compile application fits =
      some (ProductionRelation.Plan.append leftPlan rightPlan bounded) := by
  simp [Program.compile, leftCompiled, rightCompiled, bounded]

private theorem compile_piCcsPoseidonPrefixProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    piCcsPoseidonPrefixProgram.compile application fits =
      some (DirectPiDECPrefixPlan.piCcsPoseidonPrefix
        (piDecGeometry application)) := by
  unfold piCcsPoseidonPrefixProgram
  have bounded :
      (DirectPiDECPrefixPlan.pilotPlan
          (piDecGeometry application)).rowCount +
        (DirectPiDECPrefixPlan.piCcsPoseidonPlan
          (piDecGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiDECPrefixPlan.piCcsPoseidonPrefix
      (piDecGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiDECPrefixPlan.pilotPlan (piDecGeometry application))
        (DirectPiDECPrefixPlan.piCcsPoseidonPlan
          (piDecGeometry application)) bounded) :=
      compile_append application fits rfl rfl bounded
    _ = _ := by
      unfold DirectPiDECPrefixPlan.piCcsPoseidonPrefix
      rfl

private theorem compile_piCcsCoreProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    piCcsCoreProgram.compile application fits =
      some (DirectPiDECPrefixPlan.piCcsCorePlan
        (relation application fits) (piDecGeometry application)) := by
  unfold piCcsCoreProgram
  have bounded :
      (DirectPiDECPrefixPlan.piCcsPoseidonPrefix
          (piDecGeometry application)).rowCount +
        (DirectPiDECPrefixPlan.piCcsOrdinaryPlan
          (relation application fits) (piDecGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiDECPrefixPlan.piCcsCorePlan
      (relation application fits) (piDecGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiDECPrefixPlan.piCcsPoseidonPrefix
          (piDecGeometry application))
        (DirectPiDECPrefixPlan.piCcsOrdinaryPlan
          (relation application fits) (piDecGeometry application)) bounded) :=
      compile_append application fits
        (compile_piCcsPoseidonPrefixProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiDECPrefixPlan.piCcsCorePlan
      rfl

private theorem compile_pilotOrdinaryPrefixProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    pilotOrdinaryPrefixProgram.compile application fits =
      some (DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan
        (relation application fits) (piDecGeometry application)) := by
  unfold pilotOrdinaryPrefixProgram
  have bounded :
      (DirectPiDECPrefixPlan.piCcsCorePlan
          (relation application fits) (piDecGeometry application)).rowCount +
        (DirectPiDECPrefixPlan.pilotOrdinaryPlan
          (piDecGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan
      (relation application fits) (piDecGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiDECPrefixPlan.piCcsCorePlan
          (relation application fits) (piDecGeometry application))
        (DirectPiDECPrefixPlan.pilotOrdinaryPlan
          (piDecGeometry application)) bounded) :=
      compile_append application fits
        (compile_piCcsCoreProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan
      rfl

private theorem compile_pilotBindingPrefixProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    pilotBindingPrefixProgram.compile application fits =
      some (DirectPiDECPrefixPlan.pilotBindingPrefixPlan
        (relation application fits) (piDecGeometry application)) := by
  unfold pilotBindingPrefixProgram
  have bounded :
      (DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan
          (relation application fits) (piDecGeometry application)).rowCount +
        (DirectPiDECPrefixPlan.pilotBindingPlan
          (piDecGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiDECPrefixPlan.pilotBindingPrefixPlan
      (relation application fits) (piDecGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan
          (relation application fits) (piDecGeometry application))
        (DirectPiDECPrefixPlan.pilotBindingPlan
          (piDecGeometry application)) bounded) :=
      compile_append application fits
        (compile_pilotOrdinaryPrefixProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiDECPrefixPlan.pilotBindingPrefixPlan
      rfl

private theorem compile_piCcsCompleteProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    piCcsCompleteProgram.compile application fits =
      some (DirectPiDECPrefixPlan.piCcsCompletePlan
        (relation application fits) (piDecGeometry application)) := by
  unfold piCcsCompleteProgram
  have bounded :
      (DirectPiDECPrefixPlan.pilotBindingPrefixPlan
          (relation application fits) (piDecGeometry application)).rowCount +
        (DirectPiDECPrefixPlan.piCcsEndpointPlan
          (piDecGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiDECPrefixPlan.piCcsCompletePlan
      (relation application fits) (piDecGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiDECPrefixPlan.pilotBindingPrefixPlan
          (relation application fits) (piDecGeometry application))
        (DirectPiDECPrefixPlan.piCcsEndpointPlan
          (piDecGeometry application)) bounded) :=
      compile_append application fits
        (compile_pilotBindingPrefixProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiDECPrefixPlan.piCcsCompletePlan
      rfl

private theorem compile_samplerPrefixProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    samplerPrefixProgram.compile application fits =
      some (DirectPiDECPrefixPlan.samplerPrefixPlan
        (relation application fits) (piDecGeometry application)) := by
  unfold samplerPrefixProgram
  have bounded :
      (DirectPiDECPrefixPlan.piCcsCompletePlan
          (relation application fits) (piDecGeometry application)).rowCount +
        (DirectPiDECPrefixPlan.samplerPlan
          (piDecGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiDECPrefixPlan.samplerPrefixPlan
      (relation application fits) (piDecGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiDECPrefixPlan.piCcsCompletePlan
          (relation application fits) (piDecGeometry application))
        (DirectPiDECPrefixPlan.samplerPlan
          (piDecGeometry application)) bounded) :=
      compile_append application fits
        (compile_piCcsCompleteProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiDECPrefixPlan.samplerPrefixPlan
      rfl

private theorem compile_samplerCompleteProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    samplerCompleteProgram.compile application fits =
      some (DirectPiRLCSamplerCompletePrefixPlan.samplerCompletePlan
        (relation application fits) (samplerGeometry application)) := by
  unfold samplerCompleteProgram
  have leftEq :
      DirectPiDECPrefixPlan.samplerPrefixPlan
          (relation application fits) (piDecGeometry application) =
        DirectPiRLCSamplerCompletePrefixPlan.samplerPrefixPlan
          (relation application fits) (samplerGeometry application) := by
    rfl
  have bounded :
      (DirectPiRLCSamplerCompletePrefixPlan.samplerPrefixPlan
          (relation application fits) (samplerGeometry application)).rowCount +
        (DirectPiRLCSamplerCompletePrefixPlan.samplerOrdinaryPlan
          (relation application fits) (samplerGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiRLCSamplerCompletePrefixPlan.samplerCompletePlan
      (relation application fits) (samplerGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiRLCSamplerCompletePrefixPlan.samplerPrefixPlan
          (relation application fits) (samplerGeometry application))
        (DirectPiRLCSamplerCompletePrefixPlan.samplerOrdinaryPlan
          (relation application fits) (samplerGeometry application)) bounded) :=
      compile_append application fits
        (compile_samplerPrefixProgram application fits |>.trans
          (congrArg some leftEq)) rfl bounded
    _ = _ := by
      unfold DirectPiRLCSamplerCompletePrefixPlan.samplerCompletePlan
      rfl

private theorem compile_piRlcCompleteProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    piRlcCompleteProgram.compile application fits =
      some (DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan
        (relation application fits) (samplerGeometry application)) := by
  unfold piRlcCompleteProgram
  have bounded :
      (DirectPiRLCSamplerCompletePrefixPlan.samplerCompletePlan
          (relation application fits) (samplerGeometry application)).rowCount +
        (DirectPiRLCSamplerCompletePrefixPlan.piRlcPlan
          (samplerGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan
      (relation application fits) (samplerGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiRLCSamplerCompletePrefixPlan.samplerCompletePlan
          (relation application fits) (samplerGeometry application))
        (DirectPiRLCSamplerCompletePrefixPlan.piRlcPlan
          (samplerGeometry application)) bounded) :=
      compile_append application fits
        (compile_samplerCompleteProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan
      rfl

private theorem compile_piDecCompleteProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    piDecCompleteProgram.compile application fits =
      some (DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan
        (relation application fits) (samplerGeometry application)) := by
  unfold piDecCompleteProgram
  have bounded :
      (DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan
          (relation application fits) (samplerGeometry application)).rowCount +
        (DirectPiRLCSamplerCompletePrefixPlan.piDecPlan
          (relation application fits) (samplerGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan
      (relation application fits) (samplerGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan
          (relation application fits) (samplerGeometry application))
        (DirectPiRLCSamplerCompletePrefixPlan.piDecPlan
          (relation application fits) (samplerGeometry application)) bounded) :=
      compile_append application fits
        (compile_piRlcCompleteProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan
      rfl

private theorem compile_runningCompleteProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    runningCompleteProgram.compile application fits =
      some (DirectPiRLCSamplerCompletePrefixPlan.plan
        (relation application fits) (samplerGeometry application)) := by
  unfold runningCompleteProgram
  have bounded :
      (DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan
          (relation application fits) (samplerGeometry application)).rowCount +
        (DirectPiRLCSamplerCompletePrefixPlan.transitionPlan
          (relation application fits) (samplerGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    exact (DirectPiRLCSamplerCompletePrefixPlan.plan
      (relation application fits) (samplerGeometry application)).rowCount_le
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan
          (relation application fits) (samplerGeometry application))
        (DirectPiRLCSamplerCompletePrefixPlan.transitionPlan
          (relation application fits) (samplerGeometry application)) bounded) :=
      compile_append application fits
        (compile_piDecCompleteProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectPiRLCSamplerCompletePrefixPlan.plan
      rfl

private theorem compile_applicationCompleteProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    applicationCompleteProgram.compile application fits =
      some (DirectApplicationPrefixPlan.prefixApplicationPlan
        (relation application fits) fits.package
        (applicationGeometry application)) := by
  unfold applicationCompleteProgram
  have leftEq :
      DirectPiRLCSamplerCompletePrefixPlan.plan
          (relation application fits) (samplerGeometry application) =
        DirectApplicationPrefixPlan.prefixPlan
          (relation application fits) (applicationGeometry application) := by
    rfl
  have leftCompiled :
      runningCompleteProgram.compile application fits =
        some (DirectApplicationPrefixPlan.prefixPlan
          (relation application fits) (applicationGeometry application)) :=
    (compile_runningCompleteProgram application fits).trans
      (congrArg some leftEq)
  have complete := DirectApplicationPrefixPlan.rowCount_le
    (relation application fits) fits.package (applicationGeometry application)
  have bounded :
      (DirectApplicationPrefixPlan.prefixPlan
          (relation application fits) (applicationGeometry application)).rowCount +
        (DirectApplicationPrefixPlan.applicationPlan fits.package
          (applicationGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    omega
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectApplicationPrefixPlan.prefixPlan
          (relation application fits) (applicationGeometry application))
        (DirectApplicationPrefixPlan.applicationPlan fits.package
          (applicationGeometry application)) bounded) :=
      compile_append application fits leftCompiled rfl bounded
    _ = _ := by
      unfold DirectApplicationPrefixPlan.prefixApplicationPlan
      rfl

private theorem compile_throughNextPreimageProgram
    (application : ProgramApplication) (fits : FitsTwoPow28 application) :
    throughNextPreimageProgram.compile application fits =
      some (DirectApplicationPrefixPlan.throughNextPreimagePlan
        (relation application fits) fits.package
        (applicationGeometry application)) := by
  unfold throughNextPreimageProgram
  have complete := DirectApplicationPrefixPlan.rowCount_le
    (relation application fits) fits.package (applicationGeometry application)
  have bounded :
      (DirectApplicationPrefixPlan.prefixApplicationPlan
          (relation application fits) fits.package
          (applicationGeometry application)).rowCount +
        (DirectApplicationPrefixPlan.nextPreimagePlan
          (applicationGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables := by
    rw [DirectApplicationPrefixPlan.prefixApplicationPlan,
      ProductionRelation.Plan.append_rowCount]
    omega
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectApplicationPrefixPlan.prefixApplicationPlan
          (relation application fits) fits.package
          (applicationGeometry application))
        (DirectApplicationPrefixPlan.nextPreimagePlan
          (applicationGeometry application)) bounded) :=
      compile_append application fits
        (compile_applicationCompleteProgram application fits) rfl bounded
    _ = _ := by
      unfold DirectApplicationPrefixPlan.throughNextPreimagePlan
      rfl

/-- The compact instruction program expands to the exact self-derived
14-matrix plan. -/
theorem compile_canonical (application : ProgramApplication)
    (fits : FitsTwoPow28 application) :
    canonical.compile application fits =
      some (PerApplicationFixedPoint.structuralPlan application fits) := by
  unfold canonical
  have bounded :
      (DirectApplicationPrefixPlan.throughNextPreimagePlan
          (relation application fits) fits.package
          (applicationGeometry application)).rowCount +
        (DirectApplicationPrefixPlan.publicOutputPlan
          (applicationGeometry application)).rowCount ≤
        2 ^ Lifecycle.cubeVariables :=
    DirectApplicationPrefixPlan.rowCount_le (relation application fits)
      fits.package (applicationGeometry application)
  calc
    _ = some (ProductionRelation.Plan.append
        (DirectApplicationPrefixPlan.throughNextPreimagePlan
          (relation application fits) fits.package
          (applicationGeometry application))
        (DirectApplicationPrefixPlan.publicOutputPlan
          (applicationGeometry application)) bounded) :=
      compile_append application fits
        (compile_throughNextPreimageProgram application fits) rfl bounded
    _ = some (PerApplicationFixedPoint.structuralPlan application fits) := by
      unfold PerApplicationFixedPoint.structuralPlan
        DirectApplicationPrefixPlan.plan
      rfl

end NightstreamFPrime.Export.Stage1.PerApplicationProductionPlan
