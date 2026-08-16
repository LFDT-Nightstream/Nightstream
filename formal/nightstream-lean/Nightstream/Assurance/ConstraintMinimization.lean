import Mathlib.Algebra.MvPolynomial.Eval
import Mathlib.Data.ZMod.Basic
import Nightstream.Assurance.TerminalContextBoundary
import Nightstream.Assurance.TerminalProofBoundary
import Nightstream.Assurance.TerminalStatementBoundary
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsSchema
import Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards
import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.SuperNeo.Concrete.Algebra
import Nightstream.SuperNeo.CheckPlan

/-!
Artifact-bound checking for recursive-verifier constraint classifications.

Assurance tier: model-level for the generic polynomial theorems. A concrete
result is artifact-checked only after its complete `Artifact` value and
certificate pass the checkers in this module. Rust-conformant and
security-reduced claims require separate composition theorems.

Owns: exact scalar polynomial-combination checking, family-level redundancy
transport, exact artifact equality, and executable removal counterexamples.

Does not own: cvc5 trust, Rust export conformance, protocol soundness,
recursive fixed-point costs, or a global minimum claim.

Emits constraints: no.
-/

namespace Nightstream.Assurance.ConstraintMinimization

open MvPolynomial
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.CheckPlan

namespace Numeric

abbrev Row := Nightstream.Implementation.R1CS.Row
def modulus := Nightstream.Implementation.R1CS.goldilocksP

end Numeric

local instance : NeZero Numeric.modulus := ⟨by decide⟩

abbrev Field := ZMod Numeric.modulus
abbrev Polynomial := MvPolynomial Nat Field

/-- One source row with its stable source index and semantic family. -/
structure IndexedRow where
  sourceIndex : Nat
  family : String
  row : Numeric.Row
deriving DecidableEq, Repr

/-- Complete value checked at the minimization boundary. The diagnostic digest
is metadata. Exact equality plus `WellFormed` is the artifact authority. -/
structure Artifact where
  schema : String
  profile : String
  scope : String
  diagnosticDigest : String
  fieldModulus : String
  totalRows : Nat
  columnCount : Nat
  constantOneColumn : Nat
  publicInputCount : Nat
  completeFamilies : List String
  rows : List IndexedRow
deriving DecidableEq, Repr

namespace Artifact

/-- Supported Rust problem schema at this proof boundary. -/
def supportedSchema : String := "nightstream/r1cs-redundancy-problem/v3"

/-- Canonical decimal Goldilocks modulus carried by the Rust artifact. -/
def goldilocksModulusDecimal : String := toString Numeric.modulus

def scopes : List String := ["local", "branch", "lifecycle"]

def strictlyIncreasingColumns : List (Nat × Nat) → Bool
  | [] | [_] => true
  | first :: second :: tail =>
      decide (first.1 < second.1) &&
        strictlyIncreasingColumns (second :: tail)

def termsWellFormed (columnCount : Nat) (terms : List (Nat × Nat)) : Bool :=
  strictlyIncreasingColumns terms &&
    terms.all fun term => decide
      (term.1 < columnCount ∧ 0 < term.2 ∧ term.2 < Numeric.modulus)

def rowWellFormed (artifact : Artifact) (row : IndexedRow) : Bool :=
  decide (row.sourceIndex < artifact.totalRows ∧ row.family ≠ "") &&
    termsWellFormed artifact.columnCount row.row.a &&
      termsWellFormed artifact.columnCount row.row.b &&
        termsWellFormed artifact.columnCount row.row.c

def strictlyIncreasingSourceRows : List IndexedRow → Bool
  | [] | [_] => true
  | first :: second :: tail =>
      decide (first.sourceIndex < second.sourceIndex) &&
        strictlyIncreasingSourceRows (second :: tail)

theorem strictlyIncreasingSourceRows_of_pairwise
    {rows : List IndexedRow}
    (pairwise : rows.Pairwise fun left right =>
      left.sourceIndex < right.sourceIndex) :
    strictlyIncreasingSourceRows rows = true := by
  induction rows with
  | nil => rfl
  | cons first tail inductionHypothesis =>
      cases tail with
      | nil => rfl
      | cons second rest =>
          rcases List.pairwise_cons.mp pairwise with
            ⟨firstBefore, tailPairwise⟩
          have firstBeforeSecond : first.sourceIndex < second.sourceIndex :=
            firstBefore second (by simp)
          simp [strictlyIncreasingSourceRows, firstBeforeSecond,
            inductionHypothesis tailPairwise]

/-- Structural checks that make the carried Rust problem unambiguous for the
Goldilocks checker in this module. The diagnostic digest remains metadata. -/
def WellFormed (artifact : Artifact) : Prop :=
  artifact.schema = supportedSchema ∧
    artifact.profile ≠ "" ∧
    artifact.scope ∈ scopes ∧
    artifact.diagnosticDigest ≠ "" ∧
    artifact.fieldModulus = goldilocksModulusDecimal ∧
    0 < artifact.totalRows ∧
    0 < artifact.columnCount ∧
    0 < artifact.publicInputCount ∧
    artifact.publicInputCount ≤ artifact.columnCount ∧
    artifact.constantOneColumn < artifact.publicInputCount ∧
    artifact.rows ≠ [] ∧
    artifact.completeFamilies.Nodup ∧
    artifact.completeFamilies.all (fun family => decide (family ≠ "")) = true ∧
    artifact.completeFamilies.all (fun family =>
      artifact.rows.any (fun row => decide (row.family = family))) = true ∧
    strictlyIncreasingSourceRows artifact.rows = true ∧
    artifact.rows.all (rowWellFormed artifact) = true

instance (artifact : Artifact) : Decidable artifact.WellFormed := by
  unfold WellFormed
  infer_instance

/-- Structural input for an artifact validity proof. Large row predicates are
supplied as reusable leaf facts instead of one complete Boolean decision. -/
structure StructuralCertificate (artifact : Artifact) : Prop where
  schemaExact : artifact.schema = supportedSchema
  profilePresent : artifact.profile ≠ ""
  scopeSupported : artifact.scope ∈ scopes
  diagnosticDigestPresent : artifact.diagnosticDigest ≠ ""
  fieldModulusExact : artifact.fieldModulus = goldilocksModulusDecimal
  totalRowsPositive : 0 < artifact.totalRows
  columnCountPositive : 0 < artifact.columnCount
  publicInputCountPositive : 0 < artifact.publicInputCount
  publicInputCountBounded : artifact.publicInputCount ≤ artifact.columnCount
  constantOneColumnPublic :
    artifact.constantOneColumn < artifact.publicInputCount
  rowsPresent : artifact.rows ≠ []
  completeFamiliesNodup : artifact.completeFamilies.Nodup
  completeFamiliesNonempty :
    ∀ family ∈ artifact.completeFamilies, family ≠ ""
  completeFamilyWitness :
    ∀ family ∈ artifact.completeFamilies,
      ∃ row ∈ artifact.rows, row.family = family
  sourceRowsPairwise : artifact.rows.Pairwise fun left right =>
    left.sourceIndex < right.sourceIndex
  rowsWellFormed :
    ∀ row ∈ artifact.rows, rowWellFormed artifact row = true

theorem StructuralCertificate.sound
    {artifact : Artifact}
    (certificate : StructuralCertificate artifact) :
    artifact.WellFormed := by
  refine
    ⟨certificate.schemaExact,
      certificate.profilePresent,
      certificate.scopeSupported,
      certificate.diagnosticDigestPresent,
      certificate.fieldModulusExact,
      certificate.totalRowsPositive,
      certificate.columnCountPositive,
      certificate.publicInputCountPositive,
      certificate.publicInputCountBounded,
      certificate.constantOneColumnPublic,
      certificate.rowsPresent,
      certificate.completeFamiliesNodup,
      ?_, ?_, ?_, ?_⟩
  · apply List.all_eq_true.mpr
    intro family member
    simp [certificate.completeFamiliesNonempty family member]
  · apply List.all_eq_true.mpr
    intro family member
    rcases certificate.completeFamilyWitness family member with
      ⟨row, rowMember, rowFamily⟩
    exact List.any_eq_true.mpr ⟨row, rowMember, by simp [rowFamily]⟩
  · exact strictlyIncreasingSourceRows_of_pairwise
      certificate.sourceRowsPairwise
  · exact List.all_eq_true.mpr certificate.rowsWellFormed

/-- Exact coverage of every source-row index and every row-family owner.
This separates a complete relation artifact from a bounded query slice. -/
def CoversFullRelation (artifact : Artifact) : Prop :=
  artifact.rows.map (fun row => row.sourceIndex) =
      List.range artifact.totalRows ∧
    ∀ row ∈ artifact.rows, row.family ∈ artifact.completeFamilies

instance (artifact : Artifact) : Decidable artifact.CoversFullRelation := by
  unfold CoversFullRelation
  infer_instance

/-- Structural source-row coverage. Each row-family membership can be proved
in the leaf that owns that row. -/
structure CoverageCertificate (artifact : Artifact) : Prop where
  sourceIndicesExact :
    artifact.rows.map (fun row => row.sourceIndex) =
      List.range artifact.totalRows
  rowFamiliesCovered :
    ∀ row ∈ artifact.rows, row.family ∈ artifact.completeFamilies

theorem CoverageCertificate.sound
    {artifact : Artifact}
    (certificate : CoverageCertificate artifact) :
    artifact.CoversFullRelation :=
  ⟨certificate.sourceIndicesExact, certificate.rowFamiliesCovered⟩

def ExactValidation (authoritative carried : Artifact) : Bool :=
  decide (carried = authoritative ∧ carried.WellFormed)

theorem exactValidation_self (artifact : Artifact)
    (wellFormed : artifact.WellFormed) :
    ExactValidation artifact artifact = true := by
  simp [ExactValidation, wellFormed]

theorem exactValidation_eq_true_iff
    {authoritative carried : Artifact} :
    ExactValidation authoritative carried = true ↔
      carried = authoritative ∧ carried.WellFormed := by
  simp [ExactValidation]

theorem accepted_eq_authoritative
    {authoritative carried : Artifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried = authoritative :=
  (exactValidation_eq_true_iff.mp accepted).1

theorem accepted_wellFormed
    {authoritative carried : Artifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried.WellFormed :=
  (exactValidation_eq_true_iff.mp accepted).2

end Artifact

/-- One half-open interval in a compiler source or emitted-row ledger. -/
structure RowRange where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

/-- One compact geometric term run in a final selective CCS port. -/
structure FinalGeometricRun where
  columnStart : Nat
  length : Nat
  initial : Nat
  ratio : Nat
deriving DecidableEq, Repr

namespace FinalGeometricRun

def WellFormed (columnCount : Nat) (run : FinalGeometricRun) : Bool :=
  decide (0 < run.length ∧
    run.columnStart + run.length ≤ columnCount ∧
    run.initial < Numeric.modulus ∧
    run.ratio < Numeric.modulus)

end FinalGeometricRun

/-- Existing compact Rust wire payload for one seeded Phi81 linear block. -/
abbrev FinalSeededBlock :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Wire.RawSeededBlock

namespace FinalSeededBlock

private def seedWellFormed (seed : List Nat) : Bool :=
  decide (seed.length = 32) &&
    seed.all (fun byte => decide (byte < 256))

/-- Structural checks for an exact compact block attached to one projected
row. This does not execute the sampler or prove Rust sampler conformance. -/
def WellFormed (rowCount columnCount emittedRow : Nat)
    (block : FinalSeededBlock) : Bool :=
  decide (block.wordStarts ≠ [] ∧
    0 < block.wordWidth ∧
    0 < block.kappa ∧
    0 < block.chunkSize ∧
    block.rowStart ≤ emittedRow ∧
    emittedRow < block.rowStart + ringDegree * block.kappa ∧
    block.rowStart + ringDegree * block.kappa ≤ rowCount ∧
    block.messageCols =
      (block.wordStarts.length * block.wordWidth + ringDegree - 1) /
        ringDegree ∧
    block.chunkSeedsByRow.length = block.kappa ∧
    (block.superneoTransformedColumns = true →
      columnCount % ringDegree = 0)) &&
  block.wordStarts.all (fun start =>
    decide (start + block.wordWidth ≤ columnCount)) &&
  block.chunkSeedsByRow.all (fun seeds =>
    decide (seeds.length =
      (block.messageCols + block.chunkSize - 1) / block.chunkSize) &&
    seeds.all seedWellFormed)

end FinalSeededBlock

/-- One exact port of a projected selective CCS row. Coefficients are
canonical Goldilocks residues represented as natural numbers. -/
structure FinalPort where
  explicit : List (Nat × Nat)
  geometricRuns : List FinalGeometricRun
  seededBlocks : List FinalSeededBlock
deriving DecidableEq, Repr

namespace FinalPort

def WellFormed (rowCount columnCount emittedRow : Nat)
    (port : FinalPort) : Bool :=
  Artifact.termsWellFormed columnCount port.explicit &&
    port.geometricRuns.all (FinalGeometricRun.WellFormed columnCount) &&
    port.seededBlocks.all
      (FinalSeededBlock.WellFormed rowCount columnCount emittedRow)

end FinalPort

/-- One exact projected row from the final thirteen-port selective relation. -/
structure FinalRow where
  emittedRow : Nat
  runIndex : Nat
  family : String
  arm : Option Nat
  ports : List FinalPort
deriving DecidableEq, Repr

/-- One retained source row and its monotone final-row image. -/
structure RetainedRowBinding where
  sourceRow : Nat
  emittedRow : Nat
  stageOccurrence : Option Nat
deriving DecidableEq, Repr

/-- One complete compiler rewrite touched by the requested semantic family. -/
structure RewriteBinding where
  rewriteId : Nat
  kind : String
  sourceRows : List RowRange
  emittedRows : RowRange
  stageOccurrence : Option Nat
deriving DecidableEq, Repr

/-- Exact source-to-final binding carried with a minimization input. Digest
fields are diagnostic. The row ledgers and projected terms are authority. -/
structure SelectiveBinding where
  branch : String
  requestedSourceRows : List Nat
  closureSourceRows : List Nat
  additionalSourceRows : List Nat
  retainedRows : List RetainedRowBinding
  rewrites : List RewriteBinding
  emittedRows : List Nat
  finalRows : Nat
  finalColumns : Nat
  finalPublicInputCount : Nat
  finalPlanDigest : String
  projectedSliceDigest : String
  projectedRows : List FinalRow
deriving DecidableEq, Repr

/-- One source R1CS slice and its exact final selective-row binding. -/
structure BoundArtifact where
  source : Artifact
  binding : SelectiveBinding
deriving DecidableEq, Repr

namespace BoundArtifact

def Coherent (artifact : BoundArtifact) : Prop :=
  artifact.source.WellFormed ∧
    artifact.source.rows.map (fun row => row.sourceIndex) =
      artifact.binding.requestedSourceRows ∧
    artifact.binding.projectedRows.map (fun row => row.emittedRow) =
      artifact.binding.emittedRows ∧
    artifact.binding.additionalSourceRows =
      artifact.binding.closureSourceRows.filter
        (fun row => decide (row ∉ artifact.binding.requestedSourceRows)) ∧
    0 < artifact.binding.finalRows ∧
    0 < artifact.binding.finalColumns ∧
    0 < artifact.binding.finalPublicInputCount ∧
    artifact.binding.finalPublicInputCount ≤ artifact.binding.finalColumns ∧
    artifact.binding.projectedRows.all
      (fun row =>
        decide (row.emittedRow < artifact.binding.finalRows ∧
          row.ports.length = 13) &&
        row.ports.all (FinalPort.WellFormed
          artifact.binding.finalRows artifact.binding.finalColumns
          row.emittedRow)) = true

instance (artifact : BoundArtifact) : Decidable artifact.Coherent := by
  unfold Coherent
  infer_instance

/-- Structural input for one source-to-selective binding proof. Projected-row
geometry is supplied per row and per port, so the assembly cost does not
depend on evaluation of the complete generated artifact. -/
structure StructuralCertificate (artifact : BoundArtifact) : Prop where
  sourceWellFormed : artifact.source.WellFormed
  sourceRowsExact :
    artifact.source.rows.map (fun row => row.sourceIndex) =
      artifact.binding.requestedSourceRows
  projectedRowsExact :
    artifact.binding.projectedRows.map (fun row => row.emittedRow) =
      artifact.binding.emittedRows
  additionalRowsExact :
    artifact.binding.additionalSourceRows =
      artifact.binding.closureSourceRows.filter
        (fun row => decide (row ∉ artifact.binding.requestedSourceRows))
  finalRowsPositive : 0 < artifact.binding.finalRows
  finalColumnsPositive : 0 < artifact.binding.finalColumns
  finalPublicInputCountPositive :
    0 < artifact.binding.finalPublicInputCount
  finalPublicInputCountBounded :
    artifact.binding.finalPublicInputCount ≤ artifact.binding.finalColumns
  projectedRowsWellFormed :
    ∀ row ∈ artifact.binding.projectedRows,
      row.emittedRow < artifact.binding.finalRows ∧
        row.ports.length = 13 ∧
          ∀ port ∈ row.ports,
            FinalPort.WellFormed artifact.binding.finalRows
              artifact.binding.finalColumns row.emittedRow port = true

theorem StructuralCertificate.sound
    {artifact : BoundArtifact}
    (certificate : StructuralCertificate artifact) :
    artifact.Coherent := by
  refine
    ⟨certificate.sourceWellFormed,
      certificate.sourceRowsExact,
      certificate.projectedRowsExact,
      certificate.additionalRowsExact,
      certificate.finalRowsPositive,
      certificate.finalColumnsPositive,
      certificate.finalPublicInputCountPositive,
      certificate.finalPublicInputCountBounded,
      ?_⟩
  apply List.all_eq_true.mpr
  intro row rowMember
  have rowFacts := certificate.projectedRowsWellFormed row rowMember
  rw [Bool.and_eq_true]
  refine ⟨?_, List.all_eq_true.mpr rowFacts.2.2⟩
  simp [rowFacts.1, rowFacts.2.1]

/-- A coherent source-to-final binding whose source covers the full branch
relation, not only a cvc5 query slice. -/
def CoversFullRelation (artifact : BoundArtifact) : Prop :=
  artifact.Coherent ∧ artifact.source.CoversFullRelation

instance (artifact : BoundArtifact) : Decidable artifact.CoversFullRelation := by
  unfold CoversFullRelation
  infer_instance

theorem coversFullRelation_of_structural
    {artifact : BoundArtifact}
    (coherent : StructuralCertificate artifact)
    (coverage : Artifact.CoverageCertificate artifact.source) :
    artifact.CoversFullRelation :=
  ⟨coherent.sound, coverage.sound⟩

def ExactValidation (authoritative carried : BoundArtifact) : Bool :=
  decide (carried = authoritative ∧ carried.Coherent)

theorem exactValidation_self (artifact : BoundArtifact)
    (coherent : artifact.Coherent) :
    ExactValidation artifact artifact = true := by
  simp [ExactValidation, coherent]

theorem accepted_eq_authoritative
    {authoritative carried : BoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried = authoritative := by
  have both : carried = authoritative ∧ carried.Coherent := by
    simpa [ExactValidation] using of_decide_eq_true accepted
  exact both.1

theorem accepted_coherent
    {authoritative carried : BoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried.Coherent := by
  have both : carried = authoritative ∧ carried.Coherent := by
    simpa [ExactValidation] using of_decide_eq_true accepted
  exact both.2

end BoundArtifact

/-- Compact exact map from normalized source columns `[one, public, private]`
to Spartan columns `[padded private, one, public]`. -/
structure TerminalColumnLayout where
  sourcePublicColumns : Nat
  sourcePrivateColumns : Nat
  spartanPrivateColumns : Nat
deriving DecidableEq, Repr

namespace TerminalColumnLayout

def mapColumn (layout : TerminalColumnLayout) (sourceColumn : Nat) : Nat :=
  if sourceColumn = 0 then
    layout.spartanPrivateColumns
  else if sourceColumn < layout.sourcePublicColumns then
    layout.spartanPrivateColumns + sourceColumn
  else
    sourceColumn - layout.sourcePublicColumns

end TerminalColumnLayout

/-- One exact terminal source row projected into the padded Spartan columns. -/
structure TerminalProjectedRow where
  sourceRow : Nat
  spartanRow : Nat
  row : Numeric.Row
deriving DecidableEq, Repr

/-- Exact terminal source-to-Spartan binding. The digest is metadata. -/
structure TerminalBinding where
  requestedSourceRows : List Nat
  verifierNativeGuards : List String
  columnLayout : TerminalColumnLayout
  projectedRows : List TerminalProjectedRow
  spartanRows : Nat
  spartanColumns : Nat
  spartanPaddingRows : RowRange
  spartanPrivatePaddingColumns : Nat
  diagnosticDigest : String
deriving DecidableEq, Repr

/-- Exact verifier-native guard order shared with the Rust terminal binding.
These guards are never polynomial-family candidates. -/
def terminalNativeGuardNames : List String :=
  TerminalContextBoundary.guardNames ++
    TerminalStatementBoundary.guardNames ++
      TerminalProofBoundary.guardNames

/-- Artifact-checked equality between the ordered Rust ledger and the three
independent model-level Lean guard ledgers. This proves name and order
agreement only, not Rust semantic refinement. -/
theorem rust_terminal_native_guard_names_exact :
    Nightstream.Implementation.R1CS.Artifacts.TerminalVerifierNativeGuards.names =
      terminalNativeGuardNames := by
  rfl

/-- One terminal source R1CS slice and its padded Spartan binding. -/
structure TerminalBoundArtifact where
  source : Artifact
  binding : TerminalBinding
deriving DecidableEq, Repr

namespace TerminalBoundArtifact

def termColumnsInRange (bound : Nat) (terms : List (Nat × Nat)) : Bool :=
  terms.all fun term => decide (term.1 < bound)

def rowColumnsInRange (bound : Nat) (row : Numeric.Row) : Bool :=
  termColumnsInRange bound row.a &&
    termColumnsInRange bound row.b &&
      termColumnsInRange bound row.c

def projectTerms (layout : TerminalColumnLayout) :
    List (Nat × Nat) → List (Nat × Nat) :=
  List.map fun term => (layout.mapColumn term.1, term.2)

def termsMatch (layout : TerminalColumnLayout)
    (source projected : List (Nat × Nat)) : Bool :=
  decide ((projectTerms layout source).Perm projected) &&
    Artifact.strictlyIncreasingColumns projected

def projectionMatches (layout : TerminalColumnLayout)
    (source : IndexedRow) (projected : TerminalProjectedRow) : Bool :=
  decide (source.sourceIndex = projected.sourceRow ∧
      projected.sourceRow = projected.spartanRow) &&
    termsMatch layout source.row.a projected.row.a &&
      termsMatch layout source.row.b projected.row.b &&
        termsMatch layout source.row.c projected.row.c

def Coherent (artifact : TerminalBoundArtifact) : Prop :=
  artifact.source.WellFormed ∧
    artifact.binding.verifierNativeGuards = terminalNativeGuardNames ∧
    artifact.binding.verifierNativeGuards.Nodup ∧
    (∀ guard ∈ artifact.binding.verifierNativeGuards, guard ≠ "") ∧
    (∀ guard ∈ artifact.binding.verifierNativeGuards,
      guard ∉ artifact.source.completeFamilies) ∧
    artifact.source.rows.map (fun row => row.sourceIndex) =
      artifact.binding.requestedSourceRows ∧
    artifact.binding.projectedRows.map (fun row => row.sourceRow) =
      artifact.binding.requestedSourceRows ∧
    artifact.source.rows.length = artifact.binding.projectedRows.length ∧
    artifact.binding.columnLayout.sourcePublicColumns =
      artifact.source.publicInputCount ∧
    artifact.binding.columnLayout.sourcePublicColumns +
        artifact.binding.columnLayout.sourcePrivateColumns =
      artifact.source.columnCount ∧
    artifact.binding.columnLayout.sourcePrivateColumns ≤
      artifact.binding.columnLayout.spartanPrivateColumns ∧
    artifact.binding.spartanColumns =
      artifact.binding.columnLayout.spartanPrivateColumns +
        artifact.binding.columnLayout.sourcePublicColumns ∧
    artifact.binding.spartanPrivatePaddingColumns =
      artifact.binding.columnLayout.spartanPrivateColumns -
        artifact.binding.columnLayout.sourcePrivateColumns ∧
    artifact.binding.spartanPaddingRows.start = artifact.source.totalRows ∧
    artifact.binding.spartanPaddingRows.stop = artifact.binding.spartanRows ∧
    artifact.source.totalRows ≤ artifact.binding.spartanRows ∧
    artifact.source.rows.all
      (fun row => rowColumnsInRange artifact.source.columnCount row.row) = true ∧
    artifact.binding.projectedRows.all
      (fun row => decide (row.spartanRow < artifact.binding.spartanRows)) = true ∧
    (artifact.source.rows.zip artifact.binding.projectedRows).all
      (fun pair => projectionMatches artifact.binding.columnLayout
        pair.1 pair.2) = true

instance (artifact : TerminalBoundArtifact) : Decidable artifact.Coherent := by
  unfold Coherent
  infer_instance

/-- Structural input for a terminal source-to-Spartan binding proof. Source
rows and projected rows are checked in bounded leaves before this theorem is
used. -/
structure StructuralCertificate (artifact : TerminalBoundArtifact) : Prop where
  sourceWellFormed : artifact.source.WellFormed
  nativeGuardsExact :
    artifact.binding.verifierNativeGuards = terminalNativeGuardNames
  nativeGuardsNodup : artifact.binding.verifierNativeGuards.Nodup
  nativeGuardsNonempty :
    ∀ guard ∈ artifact.binding.verifierNativeGuards, guard ≠ ""
  nativeGuardsNotPolynomial :
    ∀ guard ∈ artifact.binding.verifierNativeGuards,
      guard ∉ artifact.source.completeFamilies
  sourceRowsExact :
    artifact.source.rows.map (fun row => row.sourceIndex) =
      artifact.binding.requestedSourceRows
  projectedRowsExact :
    artifact.binding.projectedRows.map (fun row => row.sourceRow) =
      artifact.binding.requestedSourceRows
  rowCountsExact :
    artifact.source.rows.length = artifact.binding.projectedRows.length
  sourcePublicColumnsExact :
    artifact.binding.columnLayout.sourcePublicColumns =
      artifact.source.publicInputCount
  sourceColumnsExact :
    artifact.binding.columnLayout.sourcePublicColumns +
        artifact.binding.columnLayout.sourcePrivateColumns =
      artifact.source.columnCount
  privateColumnsBounded :
    artifact.binding.columnLayout.sourcePrivateColumns ≤
      artifact.binding.columnLayout.spartanPrivateColumns
  spartanColumnsExact :
    artifact.binding.spartanColumns =
      artifact.binding.columnLayout.spartanPrivateColumns +
        artifact.binding.columnLayout.sourcePublicColumns
  privatePaddingExact :
    artifact.binding.spartanPrivatePaddingColumns =
      artifact.binding.columnLayout.spartanPrivateColumns -
        artifact.binding.columnLayout.sourcePrivateColumns
  paddingStartExact :
    artifact.binding.spartanPaddingRows.start = artifact.source.totalRows
  paddingStopExact :
    artifact.binding.spartanPaddingRows.stop = artifact.binding.spartanRows
  sourceRowsBounded : artifact.source.totalRows ≤ artifact.binding.spartanRows
  sourceRowColumnsInRange :
    ∀ row ∈ artifact.source.rows,
      rowColumnsInRange artifact.source.columnCount row.row = true
  projectedRowsInRange :
    ∀ row ∈ artifact.binding.projectedRows,
      row.spartanRow < artifact.binding.spartanRows
  projectionsMatch :
    ∀ pair ∈ artifact.source.rows.zip artifact.binding.projectedRows,
      projectionMatches artifact.binding.columnLayout pair.1 pair.2 = true

theorem StructuralCertificate.sound
    {artifact : TerminalBoundArtifact}
    (certificate : StructuralCertificate artifact) :
    artifact.Coherent := by
  refine
    ⟨certificate.sourceWellFormed,
      certificate.nativeGuardsExact,
      certificate.nativeGuardsNodup,
      certificate.nativeGuardsNonempty,
      certificate.nativeGuardsNotPolynomial,
      certificate.sourceRowsExact,
      certificate.projectedRowsExact,
      certificate.rowCountsExact,
      certificate.sourcePublicColumnsExact,
      certificate.sourceColumnsExact,
      certificate.privateColumnsBounded,
      certificate.spartanColumnsExact,
      certificate.privatePaddingExact,
      certificate.paddingStartExact,
      certificate.paddingStopExact,
      certificate.sourceRowsBounded,
      List.all_eq_true.mpr certificate.sourceRowColumnsInRange,
      ?_,
      List.all_eq_true.mpr certificate.projectionsMatch⟩
  apply List.all_eq_true.mpr
  intro row rowMember
  simp [certificate.projectedRowsInRange row rowMember]

/-- A coherent source-to-Spartan binding whose source covers the complete
terminal polynomial relation. -/
def CoversFullRelation (artifact : TerminalBoundArtifact) : Prop :=
  artifact.Coherent ∧ artifact.source.CoversFullRelation

instance (artifact : TerminalBoundArtifact) :
    Decidable artifact.CoversFullRelation := by
  unfold CoversFullRelation
  infer_instance

theorem coversFullRelation_of_structural
    {artifact : TerminalBoundArtifact}
    (coherent : StructuralCertificate artifact)
    (coverage : Artifact.CoverageCertificate artifact.source) :
    artifact.CoversFullRelation :=
  ⟨coherent.sound, coverage.sound⟩

def ExactValidation (authoritative carried : TerminalBoundArtifact) : Bool :=
  decide (carried = authoritative ∧ carried.Coherent)

theorem exactValidation_self (artifact : TerminalBoundArtifact)
    (coherent : artifact.Coherent) :
    ExactValidation artifact artifact = true := by
  simp [ExactValidation, coherent]

theorem accepted_eq_authoritative
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried = authoritative := by
  have both : carried = authoritative ∧ carried.Coherent := by
    simpa [ExactValidation] using of_decide_eq_true accepted
  exact both.1

theorem accepted_coherent
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried.Coherent := by
  have both : carried = authoritative ∧ carried.Coherent := by
    simpa [ExactValidation] using of_decide_eq_true accepted
  exact both.2

/-- Artifact-checked: exact validation cannot drop a verifier-native guard. -/
theorem accepted_retains_native_guard
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true)
    {guard : String}
    (retained : guard ∈ authoritative.binding.verifierNativeGuards) :
    guard ∈ carried.binding.verifierNativeGuards := by
  have equal := accepted_eq_authoritative accepted
  subst carried
  exact retained

/-- Artifact-checked: retained native guards are outside the polynomial family
set that the cvc5 query can remove. -/
theorem accepted_native_guard_not_polynomial
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true)
    {guard : String}
    (retained : guard ∈ authoritative.binding.verifierNativeGuards) :
    guard ∉ carried.source.completeFamilies := by
  have retainedCarried := accepted_retains_native_guard accepted retained
  exact (accepted_coherent accepted).2.2.2.2.1 guard retainedCarried

/-- Artifact-checked: exact validation preserves the complete ordered native
guard ledger used by the three Lean boundary models. -/
theorem accepted_native_guard_ledger_exact
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried.binding.verifierNativeGuards = terminalNativeGuardNames :=
  (accepted_coherent accepted).2.1

/-- Artifact-checked: exact terminal bindings retain all four context guards. -/
theorem accepted_retains_context_guards
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    ∀ guard ∈ TerminalContextBoundary.guardNames,
      guard ∈ carried.binding.verifierNativeGuards := by
  rw [accepted_native_guard_ledger_exact accepted]
  intro guard member
  simp [terminalNativeGuardNames, member]

/-- Artifact-checked: exact terminal bindings retain all eleven statement
guards. -/
theorem accepted_retains_statement_guards
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    ∀ guard ∈ TerminalStatementBoundary.guardNames,
      guard ∈ carried.binding.verifierNativeGuards := by
  rw [accepted_native_guard_ledger_exact accepted]
  intro guard member
  simp [terminalNativeGuardNames, member]

/-- Artifact-checked: exact terminal bindings retain all three proof-boundary
guards whose semantic plan is proved inclusion-minimal. -/
theorem accepted_retains_proof_guards
    {authoritative carried : TerminalBoundArtifact}
    (accepted : ExactValidation authoritative carried = true) :
    ∀ guard ∈ TerminalProofBoundary.guardNames,
      guard ∈ carried.binding.verifierNativeGuards := by
  rw [accepted_native_guard_ledger_exact accepted]
  intro guard member
  simp [terminalNativeGuardNames, member]

end TerminalBoundArtifact

namespace Algebraic

/-- Sparse linear combination as a formal polynomial. -/
noncomputable def linearPolynomial : List (Nat × Nat) → Polynomial
  | [] => 0
  | term :: tail =>
      C (term.2 : Field) * X term.1 + linearPolynomial tail

/-- Formal residual `(A z) * (B z) - C z`. -/
noncomputable def residual (row : Numeric.Row) : Polynomial :=
  linearPolynomial row.a * linearPolynomial row.b -
    linearPolynomial row.c

def linearEval (assignment : Nat → Field) : List (Nat × Nat) → Field
  | [] => 0
  | term :: tail =>
      (term.2 : Field) * assignment term.1 + linearEval assignment tail

def Holds (assignment : Nat → Field) (row : Numeric.Row) : Prop :=
  linearEval assignment row.a * linearEval assignment row.b =
    linearEval assignment row.c

instance (assignment : Nat → Field) (row : Numeric.Row) :
    Decidable (Holds assignment row) := by
  unfold Holds
  infer_instance

theorem eval_linearPolynomial
    (assignment : Nat → Field) (terms : List (Nat × Nat)) :
    eval assignment (linearPolynomial terms) =
      linearEval assignment terms := by
  induction terms with
  | nil => simp [linearPolynomial, linearEval]
  | cons head tail inductionHypothesis =>
      simp [linearPolynomial, linearEval, inductionHypothesis]

theorem eval_residual
    (assignment : Nat → Field) (row : Numeric.Row) :
    eval assignment (residual row) =
      linearEval assignment row.a * linearEval assignment row.b -
        linearEval assignment row.c := by
  simp [residual, eval_linearPolynomial]

end Algebraic

/-- One retained residual and its scalar coefficient. -/
structure ScalarSupport where
  source : IndexedRow
  coefficient : Field
deriving DecidableEq, Repr

noncomputable def scalarCombination : List ScalarSupport → Polynomial
  | [] => 0
  | support :: tail =>
      C support.coefficient * Algebraic.residual support.source.row +
        scalarCombination tail

/-- A small, proof-producing certificate grammar. It deliberately covers only
constant scalar combinations of retained residual polynomials. -/
structure ScalarCertificate where
  candidate : IndexedRow
  support : List ScalarSupport
deriving DecidableEq, Repr

namespace ScalarCertificate

def Valid (certificate : ScalarCertificate) : Prop :=
  Algebraic.residual certificate.candidate.row =
    scalarCombination certificate.support

theorem candidate_holds_of_valid
    (certificate : ScalarCertificate)
    (valid : certificate.Valid)
    (assignment : Nat → Field)
    (supportHolds : ∀ support ∈ certificate.support,
      Algebraic.Holds assignment support.source.row) :
    Algebraic.Holds assignment certificate.candidate.row := by
  have combinationZero : ∀ supports : List ScalarSupport,
      (∀ support ∈ supports,
        Algebraic.Holds assignment support.source.row) →
      eval assignment (scalarCombination supports) = 0 := by
    intro supports
    induction supports with
    | nil =>
        intro _
        simp [scalarCombination]
    | cons head tail inductionHypothesis =>
        intro holds
        have headHolds := holds head (by simp)
        have headResidual :
            eval assignment (Algebraic.residual head.source.row) = 0 := by
          rw [Algebraic.eval_residual]
          exact sub_eq_zero.mpr headHolds
        have tailHolds : ∀ support ∈ tail,
            Algebraic.Holds assignment support.source.row := by
          intro support member
          exact holds support (by simp [member])
        simp [scalarCombination, headResidual,
          inductionHypothesis tailHolds]
  have candidateResidual :
      eval assignment (Algebraic.residual certificate.candidate.row) = 0 := by
    rw [valid]
    exact combinationZero certificate.support supportHolds
  rw [Algebraic.eval_residual] at candidateResidual
  exact sub_eq_zero.mp candidateResidual

end ScalarCertificate

def candidateRows (artifact : Artifact) (family : String) :
    List IndexedRow :=
  artifact.rows.filter fun row => decide (row.family = family)

def FamilyHolds (artifact : Artifact) (family : String)
    (assignment : Nat → Field) : Prop :=
  ∀ row ∈ artifact.rows, row.family = family →
    Algebraic.Holds assignment row.row

instance (artifact : Artifact) (family : String)
    (assignment : Nat → Field) :
    Decidable (FamilyHolds artifact family assignment) := by
  unfold FamilyHolds
  infer_instance

def Target (artifact : Artifact) (assignment : Nat → Field) : Prop :=
  assignment artifact.constantOneColumn = 1 ∧
    ∀ row ∈ artifact.rows, Algebraic.Holds assignment row.row

instance (artifact : Artifact) (assignment : Nat → Field) :
    Decidable (Target artifact assignment) := by
  unfold Target
  infer_instance

/-- R1CS assignments at the verifier boundary. The constant-one coordinate is
part of the assignment domain, not a removable polynomial family. -/
def NormalizedAssignment (artifact : Artifact) :=
  { assignment : Nat → Field //
      assignment artifact.constantOneColumn = 1 }

def NormalizedFamilyHolds (artifact : Artifact) (family : String)
    (assignment : NormalizedAssignment artifact) : Prop :=
  FamilyHolds artifact family assignment.1

instance (artifact : Artifact) (family : String)
    (assignment : NormalizedAssignment artifact) :
    Decidable (NormalizedFamilyHolds artifact family assignment) := by
  unfold NormalizedFamilyHolds
  infer_instance

def NormalizedTarget (artifact : Artifact)
    (assignment : NormalizedAssignment artifact) : Prop :=
  ∀ row ∈ artifact.rows, Algebraic.Holds assignment.1 row.row

instance (artifact : Artifact)
    (assignment : NormalizedAssignment artifact) :
    Decidable (NormalizedTarget artifact assignment) := by
  unfold NormalizedTarget
  infer_instance

theorem normalizedTarget_iff_target
    (artifact : Artifact) (assignment : NormalizedAssignment artifact) :
    NormalizedTarget artifact assignment ↔
      Target artifact assignment.1 := by
  simp [NormalizedTarget, Target, assignment.property]

namespace Artifact

/-- Model-level soundness of the complete family ledger over normalized R1CS
assignments. Full row and family coverage is the authority premise. -/
theorem normalizedFullPlanSound
    (artifact : Artifact) (coverage : artifact.CoversFullRelation) :
    Sound (NormalizedFamilyHolds artifact)
      (NormalizedTarget artifact) artifact.completeFamilies := by
  intro assignment accepted row rowMember
  exact accepted row.family (coverage.2 row rowMember)
    row rowMember rfl

/-- Model-level completeness of the complete family ledger. -/
theorem normalizedFullPlanComplete (artifact : Artifact) :
    Complete (NormalizedFamilyHolds artifact)
      (NormalizedTarget artifact) artifact.completeFamilies := by
  intro assignment target family _ row rowMember _
  exact target row rowMember

theorem normalizedFullPlanExact
    (artifact : Artifact) (coverage : artifact.CoversFullRelation) :
    Exact (NormalizedFamilyHolds artifact)
      (NormalizedTarget artifact) artifact.completeFamilies :=
  exact_iff_sound_and_complete.mpr
    ⟨artifact.normalizedFullPlanSound coverage,
      artifact.normalizedFullPlanComplete⟩

/-- Final model-level assembly rule. Each retained family still needs an
artifact-bound removal witness. -/
theorem normalizedFullPlanInclusionMinimalSound
    (artifact : Artifact) (coverage : artifact.CoversFullRelation)
    (necessary : ∀ family ∈ artifact.completeFamilies,
      NecessaryForSoundness (NormalizedFamilyHolds artifact)
        (NormalizedTarget artifact) artifact.completeFamilies family) :
    InclusionMinimalSound (NormalizedFamilyHolds artifact)
      (NormalizedTarget artifact) artifact.completeFamilies :=
  inclusionMinimalSound_of_witnesses
    (artifact.normalizedFullPlanSound coverage) necessary

end Artifact

/-- A universal row-family redundancy proof also holds on the normalized
assignment domain used by the verifier. -/
theorem normalizedRedundant_of_redundant
    (artifact : Artifact) (plan : List String) (family : String)
    (redundant : Redundant (FamilyHolds artifact) plan family) :
    Redundant (NormalizedFamilyHolds artifact) plan family := by
  intro assignment accepted
  simpa [NormalizedFamilyHolds] using redundant assignment.1 (by
    intro retained retainedMember
    simpa [NormalizedFamilyHolds] using
      accepted retained retainedMember)

/-- Exact family coverage plus artifact-bound support rows. Every support must
remain in the plan and must have a different family from the removed family. -/
structure FamilyCertificate where
  family : String
  certificates : List ScalarCertificate
deriving DecidableEq, Repr

namespace FamilyCertificate

def Valid (certificate : FamilyCertificate)
    (artifact : Artifact) (plan : List String) : Prop :=
  certificate.family ∈ artifact.completeFamilies ∧
    certificate.certificates.map (fun scalar => scalar.candidate) =
        candidateRows artifact certificate.family ∧
      ∀ scalar ∈ certificate.certificates,
        scalar.Valid ∧
          ∀ support ∈ scalar.support,
            support.source ∈ artifact.rows ∧
              support.source.family ∈ plan ∧
                support.source.family ≠ certificate.family

/-- An artifact-checked family certificate becomes a semantic redundancy
proof. cvc5 does not occur in this theorem or its assumptions. -/
theorem redundant_of_valid
    (certificate : FamilyCertificate)
    (artifact : Artifact) (plan : List String)
    (valid : certificate.Valid artifact plan) :
    Redundant (FamilyHolds artifact) plan certificate.family := by
  intro assignment accepted row rowMember rowFamily
  have candidateMember : row ∈ candidateRows artifact certificate.family := by
    simp [candidateRows, rowMember, rowFamily]
  have mappedMember :
      row ∈ certificate.certificates.map (fun scalar => scalar.candidate) := by
    rw [valid.2.1]
    exact candidateMember
  rcases List.mem_map.mp mappedMember with
    ⟨scalar, scalarMember, scalarCandidate⟩
  have scalarFacts := valid.2.2 scalar scalarMember
  have candidateHolds := ScalarCertificate.candidate_holds_of_valid
    scalar scalarFacts.1 assignment (by
      intro support supportMember
      have supportFacts := scalarFacts.2 support supportMember
      have familyAccepted := accepted support.source.family
        (mem_without_iff.mpr ⟨supportFacts.2.1, supportFacts.2.2⟩)
      exact familyAccepted support.source supportFacts.1 rfl)
  simpa [scalarCandidate] using candidateHolds

/-- A certificate checked against an exact source-to-final artifact becomes a
redundancy proof for the verifier-owned source artifact. -/
theorem redundant_of_bound_valid
    (certificate : FamilyCertificate)
    (authoritative carried : BoundArtifact) (plan : List String)
    (exact : BoundArtifact.ExactValidation authoritative carried = true)
    (valid : certificate.Valid carried.source plan) :
    Redundant (FamilyHolds authoritative.source) plan certificate.family := by
  have equal := BoundArtifact.accepted_eq_authoritative exact
  subst carried
  exact certificate.redundant_of_valid authoritative.source plan valid

/-- A certificate checked against an exact terminal Spartan artifact becomes
a redundancy proof for its verifier-owned source R1CS. -/
theorem redundant_of_terminal_bound_valid
    (certificate : FamilyCertificate)
    (authoritative carried : TerminalBoundArtifact) (plan : List String)
    (exact : TerminalBoundArtifact.ExactValidation authoritative carried = true)
    (valid : certificate.Valid carried.source plan) :
    Redundant (FamilyHolds authoritative.source) plan certificate.family := by
  have equal := TerminalBoundArtifact.accepted_eq_authoritative exact
  subst carried
  exact certificate.redundant_of_valid authoritative.source plan valid

/-- A full-branch variant used by generated production classifications. The
coverage premise prevents a bounded query slice from becoming the theorem
target. -/
theorem redundant_of_full_bound_valid
    (certificate : FamilyCertificate)
    (authoritative carried : BoundArtifact) (plan : List String)
    (coverage : authoritative.CoversFullRelation)
    (exact : BoundArtifact.ExactValidation authoritative carried = true)
    (valid : certificate.Valid carried.source plan) :
    Redundant (FamilyHolds authoritative.source) plan certificate.family := by
  rcases coverage with ⟨_, _⟩
  exact certificate.redundant_of_bound_valid
    authoritative carried plan exact valid

/-- A full-terminal variant used by generated production classifications. -/
theorem redundant_of_full_terminal_bound_valid
    (certificate : FamilyCertificate)
    (authoritative carried : TerminalBoundArtifact) (plan : List String)
    (coverage : authoritative.CoversFullRelation)
    (exact : TerminalBoundArtifact.ExactValidation authoritative carried = true)
    (valid : certificate.Valid carried.source plan) :
    Redundant (FamilyHolds authoritative.source) plan certificate.family := by
  rcases coverage with ⟨_, _⟩
  exact certificate.redundant_of_terminal_bound_valid
    authoritative carried plan exact valid

end FamilyCertificate

/-- Finite model record used for a checked removal counterexample. -/
structure RemovalCounterexample where
  removedFamily : String
  values : List Field
deriving DecidableEq, Repr

namespace RemovalCounterexample

def assignment (counterexample : RemovalCounterexample) : Nat → Field :=
  fun column => counterexample.values.getD column 0

def Valid (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) : Prop :=
  (∀ family ∈ plan, family ∈ artifact.completeFamilies) ∧
    counterexample.values.length = artifact.columnCount ∧
      counterexample.assignment artifact.constantOneColumn = 1 ∧
        Accepts (FamilyHolds artifact)
          (without plan counterexample.removedFamily)
          counterexample.assignment ∧
          ¬ Target artifact counterexample.assignment

/-- Row-level replay facts used by a structural removal certificate. -/
def RetainedRowsHold (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) : Prop :=
  ∀ row ∈ artifact.rows,
    row.family ∈ without plan counterexample.removedFamily →
      Algebraic.Holds counterexample.assignment row.row

/-- One exact failed source row is enough to refute the complete target. -/
def HasViolatedRow (counterexample : RemovalCounterexample)
    (artifact : Artifact) : Prop :=
  ∃ row ∈ artifact.rows,
    ¬ Algebraic.Holds counterexample.assignment row.row

/-- Assemble a removal counterexample from bounded row-replay leaves. This
theorem does not evaluate the complete assignment or source-row list. -/
theorem valid_of_structural_replay
    (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String)
    (planCovered : ∀ family ∈ plan, family ∈ artifact.completeFamilies)
    (assignmentWidth :
      counterexample.values.length = artifact.columnCount)
    (constantOne :
      counterexample.assignment artifact.constantOneColumn = 1)
    (retainedRowsHold : counterexample.RetainedRowsHold artifact plan)
    (violatedRow : counterexample.HasViolatedRow artifact) :
    counterexample.Valid artifact plan := by
  refine ⟨planCovered, assignmentWidth, constantOne, ?_, ?_⟩
  · intro family familyMember row rowMember rowFamily
    exact retainedRowsHold row rowMember (by simpa [rowFamily] using familyMember)
  · rcases violatedRow with ⟨row, rowMember, rowFails⟩
    intro target
    exact rowFails (target.2 row rowMember)

instance (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) :
    Decidable (counterexample.Valid artifact plan) := by
  unfold Valid Accepts
  infer_instance

def check (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) : Bool :=
  decide (counterexample.Valid artifact plan)

theorem valid_of_check
    {counterexample : RemovalCounterexample}
    {artifact : Artifact} {plan : List String}
    (checked : counterexample.check artifact plan = true) :
    counterexample.Valid artifact plan := by
  simpa [check] using of_decide_eq_true checked

theorem necessary_of_valid
    (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String)
    (valid : counterexample.Valid artifact plan) :
    NecessaryForSoundness (FamilyHolds artifact) (Target artifact)
      plan counterexample.removedFamily :=
  ⟨counterexample.assignment, valid.2.2.2.1, valid.2.2.2.2⟩

/-- A checked model gives necessity on the normalized assignment domain. -/
theorem necessary_normalized_of_valid
    (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String)
    (valid : counterexample.Valid artifact plan) :
    NecessaryForSoundness (NormalizedFamilyHolds artifact)
      (NormalizedTarget artifact) plan counterexample.removedFamily := by
  let normalized : NormalizedAssignment artifact :=
    ⟨counterexample.assignment, valid.2.2.1⟩
  refine ⟨normalized, ?_, ?_⟩
  · intro family familyMember
    simpa [NormalizedFamilyHolds, normalized] using
      valid.2.2.2.1 family familyMember
  · intro target
    apply valid.2.2.2.2
    exact ⟨valid.2.2.1, target⟩

/-- A complete checked model for an exact source-to-final artifact becomes a
removal counterexample for the verifier-owned source artifact. -/
theorem necessary_of_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : BoundArtifact) (plan : List String)
    (exact : BoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness (FamilyHolds authoritative.source)
      (Target authoritative.source) plan counterexample.removedFamily := by
  have equal := BoundArtifact.accepted_eq_authoritative exact
  subst carried
  exact counterexample.necessary_of_valid authoritative.source plan valid

theorem necessary_normalized_of_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : BoundArtifact) (plan : List String)
    (exact : BoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness
      (NormalizedFamilyHolds authoritative.source)
      (NormalizedTarget authoritative.source)
      plan counterexample.removedFamily := by
  have equal := BoundArtifact.accepted_eq_authoritative exact
  subst carried
  exact counterexample.necessary_normalized_of_valid
    authoritative.source plan valid

/-- A complete checked terminal model becomes a removal counterexample for
the verifier-owned terminal source R1CS. -/
theorem necessary_of_terminal_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : TerminalBoundArtifact) (plan : List String)
    (exact : TerminalBoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness (FamilyHolds authoritative.source)
      (Target authoritative.source) plan counterexample.removedFamily := by
  have equal := TerminalBoundArtifact.accepted_eq_authoritative exact
  subst carried
  exact counterexample.necessary_of_valid authoritative.source plan valid

theorem necessary_normalized_of_terminal_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : TerminalBoundArtifact) (plan : List String)
    (exact : TerminalBoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness
      (NormalizedFamilyHolds authoritative.source)
      (NormalizedTarget authoritative.source)
      plan counterexample.removedFamily := by
  have equal := TerminalBoundArtifact.accepted_eq_authoritative exact
  subst carried
  exact counterexample.necessary_normalized_of_valid
    authoritative.source plan valid

/-- A complete fixed-point branch counterexample. The coverage premise keeps
partial cvc5 query artifacts outside the proof target. -/
theorem necessary_of_full_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : BoundArtifact) (plan : List String)
    (coverage : authoritative.CoversFullRelation)
    (exact : BoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness (FamilyHolds authoritative.source)
      (Target authoritative.source) plan counterexample.removedFamily := by
  rcases coverage with ⟨_, _⟩
  exact counterexample.necessary_of_bound_valid
    authoritative carried plan exact valid

theorem necessary_normalized_of_full_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : BoundArtifact) (plan : List String)
    (coverage : authoritative.CoversFullRelation)
    (exact : BoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness
      (NormalizedFamilyHolds authoritative.source)
      (NormalizedTarget authoritative.source)
      plan counterexample.removedFamily := by
  rcases coverage with ⟨_, _⟩
  exact counterexample.necessary_normalized_of_bound_valid
    authoritative carried plan exact valid

/-- A complete terminal polynomial-relation counterexample. -/
theorem necessary_of_full_terminal_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : TerminalBoundArtifact) (plan : List String)
    (coverage : authoritative.CoversFullRelation)
    (exact : TerminalBoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness (FamilyHolds authoritative.source)
      (Target authoritative.source) plan counterexample.removedFamily := by
  rcases coverage with ⟨_, _⟩
  exact counterexample.necessary_of_terminal_bound_valid
    authoritative carried plan exact valid

theorem necessary_normalized_of_full_terminal_bound_valid
    (counterexample : RemovalCounterexample)
    (authoritative carried : TerminalBoundArtifact) (plan : List String)
    (coverage : authoritative.CoversFullRelation)
    (exact : TerminalBoundArtifact.ExactValidation authoritative carried = true)
    (valid : counterexample.Valid carried.source plan) :
    NecessaryForSoundness
      (NormalizedFamilyHolds authoritative.source)
      (NormalizedTarget authoritative.source)
      plan counterexample.removedFamily := by
  rcases coverage with ⟨_, _⟩
  exact counterexample.necessary_normalized_of_terminal_bound_valid
    authoritative carried plan exact valid

end RemovalCounterexample

end Nightstream.Assurance.ConstraintMinimization
