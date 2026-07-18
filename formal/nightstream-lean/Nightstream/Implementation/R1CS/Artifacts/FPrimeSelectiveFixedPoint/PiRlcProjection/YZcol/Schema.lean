import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.SourceMap
import Nightstream.Implementation.R1CS.Artifacts.Projection.ArtifactProgram

/-!
Handwritten artifact schema for the bounded tiny-lifecycle fixed-point PiRLC
cross-branch projection certificate.

Owns: protocol → phase → arithmetic-leaf ownership across the shared and
`y_zcol` identity branches; exact
reconstruction of projection traces; and the indexed definition/check schedule
used by the profile-neutral artifact certificate.

Does not own: generated rows or coordinates, serializer semantics, producer ↔
consumer equality, assignment satisfaction, transcript or PiCCS authority,
bad-root bounds, selective lowering, final costs, or row removal.

Emits constraints: no.

| Branch | Mathematical obligation | Multiplicity | Rows |
|---|---|---:|---:|
| `shared.betaLadder` | `p[0] = 1`; `p[j+1] = p[j] * beta` | 54 products | 272 |
| `shared.rhoEvaluations` | `rho_s(beta) = sum_j rho_s[j] beta^j` | 15 evaluations | 1,620 |
| `limbs[*].pairs[*].inputEvaluation` | `input_s(beta)` | 15 per limb | 1,620 per limb |
| `limbs[*].pairs[*].rhoProduct` | `rho_s(beta) * input_s(beta)` | 15 per limb | 75 per limb |
| `limbs[*].parentEvaluation` | `parent(beta)` | one per limb | 108 per limb |
| `limbs[*].quotientEvaluation` | `q(beta)` | one per limb | 106 per limb |
| `limbs[*].quotientPhiProduct` | `q(beta) * (beta^54 + beta^27 + 1)` | one per limb | 5 per limb |
| `limbs[*].finalRows` | `sum_s rho_s(beta)input_s(beta) = q(beta)Phi81(beta) + parent(beta)` | two per limb | 2 per limb |
| shared total | beta ladder plus rho evaluations | 1,892 definitions | 1,892 |
| `y_zcol` limb total | two identities excluding shared work | 3,828 definitions + 4 checks | 3,832 |
| certificate total | selected cross-branch source bundle | 5,720 definitions + 4 checks | 5,724 |

Assurance tier: schema only. The generated instance uses a tiny application
and lifecycle with the active 15-source/13-matrix protocol dimensions. It
remains production-untrusted until exact source-row satisfaction is derived
from accepted selectively lowered rows and its inputs are bound to independent
protocol semantics.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram

def activeDefinitionCount : Nat := 5720

def activeCheckCount : Nat := 4

def activeRowCount : Nat := activeDefinitionCount + activeCheckCount

/-- Linear-time certificate that adjacent natural-number coordinates increase.
Unlike `List.Nodup`, deciding this proposition does not rescan the full tail
for every element. -/
def StrictlyIncreasing : List Nat -> Prop
  | [] => True
  | [_] => True
  | first :: second :: rest =>
      first < second ∧ StrictlyIncreasing (second :: rest)

private theorem head_lt_of_strictlyIncreasing
    {head : Nat} {tail : List Nat}
    (ordered : StrictlyIncreasing (head :: tail)) :
    ∀ value ∈ tail, head < value := by
  induction tail generalizing head with
  | nil => simp
  | cons next rest inductionHypothesis =>
      simp only [StrictlyIncreasing] at ordered
      intro value member
      simp only [List.mem_cons] at member
      rcases member with rfl | inRest
      · exact ordered.1
      · exact Nat.lt_trans ordered.1
          (inductionHypothesis ordered.2 value inRest)

theorem strictlyIncreasing_nodup {values : List Nat}
    (ordered : StrictlyIncreasing values) : values.Nodup := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.nodup_cons]
      constructor
      · intro headMember
        exact Nat.lt_irrefl head
          (head_lt_of_strictlyIncreasing ordered head headMember)
      · cases tail with
        | nil => simp
        | cons next rest =>
            simp only [StrictlyIncreasing] at ordered
            exact inductionHypothesis ordered.2

/-- Profile coordinates retained by the bounded tiny fixed-point fixture. -/
structure Scope where
  parameterConstraintCount : Nat
  commitmentWidth : Nat
  securityBits : Nat
  applicationRowCount : Nat
  applicationColumnCount : Nat
  applicationPublicInputCount : Nat
  sourceCount : Nat
  matrixCount : Nat
  serializerFieldCount : Nat
  sourceArmRowCount : Nat
  sourceArmColumnCount : Nat
  laneCount : Nat
  powerCount : Nat
  quotientCount : Nat
  maxDegree : Nat
deriving DecidableEq, Repr

namespace Scope

/-- Tiny application/lifecycle fixture using the active 15-source/13-matrix
protocol dimensions. Absolute source-arm dimensions remain artifact data;
they are not final selectively lowered dimensions. -/
def IsTinyFixture (scope : Scope) : Prop :=
  scope.parameterConstraintCount = 65536 ∧
    scope.commitmentWidth = 4 ∧
    scope.securityBits = 60 ∧
    scope.applicationRowCount = 1 ∧
    scope.applicationColumnCount = 54 ∧
    scope.applicationPublicInputCount = 1 ∧
    scope.sourceCount = 15 ∧
    scope.matrixCount = 13 ∧
    scope.serializerFieldCount = 23033 ∧
    activeRowCount < scope.sourceArmRowCount ∧
    scope.serializerFieldCount < scope.sourceArmColumnCount ∧
    scope.laneCount = 54 ∧
    scope.powerCount = 55 ∧
    scope.quotientCount = 53 ∧
    scope.maxDegree = 106

end Scope

/-- Stable Rust stage paths for the two shared projection leaves. -/
structure SharedStagePaths where
  betaLadder : String
  rhoEvaluations : String
deriving DecidableEq, Repr

/-- Stable Rust stage paths for one coefficient-limb identity. -/
structure LimbStagePaths where
  inputEvaluations : String
  rhoTimesInput : String
  outputEvaluation : String
  quotientEvaluation : String
  quotientTimesPhi : String
  finalChecks : String
deriving DecidableEq, Repr

/-- Complete path vocabulary for the selected cross-branch certificate. -/
structure SourceStagePaths where
  shared : SharedStagePaths
  limbs : List LimbStagePaths
deriving DecidableEq, Repr

/-- One physical source-R1CS leaf. Definitions and checks are retained rather
than only their totals, so the leaf ends at the actual reconstructed R1CS
constraints. Fresh columns are definition outputs; assertion rows allocate no
column. -/
structure SourceStageLeaf where
  stagePath : String
  indexedDefinitions : List (Nat × Program.Definition)
  indexedChecks : List (Nat × Row)
deriving DecidableEq, Repr

namespace SourceStageLeaf

def definitionCount (leaf : SourceStageLeaf) : Nat :=
  leaf.indexedDefinitions.length

def checkCount (leaf : SourceStageLeaf) : Nat :=
  leaf.indexedChecks.length

def rowCount (leaf : SourceStageLeaf) : Nat :=
  leaf.definitionCount + leaf.checkCount

def freshColumns (leaf : SourceStageLeaf) : List Nat :=
  leaf.indexedDefinitions.map fun entry => entry.2.output

def freshColumnCount (leaf : SourceStageLeaf) : Nat :=
  leaf.freshColumns.length

def rowIndices (leaf : SourceStageLeaf) : List Nat :=
  leaf.indexedDefinitions.map Prod.fst ++ leaf.indexedChecks.map Prod.fst

end SourceStageLeaf

/-- Half-open absolute source-row interval. -/
structure RowBlock where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

namespace RowBlock

def count (rows : RowBlock) : Nat := rows.stop - rows.start

def indices (rows : RowBlock) : List Nat :=
  List.range' rows.start rows.count

def indexDefinitions (rows : RowBlock)
    (definitions : List Program.Definition) :
    List (Nat × Program.Definition) :=
  List.zip rows.indices definitions

def indexChecks (rows : RowBlock) (checks : List Row) : List (Nat × Row) :=
  List.zip rows.indices checks

/-- Local schedule shape. This says nothing about the source R1CS. -/
def Fits (rows : RowBlock) (length : Nat) : Prop :=
  rows.start ≤ rows.stop ∧ rows.count = length

end RowBlock

/-- Exact retained coordinates for one extension-field polynomial evaluator. -/
structure EvaluationOwner where
  rows : RowBlock
  coefficients : List Nat
  powers : List KColumns
  products : List KColumns
  output : KColumns
deriving DecidableEq, Repr

namespace EvaluationOwner

def trace (owner : EvaluationOwner) : EvalTrace where
  coefficients := owner.coefficients
  powers := owner.powers
  products := owner.products
  output := owner.output

def allocatedColumns (owner : EvaluationOwner) : List Nat :=
  (owner.products.flatMap fun product => [product.c0, product.c1]) ++
    [owner.output.c0, owner.output.c1]

def indexedDefinitions (owner : EvaluationOwner) :
    List (Nat × Program.Definition) :=
  owner.rows.indexDefinitions owner.trace.definitions

def Valid (owner : EvaluationOwner) (coefficientCount : Nat) : Prop :=
  owner.coefficients.length = coefficientCount ∧
    owner.trace.LayoutValid ∧
    owner.rows.Fits owner.trace.definitions.length

end EvaluationOwner

/-- Exact retained coordinates and linear combinations for one five-row
extension-field Karatsuba product. -/
structure KProductOwner where
  rows : RowBlock
  left : KTerms
  right : KTerms
  sumLeft : List (Nat × Nat)
  sumRight : List (Nat × Nat)
  productC0 : Nat
  productC1 : Nat
  productSum : Nat
  output : KColumns
deriving DecidableEq, Repr

namespace KProductOwner

def trace (owner : KProductOwner) : KMulTrace where
  left := owner.left
  right := owner.right
  sumLeft := owner.sumLeft
  sumRight := owner.sumRight
  productC0 := owner.productC0
  productC1 := owner.productC1
  productSum := owner.productSum
  output := owner.output

def allocatedColumns (owner : KProductOwner) : List Nat :=
  [owner.productC0, owner.productC1, owner.productSum,
    owner.output.c0, owner.output.c1]

def indexedDefinitions (owner : KProductOwner) :
    List (Nat × Program.Definition) :=
  owner.rows.indexDefinitions owner.trace.definitions

def Valid (owner : KProductOwner) : Prop :=
  owner.trace.SumLayoutValid ∧
    owner.rows.Fits owner.trace.definitions.length

end KProductOwner

/-- One source-specific limb pair. The matching rho evaluator remains in the
shared owner so its rows have exactly one physical owner. -/
structure PairOwner where
  sourceIndex : Nat
  inputEvaluation : EvaluationOwner
  rhoProduct : KProductOwner
deriving DecidableEq, Repr

namespace PairOwner

def trace (owner : PairOwner) (rhoEvaluation : EvaluationOwner) : PairTrace where
  rhoColumns := rhoEvaluation.coefficients
  inputColumns := owner.inputEvaluation.coefficients
  rhoEvaluation := rhoEvaluation.trace
  inputEvaluation := owner.inputEvaluation.trace
  product := owner.rhoProduct.trace

def Valid (owner : PairOwner) (rhoEvaluation : EvaluationOwner)
    (laneCount : Nat) : Prop :=
  owner.inputEvaluation.Valid laneCount ∧
    owner.rhoProduct.Valid ∧
    owner.rhoProduct.left = KTerms.ofColumns rhoEvaluation.output ∧
    owner.rhoProduct.right = KTerms.ofColumns owner.inputEvaluation.output

end PairOwner

/-- Shared beta ladder and rho evaluators used by both coefficient limbs. -/
structure SharedOwner where
  betaLadderRows : RowBlock
  beta : KColumns
  powers : List KColumns
  ladderProducts : List KProductOwner
  rhoEvaluations : List EvaluationOwner
deriving DecidableEq, Repr

namespace SharedOwner

def ladderTrace (owner : SharedOwner) : LadderTrace where
  beta := owner.beta
  powers := owner.powers
  multiplications := owner.ladderProducts.map KProductOwner.trace

def ladderBaseRows (owner : SharedOwner) : RowBlock :=
  ⟨owner.betaLadderRows.start, owner.betaLadderRows.start + 2⟩

def ladderBaseDefinitions (owner : SharedOwner) :
    List (Nat × Program.Definition) :=
  owner.ladderBaseRows.indexDefinitions (owner.ladderTrace.definitions.take 2)

def indexedDefinitions (owner : SharedOwner) :
    List (Nat × Program.Definition) :=
  owner.ladderBaseDefinitions ++
    owner.ladderProducts.flatMap KProductOwner.indexedDefinitions ++
    owner.rhoEvaluations.flatMap EvaluationOwner.indexedDefinitions

def betaLadderDefinitions (owner : SharedOwner) :
    List (Nat × Program.Definition) :=
  owner.ladderBaseDefinitions ++
    owner.ladderProducts.flatMap KProductOwner.indexedDefinitions

def rhoEvaluationDefinitions (owner : SharedOwner) :
    List (Nat × Program.Definition) :=
  owner.rhoEvaluations.flatMap EvaluationOwner.indexedDefinitions

def sourceStageLeaves (owner : SharedOwner)
    (paths : SharedStagePaths) : List SourceStageLeaf :=
  [ { stagePath := paths.betaLadder
      indexedDefinitions := owner.betaLadderDefinitions
      indexedChecks := [] },
    { stagePath := paths.rhoEvaluations
      indexedDefinitions := owner.rhoEvaluationDefinitions
      indexedChecks := [] } ]

def allocatedColumns (owner : SharedOwner) : List Nat :=
  (owner.powers.take 1).flatMap (fun power => [power.c0, power.c1]) ++
    owner.ladderProducts.flatMap KProductOwner.allocatedColumns ++
    owner.rhoEvaluations.flatMap EvaluationOwner.allocatedColumns

def definitionIndices (owner : SharedOwner) : List Nat :=
  owner.indexedDefinitions.map Prod.fst

def ladderDefinitionIndices (owner : SharedOwner) : List Nat :=
  (owner.ladderBaseDefinitions ++
    owner.ladderProducts.flatMap KProductOwner.indexedDefinitions).map Prod.fst

def Valid (owner : SharedOwner) (scope : Scope) : Prop :=
  owner.powers.length = scope.powerCount ∧
    owner.ladderProducts.length + 1 = scope.powerCount ∧
    owner.rhoEvaluations.length = scope.sourceCount ∧
    owner.betaLadderRows.Fits owner.ladderTrace.definitions.length ∧
    owner.ladderDefinitionIndices = owner.betaLadderRows.indices ∧
    owner.definitionIndices = List.range' owner.betaLadderRows.start
      owner.indexedDefinitions.length ∧
    owner.ladderTrace.LayoutValid ∧
    (∀ product ∈ owner.ladderProducts, product.Valid) ∧
    (∀ evaluation ∈ owner.rhoEvaluations,
      evaluation.Valid scope.laneCount ∧
        evaluation.powers = owner.powers.take scope.laneCount)

end SharedOwner

/-- One complete base-field limb of the batched projection identity. -/
structure LimbOwner where
  limb : Nat
  pairs : List PairOwner
  parentEvaluation : EvaluationOwner
  quotientEvaluation : EvaluationOwner
  quotientPhiProduct : KProductOwner
  finalRows : RowBlock
  maxDegree : Nat
deriving DecidableEq, Repr

namespace LimbOwner

def pairTraces (owner : LimbOwner) (shared : SharedOwner) : List PairTrace :=
  (List.zip owner.pairs shared.rhoEvaluations).map fun pair =>
    pair.1.trace pair.2

def trace (owner : LimbOwner) (shared : SharedOwner) : ProjectionTrace where
  ladder := shared.ladderTrace
  pairs := owner.pairTraces shared
  outputColumns := owner.parentEvaluation.coefficients
  quotientColumns := owner.quotientEvaluation.coefficients
  outputEvaluation := owner.parentEvaluation.trace
  quotientEvaluation := owner.quotientEvaluation.trace
  quotientPhiProduct := owner.quotientPhiProduct.trace
  maxDegree := owner.maxDegree

/-- Physical source-emitter order: each input evaluator is immediately
followed by its rho/input product, then come parent, quotient, and quotient/Phi
leaves. -/
def indexedDefinitions (owner : LimbOwner) :
    List (Nat × Program.Definition) :=
  (owner.pairs.flatMap fun pair =>
      pair.inputEvaluation.indexedDefinitions ++
        pair.rhoProduct.indexedDefinitions) ++
    owner.parentEvaluation.indexedDefinitions ++
    owner.quotientEvaluation.indexedDefinitions ++
    owner.quotientPhiProduct.indexedDefinitions

def indexedChecks (owner : LimbOwner) (shared : SharedOwner) :
    List (Nat × Row) :=
  owner.finalRows.indexChecks (owner.trace shared).checks

def inputEvaluationDefinitions (owner : LimbOwner) :
    List (Nat × Program.Definition) :=
  owner.pairs.flatMap fun pair => pair.inputEvaluation.indexedDefinitions

def rhoProductDefinitions (owner : LimbOwner) :
    List (Nat × Program.Definition) :=
  owner.pairs.flatMap fun pair => pair.rhoProduct.indexedDefinitions

def sourceStageLeaves (owner : LimbOwner) (shared : SharedOwner)
    (paths : LimbStagePaths) : List SourceStageLeaf :=
  [ { stagePath := paths.inputEvaluations
      indexedDefinitions := owner.inputEvaluationDefinitions
      indexedChecks := [] },
    { stagePath := paths.rhoTimesInput
      indexedDefinitions := owner.rhoProductDefinitions
      indexedChecks := [] },
    { stagePath := paths.outputEvaluation
      indexedDefinitions := owner.parentEvaluation.indexedDefinitions
      indexedChecks := [] },
    { stagePath := paths.quotientEvaluation
      indexedDefinitions := owner.quotientEvaluation.indexedDefinitions
      indexedChecks := [] },
    { stagePath := paths.quotientTimesPhi
      indexedDefinitions := owner.quotientPhiProduct.indexedDefinitions
      indexedChecks := [] },
    { stagePath := paths.finalChecks
      indexedDefinitions := []
      indexedChecks := owner.indexedChecks shared } ]

def allocatedColumns (owner : LimbOwner) : List Nat :=
  (owner.pairs.flatMap fun pair =>
      pair.inputEvaluation.allocatedColumns ++
        pair.rhoProduct.allocatedColumns) ++
    owner.parentEvaluation.allocatedColumns ++
    owner.quotientEvaluation.allocatedColumns ++
    owner.quotientPhiProduct.allocatedColumns

def ownedIndices (owner : LimbOwner) (shared : SharedOwner) : List Nat :=
  owner.indexedDefinitions.map Prod.fst ++
    (owner.indexedChecks shared).map Prod.fst

def ScheduleValid (owner : LimbOwner) (shared : SharedOwner) : Prop :=
  match owner.pairs with
  | [] => False
  | first :: _ =>
      owner.ownedIndices shared =
        List.range' first.inputEvaluation.rows.start
          (owner.ownedIndices shared).length

def consumerColumns (owner : LimbOwner) : List (Nat × List Nat) :=
  owner.pairs.map fun pair =>
    (pair.sourceIndex, pair.inputEvaluation.coefficients)

def Valid (owner : LimbOwner) (scope : Scope) (shared : SharedOwner) : Prop :=
  owner.limb < 2 ∧
    owner.pairs.length = scope.sourceCount ∧
    owner.pairs.map PairOwner.sourceIndex = List.range scope.sourceCount ∧
    (∀ pair ∈ List.zip owner.pairs shared.rhoEvaluations,
      pair.1.Valid pair.2 scope.laneCount) ∧
    owner.parentEvaluation.Valid scope.laneCount ∧
    owner.parentEvaluation.powers = shared.powers.take scope.laneCount ∧
    owner.quotientEvaluation.Valid scope.quotientCount ∧
    owner.quotientEvaluation.powers =
      shared.powers.take scope.quotientCount ∧
    owner.quotientPhiProduct.Valid ∧
    owner.quotientPhiProduct.left =
      KTerms.ofColumns owner.quotientEvaluation.output ∧
    owner.quotientPhiProduct.right = phiTerms shared.powers ∧
    owner.finalRows.Fits (owner.trace shared).checks.length ∧
    owner.ScheduleValid shared ∧
    owner.maxDegree = scope.maxDegree ∧
    (owner.trace shared).LayoutValid

end LimbOwner

/-- Complete selected cross-branch source artifact. `sourceRows` is the one
raw physical-row list; definition/check classification is recomputed from the
owner schedule. -/
structure Artifact where
  scope : Scope
  sourceRows : List (Nat × Row)
  shared : SharedOwner
  limbs : List LimbOwner
  producers : List ProducerVector
deriving DecidableEq, Repr

namespace Artifact

def traces (artifact : Artifact) : List ProjectionTrace :=
  artifact.limbs.map fun limb => limb.trace artifact.shared

/-- Exact protocol → phase → family → source-constraint ownership. Shared
rows occur once even though both limb traces consume them. `List.zip` fails
closed by truncation; the checked census separately proves the exact two-limb
path/owner shape. -/
def sourceStageLeaves (artifact : Artifact)
    (paths : SourceStagePaths) : List SourceStageLeaf :=
  artifact.shared.sourceStageLeaves paths.shared ++
    (List.zip artifact.limbs paths.limbs).flatMap fun owner =>
      owner.1.sourceStageLeaves artifact.shared owner.2

def indexedDefinitions (artifact : Artifact) :
    List (Nat × Program.Definition) :=
  artifact.shared.indexedDefinitions ++
    artifact.limbs.flatMap LimbOwner.indexedDefinitions

def indexedChecks (artifact : Artifact) : List (Nat × Row) :=
  artifact.limbs.flatMap fun limb => limb.indexedChecks artifact.shared

def definitionIndices (artifact : Artifact) : List Nat :=
  artifact.indexedDefinitions.map Prod.fst

def checkIndices (artifact : Artifact) : List Nat :=
  artifact.indexedChecks.map Prod.fst

def definitionSourceRows (artifact : Artifact) : List (Nat × Row) :=
  artifact.sourceRows.filter fun row =>
    !(artifact.checkIndices.contains row.1)

def checkSourceRows (artifact : Artifact) : List (Nat × Row) :=
  artifact.sourceRows.filter fun row => artifact.checkIndices.contains row.1

/-- Profile-neutral certificate input reconstructed from one raw source-row
list and the handwritten schedule. This is data, not a proof of `ExactRows`. -/
def certificate (artifact : Artifact) : ProjectionArtifactProgram.Certificate where
  definitionSourceRows := artifact.definitionSourceRows
  indexedDefinitions := artifact.indexedDefinitions
  checkSourceRows := artifact.checkSourceRows
  indexedChecks := artifact.indexedChecks

def allocatedColumns (artifact : Artifact) : List Nat :=
  artifact.shared.allocatedColumns ++
    artifact.limbs.flatMap LimbOwner.allocatedColumns

def producerEntries (artifact : Artifact) : List ProducerEntry :=
  artifact.producers.flatMap ProducerVector.entries

def producerFieldIndices (artifact : Artifact) : List Nat :=
  artifact.producerEntries.map ProducerEntry.serializerFieldIndex

def producerSourceColumns (artifact : Artifact) : List Nat :=
  artifact.producerEntries.map ProducerEntry.sourceColumn

def expectedProducerKeys (scope : Scope) : List (Nat × Nat) :=
  (List.range 2).flatMap fun limb =>
    (List.range scope.sourceCount).map fun source => (limb, source)

def producerKeys (artifact : Artifact) : List (Nat × Nat) :=
  artifact.producers.map fun producer =>
    (producer.limb, producer.sourceIndex)

/-- Producer coordinates are shape-checked here but never equated to consumer
columns. That equality is the independent source-binding theorem's job. -/
def ProducerShapeValid (artifact : Artifact) : Prop :=
  artifact.producerKeys = expectedProducerKeys artifact.scope ∧
    (∀ producer ∈ artifact.producers,
      producer.HasShape artifact.scope.laneCount) ∧
    (∀ index ∈ artifact.producerFieldIndices,
      index < artifact.scope.serializerFieldCount) ∧
    (∀ column ∈ artifact.producerSourceColumns,
      column ≠ 0 ∧ column < artifact.scope.sourceArmColumnCount)

def SparseTermsBounded (terms : List (Nat × Nat)) (width : Nat) : Prop :=
  ∀ term ∈ terms, term.1 < width

def RowBounded (row : Row) (width : Nat) : Prop :=
  SparseTermsBounded row.a width ∧
    SparseTermsBounded row.b width ∧
    SparseTermsBounded row.c width

/-- Every raw row has exactly one scheduled owner, and schedule-derived
definition/check filtering preserves their expected order. -/
def RowsOwned (artifact : Artifact) : Prop :=
  StrictlyIncreasing (artifact.sourceRows.map Prod.fst) ∧
    artifact.definitionIndices.length = activeDefinitionCount ∧
    artifact.checkIndices.length = activeCheckCount ∧
    artifact.sourceRows.length = activeRowCount ∧
    (∀ entry ∈ artifact.sourceRows,
      entry.1 < artifact.scope.sourceArmRowCount ∧
        RowBounded entry.2 artifact.scope.sourceArmColumnCount) ∧
    artifact.definitionSourceRows.map Prod.fst =
      artifact.definitionIndices ∧
    artifact.checkSourceRows.map Prod.fst =
      artifact.checkIndices

/-- Every definition owns one fresh, nonconstant in-bounds column. This is the
physical column census; it is deliberately separate from row semantics. -/
def ColumnsOwned (artifact : Artifact) : Prop :=
  artifact.allocatedColumns.length = activeDefinitionCount ∧
    StrictlyIncreasing artifact.allocatedColumns ∧
    (∀ column ∈ artifact.allocatedColumns,
      column ≠ 0 ∧ column < artifact.scope.sourceArmColumnCount)

/-- Structural gate only; it makes no semantic or Rust-conformance claim. -/
def StructureValid (artifact : Artifact) : Prop :=
  artifact.scope.IsTinyFixture ∧
    artifact.shared.Valid artifact.scope ∧
    artifact.limbs.length = 2 ∧
    artifact.limbs.map LimbOwner.limb = [0, 1] ∧
    (∀ limb ∈ artifact.limbs,
      limb.Valid artifact.scope artifact.shared) ∧
    artifact.ProducerShapeValid ∧
    artifact.RowsOwned ∧
    artifact.ColumnsOwned

end Artifact

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
