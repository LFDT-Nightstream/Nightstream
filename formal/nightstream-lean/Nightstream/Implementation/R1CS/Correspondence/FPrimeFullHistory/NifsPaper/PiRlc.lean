import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicInputBoundary
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryProjectionArtifact
import Nightstream.SuperNeo.Folding.PiRLC

/-!
Conditional model-level Pi_RLC refinement for the fixed full-history F' NIFS
profiles.

Owns: the fixed public projection tree, a shared-carrier Pi_RLC algebra, paper
attempt construction, and the implication from four explicit artifact
obligations to `PiRLC.Accepted`.

Does not own: a generated inhabitant of any obligation below, private CE
membership, `CE.Holds`, extraction, phase composition, row removal, or a
production proof of the strong sampling set. `y_zcol` remains a separately
typed delayed-NC sidecar and is never inserted into the paper CE evaluations.
Likewise `s_col` and `fold_digest` are transcript/NC context, not the paper CE
point; they live in an optional sidecar below and are not premises of paper
acceptance. Production's five packed x rings retain all 270 coefficients.
`PublicInputBoundary.productionPublicWidth_not_aligned` and
`publicProjection_not_injective` show that generic homogeneous `PublicInput`
cannot express production's fresh-257 CCS to packed-270 CE boundary.
`PublicInputBoundary.ringAction_enters_extra_coefficient` additionally proves
that the actual Phi81 ring action moves coefficient 40 into extra coefficient
41, so truncation is not closed under the operation verified here. This file
proves only the packed CE-to-CE arithmetic and does not relabel that unresolved
heterogeneous NIFS boundary as paper public-input refinement.

| Gap / theorem | Production owner | Mathematical obligation | Status |
|---|---|---|---|
| `SamplerArtifact` | Pi_RLC transcript sampler | all 29 paper-public traces use one decoded rho per input, and every rho satisfies the algebra's unary membership predicate | missing generated/refinement data; pairwise strong-set security is a separate theorem |
| `CarrierArtifact` | projection-role and affine glue exporters | exact 18 commitment, 5 active-x, and 3 x 2 y-ring trace-to-carrier wiring, with exactly 54 coefficients in every input/output leaf | missing typed facade |
| `ParentArtifact` | strict-PiDEC parent allocation | combined public carrier and point are exactly the strict-PiDEC parent columns | missing typed facade |
| `ReductionArtifact` | exact projection identities plus Phi81 algebra | exact polynomial equality implies the selected 54-coefficient quotient-ring combination at every public leaf | missing deterministic refinement lemma |
| `AlgebraRefinement` | independent paper algebra instantiation | the actual algebra's public operations are the same componentwise Phi81 combination | obligation only; no production inhabitant exported |
| `PublicInputBoundary` | heterogeneous CCS-to-CE input boundary | reconcile 257 scalar CCS inputs with the non-injective 270-cell carrier and Phi81 flow into coefficient 41 | generic homogeneous paper input is insufficient; semantic bridge open |
| `DelayedNcTraceTree` | two `y_zcol` projection traces | delayed-NC transition data only | sidecar; excluded from 29 paper-public leaves and acceptance |
| `TranscriptNcSidecarArtifact` | `sColCols`, `foldDigestCols` | transcript/NC context authority | sidecar; excluded from CE point and acceptance |
| `EvaluationPaddingSidecarArtifact` | 20 tail limbs in each y row | implementation zero padding and packed-parent placement | sidecar; excluded from active evaluations and acceptance |
| `accepted_of_refinement` | this file | the four lower-level obligations imply generic packed-carrier `PiRLC.Accepted` | model-level, conditional; not production NIFS acceptance |

`FPR-NIFS-BRIDGE` therefore remains open. In particular, a proof of
`ReductionArtifact.exact` alone is insufficient: the quotient/remainder
refinement field is intentionally separate so coefficient equality cannot be
silently treated as a typed CE equation.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

set_option maxRecDepth 4096

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

/-! ## Public projection tree -/

abbrev Ring := List Scalar
abbrev CommitmentRings := Fin 18 -> Ring
abbrev XRings := Fin 5 -> Ring
abbrev YRingRings := Fin 3 -> Fin 2 -> Ring

/-- Exactly the projection components carried by the paper CE statement. -/
structure ProjectionOpening where
  commitment : CommitmentRings
  x : XRings
  yRing : YRingRings

/-- Fixed public leaves in review order. `y_zcol` is deliberately absent. -/
inductive PublicRole where
  | commitment (lane : Fin 18)
  | x (column : Fin 5)
  | yRing (row : Fin 3) (limb : Fin 2)
deriving DecidableEq, Repr

def publicOrder : List PublicRole :=
  (List.ofFn fun lane : Fin 18 => .commitment lane) ++
  (List.ofFn fun column : Fin 5 => .x column) ++
  (List.ofFn fun row : Fin 3 =>
    List.ofFn fun limb : Fin 2 => .yRing row limb).flatten

theorem public_role_count : publicOrder.length = 29 := by
  decide

def ProjectionOpening.at (opening : ProjectionOpening) : PublicRole -> Ring
  | .commitment lane => opening.commitment lane
  | .x column => opening.x column
  | .yRing row limb => opening.yRing row limb

structure ProjectionColumns where
  commitment : Fin 18 -> List Nat
  x : Fin 5 -> List Nat
  yRing : Fin 3 -> Fin 2 -> List Nat
  /-- Twenty implementation-only tail slots per y row. -/
  yRingPadding : Fin 3 -> List Nat

def ProjectionColumns.at (columns : ProjectionColumns) : PublicRole -> List Nat
  | .commitment lane => columns.commitment lane
  | .x column => columns.x column
  | .yRing row limb => columns.yRing row limb

def decodeOpening (assignment : Nat -> Nat)
    (columns : ProjectionColumns) : ProjectionOpening where
  commitment lane := values assignment (columns.commitment lane)
  x column := values assignment (columns.x column)
  yRing row limb := values assignment (columns.yRing row limb)

@[simp] theorem decodeOpening_at (assignment : Nat -> Nat)
    (columns : ProjectionColumns) (role : PublicRole) :
    (decodeOpening assignment columns).at role =
      values assignment (columns.at role) := by
  cases role <;> rfl

/-! ## Shared public-carrier codec -/

/-- A one-way canonical representation boundary. It intentionally has no
inverse: the separate Concrete 257-field projection cannot recover all 270
packed Pi_RLC x coefficients. Concrete selection and layout are checked by
`CodecArtifact`; y padding never enters the encoded evaluation carrier. -/
structure Codec (Source Target : Type) where
  encode : Source -> Target

structure CarrierCodec where
  commitment : Codec CommitmentRings PackedCommitment
  x : Codec XRings PackedPublicInput
  yRing : Codec YRingRings (Array Evaluation)

def ProjectionOpening.commit (codec : CarrierCodec)
    (opening : ProjectionOpening) : PackedCommitment :=
  codec.commitment.encode opening.commitment

def ProjectionOpening.publicInput (codec : CarrierCodec)
    (opening : ProjectionOpening) : PackedPublicInput :=
  codec.x.encode opening.x

def ProjectionOpening.evaluations (codec : CarrierCodec)
    (opening : ProjectionOpening) : Array Evaluation :=
  codec.yRing.encode opening.yRing

/-! ## Mathematical ring combination -/

/-- Interpret an exact 54-coefficient carrier as a polynomial in the concrete
Phi81 quotient ring. Shape obligations below ensure production never relies on
the default outside the carrier. -/
def ringOfList (coefficients : Ring) : Concrete.RingF :=
  fun coefficient => coefficients.getD coefficient.val 0

/-- The fixed public operation: sum of challenge-times-input products in
`F[X] / (X^54 + X^27 + 1)`, returned in coefficient order. -/
def phi81Combine {n : Nat} (challenges : Fin n -> Ring)
    (inputs : Fin n -> Ring) : Ring :=
  List.ofFn fun coefficient : Fin Concrete.ringDegree =>
    (List.ofFn fun index : Fin n =>
      Concrete.ringFMul (ringOfList (challenges index))
        (ringOfList (inputs index)) coefficient).foldl (fun sum item => sum + item) 0

/-- The paper-selected strong set plus a public combination implementation.
`phi81` is deliberately an obligation: until a concrete inhabitant is
exported, this remains an abstract interface constrained to the actual Phi81
operation, not a production algebra instance. A generated sampler still has
to prove strong-set membership for its decoded columns. -/
structure RingAlgebra where
  challengeValid : Ring -> Prop
  combine : {n : Nat} -> (Fin n -> Ring) -> (Fin n -> Ring) -> Ring
  phi81 : forall {n : Nat} (challenges : Fin n -> Ring)
      (inputs : Fin n -> Ring),
    combine challenges inputs = phi81Combine challenges inputs

def combineOpening {n : Nat} (ring : RingAlgebra)
    (challenges : Fin n -> Ring)
    (openings : Fin n -> ProjectionOpening) : ProjectionOpening where
  commitment lane := ring.combine challenges fun index =>
    (openings index).commitment lane
  x column := ring.combine challenges fun index => (openings index).x column
  yRing row limb := ring.combine challenges fun index =>
    (openings index).yRing row limb

/-- Universal public-operation refinement required from the actual paper
algebra. This interface says nothing about private assignments or relation
membership. Its laws are independent of any one attempt and expose exactly
which public combine operation production must instantiate. -/
structure AlgebraRefinement
    {Assignment : Type}
    {params : GlobalParams}
    {semantics : RelationSemantics Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment}
    (algebra : PiRLC.Algebra Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment Ring semantics params)
    (codec : CarrierCodec) (ring : RingAlgebra) : Prop where
  challenge : forall value,
    ring.challengeValid value -> algebra.challengeValid value
  commitment : forall {n : Nat} (challenges : Fin n -> Ring)
      (openings : Fin n -> ProjectionOpening),
    codec.commitment.encode (combineOpening ring challenges openings).commitment =
      algebra.combineCommitment challenges
        (fun index => (openings index).commit codec)
  x : forall {n : Nat} (challenges : Fin n -> Ring)
      (openings : Fin n -> ProjectionOpening),
    codec.x.encode (combineOpening ring challenges openings).x =
      algebra.combinePublicInput challenges
        (fun index => (openings index).publicInput codec)
  yRing : forall {n : Nat} (challenges : Fin n -> Ring)
      (openings : Fin n -> ProjectionOpening),
    codec.yRing.encode (combineOpening ring challenges openings).yRing =
      algebra.combineEvaluations challenges
        (fun index => (openings index).evaluations codec)

/-! ## Attempt columns and decoding -/

structure PointColumns where
  r : List (Nat × Nat)

def decodePointColumns (assignment : Nat -> Nat)
    (columns : PointColumns) : Point :=
  extensionValues assignment columns.r

/-- Implementation context kept visible for later authority/minimality work.
It is not a paper CE point and is not consumed by `accepted_of_refinement`. -/
structure TranscriptNcContextColumns where
  sCol : List (Nat × Nat)
  foldDigest : List Nat

/-- Optional column identity for the non-CE transcript/NC context. This is
deliberately outside `Refinement`: whether these bindings are necessary must
be decided by the transcript/NC authority proof, not by Pi_RLC `samePoint`. -/
structure TranscriptNcSidecarArtifact
    (context : TranscriptNcContextColumns) : Prop where
  sCol : context.sCol = layout.parent.sColCols
  foldDigest : context.foldDigest = layout.parent.foldDigestCols

structure BatchColumns
    (params : GlobalParams) (arity : BatchArity params) where
  challenges : Fin arity.total -> List Nat
  inputs : Fin arity.total -> ProjectionColumns
  output : ProjectionColumns
  inputPoints : Fin arity.total -> PointColumns
  outputPoint : PointColumns

def decodedInstance (codec : CarrierCodec) (assignment : Nat -> Nat)
    (point : Point) (stage : NormStage) (columns : ProjectionColumns) :
    CE.Instance Unit PackedPublicInput Point Evaluation PackedCommitment where
  constraintSystem := ()
  commitment := (decodeOpening assignment columns).commit codec
  publicInput := (decodeOpening assignment columns).publicInput codec
  point := point
  evaluations := (decodeOpening assignment columns).evaluations codec
  stage := stage

def attempt {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity) :
    PiRLC.Attempt Unit PackedPublicInput Point Evaluation PackedCommitment
      Ring params arity where
  inputs index := decodedInstance codec assignment
    (decodePointColumns assignment (columns.inputPoints index))
    .fresh (columns.inputs index)
  challenges index := values assignment (columns.challenges index)
  output := decodedInstance codec assignment
    (decodePointColumns assignment columns.outputPoint) .combined columns.output

/-! ## Fixed trace tree and the four open artifact obligations -/

/-- The generated 29-trace paper-public subtree. The two delayed-NC traces
have a separate type below and cannot be consumed through this interface. -/
structure TraceTree
    {params : GlobalParams} (arity : BatchArity params) where
  publicTrace : PublicRole -> ProjectionTrace
  publicPairArity : forall role, (publicTrace role).pairs.length = arity.total

def TraceTree.flatten {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity) : List ProjectionTrace :=
  publicOrder.map tree.publicTrace

def TraceTree.publicPairAt
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity) (role : PublicRole) (index : Fin arity.total) :
    PairTrace :=
  (tree.publicTrace role).pairs.get
    (Fin.cast (tree.publicPairArity role).symm index)

/-- Gap 1: transcript columns are shared across every role and every decoded
rho satisfies the algebra's unary challenge-membership predicate. Pairwise
invertibility of distinct members is intentionally not stored under this
name; it is a separate security theorem about the chosen algebra. -/
structure SamplerArtifact
    {params : GlobalParams} {arity : BatchArity params}
    (ring : RingAlgebra) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity) (tree : TraceTree arity) : Prop where
  width : forall index,
    (columns.challenges index).length = Concrete.ringDegree
  publicShared : forall role index,
    (tree.publicPairAt role index).rhoColumns = columns.challenges index
  challengeMembership : forall index,
    ring.challengeValid (values assignment (columns.challenges index))

theorem SamplerArtifact.traceChallengeWidth
    {params : GlobalParams} {arity : BatchArity params}
    {ring : RingAlgebra} {assignment : Nat -> Nat}
    {columns : BatchColumns params arity} {tree : TraceTree arity}
    (artifact : SamplerArtifact ring assignment columns tree)
    (role : PublicRole) (index : Fin arity.total) :
    (tree.publicPairAt role index).rhoColumns.length = Concrete.ringDegree := by
  rw [artifact.publicShared role index]
  exact artifact.width index

/-- Column carrier for the two delayed-NC `y_zcol` limbs. It is not part of
`BatchColumns` and cannot enter a paper CE instance. -/
structure DelayedNcColumns
    (params : GlobalParams) (arity : BatchArity params) where
  inputs : Fin arity.total -> Fin 2 -> List Nat
  output : Fin 2 -> List Nat

structure DelayedNcTraceTree
    {params : GlobalParams} (arity : BatchArity params) where
  trace : Fin 2 -> ProjectionTrace
  pairArity : forall limb, (trace limb).pairs.length = arity.total

def DelayedNcTraceTree.pairAt
    {params : GlobalParams} {arity : BatchArity params}
    (tree : DelayedNcTraceTree arity) (limb : Fin 2)
    (index : Fin arity.total) : PairTrace :=
  (tree.trace limb).pairs.get (Fin.cast (tree.pairArity limb).symm index)

/-- Optional delayed-NC trace wiring. This proposition is deliberately not a
field of `Refinement` and is not consumed by `accepted_of_refinement`. -/
structure DelayedNcSidecarArtifact
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat) (batch : BatchColumns params arity)
    (columns : DelayedNcColumns params arity)
    (tree : DelayedNcTraceTree arity)
    (productionTraces : List ProjectionTrace) : Prop where
  census : List.ofFn tree.trace = productionTraces
  challengeShared : forall limb index,
    (tree.pairAt limb index).rhoColumns = batch.challenges index
  inputColumns : forall limb index,
    (tree.pairAt limb index).inputColumns = columns.inputs index limb
  outputColumns : forall limb,
    (tree.trace limb).outputColumns = columns.output limb
  exact : forall limb, (tree.trace limb).identity assignment |>.Exact

/-- Exact public list layout expected by the shared carrier. -/
def assembleCommitmentColumns (columns : ProjectionColumns) : List Nat :=
  (List.ofFn columns.commitment).flatten

def assembleXColumns (columns : ProjectionColumns) : List Nat :=
  (List.ofFn fun coefficient : Fin 54 =>
    List.ofFn fun column : Fin 5 =>
      (columns.x column).getD coefficient.val 0).flatten

def assembleYRingActiveColumns (columns : ProjectionColumns) : List (List Nat) :=
  List.ofFn fun row : Fin 3 =>
    (List.ofFn fun coefficient : Fin 54 =>
      List.ofFn fun limb : Fin 2 =>
        (columns.yRing row limb).getD coefficient.val 0).flatten

def assembleYRingPackedColumns (columns : ProjectionColumns) : List (List Nat) :=
  List.ofFn fun row : Fin 3 =>
    (assembleYRingActiveColumns columns).getD row.val [] ++
      columns.yRingPadding row

def decodeYRingActive (assignment : Nat -> Nat)
    (columns : ProjectionColumns) : Array Evaluation :=
  Array.ofFn fun row : Fin 3 =>
    fun coefficient : Fin 54 =>
      ⟨residue (assignment
          ((columns.yRing row ⟨0, by decide⟩).getD coefficient.val 0)),
       residue (assignment
          ((columns.yRing row ⟨1, by decide⟩).getD coefficient.val 0))⟩

/-- Deterministic codec/list-layout conformance, grouped with the typed
carrier wiring rather than treated as a fifth protocol assumption. -/
structure CodecArtifact (codec : CarrierCodec) : Prop where
  commitment : forall assignment columns,
    codec.commitment.encode (decodeOpening assignment columns).commitment =
      (PackedCommitment.mk
        (values assignment (assembleCommitmentColumns columns)))
  x : forall assignment columns,
    codec.x.encode (decodeOpening assignment columns).x =
      PackedPublicInput.mk (values assignment (assembleXColumns columns))
  yRing : forall assignment columns,
    codec.yRing.encode (decodeOpening assignment columns).yRing =
      decodeYRingActive assignment columns

/-- Gap 2: the generated trace order and every public input/output polynomial
are tied to the exact typed CE carrier leaves. Every public polynomial has
exactly 54 coefficients, so no `getD` defaulting or tail truncation can affect
an accepted refinement. -/
structure CarrierArtifact
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec) (columns : BatchColumns params arity)
    (tree : TraceTree arity) (productionTraces : List ProjectionTrace) : Prop where
  census : tree.flatten = productionTraces
  codecLayout : CodecArtifact codec
  inputWidth : forall role index,
    ((columns.inputs index).at role).length = Concrete.ringDegree
  outputWidth : forall role,
    (columns.output.at role).length = Concrete.ringDegree
  inputColumns : forall role index,
    (tree.publicPairAt role index).inputColumns = (columns.inputs index).at role
  outputColumns : forall role,
    (tree.publicTrace role).outputColumns = columns.output.at role
  pointColumns : forall index,
    columns.inputPoints index = columns.outputPoint

theorem CarrierArtifact.traceInputWidth
    {params : GlobalParams} {arity : BatchArity params}
    {codec : CarrierCodec} {columns : BatchColumns params arity}
    {tree : TraceTree arity} {productionTraces : List ProjectionTrace}
    (artifact : CarrierArtifact codec columns tree productionTraces)
    (role : PublicRole) (index : Fin arity.total) :
    (tree.publicPairAt role index).inputColumns.length = Concrete.ringDegree := by
  rw [artifact.inputColumns role index]
  exact artifact.inputWidth role index

theorem CarrierArtifact.traceOutputWidth
    {params : GlobalParams} {arity : BatchArity params}
    {codec : CarrierCodec} {columns : BatchColumns params arity}
    {tree : TraceTree arity} {productionTraces : List ProjectionTrace}
    (artifact : CarrierArtifact codec columns tree productionTraces)
    (role : PublicRole) :
    (tree.publicTrace role).outputColumns.length = Concrete.ringDegree := by
  rw [artifact.outputColumns role]
  exact artifact.outputWidth role

/-- Implementation-only 20-slot y tails. This is intentionally outside
`Refinement`: zero padding is not a paper evaluation equation and must not be
counted as part of Pi_RLC acceptance. -/
structure EvaluationPaddingSidecarArtifact
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat) (columns : BatchColumns params arity) : Prop where
  inputWidth : forall index row,
    ((columns.inputs index).yRingPadding row).length = 20
  inputZero : forall index row column,
    column ∈ (columns.inputs index).yRingPadding row -> assignment column = 0
  outputWidth : forall row, (columns.output.yRingPadding row).length = 20
  outputZero : forall row column,
    column ∈ columns.output.yRingPadding row -> assignment column = 0
  parentPacked : assembleYRingPackedColumns columns.output =
    layout.parent.yRingCols

/-- Gap 3: the Π_RLC combined output is the same strict-PiDEC parent, by
column identity rather than by equality of prover-supplied values. -/
structure ParentArtifact
    {params : GlobalParams} {arity : BatchArity params}
    (columns : BatchColumns params arity) : Prop where
  commitment : assembleCommitmentColumns columns.output =
    layout.parent.commitment.dataCols
  x : assembleXColumns columns.output = layout.parent.xActiveCols
  yRing : forall row limb,
    columns.output.yRing row limb =
      (List.ofFn fun coefficient : Fin 54 =>
        (layout.parent.yRingCols.getD row.val []).getD
          (2 * coefficient.val + limb.val) 0)
  r : columns.outputPoint.r = layout.parent.rCols

/-- Gap 4: exact coefficient identities are interpreted by the concrete Phi81
quotient-ring operation. The implication is per public leaf; it is not a
disguised paper `Accepted` premise. A production inhabitant remains missing. -/
structure ReductionArtifact
    {params : GlobalParams} {arity : BatchArity params}
    (ring : RingAlgebra) (assignment : Nat -> Nat)
    (tree : TraceTree arity) : Prop where
  exact : forall role, (tree.publicTrace role).identity assignment |>.Exact
  quotientRemainder : forall role,
    (tree.publicTrace role).identity assignment |>.Exact ->
      values assignment (tree.publicTrace role).outputColumns =
        phi81Combine
          (fun index =>
            values assignment (tree.publicPairAt role index).rhoColumns)
          (fun index =>
            values assignment (tree.publicPairAt role index).inputColumns)

/-- The four production gaps, with no verifier-acceptance predicate stored as
proof data. -/
structure Refinement
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec) (ring : RingAlgebra) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity) (tree : TraceTree arity)
    (productionTraces : List ProjectionTrace) : Prop where
  sampler : SamplerArtifact ring assignment columns tree
  carrier : CarrierArtifact codec columns tree productionTraces
  parent : ParentArtifact columns
  reduction : ReductionArtifact ring assignment tree

private theorem reduction_at
    {params : GlobalParams} {arity : BatchArity params}
    {ring : RingAlgebra} {assignment : Nat -> Nat}
    {columns : BatchColumns params arity} {tree : TraceTree arity}
    {codec : CarrierCodec} {productionTraces : List ProjectionTrace}
    (sampler : SamplerArtifact ring assignment columns tree)
    (carrier : CarrierArtifact codec columns tree productionTraces)
    (reduction : ReductionArtifact ring assignment tree)
    (role : PublicRole) :
    (decodeOpening assignment columns.output).at role =
      ring.combine
        (fun index => values assignment (columns.challenges index))
        (fun index =>
          (decodeOpening assignment (columns.inputs index)).at role) := by
  rw [decodeOpening_at, ← carrier.outputColumns role,
    reduction.quotientRemainder role (reduction.exact role)]
  have challenges :
      (fun index =>
        values assignment (tree.publicPairAt role index).rhoColumns) =
      (fun index => values assignment (columns.challenges index)) := by
    funext index
    rw [sampler.publicShared role index]
  have inputs :
      (fun index =>
        values assignment (tree.publicPairAt role index).inputColumns) =
      (fun index =>
        (decodeOpening assignment (columns.inputs index)).at role) := by
    funext index
    rw [carrier.inputColumns role index, decodeOpening_at]
  rw [challenges, inputs, ← ring.phi81]

/-- The exact per-leaf reduction plus sampler membership constructs the
generic paper Π_RLC verifier predicate over a packed candidate carrier. This
is not a production paper-acceptance theorem and does not close the separate
fresh-257 to CE-270 public-input boundary. -/
theorem accepted_of_refinement
    {Assignment : Type}
    {params : GlobalParams} {arity : BatchArity params}
    {semantics : RelationSemantics Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment}
    (algebra : PiRLC.Algebra Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment Ring semantics params)
    (codec : CarrierCodec) (ring : RingAlgebra)
    (algebraRefinement : AlgebraRefinement algebra codec ring)
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity) (tree : TraceTree arity)
    {productionTraces : List ProjectionTrace}
    (refinement : Refinement codec ring assignment columns tree productionTraces) :
    PiRLC.Accepted algebra (attempt codec assignment columns) := by
  refine {
    inputFresh := ?_
    sameStructure := ?_
    samePoint := ?_
    challengesValid := fun index =>
      algebraRefinement.challenge _
        (refinement.sampler.challengeMembership index)
    outputCombined := ?_
    commitmentEquation := ?_
    publicInputEquation := ?_
    evaluationEquation := ?_
  }
  · intro index
    rfl
  · intro index
    rfl
  · intro index
    exact congrArg (decodePointColumns assignment)
      (refinement.carrier.pointColumns index)
  · rfl
  · change codec.commitment.encode
        (decodeOpening assignment columns.output).commitment =
      algebra.combineCommitment _ _
    calc
      _ = codec.commitment.encode
          (combineOpening ring
            (fun index => values assignment (columns.challenges index))
            (fun index => decodeOpening assignment
              (columns.inputs index))).commitment := by
        apply congrArg codec.commitment.encode
        funext lane
        exact reduction_at refinement.sampler refinement.carrier
          refinement.reduction (.commitment lane)
      _ = _ := algebraRefinement.commitment _ _
  · change codec.x.encode (decodeOpening assignment columns.output).x =
      algebra.combinePublicInput _ _
    calc
      _ = codec.x.encode
          (combineOpening ring
            (fun index => values assignment (columns.challenges index))
            (fun index => decodeOpening assignment (columns.inputs index))).x := by
        apply congrArg codec.x.encode
        funext column
        exact reduction_at refinement.sampler refinement.carrier
          refinement.reduction (.x column)
      _ = _ := algebraRefinement.x _ _
  · change codec.yRing.encode (decodeOpening assignment columns.output).yRing =
      algebra.combineEvaluations _ _
    calc
      _ = codec.yRing.encode
          (combineOpening ring
            (fun index => values assignment (columns.challenges index))
            (fun index => decodeOpening assignment
              (columns.inputs index))).yRing := by
        apply congrArg codec.yRing.encode
        funext row limb
        exact reduction_at refinement.sampler refinement.carrier
          refinement.reduction (.yRing row limb)
      _ = _ := algebraRefinement.yRing _ _

private theorem getD_ofFn {Carrier : Type} {size : Nat}
    (items : Fin size → Carrier) (index : Fin size) (default : Carrier) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem decodeYRingActive_eq_parent
    (assignment : Nat -> Nat) (columns : ProjectionColumns)
    (wiring : forall row limb,
      columns.yRing row limb =
        (List.ofFn fun coefficient : Fin 54 =>
          (layout.parent.yRingCols.getD row.val []).getD
            (2 * coefficient.val + limb.val) 0)) :
    decodeYRingActive assignment columns =
      FPrimeFullHistoryNifsPaper.decodedEvaluations assignment layout.parent := by
  apply Array.ext
  · simp [decodeYRingActive, FPrimeFullHistoryNifsPaper.decodedEvaluations,
      FPrimeFullHistoryPiDec.layout]
  · intro index left right
    have indexLt : index < 3 := by
      simpa [decodeYRingActive] using left
    let row : Fin 3 := ⟨index, indexLt⟩
    simp only [decodeYRingActive, Array.getElem_ofFn]
    change (fun coefficient : Fin 54 =>
      Concrete.K.mk
        (residue (assignment
          ((columns.yRing row ⟨0, by decide⟩).getD coefficient.val 0)))
        (residue (assignment
          ((columns.yRing row ⟨1, by decide⟩).getD coefficient.val 0)))) = _
    funext coefficient
    rw [wiring row ⟨0, by decide⟩, wiring row ⟨1, by decide⟩]
    rw [getD_ofFn, getD_ofFn]
    change Concrete.K.mk
      (residue (assignment
        ((layout.parent.yRingCols.getD row.val []).getD
          (2 * coefficient.val) 0)))
      (residue (assignment
        ((layout.parent.yRingCols.getD row.val []).getD
          (2 * coefficient.val + 1) 0))) = _
    have parentRows : layout.parent.yRingCols.length = 3 := by
      decide
    have rowLt : row.val < layout.parent.yRingCols.length := by
      rw [parentRows]
      exact row.isLt
    simp only [FPrimeFullHistoryNifsPaper.decodedEvaluations,
      List.getElem_toArray, List.getElem_map, FPrimeFullHistoryNifsPaper.decodedEvaluation]
    rw [← List.getElem_eq_getD
      (l := layout.parent.yRingCols) (i := row.val) []]

/-- The separate output-column obligation identifies the accepted combined CE
with the strict-PiDEC parent. This is linkage only, not CE membership. -/
theorem output_eq_piDecParent_of_refinement
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec) (ring : RingAlgebra) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity) (tree : TraceTree arity)
    {productionTraces : List ProjectionTrace}
    (refinement : Refinement codec ring assignment columns tree productionTraces) :
    (attempt codec assignment columns).output =
      FPrimeFullHistoryNifsPaper.decodedInstance () assignment
        layout.parent .combined := by
  have commitment := refinement.carrier.codecLayout.commitment assignment columns.output
  have input := refinement.carrier.codecLayout.x assignment columns.output
  have evaluations := refinement.carrier.codecLayout.yRing assignment columns.output
  rw [refinement.parent.commitment] at commitment
  rw [refinement.parent.x] at input
  rw [decodeYRingActive_eq_parent assignment columns.output
    refinement.parent.yRing] at evaluations
  have point : decodePointColumns assignment columns.outputPoint =
      FPrimeFullHistoryNifsPaper.decodedPoint assignment layout.parent := by
    exact congrArg (extensionValues assignment) refinement.parent.r
  simp only [attempt, decodedInstance,
    ProjectionOpening.commit, ProjectionOpening.publicInput,
    ProjectionOpening.evaluations,
    FPrimeFullHistoryNifsPaper.decodedInstance,
    FPrimeFullHistoryNifsPaper.decodedPackedCommitment,
    FPrimeFullHistoryNifsPaper.decodedPackedInput]
  rw [commitment, input, point, evaluations]

/-! ## Fixed arity wrappers -/

def recursiveArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.bootstrap Concrete.productionGlobalParams 1 (by decide) (by decide)

def terminalArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.active Concrete.productionGlobalParams 1 (by decide) (by decide)

def recursiveAttempt (codec : CarrierCodec) (assignment : Nat -> Nat)
    (columns : BatchColumns Concrete.productionGlobalParams recursiveArity) :=
  attempt codec assignment columns

def terminalAttempt (codec : CarrierCodec) (assignment : Nat -> Nat)
    (columns : BatchColumns Concrete.productionGlobalParams terminalArity) :=
  attempt codec assignment columns

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
