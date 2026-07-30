import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicInputBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PublicColumns
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryProjectionArtifact
import Nightstream.SuperNeo.Folding.PiRLC

/-!
Matrix-count-parametric Π_RLC public-carrier refinement.

Assurance tier: model-level. Fixed recursive and terminal artifacts instantiate
this interface only as explicitly labelled diagnostics; this file is neither
Rust-conformant nor security-reduced.

Owns: the ordered public projection carrier, its canonical codec boundary, the
Phi81 public combination, typed attempt construction, structural artifacts,
and the implication from exact leaf equations to `PiRLC.Equations`.

Does not own: transcript-derived challenge membership, private CE membership,
Π_CCS output authority, Π_DEC child membership, delayed `y_zcol`, `s_col`,
`fold_digest`, extraction, full NIFS acceptance, row removal, or costs.

Emits constraints: no.

Authority boundary: `matrixCount` is a type index throughout this interface.
The active specialization must derive it from the independent relation shape.
The legacy three-row fixtures live in `DiagnosticProfile` and cannot establish
active 13-matrix conformance.

| Child/stage | Mathematical obligation | Authority class | Lean owner | Rust/R1CS owner | Multiplicity source |
|---|---|---|---|---|---|
| public carrier | commitment, five packed `X` rings, and two limbs per matrix evaluation | direct dataflow | `ProjectionOpening`, `PublicRole` | projection-role emitter | `public_role_count` |
| codec | one canonical representation of each public carrier family | computed | `CarrierCodec`, `CodecArtifact` | carrier decoder/refinement | semantic matrix count |
| algebra | componentwise Φ81 challenge-times-input sum | checked | `phi81Combine`, `AlgebraRefinement` | Π_RLC ring arithmetic | ring degree and batch arity |
| attempt | common structure, point, stages, and combined output | computed/direct dataflow | `attempt`, `EquationWiringArtifact` | NIFS Π_RLC orchestration | batch arity |
| public identities | every output leaf equals the Φ81 combination of its inputs | checked | `ReductionArtifact`, `equations_of_refinement` | projection identity rows | public-role tree |
| parent binding | combined claim is the exact strict-Π_DEC parent | direct dataflow | `ParentArtifact`, `output_eq_parent_of_artifacts` | parent allocation/binding | semantic matrix count |

Challenge sampling, active claim-shape alignment, arithmetic transport, and
fixed diagnostic artifacts are separate child modules. Their completion state
belongs in the bridge specification, not in this semantic parent.
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
abbrev YRingRings (matrixCount : Nat) := Fin matrixCount -> Fin 2 -> Ring

/-- Exactly the projection components carried by the paper CE statement. -/
structure ProjectionOpening (matrixCount : Nat) where
  commitment : CommitmentRings
  x : XRings
  yRing : YRingRings matrixCount

def ProjectionOpening.at {matrixCount : Nat}
    (opening : ProjectionOpening matrixCount) : PublicRole matrixCount -> Ring
  | .commitment lane => opening.commitment lane
  | .x column => opening.x column
  | .yRing row limb => opening.yRing row limb

def decodeOpening {matrixCount : Nat} (assignment : Nat -> Nat)
    (columns : ProjectionColumns matrixCount) :
    ProjectionOpening matrixCount where
  commitment lane := values assignment (columns.commitment lane)
  x column := values assignment (columns.x column)
  yRing row limb := values assignment (columns.yRing row limb)

@[simp] theorem decodeOpening_at {matrixCount : Nat}
    (assignment : Nat -> Nat)
    (columns : ProjectionColumns matrixCount)
    (role : PublicRole matrixCount) :
    (decodeOpening assignment columns).at role =
      values assignment (columns.at role) := by
  cases role <;> rfl

/-! ## Shared public-carrier codec -/

/-- A one-way canonical representation boundary. It intentionally has no
inverse from the legacy 257-field projection to all 270 packed Pi_RLC x
coefficients. Concrete selection and layout are checked by `CodecArtifact`;
y padding never enters the encoded evaluation carrier. -/
structure Codec (Source Target : Type) where
  encode : Source -> Target

structure CarrierCodec (matrixCount : Nat) where
  commitment : Codec CommitmentRings PackedCommitment
  x : Codec XRings PackedPublicInput
  yRing : Codec (YRingRings matrixCount) (Array Evaluation)

def ProjectionOpening.commit {matrixCount : Nat}
    (codec : CarrierCodec matrixCount)
    (opening : ProjectionOpening matrixCount) : PackedCommitment :=
  codec.commitment.encode opening.commitment

def ProjectionOpening.publicInput {matrixCount : Nat}
    (codec : CarrierCodec matrixCount)
    (opening : ProjectionOpening matrixCount) : PackedPublicInput :=
  codec.x.encode opening.x

def ProjectionOpening.evaluations {matrixCount : Nat}
    (codec : CarrierCodec matrixCount)
    (opening : ProjectionOpening matrixCount) : Array Evaluation :=
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

def combineOpening {matrixCount n : Nat} (ring : RingAlgebra)
    (challenges : Fin n -> Ring)
    (openings : Fin n -> ProjectionOpening matrixCount) :
    ProjectionOpening matrixCount where
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
    {matrixCount : Nat}
    {Assignment : Type}
    {params : GlobalParams}
    {semantics : RelationSemantics Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment}
    (algebra : PiRLC.Algebra Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment Ring semantics params)
    (codec : CarrierCodec matrixCount) (ring : RingAlgebra) : Prop where
  commitment : forall {n : Nat} (challenges : Fin n -> Ring)
      (openings : Fin n -> ProjectionOpening matrixCount),
    codec.commitment.encode (combineOpening ring challenges openings).commitment =
      algebra.combineCommitment challenges
        (fun index => (openings index).commit codec)
  x : forall {n : Nat} (challenges : Fin n -> Ring)
      (openings : Fin n -> ProjectionOpening matrixCount),
    codec.x.encode (combineOpening ring challenges openings).x =
      algebra.combinePublicInput challenges
        (fun index => (openings index).publicInput codec)
  yRing : forall {n : Nat} (challenges : Fin n -> Ring)
      (openings : Fin n -> ProjectionOpening matrixCount),
    codec.yRing.encode (combineOpening ring challenges openings).yRing =
      algebra.combineEvaluations challenges
        (fun index => (openings index).evaluations codec)

/-! ## Attempt columns and decoding -/

structure PointColumns where
  r : List (Nat × Nat)

def decodePointColumns (assignment : Nat -> Nat)
    (columns : PointColumns) : Point :=
  extensionValues assignment columns.r

structure BatchColumns
    (params : GlobalParams) (arity : BatchArity params)
    (matrixCount : Nat) where
  /-- Exact strict-PiDEC parent claim named by this profile. Local and global
  column spaces use different values and must be connected explicitly. -/
  parentClaim : ClaimLayout
  challenges : Fin arity.total -> List Nat
  inputs : Fin arity.total -> ProjectionColumns matrixCount
  output : ProjectionColumns matrixCount
  inputPoints : Fin arity.total -> PointColumns
  outputPoint : PointColumns

def decodedInstance {matrixCount : Nat}
    (codec : CarrierCodec matrixCount) (assignment : Nat -> Nat)
    (point : Point) (stage : NormStage)
    (columns : ProjectionColumns matrixCount) :
    CE.Instance Unit PackedPublicInput Point Evaluation PackedCommitment where
  constraintSystem := ()
  commitment := (decodeOpening assignment columns).commit codec
  publicInput := (decodeOpening assignment columns).publicInput codec
  point := point
  evaluations := (decodeOpening assignment columns).evaluations codec
  stage := stage

def attempt {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec matrixCount) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount) :
    PiRLC.Attempt Unit PackedPublicInput Point Evaluation PackedCommitment
      Ring params arity where
  inputs index := decodedInstance codec assignment
    (decodePointColumns assignment (columns.inputPoints index))
    .fresh (columns.inputs index)
  challenges index := values assignment (columns.challenges index)
  output := decodedInstance codec assignment
    (decodePointColumns assignment columns.outputPoint) .combined columns.output

/-! ## Matrix-indexed public trace tree and artifacts -/

/-- The paper-public subtree. It has `23 + 2 * matrixCount` leaves; delayed-NC
traces remain outside this interface and cannot be consumed through it. -/
structure TraceTree
    {params : GlobalParams} (arity : BatchArity params)
    (matrixCount : Nat) where
  publicTrace : PublicRole matrixCount -> ProjectionTrace
  publicPairArity : forall role, (publicTrace role).pairs.length = arity.total

def TraceTree.flatten {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) : List ProjectionTrace :=
  (publicOrder matrixCount).map tree.publicTrace

def TraceTree.publicPairAt
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount)
    (role : PublicRole matrixCount) (index : Fin arity.total) :
    PairTrace :=
  (tree.publicTrace role).pairs.get
    (Fin.cast (tree.publicPairArity role).symm index)

/-- Static challenge-column wiring needed by the public PiRLC equations.
Challenge membership is deliberately absent: it is derived by transcript
replay and is needed only when equations are promoted to acceptance. -/
structure ChallengeWiringArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount) : Prop where
  publicShared : forall role index,
    (tree.publicPairAt role index).rhoColumns = columns.challenges index

/-- Exact public list layout expected by the shared carrier. -/
def assembleCommitmentColumns {matrixCount : Nat}
    (columns : ProjectionColumns matrixCount) : List Nat :=
  (List.ofFn columns.commitment).flatten

def assembleXColumns {matrixCount : Nat}
    (columns : ProjectionColumns matrixCount) : List Nat :=
  (List.ofFn fun coefficient : Fin 54 =>
    List.ofFn fun column : Fin 5 =>
      (columns.x column).getD coefficient.val 0).flatten

def decodeYRingActive {matrixCount : Nat} (assignment : Nat -> Nat)
    (columns : ProjectionColumns matrixCount) : Array Evaluation :=
  Array.ofFn fun row : Fin matrixCount =>
    fun coefficient : Fin 54 =>
      ⟨residue (assignment
          ((columns.yRing row ⟨0, by decide⟩).getD coefficient.val 0)),
       residue (assignment
          ((columns.yRing row ⟨1, by decide⟩).getD coefficient.val 0))⟩

/-- Deterministic codec/list-layout conformance, grouped with the typed
carrier wiring rather than treated as a fifth protocol assumption. -/
structure CodecArtifact {matrixCount : Nat}
    (codec : CarrierCodec matrixCount) : Prop where
  commitment : forall assignment (columns : ProjectionColumns matrixCount),
    codec.commitment.encode (decodeOpening assignment columns).commitment =
      (PackedCommitment.mk
        (values assignment (assembleCommitmentColumns columns)))
  x : forall assignment (columns : ProjectionColumns matrixCount),
    (forall block, (columns.x block).length = Concrete.ringDegree) ->
    codec.x.encode (decodeOpening assignment columns).x =
      PackedPublicInput.mk (values assignment (assembleXColumns columns))
  yRing : forall assignment (columns : ProjectionColumns matrixCount),
    (forall row limb,
      (columns.yRing row limb).length = Concrete.ringDegree) ->
    codec.yRing.encode (decodeOpening assignment columns).yRing =
      decodeYRingActive assignment columns

/-- The three column equalities used by the public PiRLC equations. This
contains no trace census, codec theorem, width fact, or parent binding. -/
structure EquationWiringArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount) : Prop where
  inputColumns : forall role index,
    (tree.publicPairAt role index).inputColumns = (columns.inputs index).at role
  outputColumns : forall role,
    (tree.publicTrace role).outputColumns = columns.output.at role
  pointColumns : forall index,
    columns.inputPoints index = columns.outputPoint

/-- Generated-trace census, codec layout, exact widths, and the
separately reusable equation wiring for the typed CE carrier leaves. -/
structure CarrierArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec matrixCount)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount)
    (productionTraces : List ProjectionTrace) : Prop where
  census : tree.flatten = productionTraces
  codecLayout : CodecArtifact codec
  inputWidth : forall role index,
    ((columns.inputs index).at role).length = Concrete.ringDegree
  outputWidth : forall role,
    (columns.output.at role).length = Concrete.ringDegree
  wiring : EquationWiringArtifact columns tree

/-- The Π_RLC combined output is the same strict-PiDEC parent, by
column identity rather than by equality of prover-supplied values. -/
structure ParentArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (columns : BatchColumns params arity matrixCount) : Prop where
  commitment : assembleCommitmentColumns columns.output =
    columns.parentClaim.commitment.dataCols
  x : assembleXColumns columns.output = columns.parentClaim.xActiveCols
  evaluationRows : columns.parentClaim.yRingCols.length = matrixCount
  yRing : forall row limb,
    columns.output.yRing row limb =
      (List.ofFn fun coefficient : Fin 54 =>
        (columns.parentClaim.yRingCols.getD row.val []).getD
          (2 * coefficient.val + limb.val) 0)
  r : columns.outputPoint.r = columns.parentClaim.rCols

/-- The exact coefficient identities have already been
interpreted as unconditional concrete Phi81 quotient-ring equations. It is
not a disguised paper `Accepted` premise. -/
structure ReductionArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (tree : TraceTree arity matrixCount) : Prop where
  equation : forall role,
    values assignment (tree.publicTrace role).outputColumns =
      phi81Combine
        (fun index =>
          values assignment (tree.publicPairAt role index).rhoColumns)
        (fun index =>
          values assignment (tree.publicPairAt role index).inputColumns)

/-- Minimal implementation refinement of the public PiRLC equations.
Challenge membership, trace census, codec layout, widths, and parent binding
are absent because none is an equation obligation. -/
structure EquationRefinement
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount) : Prop where
  challengeWiring : ChallengeWiringArtifact columns tree
  wiring : EquationWiringArtifact columns tree
  reduction : ReductionArtifact assignment tree

/-- Exact per-leaf reduction expressed directly through the independent
Phi81 combination. This is the equation-level core and has no algebra codec. -/
theorem phi81_reduction_at
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {assignment : Nat -> Nat}
    {columns : BatchColumns params arity matrixCount}
    {tree : TraceTree arity matrixCount}
    (challengeWiring : ChallengeWiringArtifact columns tree)
    (wiring : EquationWiringArtifact columns tree)
    (reduction : ReductionArtifact assignment tree)
    (role : PublicRole matrixCount) :
    (decodeOpening assignment columns.output).at role =
      phi81Combine
        (fun index => values assignment (columns.challenges index))
        (fun index =>
          (decodeOpening assignment (columns.inputs index)).at role) := by
  rw [decodeOpening_at, ← wiring.outputColumns role,
    reduction.equation role]
  have challenges :
      (fun index =>
        values assignment (tree.publicPairAt role index).rhoColumns) =
      (fun index => values assignment (columns.challenges index)) := by
    funext index
    rw [challengeWiring.publicShared role index]
  have inputs :
      (fun index =>
        values assignment (tree.publicPairAt role index).inputColumns) =
      (fun index =>
        (decodeOpening assignment (columns.inputs index)).at role) := by
    funext index
    rw [wiring.inputColumns role index, decodeOpening_at]
  rw [challenges, inputs]

theorem reduction_at
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {ring : RingAlgebra} {assignment : Nat -> Nat}
    {columns : BatchColumns params arity matrixCount}
    {tree : TraceTree arity matrixCount}
    (challengeWiring : ChallengeWiringArtifact columns tree)
    (wiring : EquationWiringArtifact columns tree)
    (reduction : ReductionArtifact assignment tree)
    (role : PublicRole matrixCount) :
    (decodeOpening assignment columns.output).at role =
      ring.combine
        (fun index => values assignment (columns.challenges index))
        (fun index =>
          (decodeOpening assignment (columns.inputs index)).at role) := by
  rw [phi81_reduction_at challengeWiring wiring reduction role,
    ← ring.phi81]

/-- Exact per-leaf reduction and static wiring construct the generic public
PiRLC equations. No challenge-membership premise is consumed here. -/
theorem equations_of_refinement
    {Assignment : Type}
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {semantics : RelationSemantics Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment}
    (algebra : PiRLC.Algebra Unit Assignment PackedPublicInput Point Evaluation
      PackedCommitment Ring semantics params)
    (codec : CarrierCodec matrixCount) (ring : RingAlgebra)
    (algebraRefinement : AlgebraRefinement algebra codec ring)
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (tree : TraceTree arity matrixCount)
    (refinement : EquationRefinement assignment columns tree) :
    PiRLC.Equations algebra (attempt codec assignment columns) := by
  refine {
    inputFresh := ?_
    sameStructure := ?_
    samePoint := ?_
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
      (refinement.wiring.pointColumns index)
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
        exact reduction_at refinement.challengeWiring refinement.wiring
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
        exact reduction_at refinement.challengeWiring refinement.wiring
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
        exact reduction_at refinement.challengeWiring refinement.wiring
          refinement.reduction (.yRing row limb)
      _ = _ := algebraRefinement.yRing _ _

private theorem getD_ofFn {Carrier : Type} {size : Nat}
    (items : Fin size → Carrier) (index : Fin size) (default : Carrier) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem decodeYRingActive_eq_claim
    {matrixCount : Nat}
    (assignment : Nat -> Nat) (columns : ProjectionColumns matrixCount)
    (claim : ClaimLayout)
    (rowCount : claim.yRingCols.length = matrixCount)
    (wiring : forall row limb,
      columns.yRing row limb =
        (List.ofFn fun coefficient : Fin 54 =>
          (claim.yRingCols.getD row.val []).getD
            (2 * coefficient.val + limb.val) 0)) :
    decodeYRingActive assignment columns =
      FPrimeFullHistoryNifsPaper.decodedEvaluations assignment claim := by
  apply Array.ext
  · simp [decodeYRingActive, FPrimeFullHistoryNifsPaper.decodedEvaluations,
      rowCount]
  · intro index left right
    have indexLt : index < matrixCount := by
      simpa [decodeYRingActive] using left
    let row : Fin matrixCount := ⟨index, indexLt⟩
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
        ((claim.yRingCols.getD row.val []).getD
          (2 * coefficient.val) 0)))
      (residue (assignment
        ((claim.yRingCols.getD row.val []).getD
          (2 * coefficient.val + 1) 0))) = _
    have rowLt : row.val < claim.yRingCols.length := by
      rw [rowCount]
      exact row.isLt
    simp only [FPrimeFullHistoryNifsPaper.decodedEvaluations,
      List.getElem_toArray, List.getElem_map, FPrimeFullHistoryNifsPaper.decodedEvaluation]
    rw [← List.getElem_eq_getD
      (l := claim.yRingCols) (i := row.val) []]

/-- Codec layout, exact output widths, and parent-column binding identify the
combined CE with its explicitly named strict-PiDEC parent claim. No public
equation, challenge, reduction, input wiring, or trace-census premise is used. -/
theorem output_eq_parent_of_artifacts
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec matrixCount) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (codecLayout : CodecArtifact codec)
    (outputWidth : forall role,
      (columns.output.at role).length = Concrete.ringDegree)
    (parent : ParentArtifact columns) :
    (attempt codec assignment columns).output =
      FPrimeFullHistoryNifsPaper.decodedInstance () assignment
        columns.parentClaim .combined := by
  have commitment := codecLayout.commitment assignment columns.output
  have input := codecLayout.x assignment columns.output
    (fun block => outputWidth (.x block))
  have evaluations := codecLayout.yRing assignment columns.output
    (fun row limb => outputWidth (.yRing row limb))
  rw [parent.commitment] at commitment
  rw [parent.x] at input
  rw [decodeYRingActive_eq_claim assignment columns.output
    columns.parentClaim parent.evaluationRows parent.yRing] at evaluations
  have point : decodePointColumns assignment columns.outputPoint =
      FPrimeFullHistoryNifsPaper.decodedPoint assignment
        columns.parentClaim := by
    exact congrArg (extensionValues assignment) parent.r
  simp only [attempt, decodedInstance,
    ProjectionOpening.commit, ProjectionOpening.publicInput,
    ProjectionOpening.evaluations,
    FPrimeFullHistoryNifsPaper.decodedInstance,
    FPrimeFullHistoryNifsPaper.decodedPackedCommitment,
    FPrimeFullHistoryNifsPaper.decodedPackedInput]
  rw [commitment, input, point, evaluations]

/-- Local-layout specialization. A production profile must prove this claim
identity or use the explicit relabeled theorem instead. -/
theorem output_eq_piDecParent_of_artifacts
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec matrixCount) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (codecLayout : CodecArtifact codec)
    (outputWidth : forall role,
      (columns.output.at role).length = Concrete.ringDegree)
    (parent : ParentArtifact columns)
    (parentClaim : columns.parentClaim = layout.parent) :
    (attempt codec assignment columns).output =
      FPrimeFullHistoryNifsPaper.decodedInstance () assignment
        layout.parent .combined := by
  simpa [parentClaim] using output_eq_parent_of_artifacts
    codec assignment columns codecLayout outputWidth parent

/-! ## Fixed arity wrappers -/

def recursiveArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.bootstrap Concrete.productionGlobalParams 1 (by decide) (by decide)

def terminalArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.active Concrete.productionGlobalParams 1 (by decide) (by decide)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
