import Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.TraceNormalForm

/-!
Contract: bind the Lean-owned public-PiRLC coefficient program to the selected
paper NIFS public batch.

Owns:
- a source carrier whose challenges, inputs, output, and shared CE point are
  the exact fields consumed by `PiRLC.attempt`;
- construction of `KPiRlcTrace.Columns` from that source carrier;
- the profile-neutral Phi81 normal form for the minimal canonical trace; and
- the implication from occurrence-bound coefficient exactness to the existing
  paper `PiRLC.Equations`.

Does not own the extra projection challenge's transcript derivation, private
PiCCS/PiDEC rows, the parent binding, or a probability bound for `BatchBadRoot`.

No equation, acceptance proposition, or reduction witness is a field of the
source carrier.
-/

set_option autoImplicit false
set_option maxRecDepth 8192

namespace Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.ProjectionCheck
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
open Nightstream.Implementation.R1CS.Canonical.KPiRlcTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.ProjectionPhi81.PolynomialNormalForm

/-! ## Selected public source carrier -/

/-- Public NIFS columns with one shared CE point by construction.

`PiRLC.BatchColumns` permits a separate point-column record for every input.
The paper equation requires them all to be the output point.  Storing only one
record here makes that invariant definitional instead of a caller premise. -/
structure SourceColumns
    (params : GlobalParams) (arity : BatchArity params)
    (matrixCount : Nat) where
  parentClaim : PiDecStrictCompiler.ClaimLayout
  challenges : Fin arity.total → List Nat
  inputs : Fin arity.total → ProjectionColumns matrixCount
  output : ProjectionColumns matrixCount
  sharedPoint : PointColumns

/-- Exact logical widths of the selected NIFS public coefficient carrier. -/
structure SourceColumns.Valid
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (source : SourceColumns params arity matrixCount) : Prop where
  challengeWidth :
    ∀ index, (source.challenges index).length = Concrete.ringDegree
  inputWidth :
    ∀ index role, ((source.inputs index).at role).length =
      Concrete.ringDegree
  outputWidth :
    ∀ role, (source.output.at role).length = Concrete.ringDegree

def SourceColumns.toBatchColumns
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (source : SourceColumns params arity matrixCount) :
    BatchColumns params arity matrixCount where
  parentClaim := source.parentClaim
  challenges := source.challenges
  inputs := source.inputs
  output := source.output
  inputPoints := fun _ => source.sharedPoint
  outputPoint := source.sharedPoint

/-- Add only the data owned by the quotient-identity implementation:
one verifier-selected projection point and one prover quotient per public
role.  Neither changes the NIFS public sources above. -/
structure ProjectionColumns
    (params : GlobalParams) (arity : BatchArity params)
    (matrixCount : Nat) where
  source : SourceColumns params arity matrixCount
  beta : KColumns
  quotients : PublicRole matrixCount → List Nat

structure ProjectionColumns.Valid
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : ProjectionColumns params arity matrixCount) : Prop where
  source : columns.source.Valid
  quotientWidth : ∀ role, (columns.quotients role).length = 53

def ProjectionColumns.toColumns
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : ProjectionColumns params arity matrixCount) :
    KPiRlcTrace.Columns arity.total matrixCount where
  beta := columns.beta
  challenges := columns.source.challenges
  inputs := columns.source.inputs
  output := columns.source.output
  quotients := columns.quotients

theorem ProjectionColumns.toColumns_valid
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid) :
    columns.toColumns.Valid := by
  refine ⟨arity.totalPositive, ?_, ?_, ?_, valid.quotientWidth⟩
  · intro index
    simpa only [ProjectionColumns.toColumns, Concrete.ringDegree] using
      valid.source.challengeWidth index
  · intro index role
    simpa only [ProjectionColumns.toColumns, Concrete.ringDegree] using
      valid.source.inputWidth index role
  · intro role
    simpa only [ProjectionColumns.toColumns, Concrete.ringDegree] using
      valid.source.outputWidth role

def ProjectionColumns.occurrence
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid) (base : Nat) :
    KTraceProgram.Occurrence :=
  KPiRlcTrace.occurrence base columns.toColumns
    (columns.toColumns_valid valid)

@[simp] theorem SourceColumns.toBatchColumns_inputPoint
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (source : SourceColumns params arity matrixCount)
    (index : Fin arity.total) :
    (source.toBatchColumns.inputPoints index) = source.sharedPoint :=
  rfl

@[simp] theorem SourceColumns.toBatchColumns_outputPoint
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (source : SourceColumns params arity matrixCount) :
    source.toBatchColumns.outputPoint = source.sharedPoint :=
  rfl

/-! ## Minimal-trace Phi81 normal form -/

private theorem remainderRing_sum
    (polynomials : List (List ProjectionProgram.K)) :
    remainderRing (Polynomial.sum polynomials) =
      polynomials.foldr
        (fun polynomial suffix =>
          Concrete.ringFAdd (remainderRing polynomial) suffix)
        Concrete.ringFZero := by
  induction polynomials with
  | nil => exact remainderRing_nil
  | cons polynomial polynomials inductionHypothesis =>
      simp only [Polynomial.sum, List.foldr_cons]
      rw [remainderRing_add, inductionHypothesis]

private theorem remainderRing_sum_products
    (assignment : Nat → Nat) (pairs : List PairColumns)
    (rhoWidth : ∀ pair ∈ pairs,
      pair.rho.length = Concrete.ringDegree)
    (inputWidth : ∀ pair ∈ pairs,
      pair.input.length = Concrete.ringDegree) :
    remainderRing
        (Polynomial.sum (pairs.map fun pair =>
          pair.productPolynomial assignment)) =
      fun coefficient =>
        ProjectionPhi81.scalarSum (pairs.map fun pair =>
          Concrete.ringFMul
            (ProjectionPhi81.ringOfList
              (ProjectionPhi81.values assignment pair.rho))
            (ProjectionPhi81.ringOfList
              (ProjectionPhi81.values assignment pair.input))
            coefficient) := by
  induction pairs with
  | nil =>
      rw [List.map_nil, Polynomial.sum, remainderRing_nil]
      rfl
  | cons pair pairs inductionHypothesis =>
      have headRhoWidth := rhoWidth pair (by simp)
      have headInputWidth := inputWidth pair (by simp)
      have headRhoValueWidth :
          (ProjectionPhi81.values assignment pair.rho).length = 54 := by
        simpa [ProjectionPhi81.values, Concrete.ringDegree] using headRhoWidth
      have headInputValueWidth :
          (ProjectionPhi81.values assignment pair.input).length = 54 := by
        simpa [ProjectionPhi81.values, Concrete.ringDegree] using
          headInputWidth
      have tailRhoWidth : ∀ candidate ∈ pairs,
          candidate.rho.length = Concrete.ringDegree := by
        intro candidate member
        exact rhoWidth candidate (by simp [member])
      have tailInputWidth : ∀ candidate ∈ pairs,
          candidate.input.length = Concrete.ringDegree := by
        intro candidate member
        exact inputWidth candidate (by simp [member])
      have headProduct :
          remainderRing (pair.productPolynomial assignment) =
            Concrete.ringFMul
              (ProjectionPhi81.ringOfList
                (ProjectionPhi81.values assignment pair.rho))
              (ProjectionPhi81.ringOfList
                (ProjectionPhi81.values assignment pair.input)) := by
        unfold PairColumns.productPolynomial
        rw [ProjectionPhi81.basePolynomial_eq_embedded,
          ProjectionPhi81.basePolynomial_eq_embedded]
        exact product_remainder_eq_ringFMul _ _
          headRhoValueWidth headInputValueWidth
      simp only [List.map_cons, Polynomial.sum, ProjectionPhi81.scalarSum]
      rw [remainderRing_add, headProduct,
        inductionHypothesis tailRhoWidth tailInputWidth]
      rfl

private theorem map_c0_embedded
    (coefficients : List ProjectionPhi81.Scalar) :
    List.map ProjectionProgram.K.c0
        (Nightstream.Implementation.R1CS.ProjectionPhi81.PolynomialNormalForm.embedded
          coefficients) =
      coefficients := by
  unfold Nightstream.Implementation.R1CS.ProjectionPhi81.PolynomialNormalForm.embedded
  simpa only [List.map_map, Function.comp_apply] using
    List.map_id coefficients

def pairAt {count : Nat} (trace : Trace)
    (pairArity : trace.pairs.length = count)
    (index : Fin count) : PairColumns :=
  trace.pairs.get (Fin.cast pairArity.symm index)

theorem ofFn_pairAt_eq_pairs {count : Nat} (trace : Trace)
    (pairArity : trace.pairs.length = count) :
    List.ofFn (pairAt trace pairArity) = trace.pairs := by
  subst count
  change List.ofFn (fun index : Fin trace.pairs.length =>
    trace.pairs.get index) = trace.pairs
  exact List.ofFn_getElem

/-- Coefficient exactness of the minimal canonical trace determines its
concrete Phi81 remainder.  No legacy ladder or evaluation trace is involved. -/
theorem exact_output_eq_phi81Combine
    {count : Nat} (assignment : Nat → Nat) (trace : Trace)
    (pairArity : trace.pairs.length = count)
    (rhoWidth : ∀ index,
      (pairAt trace pairArity index).rho.length = Concrete.ringDegree)
    (inputWidth : ∀ index,
      (pairAt trace pairArity index).input.length = Concrete.ringDegree)
    (outputWidth : trace.output.length = Concrete.ringDegree)
    (quotientWidth : trace.quotient.length = 53)
    (maxDegree : trace.maxDegree = 106)
    (exact : (trace.identity assignment).Exact) :
    ProjectionPhi81.values assignment trace.output =
      ProjectionPhi81.phi81Combine
        (fun index =>
          ProjectionPhi81.values assignment
            (pairAt trace pairArity index).rho)
        (fun index =>
          ProjectionPhi81.values assignment
            (pairAt trace pairArity index).input) := by
  have exact107 :
      Polynomial.sum (trace.pairs.map fun pair =>
          pair.productPolynomial assignment) =
        Polynomial.add
          (Polynomial.mul
            (basePolynomial assignment trace.quotient)
            Polynomial.phi81)
          (Polynomial.padRight 107
            (basePolynomial assignment trace.output)) := by
    simpa [Identity.Exact, Trace.identity, maxDegree] using exact
  have quotientPolynomialWidth :
      (basePolynomial assignment trace.quotient).length = 53 := by
    simpa [basePolynomial] using quotientWidth
  have outputPolynomialWidth :
      (basePolynomial assignment trace.output).length = 54 := by
    simpa [basePolynomial] using outputWidth
  have outputNormal := exact_output_eq_remainder
    (Polynomial.sum (trace.pairs.map fun pair =>
      pair.productPolynomial assignment))
    (basePolynomial assignment trace.quotient)
    (basePolynomial assignment trace.output)
    quotientPolynomialWidth outputPolynomialWidth exact107
  have outputRemainder :
      ProjectionPhi81.values assignment trace.output =
        List.ofFn (remainderRing
          (Polynomial.sum (trace.pairs.map fun pair =>
            pair.productPolynomial assignment))) := by
    rw [ProjectionPhi81.basePolynomial_eq_embedded] at outputNormal
    exact (map_c0_embedded _).symm.trans outputNormal
  have pairCensus := ofFn_pairAt_eq_pairs trace pairArity
  have allRhoWidth : ∀ pair ∈ trace.pairs,
      pair.rho.length = Concrete.ringDegree := by
    intro pair member
    rw [← pairCensus, List.mem_ofFn] at member
    rcases member with ⟨index, rfl⟩
    exact rhoWidth index
  have allInputWidth : ∀ pair ∈ trace.pairs,
      pair.input.length = Concrete.ringDegree := by
    intro pair member
    rw [← pairCensus, List.mem_ofFn] at member
    rcases member with ⟨index, rfl⟩
    exact inputWidth index
  have sumRemainder := remainderRing_sum_products assignment trace.pairs
    allRhoWidth allInputWidth
  calc
    ProjectionPhi81.values assignment trace.output =
        List.ofFn (remainderRing
          (Polynomial.sum (trace.pairs.map fun pair =>
            pair.productPolynomial assignment))) := outputRemainder
    _ = List.ofFn (fun coefficient =>
        ProjectionPhi81.scalarSum (trace.pairs.map fun pair =>
          Concrete.ringFMul
            (ProjectionPhi81.ringOfList
              (ProjectionPhi81.values assignment pair.rho))
            (ProjectionPhi81.ringOfList
              (ProjectionPhi81.values assignment pair.input))
            coefficient)) :=
      congrArg List.ofFn sumRemainder
    _ = List.ofFn (fun coefficient =>
        ProjectionPhi81.scalarSum (List.ofFn fun index =>
          Concrete.ringFMul
            (ProjectionPhi81.ringOfList
              (ProjectionPhi81.values assignment
                (pairAt trace pairArity index).rho))
            (ProjectionPhi81.ringOfList
              (ProjectionPhi81.values assignment
                (pairAt trace pairArity index).input))
            coefficient)) := by
      apply congrArg List.ofFn
      funext coefficient
      apply congrArg ProjectionPhi81.scalarSum
      let contribution := fun pair : PairColumns =>
            Concrete.ringFMul
              (ProjectionPhi81.ringOfList
                (ProjectionPhi81.values assignment pair.rho))
              (ProjectionPhi81.ringOfList
                (ProjectionPhi81.values assignment pair.input))
              coefficient
      calc
        trace.pairs.map contribution =
            (List.ofFn (pairAt trace pairArity)).map contribution :=
          congrArg (List.map contribution) pairCensus.symm
        _ = List.ofFn (fun index =>
            contribution (pairAt trace pairArity index)) := by
          simpa only [Function.comp_apply] using
            (List.map_ofFn
              (f := pairAt trace pairArity) (g := contribution))
    _ = ProjectionPhi81.phi81Combine
          (fun index =>
            ProjectionPhi81.values assignment
              (pairAt trace pairArity index).rho)
          (fun index =>
            ProjectionPhi81.values assignment
              (pairAt trace pairArity index).input) :=
      (ProjectionPhi81.phi81Combine_eq_scalarSum _ _).symm

/-! ## Occurrence exactness to selected NIFS equations -/

private theorem role_mem_publicOrder
    {matrixCount : Nat} (role : PublicRole matrixCount) :
    role ∈ publicOrder matrixCount := by
  cases role with
  | commitment lane =>
      unfold publicOrder
      exact List.mem_append_left _
        (List.mem_append_left _ (List.mem_ofFn.mpr ⟨lane, rfl⟩))
  | x column =>
      unfold publicOrder
      exact List.mem_append_left _
        (List.mem_append_right _ (List.mem_ofFn.mpr ⟨column, rfl⟩))
  | yRing row limb =>
      unfold publicOrder
      apply List.mem_append_right
      apply List.mem_flatten.mpr
      refine ⟨List.ofFn (fun candidate : Fin 2 =>
        PublicRole.yRing row candidate), List.mem_ofFn.mpr ⟨row, rfl⟩, ?_⟩
      exact List.mem_ofFn.mpr ⟨limb, rfl⟩

private theorem exact_trace
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (assignment : Nat → Nat)
    (exact :
      (columns.occurrence valid base).Exact assignment)
    (role : PublicRole matrixCount) :
    ((KPiRlcTrace.trace columns.toColumns role).identity assignment).Exact := by
  apply exact
  unfold KTraceProgram.Occurrence.identities
    ProjectionColumns.occurrence KProjectionTrace.BatchIdentity
    KPiRlcTrace.occurrence
    KPiRlcTrace.batchLayout
  exact List.mem_map.mpr
    ⟨KPiRlcTrace.trace columns.toColumns role,
      List.mem_map.mpr ⟨role, role_mem_publicOrder role, rfl⟩, rfl⟩

private theorem pairAt_trace
    {arity matrixCount : Nat}
    (columns : KPiRlcTrace.Columns arity matrixCount)
    (role : PublicRole matrixCount) (index : Fin arity) :
    pairAt (KPiRlcTrace.trace columns role)
        (KPiRlcTrace.trace_pairs_length columns role) index =
      KPiRlcTrace.pair columns role index := by
  unfold pairAt KPiRlcTrace.trace
  simp only [List.get_eq_getElem, List.getElem_ofFn]
  apply congrArg (fun selected =>
    KPiRlcTrace.pair columns role selected)
  apply Fin.ext
  rfl

/-- Exactness of the occurrence's role trace is the Phi81 combination of the
same challenges and public input columns stored in `SourceColumns`. -/
theorem exact_output_at
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (assignment : Nat → Nat)
    (exact : (columns.occurrence valid base).Exact assignment)
    (role : PublicRole matrixCount) :
    (decodeOpening assignment columns.source.output).at role =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.phi81Combine
        (fun index =>
          Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.values
            assignment (columns.source.challenges index))
        (fun index =>
          (decodeOpening assignment (columns.source.inputs index)).at role) := by
  have normal :=
    exact_output_eq_phi81Combine assignment
      (KPiRlcTrace.trace columns.toColumns role)
      (KPiRlcTrace.trace_pairs_length columns.toColumns role)
      (fun index => by
        rw [pairAt_trace]
        exact valid.source.challengeWidth index)
      (fun index => by
        rw [pairAt_trace]
        exact valid.source.inputWidth index role)
      (valid.source.outputWidth role)
      (valid.quotientWidth role)
      rfl
      (exact_trace columns valid base assignment exact role)
  simpa only [decodeOpening_at, KPiRlcTrace.trace_pairs_length,
    KPiRlcTrace.trace, KPiRlcTrace.pair, ProjectionColumns.toColumns,
    ProjectionPhi81.values,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.values,
    ProjectionPhi81.phi81Combine,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.phi81Combine,
    pairAt, List.get_eq_getElem, List.getElem_ofFn] using normal

theorem ring_reduction_at
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    (ring : RingAlgebra)
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (assignment : Nat → Nat)
    (exact : (columns.occurrence valid base).Exact assignment)
    (role : PublicRole matrixCount) :
    (decodeOpening assignment columns.source.output).at role =
      ring.combine
        (fun index =>
          Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.values
            assignment (columns.source.challenges index))
        (fun index =>
          (decodeOpening assignment (columns.source.inputs index)).at role) := by
  rw [exact_output_at columns valid base assignment exact role, ← ring.phi81]

/-- The source-bound exact branch constructs the existing paper PiRLC
equations.  The shared point equality is definitional and no legacy
`EquationWiringArtifact` or `ReductionArtifact` is supplied. -/
theorem equations_of_exact
    {Assignment : Type}
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    {semantics :
      RelationSemantics Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment}
    (algebra :
      PiRLC.Algebra Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment Ring semantics params)
    (codec : CarrierCodec matrixCount)
    (ring : RingAlgebra)
    (algebraRefinement : AlgebraRefinement algebra codec ring)
    (assignment : Nat → Nat)
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (exact : (columns.occurrence valid base).Exact assignment) :
    PiRLC.Equations algebra
      (attempt codec assignment columns.source.toBatchColumns) := by
  refine
    { inputFresh := ?_
      sameStructure := ?_
      samePoint := ?_
      outputCombined := ?_
      commitmentEquation := ?_
      publicInputEquation := ?_
      evaluationEquation := ?_ }
  · intro index
    rfl
  · intro index
    rfl
  · intro index
    rfl
  · rfl
  · change codec.commitment.encode
        (decodeOpening assignment columns.source.output).commitment =
      algebra.combineCommitment _ _
    calc
      _ = codec.commitment.encode
          (combineOpening ring
            (fun index =>
              Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.values
                assignment (columns.source.challenges index))
            (fun index =>
              decodeOpening assignment
                (columns.source.inputs index))).commitment := by
        apply congrArg codec.commitment.encode
        funext lane
        exact ring_reduction_at ring columns valid base assignment exact
          (.commitment lane)
      _ = _ := algebraRefinement.commitment _ _
  · change codec.x.encode
        (decodeOpening assignment columns.source.output).x =
      algebra.combinePublicInput _ _
    calc
      _ = codec.x.encode
          (combineOpening ring
            (fun index =>
              Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.values
                assignment (columns.source.challenges index))
            (fun index =>
              decodeOpening assignment
                (columns.source.inputs index))).x := by
        apply congrArg codec.x.encode
        funext column
        exact ring_reduction_at ring columns valid base assignment exact
          (.x column)
      _ = _ := algebraRefinement.x _ _
  · change codec.yRing.encode
        (decodeOpening assignment columns.source.output).yRing =
      algebra.combineEvaluations _ _
    calc
      _ = codec.yRing.encode
          (combineOpening ring
            (fun index =>
              Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.values
                assignment (columns.source.challenges index))
            (fun index =>
              decodeOpening assignment
                (columns.source.inputs index))).yRing := by
        apply congrArg codec.yRing.encode
        funext row limb
        exact ring_reduction_at ring columns valid base assignment exact
          (.yRing row limb)
      _ = _ := algebraRefinement.yRing _ _

/-- Numeric row satisfaction for the source-bound occurrence yields the paper
PiRLC equations or the exact occurrence-bound projection-root event. -/
theorem equations_or_badRoot_of_rows
    {Assignment : Type}
    {params : GlobalParams} {arity : BatchArity params}
    {matrixCount : Nat}
    {semantics :
      RelationSemantics Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment}
    (algebra :
      PiRLC.Algebra Unit Assignment PackedPublicInput
        FPrimeFullHistoryNifsPaper.Point
        FPrimeFullHistoryNifsPaper.Evaluation
        PackedCommitment Ring semantics params)
    (codec : CarrierCodec matrixCount)
    (ring : RingAlgebra)
    (algebraRefinement : AlgebraRefinement algebra codec ring)
    (assignment : Nat → Nat)
    (columns : ProjectionColumns params arity matrixCount)
    (valid : columns.Valid)
    (base : Nat)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Nightstream.Implementation.R1CS.Satisfies
        (columns.occurrence valid base).rows assignment) :
    PiRLC.Equations algebra
        (attempt codec assignment columns.source.toBatchColumns) ∨
      (columns.occurrence valid base).BadRoot assignment := by
  rcases (columns.occurrence valid base).exact_or_badRoot
      assignment constantWire satisfied with exact | badRoot
  · exact Or.inl
      (equations_of_exact algebra codec ring algebraRefinement assignment
        columns valid base exact)
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.Canonical.KPiRlcSemanticBinding
