import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.Reduction.PolynomialNormalForm
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RingTransport

/-!
Exact production projection traces refine the independent Phi81 ring action.

Assurance tier: model-level. Profile-specific generated-row facts are owned by
`Reduction.Profiles`.

Owns: the structural bridge from assignment-backed projection polynomials to
typed coefficient lists; additive interpretation of a complete product sum;
the exact trace-pair census; and construction of `ReductionArtifact` from
per-role coefficient exactness.

Does not own: sampled-point soundness, transcript probability, generated-row
satisfaction, profile-specific trace layouts, sampler acceptance, parent
binding, Rust conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: quotient and output columns are interpreted only through
the exact polynomial identity. The result is the independently defined
`phi81Combine`; no native-verifier result or carried digest is a premise.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | assignment-backed base polynomials are canonical embedded coefficient lists | computed | `basePolynomial_eq_embedded` |
| `nifs.pi_rlc.verify.identities.public` | remainder extraction distributes over the complete product sum | derived | `remainderRing_sum_products` |
| `nifs.pi_rlc.verify.identities.public` | the typed finite pair function enumerates the trace pair list exactly once | direct dataflow | `ofFn_pairAt_eq_pairs` |
| `nifs.pi_rlc.verify.identities.public` | quotient width 53 and asserted degree 106 are explicit reduction premises | checked | `TraceShapeArtifact` |
| `nifs.pi_rlc.verify.identities.public` | exact trace identity plus only the named shape facts implies the concrete Phi81 combination | derived | `exact_output_eq_phi81Combine` |
| `nifs.pi_rlc.verify.identities.public` | all exact public leaves construct `ReductionArtifact` | derived | `reductionArtifact_of_exact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport

set_option maxRecDepth 16384
set_option maxHeartbeats 1000000

/-! ## Assignment-backed coefficient decoding -/

/-- The projection interpreter and paper-facing carrier use the same
Goldilocks residue and the same coefficient order. -/
theorem basePolynomial_eq_embedded
    (assignment : Nat → Nat) (columns : List Nat) :
    basePolynomial assignment columns =
      embedded (values assignment columns) := by
  induction columns with
  | nil => rfl
  | cons column columns inductionHypothesis =>
      change
        { c0 := baseAt assignment column, c1 := 0 } ::
            basePolynomial assignment columns =
          { c0 := residue (assignment column), c1 := 0 } ::
            embedded (values assignment columns)
      rw [inductionHypothesis]
      rfl

/-! ## Additive product-sum interpretation -/

private theorem remainderRing_sum (polynomials : List (List ProjectionProgram.K)) :
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

/-- The complete list of schoolbook trace products has exactly the
coefficientwise sum of the independent concrete ring products as remainder. -/
theorem remainderRing_sum_products
    (assignment : Nat → Nat) (pairs : List PairTrace)
    (rhoWidth : ∀ pair ∈ pairs,
      pair.rhoColumns.length = Concrete.ringDegree)
    (inputWidth : ∀ pair ∈ pairs,
      pair.inputColumns.length = Concrete.ringDegree) :
    remainderRing
        (Polynomial.sum (pairs.map fun pair =>
          pair.productPolynomial assignment)) =
      fun coefficient => scalarSum (pairs.map fun pair =>
        Concrete.ringFMul
          (ringOfList (values assignment pair.rhoColumns))
          (ringOfList (values assignment pair.inputColumns)) coefficient) := by
  induction pairs with
  | nil =>
      rw [List.map_nil, Polynomial.sum, remainderRing_nil]
      rfl
  | cons pair pairs inductionHypothesis =>
      have headRhoWidth := rhoWidth pair (by simp)
      have headInputWidth := inputWidth pair (by simp)
      have headRhoValueWidth :
          (values assignment pair.rhoColumns).length = 54 := by
        simpa [values, Concrete.ringDegree] using headRhoWidth
      have headInputValueWidth :
          (values assignment pair.inputColumns).length = 54 := by
        simpa [values, Concrete.ringDegree] using headInputWidth
      have tailRhoWidth : ∀ candidate ∈ pairs,
          candidate.rhoColumns.length = Concrete.ringDegree := by
        intro candidate member
        exact rhoWidth candidate (by simp [member])
      have tailInputWidth : ∀ candidate ∈ pairs,
          candidate.inputColumns.length = Concrete.ringDegree := by
        intro candidate member
        exact inputWidth candidate (by simp [member])
      have headProduct :
          remainderRing (pair.productPolynomial assignment) =
            Concrete.ringFMul
              (ringOfList (values assignment pair.rhoColumns))
              (ringOfList (values assignment pair.inputColumns)) := by
        unfold PairTrace.productPolynomial
        rw [basePolynomial_eq_embedded, basePolynomial_eq_embedded]
        exact product_remainder_eq_ringFMul _ _
          headRhoValueWidth headInputValueWidth
      simp only [List.map_cons, Polynomial.sum, scalarSum]
      rw [remainderRing_add, headProduct,
        inductionHypothesis tailRhoWidth tailInputWidth]
      rfl

/-! ## Pair census and trace-level reduction -/

private def pairAt {count : Nat} (trace : ProjectionTrace)
    (pairArity : trace.pairs.length = count) (index : Fin count) : PairTrace :=
  trace.pairs.get (Fin.cast pairArity.symm index)

/-- A length-indexed pair function enumerates the original trace list in the
same order and without duplication. -/
theorem ofFn_pairAt_eq_pairs {count : Nat} (trace : ProjectionTrace)
    (pairArity : trace.pairs.length = count) :
    List.ofFn (pairAt trace pairArity) = trace.pairs := by
  subst count
  change List.ofFn (fun index : Fin trace.pairs.length =>
    trace.pairs.get index) = trace.pairs
  exact List.ofFn_getElem

private theorem map_c0_embedded (coefficients : List Scalar) :
    List.map ProjectionProgram.K.c0 (embedded coefficients) = coefficients := by
  unfold embedded
  simpa only [List.map_map, Function.comp_apply] using List.map_id coefficients

/-- One exact, well-shaped production trace determines the concrete Phi81
combination of every typed challenge/input pair. -/
theorem exact_output_eq_phi81Combine
    {count : Nat} (assignment : Nat → Nat) (trace : ProjectionTrace)
    (pairArity : trace.pairs.length = count)
    (rhoWidth : ∀ index,
      (pairAt trace pairArity index).rhoColumns.length =
        Concrete.ringDegree)
    (inputWidth : ∀ index,
      (pairAt trace pairArity index).inputColumns.length =
        Concrete.ringDegree)
    (outputWidth : trace.outputColumns.length = Concrete.ringDegree)
    (quotientWidth : trace.quotientColumns.length = 53)
    (maxDegree : trace.maxDegree = 106)
    (exact : (trace.identity assignment).Exact) :
    values assignment trace.outputColumns =
      phi81Combine
        (fun index =>
          values assignment (pairAt trace pairArity index).rhoColumns)
        (fun index =>
          values assignment (pairAt trace pairArity index).inputColumns) := by
  have exact107 :
      Polynomial.sum (trace.pairs.map fun pair =>
          pair.productPolynomial assignment) =
        Polynomial.add
          (Polynomial.mul
            (basePolynomial assignment trace.quotientColumns)
            Polynomial.phi81)
          (Polynomial.padRight 107
            (basePolynomial assignment trace.outputColumns)) := by
    simpa [Nightstream.SuperNeo.ProjectionCheck.Identity.Exact,
      ProjectionTrace.identity, maxDegree] using exact
  have quotientPolynomialWidth :
      (basePolynomial assignment trace.quotientColumns).length = 53 := by
    simpa [basePolynomial] using quotientWidth
  have outputPolynomialWidth :
      (basePolynomial assignment trace.outputColumns).length = 54 := by
    simpa [basePolynomial] using outputWidth
  have outputNormal := exact_output_eq_remainder
    (Polynomial.sum (trace.pairs.map fun pair =>
      pair.productPolynomial assignment))
    (basePolynomial assignment trace.quotientColumns)
    (basePolynomial assignment trace.outputColumns)
    quotientPolynomialWidth outputPolynomialWidth exact107
  have outputRemainder :
      values assignment trace.outputColumns =
        List.ofFn (remainderRing
          (Polynomial.sum (trace.pairs.map fun pair =>
            pair.productPolynomial assignment))) := by
    rw [basePolynomial_eq_embedded] at outputNormal
    exact map_c0_embedded _ |>.symm.trans outputNormal
  have pairCensus := ofFn_pairAt_eq_pairs trace pairArity
  have allRhoWidth : ∀ pair ∈ trace.pairs,
      pair.rhoColumns.length = Concrete.ringDegree := by
    intro pair member
    rw [← pairCensus, List.mem_ofFn] at member
    rcases member with ⟨index, rfl⟩
    exact rhoWidth index
  have allInputWidth : ∀ pair ∈ trace.pairs,
      pair.inputColumns.length = Concrete.ringDegree := by
    intro pair member
    rw [← pairCensus, List.mem_ofFn] at member
    rcases member with ⟨index, rfl⟩
    exact inputWidth index
  have sumRemainder := remainderRing_sum_products assignment trace.pairs
    allRhoWidth allInputWidth
  calc
    values assignment trace.outputColumns =
        List.ofFn (remainderRing
          (Polynomial.sum (trace.pairs.map fun pair =>
            pair.productPolynomial assignment))) := outputRemainder
    _ = List.ofFn (fun coefficient => scalarSum (trace.pairs.map fun pair =>
          Concrete.ringFMul
            (ringOfList (values assignment pair.rhoColumns))
            (ringOfList (values assignment pair.inputColumns)) coefficient)) :=
      congrArg List.ofFn sumRemainder
    _ = List.ofFn (fun coefficient => scalarSum (List.ofFn fun index =>
          Concrete.ringFMul
            (ringOfList (values assignment
              (pairAt trace pairArity index).rhoColumns))
            (ringOfList (values assignment
              (pairAt trace pairArity index).inputColumns)) coefficient)) := by
      rw [← pairCensus]
      apply congrArg List.ofFn
      funext coefficient
      apply congrArg scalarSum
      simpa only [Function.comp_apply] using
        (List.map_ofFn
          (f := pairAt trace pairArity)
          (g := fun pair =>
            Concrete.ringFMul
              (ringOfList (values assignment pair.rhoColumns))
              (ringOfList (values assignment pair.inputColumns)) coefficient))
    _ = phi81Combine
          (fun index =>
            values assignment (pairAt trace pairArity index).rhoColumns)
          (fun index =>
            values assignment (pairAt trace pairArity index).inputColumns) :=
      (phi81Combine_eq_scalarSum _ _).symm

/-! ## Public-tree artifact construction -/

/-- Exactly the static shape facts consumed by quotient-ring semantics. This
does not bundle sampler membership, carrier acceptance, transcript state,
evaluation gadgets, or ladder layout into the reduction theorem. -/
structure TraceShapeArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) : Prop where
  challengeWidth : ∀ role index,
    (tree.publicPairAt role index).rhoColumns.length =
      Concrete.ringDegree
  inputWidth : ∀ role index,
    (tree.publicPairAt role index).inputColumns.length =
      Concrete.ringDegree
  outputWidth : ∀ role,
    (tree.publicTrace role).outputColumns.length =
      Concrete.ringDegree
  quotientWidth : ∀ role,
    (tree.publicTrace role).quotientColumns.length = 53
  maxDegree : ∀ role, (tree.publicTrace role).maxDegree = 106

theorem quotientRemainder_of_exact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {assignment : Nat → Nat} {tree : TraceTree arity matrixCount}
    (shape : TraceShapeArtifact tree)
    (role : PublicRole matrixCount)
    (exact : ((tree.publicTrace role).identity assignment).Exact) :
    values assignment (tree.publicTrace role).outputColumns =
      phi81Combine
        (fun index =>
          values assignment (tree.publicPairAt role index).rhoColumns)
        (fun index =>
          values assignment (tree.publicPairAt role index).inputColumns) := by
  exact exact_output_eq_phi81Combine assignment (tree.publicTrace role)
    (tree.publicPairArity role)
    (shape.challengeWidth role) (shape.inputWidth role)
    (shape.outputWidth role)
    (shape.quotientWidth role) (shape.maxDegree role) exact

/-- Per-role exactness plus the already-separated shape artifacts constructs
the complete deterministic reduction artifact. -/
theorem reductionArtifact_of_exact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    {assignment : Nat → Nat} {tree : TraceTree arity matrixCount}
    (shape : TraceShapeArtifact tree)
    (exact : ∀ role, ((tree.publicTrace role).identity assignment).Exact) :
    ReductionArtifact assignment tree where
  equation role := quotientRemainder_of_exact shape role (exact role)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction
