import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Baseline

/-!
Removal witnesses for selected-NIFS `Pi_CCS` semantic obligations.

Assurance tier: model-level.

Owns: counterexamples for the public polynomial/source-product bindings and,
as they are closed, the three independent paper-relation leaves.

Does not own: executable SumCheck messages, Fiat--Shamir, transcript security,
physical rows, costs, Rust/R1CS refinement, or row removal.

Emits constraints: no.

Authority boundary: public verifier inputs must be projections of independent
source data. This file mutates raw verifier-visible fields rather than asking
an implementation predicate to define the intended relation.

| Family | Stage path | Counterexample | Lean owner |
|---|---|---|---|
| input authority | `fprime.active.nifs.pi_ccs.input.polynomial.necessity` | change only the prior point and recompute outputs | `polynomialInput_necessary` |
| input authority | `fprime.active.nifs.pi_ccs.input.product.necessity` | change only the fresh statement's stage and recompute outputs | `sourceProduct_necessary` |
| paper relation | `fprime.active.nifs.pi_ccs.relation.*.necessity` | owned by the child family module | `PiCcs.PaperRelation` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

/-! ## Polynomial-input removal witness -/

/-- A visibly different verifier prior point at the exact one-coordinate
model shape. -/
def differentPriorPoint :
    PiCCS.PaperJoint.CubePoint K Sources.shape.rowVariables where
  coordinates := [K.one]
  dimension := rfl

theorem differentPriorPoint_ne_source :
    differentPriorPoint ≠
      (PiCCS.SplitNc.Verifier.PublicInput.ofSources Sources.data).priorPoint := by
  change differentPriorPoint ≠ Sources.priorPoint
  intro equal
  have coordinatesEqual :=
    congrArg PiCCS.PaperJoint.CubePoint.coordinates equal
  have oneNeZero : K.one ≠ K.zero := by decide
  exact oneNeZero (by
    simpa [differentPriorPoint, Sources.priorPoint] using
      congrArg List.head? coordinatesEqual)

/-- Preserve the structure polynomial and carried claims but change the sole
public prior point. -/
def mismatchedPolynomialInput :
    PiCCS.SplitNc.Verifier.PublicInput Sources.shape :=
  { PiCCS.SplitNc.Verifier.PublicInput.ofSources Sources.data with
    priorPoint := differentPriorPoint }

theorem mismatchedPolynomialInput_ne_source :
    mismatchedPolynomialInput ≠
      PiCCS.SplitNc.Verifier.PublicInput.ofSources Sources.data := by
  intro equal
  exact differentPriorPoint_ne_source
    (congrArg PiCCS.SplitNc.Verifier.PublicInput.priorPoint equal)

/-- Preserve every context field except the public polynomial-verifier
input. -/
def mismatchedPolynomialContext :
    FixedActive.Context Sources.shape Unit Context.publicRingColumns
      Context.publicFits Context.verifierRows :=
  { Context.context with piCcsInput := mismatchedPolynomialInput }

def mismatchedPolynomialWitness :
    SemanticFold.Witness mismatchedPolynomialContext where
  point := baselineWitness.point
  challenges := baselineWitness.challenges

/-- Incoming-parent authority is independent of the polynomial public-input
field. This reconstructs the indexed proposition instead of identifying the
two context records. -/
theorem mismatchedPolynomial_runningAccepted :
    RunningAuthority.Accepted mismatchedPolynomialContext := by
  cases Context.runningAccepted with
  | bootstrap mode _ =>
      change RunningMode.active = RunningMode.bootstrap at mode
      cases mode
  | active bound =>
      apply RunningAuthority.Accepted.active
      exact {
        active := bound.active
        parent := bound.parent
        parentBound := by
          simpa [mismatchedPolynomialContext] using bound.parentBound
        piDec := by
          simpa [RunningAuthority.attempt, mismatchedPolynomialContext] using
            bound.piDec
      }

/-- Recompute both result surfaces under the context whose polynomial input
no longer projects the authoritative sources. -/
def mismatchedPolynomialCandidate : BaselineCandidate := {
  context := mismatchedPolynomialContext
  data := Sources.data
  point := mismatchedPolynomialWitness.point
  challenges := mismatchedPolynomialWitness.challenges
  parent := SemanticFold.parentOf mismatchedPolynomialContext Sources.data
    mismatchedPolynomialWitness
  children := SemanticFold.childrenOf mismatchedPolynomialContext Sources.data
    mismatchedPolynomialWitness
}

theorem mismatchedPolynomial_not_bound :
    ¬SemanticFold.PublicInputBound mismatchedPolynomialContext Sources.data := by
  intro bound
  exact mismatchedPolynomialInput_ne_source (by
    simpa [SemanticFold.PublicInputBound, mismatchedPolynomialContext] using
      bound)

/-- The prior-point mutation changes only the polynomial-input leaf after
canonical result recomputation. -/
theorem mismatchedPolynomial_semantics_iff
    (leaf : SemanticFold.ObligationPlan.Leaf)
    (retained : leaf ≠ .polynomialInput) :
    baselineSemantics leaf mismatchedPolynomialCandidate ↔
      baselineSemantics leaf baselineCandidate := by
  cases leaf with
  | freshCcs => rfl
  | allSourceNorm => rfl
  | carriedEvaluations => rfl
  | polynomialInput => exact (retained rfl).elim
  | sourceProduct => rfl
  | incomingAuthority =>
      constructor
      · intro _
        exact baselineAccepted .incomingAuthority
          (SemanticFold.ObligationPlan.mem_checks .incomingAuthority)
      · intro _
        exact mismatchedPolynomial_runningAccepted
  | challengeStrongSet => rfl
  | parentExact =>
      constructor
      · intro _
        exact baselineAccepted .parentExact
          (SemanticFold.ObligationPlan.mem_checks .parentExact)
      · intro _
        rfl
  | childrenExact =>
      constructor
      · intro _
        exact baselineAccepted .childrenExact
          (SemanticFold.ObligationPlan.mem_checks .childrenExact)
      · intro _
        rfl

theorem mismatchedPolynomialWeakened :
    CheckPlan.Accepts baselineSemantics
      (CheckPlan.without SemanticFold.ObligationPlan.checks .polynomialInput)
      mismatchedPolynomialCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (mismatchedPolynomial_semantics_iff leaf retained).mpr
    (baselineAccepted leaf (SemanticFold.ObligationPlan.mem_checks leaf))

theorem mismatchedPolynomialRejected :
    ¬baselineTarget mismatchedPolynomialCandidate := by
  intro realized
  exact mismatchedPolynomial_not_bound realized.input.publicInput

/-- Closed inclusion-necessity of binding the polynomial verifier input to
the independent source projection. -/
theorem polynomialInput_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .polynomialInput :=
  ⟨mismatchedPolynomialCandidate, mismatchedPolynomialWeakened,
    mismatchedPolynomialRejected⟩

/-! ## Source-product removal witness -/

/-- The baseline fresh statement with only its norm stage changed. -/
def wrongStageFreshStatement :=
  { Context.freshStatement with stage := NormStage.combined }

/-- Preserve the entire public source product except the sole fresh
statement's stage. -/
def mismatchedSourceProduct :
    PiCCS.SplitNc.Verifier.SourceProduct Sources.shape
      Context.publicRingColumns Context.publicFits
      (CommitmentValue Context.verifierRows) productionGlobalParams
      FixedActive.arity :=
  { Context.context.input with fresh := fun _ => wrongStageFreshStatement }

@[simp] theorem mismatchedSourceProduct_fresh
    (source : Fin FixedActive.arity.freshCount) :
    mismatchedSourceProduct.fresh source = wrongStageFreshStatement := by
  rfl

@[simp] theorem mismatchedSourceProduct_running
    (source : Fin (FixedActive.arity.mode.count productionGlobalParams)) :
    mismatchedSourceProduct.running source = Context.context.input.running source := by
  rfl

/-- Preserve every context field except the complete public source product. -/
def mismatchedSourceContext :
    FixedActive.Context Sources.shape Unit Context.publicRingColumns
      Context.publicFits Context.verifierRows :=
  { Context.context with input := mismatchedSourceProduct }

def mismatchedSourceWitness : SemanticFold.Witness mismatchedSourceContext where
  point := baselineWitness.point
  challenges := baselineWitness.challenges

/-- Incoming authority sees only the running partition, which this mutation
preserves exactly. -/
theorem mismatchedSource_runningAccepted :
    RunningAuthority.Accepted mismatchedSourceContext := by
  cases Context.runningAccepted with
  | bootstrap mode _ =>
      change RunningMode.active = RunningMode.bootstrap at mode
      cases mode
  | active bound =>
      apply RunningAuthority.Accepted.active
      exact {
        active := bound.active
        parent := bound.parent
        parentBound := by
          simpa [mismatchedSourceContext] using bound.parentBound
        piDec := by
          simpa [RunningAuthority.attempt, RunningAuthority.children,
            mismatchedSourceContext, mismatchedSourceProduct] using bound.piDec
      }

/-- Recompute the canonical result surfaces after changing only the public
fresh source's stage. -/
def mismatchedSourceCandidate : BaselineCandidate := {
  context := mismatchedSourceContext
  data := Sources.data
  point := mismatchedSourceWitness.point
  challenges := mismatchedSourceWitness.challenges
  parent := SemanticFold.parentOf mismatchedSourceContext Sources.data
    mismatchedSourceWitness
  children := SemanticFold.childrenOf mismatchedSourceContext Sources.data
    mismatchedSourceWitness
}

/-- The malformed fresh stage contradicts source-product authority. -/
theorem mismatchedSource_not_bound :
    ¬SemanticFold.InputBound mismatchedSourceContext Sources.data := by
  intro bound
  have stage := (bound.fresh ⟨0, by decide⟩).stage
  change NormStage.combined = NormStage.fresh at stage
  cases stage

/-- The fresh-stage mutation changes only the source-product leaf after
canonical result recomputation. -/
theorem mismatchedSource_semantics_iff
    (leaf : SemanticFold.ObligationPlan.Leaf)
    (retained : leaf ≠ .sourceProduct) :
    baselineSemantics leaf mismatchedSourceCandidate ↔
      baselineSemantics leaf baselineCandidate := by
  cases leaf with
  | freshCcs => rfl
  | allSourceNorm => rfl
  | carriedEvaluations => rfl
  | polynomialInput => rfl
  | sourceProduct => exact (retained rfl).elim
  | incomingAuthority =>
      constructor
      · intro _
        exact baselineAccepted .incomingAuthority
          (SemanticFold.ObligationPlan.mem_checks .incomingAuthority)
      · intro _
        exact mismatchedSource_runningAccepted
  | challengeStrongSet => rfl
  | parentExact =>
      constructor
      · intro _
        exact baselineAccepted .parentExact
          (SemanticFold.ObligationPlan.mem_checks .parentExact)
      · intro _
        rfl
  | childrenExact =>
      constructor
      · intro _
        exact baselineAccepted .childrenExact
          (SemanticFold.ObligationPlan.mem_checks .childrenExact)
      · intro _
        rfl

theorem mismatchedSourceWeakened :
    CheckPlan.Accepts baselineSemantics
      (CheckPlan.without SemanticFold.ObligationPlan.checks .sourceProduct)
      mismatchedSourceCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  exact (mismatchedSource_semantics_iff leaf retained).mpr
    (baselineAccepted leaf (SemanticFold.ObligationPlan.mem_checks leaf))

theorem mismatchedSourceRejected :
    ¬baselineTarget mismatchedSourceCandidate := by
  intro realized
  exact mismatchedSource_not_bound realized.input.sources

/-- Closed inclusion-necessity of binding the complete public source product
to the independent source family. -/
theorem sourceProduct_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .sourceProduct :=
  ⟨mismatchedSourceCandidate, mismatchedSourceWeakened,
    mismatchedSourceRejected⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
