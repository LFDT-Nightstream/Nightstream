import NightstreamFPrime.Gadgets.SumCheck.FixedChain
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.Accepted

/-!
Paper authority: SuperNeo v1.2, Section 7.3, Step 2, `SumCheck(T; Q)`.
Obligation: Enforce all 28 equations
`p_i(0) + p_i(1) = claim_i`, then `claim_(i+1) = p_i(r_i)`, and
export the final `claim_28` for the separate `Q(r')` check.

Inputs:
- the initial claim `T`;
- 28 prover round polynomials of the fixed production degree;
- 28 challenges that are shared with the transcript leaf;

Outputs:
- the final claimed value `v`;
- the exact fixed claimed-chain predicate used by production `piCcsCheck`.

Constraint groups:
- C1: the opaque reusable `FixedChain` circuit, which stores every
  `p_i(r_i)`.

Parent coverage:
- `PiCCS.v1_2.Coverage.chain`.

This file owns only the fixed production round count and key wiring. It does
not absorb messages, derive challenges, or compute `T` or `Q(r')`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

structure Interface (degree : Nat) where
  initial : Nat → KExpr
  round : Nat → Fin productionShape.cubeVariables →
    FixedChain.Round degree

def coreInterface {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    FixedChain.Owned.Interface degree productionShape.cubeVariables where
  initial := interface.initial offset
  round := interface.round offset

/-- Number of stored base-field values of the fixed production chain. -/
def privateCount (degree : Nat) : Nat :=
  FixedChain.Owned.privateCount degree productionShape.cubeVariables

/-- The stored final `p_i(r_i)` value exported to the terminal-identity
leaf. -/
def output {degree : Nat} (interface : Interface degree)
    (offset : Nat) : KExpr :=
  FixedChain.Owned.output (coreInterface interface offset) offset

/-- The exact dimension-checked challenge vector supplied by the transcript
leaf through the shared round interface. -/
def evalRoundPoint {degree : Nat} (interface : Interface degree)
    (offset : Nat) (env : Env) : CubePoint K productionShape.cubeVariables where
  coordinates := (coreInterface interface offset).rounds.map
    fun round => round.challenge.eval env
  dimension := by
    simp [coreInterface, FixedChain.Owned.Interface.rounds]

abbrev Assumptions {degree : Nat} (interface : Interface degree)
    (offset : Nat) (env : Env) : Prop :=
  FixedChain.Owned.Assumptions (coreInterface interface offset) offset env

abbrev SpecHolds {degree : Nat} (interface : Interface degree)
    (offset : Nat) (env : Env) : Prop :=
  FixedChain.Owned.SpecHolds (coreInterface interface offset) offset env

/-- The sole logical circuit for the fixed production chain. -/
def circuit {degree : Nat} (interface : Interface degree) : FormalCircuit where
  main := fun offset =>
    (FixedChain.Owned.circuit (coreInterface interface offset)).main offset
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := by
    intro env offset assumptions rows
    exact (FixedChain.Owned.circuit
      (coreInterface interface offset)).soundness env offset assumptions rows
  completeness := by
    intro env offset assumptions specification
    exact (FixedChain.Owned.circuit
      (coreInterface interface offset)).completeness env offset assumptions
        specification

theorem soundness {degree : Nat} (interface : Interface degree)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness {degree : Nat} (interface : Interface degree)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

/-- The leaf emits exactly the reusable chain's operations. -/
theorem circuit_ops {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    Circuit.ops (circuit interface).main offset =
      FixedChain.Owned.opsAt (coreInterface interface offset) offset :=
  rfl

/-- The specification is stable when the input wires and the stored chain
interval are unchanged. -/
theorem specHolds_of_agree_below {degree : Nat}
    (interface : Interface degree) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset + privateCount degree →
      after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after :=
  (FixedChain.Owned.specHolds_eq_of_agree_below
    (coreInterface interface offset) offset before after assumptions
      (fun index below => (agrees index below).symm)).mp
      specification

theorem localLength_eq {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      privateCount degree :=
  FixedChain.Owned.localLength_eq (coreInterface interface offset) offset

theorem operations_length {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 57 := by
  change (Circuit.ops
    (FixedChain.Owned.circuit
      (coreInterface interface offset)).main offset).length = 57
  simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using!
    FixedChain.Owned.operations_length (coreInterface interface offset) offset

theorem flatConstraints_length {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      privateCount degree + 56 := by
  change (flatConstraints (Circuit.ops
    (FixedChain.Owned.circuit
      (coreInterface interface offset)).main offset)).length =
        privateCount degree + 56
  simpa [privateCount, productionShape, Phi81MatrixSource.phi81Shape,
    cubeVariables] using!
    FixedChain.Owned.flatConstraints_length
      (coreInterface interface offset) offset

/-- Every chain row reads only input wires or the stored chain interval. -/
theorem flatConstraints_varsBelow {degree : Nat}
    (interface : Interface degree) (offset : Nat)
    (assumptions : Assumptions interface offset (fun _ => 0)) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + privateCount degree) :=
  FixedChain.Owned.flatConstraints_varsBelow
    (coreInterface interface offset) offset assumptions

/-- The stored final claim lies inside the chain's interval. -/
theorem output_varsBelow {degree : Nat} (interface : Interface degree)
    (offset : Nat) (assumptions : Assumptions interface offset (fun _ => 0)) :
    (output interface offset).VarsBelow (offset + privateCount degree) :=
  FixedChain.Owned.output_varsBelow (coreInterface interface offset) offset
    assumptions

/-- Concrete parent coverage: the shared initial, round, challenge, and
terminal wires form exactly the claimed chain checked by production PiCCS. -/
theorem spec_implies_keyChain
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (initialEq : (interface.initial offset).eval env =
      (ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env =
        proof.piCcsRounds roundIndex)
    (roundPointEq : evalRoundPoint interface offset env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint)
    (terminalEq : (output interface offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)
    (specification : SpecHolds interface offset env) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
      ((ProductionKey.key relation ajtai).piCcsFixedCertificate
        running fresh proof).rounds
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output) := by
  have roundsListEq :
      (coreInterface interface offset).rounds.map
          (FixedChain.Round.semanticPolynomial env) =
        ((ProductionKey.key relation ajtai).piCcsFixedCertificate
          running fresh proof).rounds := by
    change (List.ofFn (interface.round offset)).map
        (FixedChain.Round.semanticPolynomial env) =
      List.ofFn proof.piCcsRounds
    rw [List.map_ofFn]
    apply congrArg List.ofFn
    funext roundIndex
    exact roundsEq roundIndex
  have challengeListEq :
      (coreInterface interface offset).rounds.map
          (fun round => round.challenge.eval env) =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint.coordinates := by
    simpa [evalRoundPoint] using
      congrArg (fun point => point.coordinates) roundPointEq
  have initialCoreEq :
      (coreInterface interface offset).initial.eval env =
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input.initial extensionOps
            ((ProductionKey.key relation ajtai).piCcsExecution
              running fresh proof).coins.gamma := by
    simpa [coreInterface] using initialEq
  have terminalCoreEq :
      (FixedChain.Owned.output
        (coreInterface interface offset) offset).eval env =
        ProtocolPolynomial.terminalFromMessage extensionOps
          (ChallengeDerivation.productionContext
            relation ajtai running fresh).input
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.alpha
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.roundPoint
          ((ProductionKey.key relation ajtai).piCcsCertificate
            running fresh proof).output := by
    simpa [output] using terminalEq
  unfold SpecHolds FixedChain.Owned.SpecHolds at specification
  rw [initialCoreEq, roundsListEq, challengeListEq, terminalCoreEq] at specification
  exact specification

/-- The canonical verifier chain, restated over this leaf's input wires. -/
private theorem keyChain_core
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (initialEq : (interface.initial offset).eval env =
      (ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env =
        proof.piCcsRounds roundIndex)
    (roundPointEq : evalRoundPoint interface offset env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint)
    (chain : SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
      ((ProductionKey.key relation ajtai).piCcsFixedCertificate
        running fresh proof).rounds
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((coreInterface interface offset).initial.eval env)
      ((coreInterface interface offset).rounds.map
        (FixedChain.Round.semanticPolynomial env))
      ((coreInterface interface offset).rounds.map
        fun round => round.challenge.eval env)
      (ProtocolPolynomial.terminalFromMessage extensionOps
          (ChallengeDerivation.productionContext
            relation ajtai running fresh).input
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.alpha
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.roundPoint
          ((ProductionKey.key relation ajtai).piCcsCertificate
            running fresh proof).output) := by
  have roundsListEq :
      (coreInterface interface offset).rounds.map
          (FixedChain.Round.semanticPolynomial env) =
        ((ProductionKey.key relation ajtai).piCcsFixedCertificate
          running fresh proof).rounds := by
    change (List.ofFn (interface.round offset)).map
        (FixedChain.Round.semanticPolynomial env) =
      List.ofFn proof.piCcsRounds
    rw [List.map_ofFn]
    apply congrArg List.ofFn
    funext roundIndex
    exact roundsEq roundIndex
  have challengeListEq :
      (coreInterface interface offset).rounds.map
          (fun round => round.challenge.eval env) =
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint.coordinates := by
    simpa [evalRoundPoint] using
      congrArg (fun point => point.coordinates) roundPointEq
  have initialCoreEq :
      (coreInterface interface offset).initial.eval env =
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input.initial extensionOps
            ((ProductionKey.key relation ajtai).piCcsExecution
              running fresh proof).coins.gamma := by
    simpa [coreInterface] using initialEq
  rw [initialCoreEq, roundsListEq, challengeListEq]
  exact chain

/-- Exact completeness direction: from the canonical verifier chain over the
input wires, honest execution stores every `p_i(r_i)`, satisfies the owned
round rows, and exports the terminal used by the final identity. -/
theorem keyChain_build
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (initialEq : (interface.initial offset).eval env =
      (ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env =
        proof.piCcsRounds roundIndex)
    (roundPointEq : evalRoundPoint interface offset env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint)
    (chain : SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
      ((ProductionKey.key relation ajtai).piCcsFixedCertificate
        running fresh proof).rounds
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output)) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) ∧
      (output interface offset).eval completed =
        ProtocolPolynomial.terminalFromMessage extensionOps
          (ChallengeDerivation.productionContext
            relation ajtai running fresh).input
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.alpha
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.roundPoint
          ((ProductionKey.key relation ajtai).piCcsCertificate
            running fresh proof).output := by
  have built := FixedChain.Owned.build (coreInterface interface offset) env
    offset assumptions _
    (keyChain_core relation ajtai running fresh proof interface offset env
      initialEq roundsEq roundPointEq chain)
  rw [FixedChain.Owned.main_ops] at built
  rw [circuit_ops]
  exact built

/-- With the chain rows satisfied, the stored final claim is the terminal of
the canonical verifier chain over the same input wires. -/
theorem output_eval_of_keyChain
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (initialEq : (interface.initial offset).eval env =
      (ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env =
        proof.piCcsRounds roundIndex)
    (roundPointEq : evalRoundPoint interface offset env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint)
    (chain : SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma)
      ((ProductionKey.key relation ajtai).piCcsFixedCertificate
        running fresh proof).rounds
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output))
    (rows : holdsFlat env (Circuit.ops (circuit interface).main offset)) :
    (output interface offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
          (ChallengeDerivation.productionContext
            relation ajtai running fresh).input
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.alpha
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.roundPoint
          ((ProductionKey.key relation ajtai).piCcsCertificate
            running fresh proof).output := by
  rw [circuit_ops] at rows
  unfold holdsFlat at rows
  rw [FixedChain.Owned.flatConstraints_opsAt] at rows
  exact (FixedChain.Owned.rows_of_chain env offset
    (coreInterface interface offset).initial
    (coreInterface interface offset).rounds
    ((Circuit.constraintsHold_append env _ _).mp rows).1 _
    (keyChain_core relation ajtai running fresh proof interface offset env
      initialEq roundsEq roundPointEq chain)).2

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain
