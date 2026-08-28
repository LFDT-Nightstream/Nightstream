import NightstreamFPrime.Gadgets.SumCheck.FixedChain
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.Accepted

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 2, `SumCheck(T; Q)`.
Obligation: Enforce all 26 equations
`p_i(0) + p_i(1) = claim_i`, then `claim_(i+1) = p_i(r_i)`, and
export the final `claim_26` for the separate `Q(r')` check.

Inputs:
- the initial claim `T`;
- 26 prover round polynomials of the fixed production degree;
- 26 challenges that are shared with the transcript leaf;

Outputs:
- the final claimed value `v`;
- the exact fixed claimed-chain predicate used by production `piCcsCheck`.

Constraint groups:
- C1: the opaque reusable `FixedChain` circuit.

Parent coverage:
- `PiCCS.v1_1.Coverage.chain`.

This file owns only the fixed production round count and key wiring. It does
not absorb messages, derive challenges, or compute `T` or `Q(r')`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain

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

/-- The final `p_i(r_i)` value exported to the terminal-identity leaf. -/
def output {degree : Nat} (interface : Interface degree)
    (offset : Nat) : KExpr :=
  FixedChain.Owned.output (coreInterface interface offset)

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
  FixedChain.Owned.SpecHolds (coreInterface interface offset) env

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

theorem specHolds_of_agree_below {degree : Nat}
    (interface : Interface degree) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after :=
  (FixedChain.Owned.specHolds_eq_of_agree_below
    (coreInterface interface offset) offset before after assumptions
      (fun index below => (agrees index below).symm)).mp
      specification

theorem localLength_eq {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 := by
  change localLength
    (Circuit.ops (FixedChain.Owned.main
      (coreInterface interface offset)) offset) = 0
  exact FixedChain.Owned.localLength_eq
    (coreInterface interface offset) offset

theorem operations_length {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 52 := by
  change (Circuit.ops
    (FixedChain.Owned.main
      (coreInterface interface offset)) offset).length = 52
  simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
    FixedChain.Owned.operations_length (coreInterface interface offset) offset

theorem flatConstraints_length {degree : Nat} (interface : Interface degree)
    (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      52 := by
  change (flatConstraints (Circuit.ops
    (FixedChain.Owned.main
      (coreInterface interface offset)) offset)).length = 52
  simpa [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables] using
    FixedChain.Owned.flatConstraints_length
      (coreInterface interface offset) offset

theorem flatConstraints_varsBelow {degree : Nat}
    (interface : Interface degree) (offset : Nat)
    (assumptions : Assumptions interface offset (fun _ => 0)) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow offset := by
  change ∀ expression ∈ flatConstraints (Circuit.ops
    (FixedChain.Owned.main (coreInterface interface offset)) offset),
      expression.VarsBelow offset
  exact FixedChain.Owned.flatConstraints_varsBelow
    (coreInterface interface offset) offset assumptions

theorem output_varsBelow {degree : Nat} (interface : Interface degree)
    (offset : Nat) (assumptions : Assumptions interface offset (fun _ => 0)) :
    (output interface offset).VarsBelow offset := by
  unfold output FixedChain.Owned.output
  apply FixedChain.Owned.outputFrom_varsBelow
  · exact assumptions.1
  · intro round member
    rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
    rcases member with ⟨roundIndex, rfl⟩
    exact assumptions.2 roundIndex

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
        (coreInterface interface offset)).eval env =
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

/-- Exact completeness direction: the canonical verifier chain supplies the
owned round constraints and the final claimed value used by the terminal
identity. -/
theorem keyChain_implies_spec_and_terminal
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
    SpecHolds interface offset env ∧
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
  have coreChain := chain
  rw [← initialCoreEq, ← roundsListEq, ← challengeListEq] at coreChain
  have split := (FixedChain.Owned.chain_iff_specHolds_and_output_eq
    (coreInterface interface offset) env
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
          running fresh proof).output)).mp coreChain
  exact ⟨by simpa [SpecHolds] using split.1,
    by simpa [output] using split.2⟩

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain
