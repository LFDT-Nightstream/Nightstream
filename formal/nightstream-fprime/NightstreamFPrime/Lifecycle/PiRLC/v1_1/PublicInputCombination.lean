import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Lifecycle.PaperAlgebra

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1, equation
`x = sum_i rho_i x_i` under coefficient embedding.

This leaf instantiates one exact 54-coefficient public ring. It proves that
the block/lane circuit order is the canonical flat public-column order used
by `PiRLCAlgebra.PublicInput.combinePublicInputs`.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev blockCount : Nat := publicRingColumns
abbrev cellCount : Nat := 1

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  challenge : Nat → Fin CombinationFamily.sourceCount → Fin ringDegree → Expr
  input : Nat → Fin CombinationFamily.sourceCount →
    Fin (FullShape logicalWidth publicFits).publicWidth → Expr

def publicColumn
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (block : Fin blockCount) (lane : Fin ringDegree) :
    Fin (FullShape logicalWidth publicFits).publicWidth :=
  ⟨block.val * ringDegree + lane.val, by
    have blockLt := block.isLt
    have laneLt := lane.isLt
    have blockZero : block.val = 0 := by
      simp only [blockCount, publicRingColumns] at blockLt
      omega
    change block.val * ringDegree + lane.val < ringDegree * publicRingColumns
    simpa [blockZero, publicRingColumns] using laneLt⟩

theorem publicColumn_decode
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    publicColumn
        (PiRLCAlgebra.PublicInput.publicBlockIndex
          (FullShape logicalWidth publicFits) column)
        (PiRLCAlgebra.PublicInput.publicLaneIndex column) = column := by
  apply Fin.ext
  change column.val / ringDegree * ringDegree + column.val % ringDegree =
    column.val
  simpa [Nat.mul_comm] using Nat.div_add_mod column.val ringDegree

def familyInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    CombinationFamily.Interface blockCount cellCount where
  challenge := interface.challenge
  input := fun offset source block lane _ =>
    interface.input offset source (publicColumn block lane)

def cell : Fin cellCount := ⟨0, by decide⟩

def output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  CombinationFamily.output (familyInterface interface) offset
    (PiRLCAlgebra.PublicInput.publicBlockIndex
      (FullShape logicalWidth publicFits) column)
    (PiRLCAlgebra.PublicInput.publicLaneIndex column) cell

abbrev Assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) :=
  CombinationFamily.Assumptions (familyInterface interface) offset env

abbrev SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) :=
  CombinationFamily.CanonicalHolds (familyInterface interface) offset env

def evalChallenges
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    Fin 17 → RingF :=
  fun source lane =>
    (interface.challenge offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) lane).eval env

def evalInputs
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    Fin 17 → PublicInput (FullShape logicalWidth publicFits) :=
  fun source column =>
    (interface.input offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) column).eval env

def evalOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env) :
    PublicInput (FullShape logicalWidth publicFits) :=
  fun column => (output interface offset column).eval env

private theorem rightCombination_eq_combinePublicInputs
    {shape : Phi81Relation.Shape} {count : Nat}
    (challenges : Fin count → RingF)
    (inputs : Fin count → PublicInput shape)
    (column : Fin shape.publicWidth) :
    CombinationFamily.rightCombination
        (fun source => ringFMul (challenges source)
          (PiRLCAlgebra.PublicInput.publicBlock (inputs source)
            (PiRLCAlgebra.PublicInput.publicBlockIndex shape column)))
        (PiRLCAlgebra.PublicInput.publicLaneIndex column) =
      PiRLCAlgebra.PublicInput.combinePublicInputs challenges inputs column := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      change
        ringFMul (challenges 0)
              (PiRLCAlgebra.PublicInput.publicBlock (inputs 0)
                (PiRLCAlgebra.PublicInput.publicBlockIndex shape column))
              (PiRLCAlgebra.PublicInput.publicLaneIndex column) +
            CombinationFamily.rightCombination
              (fun source => ringFMul (challenges source.succ)
                (PiRLCAlgebra.PublicInput.publicBlock (inputs source.succ)
                  (PiRLCAlgebra.PublicInput.publicBlockIndex shape column)))
              (PiRLCAlgebra.PublicInput.publicLaneIndex column) =
          ringFMul (challenges 0)
              (PiRLCAlgebra.PublicInput.publicBlock (inputs 0)
                (PiRLCAlgebra.PublicInput.publicBlockIndex shape column))
              (PiRLCAlgebra.PublicInput.publicLaneIndex column) +
            PiRLCAlgebra.PublicInput.combinePublicInputs
              (fun source => challenges source.succ)
              (fun source => inputs source.succ) column
      rw [inductionHypothesis]

theorem parentCoverage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalOutput interface offset env =
      PiRLCAlgebra.PublicInput.combinePublicInputs
        (evalChallenges interface offset env)
        (evalInputs interface offset env) := by
  funext column
  let block := PiRLCAlgebra.PublicInput.publicBlockIndex
    (FullShape logicalWidth publicFits) column
  let lane := PiRLCAlgebra.PublicInput.publicLaneIndex column
  have familyResult := specification block cell
  have laneResult := congrFun familyResult lane
  calc
    evalOutput interface offset env column =
        CombinationFamily.orderedCombination (familyInterface interface)
          offset env block cell lane := by
      simpa [evalOutput, output, block, lane,
        CombinationFamily.evalOutput] using laneResult
    _ = PiRLCAlgebra.PublicInput.combinePublicInputs
          (evalChallenges interface offset env)
          (evalInputs interface offset env) column := by
      have challengeEq : ∀ source : Fin 17,
          CombinationFamily.challengeValue (familyInterface interface)
              offset env
              (Fin.cast CombinationFamily.sourceCount_eq.symm source) =
            evalChallenges interface offset env source := by
        intro source
        rfl
      have inputEq : ∀ source : Fin 17,
          CombinationFamily.inputValue (familyInterface interface)
              offset env
              (Fin.cast CombinationFamily.sourceCount_eq.symm source)
              block cell =
            PiRLCAlgebra.PublicInput.publicBlock
              (evalInputs interface offset env source) block := by
        intro source
        funext currentLane
        rfl
      have orderedEq :
          CombinationFamily.orderedCombination (familyInterface interface)
              offset env block cell =
            CombinationFamily.rightCombination fun source =>
              ringFMul (evalChallenges interface offset env source)
                (PiRLCAlgebra.PublicInput.publicBlock
                  (evalInputs interface offset env source) block) := by
        unfold CombinationFamily.orderedCombination CombinationFamily.term
        apply congrArg CombinationFamily.rightCombination
        funext source
        rw [challengeEq source, inputEq source]
      rw [orderedEq]
      simpa [block, lane] using
        rightCombination_eq_combinePublicInputs
          (evalChallenges interface offset env)
          (evalInputs interface offset env) column

theorem logicalPrivateCount_eq :
    CombinationFamily.logicalPrivateCount blockCount cellCount = 918 := by
  rw [CombinationFamily.logicalPrivateCount,
    CombinationFamily.sourceCount_eq]
  norm_num [CombinationFamily.stepSize, CombinationStep.privateCount,
    blockCount, cellCount, publicRingColumns, ringDegree]

theorem logicalRowCount_eq :
    CombinationFamily.logicalRowCount blockCount cellCount = 918 := by
  rw [CombinationFamily.logicalRowCount, logicalPrivateCount_eq]

def circuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  CombinationFamily.circuit (familyInterface interface)

theorem localLength_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 918 := by
  rw [(circuit interface).privateCount_eq offset]
  exact logicalPrivateCount_eq

theorem flatConstraints_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      918 := by
  rw [(circuit interface).rowCount_eq offset]
  exact logicalRowCount_eq

theorem flatConstraints_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + 918) := by
  simpa [circuit, logicalPrivateCount_eq] using
    CombinationFamily.flatConstraints_varsBelow (familyInterface interface)
      offset env assumptions

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  CombinationFamily.soundness (familyInterface interface) offset env
    assumptions rows

theorem complete
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  CombinationFamily.complete (familyInterface interface) offset env assumptions

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  CombinationFamily.completeness (familyInterface interface) offset env
    assumptions specification

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination
