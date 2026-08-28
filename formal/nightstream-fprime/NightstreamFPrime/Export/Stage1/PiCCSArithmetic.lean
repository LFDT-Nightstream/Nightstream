import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FormalRows

/-!
Owns the compact ordinary-row packet for the eight non-permutation PiCCS
leaves.

The canonical PiCCS phase assembler remains
`Lifecycle.PiCCS.v1_1.Formal`. This module only lowers its arithmetic children
at their proved offsets, applies the Stage 1 Spartan column permutation, and
encodes each physical row as one witness instruction or sparse assertion.

Parent coverage:
- `PiCCS.v1_1.Formal.statementBindingCircuit`
- `PiCCS.v1_1.Formal.initialClaimCircuit`
- `PiCCS.v1_1.Formal.sumcheckCircuit`
- `PiCCS.v1_1.Formal.evalKCircuit`
- `PiCCS.v1_1.Formal.evalACircuit`
- `PiCCS.v1_1.Formal.ccsCircuit`
- `PiCCS.v1_1.Formal.normCircuit`
- `PiCCS.v1_1.Formal.finalIdentityCircuit`
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSArithmetic

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def statementBindingRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingRowStart

def initialClaimRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimRowStart
def sumcheckRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckRowStart
def evalKRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKRowStart
def evalARowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalARowStart
def ccsRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsRowStart
def normRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.normRowStart
def finalIdentityRowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityRowStart

def statementBindingFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingFreshStart

def initialClaimFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimFreshStart
def sumcheckFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckFreshStart
def evalKFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKFreshStart
def evalAFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalAFreshStart
def ccsFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsFreshStart
def normFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.normFreshStart
def finalIdentityFreshStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityFreshStart

/-- Materialized logical child starts. Their match theorems below tie every
value to the canonical PiCCS assembler without executing prior child circuits. -/
def statementBindingLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset

def initialClaimLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimLogicalStart
def sumcheckLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckLogicalStart
def evalKLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKLogicalStart
def evalALogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalALogicalStart
def ccsLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsLogicalStart
def normLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.normLogicalStart
def finalIdentityLogicalStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityLogicalStart

/-- Lower and encode one ordinary child packet without changing its rows. -/
def compilePacket (rowStart freshStart : Nat) (constraints : List Expr) :
    List Rows.CompiledRow :=
  Rows.compileRowsTR
    (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan freshStart)
    rowStart
    (NightstreamFPrime.Layout.Stage1.Spartan.remapRows
      (Rows.lowerConstraintsTR constraints freshStart).rows)

theorem compilePacket_toR1CS (rowStart freshStart : Nat)
    (constraints : List Expr) :
    (compilePacket rowStart freshStart constraints).map
      Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints constraints freshStart).rows := by
  rw [compilePacket, Rows.compileRowsTR_toR1CS,
    Rows.lowerConstraintsTR_eq]

/-- Satisfaction of one emitted arithmetic packet implies its exact source
constraint list under the proved Stage 1 column pullback. -/
theorem compilePacket_sound (rowStart freshStart : Nat)
    (constraints : List Expr) (env : Env)
    (holds : R1CS.RowsHold env
      ((compilePacket rowStart freshStart constraints).map
        Rows.CompiledRow.toR1CS)) :
    ConstraintsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) constraints := by
  rw [compilePacket_toR1CS,
    NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold] at holds
  exact R1CS.lowerConstraints_sound
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) constraints
    freshStart holds

/-- Constructive completeness of one emitted arithmetic packet. Only the
mapped R1CS-fresh interval changes; logical variables and all other packets
keep their values. -/
theorem compilePacket_complete (rowStart freshStart : Nat)
    (constraints : List Expr) (env : Env)
    (startLocal : NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
      freshStart)
    (targetEndPrivate :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan freshStart +
          R1CS.totalFreshCount constraints ≤
        NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount)
    (scope : ∀ expression ∈ constraints,
      expression.VarsBelow freshStart)
    (logical : ConstraintsHold
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) constraints) :
    ∃ completed,
      AgreesOutside env completed
          (NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan freshStart)
          (R1CS.totalFreshCount constraints) ∧
        R1CS.RowsHold completed
          ((compilePacket rowStart freshStart constraints).map
            Rows.CompiledRow.toR1CS) := by
  rcases R1CS.lowerConstraints_complete
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) constraints
      freshStart scope logical with ⟨source, sourceAgrees, sourceRows⟩
  let completed :=
    NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval env source
      freshStart (R1CS.totalFreshCount constraints)
  refine ⟨completed, ?_, ?_⟩
  · exact NightstreamFPrime.Layout.Stage1.Spartan.copyMappedInterval_agreesOutside
      env source freshStart (R1CS.totalFreshCount constraints)
  · rw [compilePacket_toR1CS]
    exact
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold_copyMappedInterval
        (R1CS.lowerConstraints constraints freshStart).rows env source
        freshStart (R1CS.totalFreshCount constraints) startLocal
        targetEndPrivate sourceAgrees sourceRows

@[simp] theorem compilePacket_length (rowStart freshStart : Nat)
    (constraints : List Expr) :
    (compilePacket rowStart freshStart constraints).length =
      R1CS.totalRowCount constraints := by
  simp [compilePacket, NightstreamFPrime.Layout.Stage1.Spartan.remapRows,
    Rows.lowerConstraintsTR_eq]

def parentInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.Interface logicalWidth 9 publicFits :=
  PiCCSInvocations.parentInterface logicalWidth publicFits

def sharedInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.Interface logicalWidth 9 publicFits :=
  PiCCSInvocations.sharedInterface logicalWidth publicFits

theorem initialClaimLogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    initialClaimLogicalStart =
      Formal.initialClaimOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  calc
    initialClaimLogicalStart =
        Formal.initialClaimRowOffset 9
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
      rfl
    _ = _ := (Formal.initialClaimOffset_eq_initialClaimRowOffset
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset).symm

theorem sumcheckLogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    sumcheckLogicalStart =
      Formal.sumcheckOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  calc
    sumcheckLogicalStart =
        Formal.sumcheckRowOffset 9
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
      rfl
    _ = _ := (Formal.sumcheckOffset_eq_sumcheckRowOffset
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset).symm

theorem evalKLogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    evalKLogicalStart =
      Formal.evalKOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  calc
    evalKLogicalStart =
        Formal.evalKRowOffset 9
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
      rfl
    _ = _ := (Formal.evalKOffset_eq_evalKRowOffset
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset).symm

theorem evalALogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    evalALogicalStart =
      Formal.evalAOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  calc
    evalALogicalStart =
        Formal.evalARowOffset 9
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
      rfl
    _ = _ := (Formal.evalAOffset_eq_evalARowOffset
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset).symm

theorem ccsLogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    ccsLogicalStart =
      Formal.ccsOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  calc
    ccsLogicalStart =
        Formal.ccsRowOffset 9
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
      rfl
    _ = _ := (Formal.ccsOffset_eq_ccsRowOffset
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset).symm

theorem normLogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    normLogicalStart =
      Formal.normRowOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  rfl

theorem finalIdentityLogicalStart_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    finalIdentityLogicalStart =
      Formal.finalIdentityRowOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset := by
  rfl

def statementBindingConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    (Formal.statementBindingCircuit (sharedInterface logicalWidth publicFits))
    statementBindingLogicalStart

def initialClaimConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    (Formal.initialClaimCircuit (sharedInterface logicalWidth publicFits))
    initialClaimLogicalStart

def sumcheckConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    (Formal.sumcheckCircuit (sharedInterface logicalWidth publicFits))
    sumcheckLogicalStart

def evalKConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    (Formal.evalKCircuit (sharedInterface logicalWidth publicFits))
    evalKLogicalStart

def evalAConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints
    (Formal.evalACircuit (sharedInterface logicalWidth publicFits))
    evalALogicalStart

def mainConstraints (main : Circuit Unit) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops main offset)

def ccsConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  mainConstraints
    (Formal.ccsRowMain (sharedInterface logicalWidth publicFits))
    ccsLogicalStart

def normConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  mainConstraints
    (Formal.normRowMain (sharedInterface logicalWidth publicFits))
    normLogicalStart

def finalIdentityConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  mainConstraints
    (Formal.finalIdentityRowMain (sharedInterface logicalWidth publicFits))
    finalIdentityLogicalStart

def statementBindingRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket statementBindingRowStart statementBindingFreshStart
    (statementBindingConstraints logicalWidth publicFits)

def initialClaimRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket initialClaimRowStart initialClaimFreshStart
    (initialClaimConstraints logicalWidth publicFits)

def sumcheckRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket sumcheckRowStart sumcheckFreshStart
    (sumcheckConstraints logicalWidth publicFits)

def evalKRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket evalKRowStart evalKFreshStart
    (evalKConstraints logicalWidth publicFits)

def evalARows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket evalARowStart evalAFreshStart
    (evalAConstraints logicalWidth publicFits)

def ccsRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket ccsRowStart ccsFreshStart
    (ccsConstraints logicalWidth publicFits)

def normRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket normRowStart normFreshStart
    (normConstraints logicalWidth publicFits)

def finalIdentityRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  compilePacket finalIdentityRowStart finalIdentityFreshStart
    (finalIdentityConstraints logicalWidth publicFits)

/-- The eight non-permutation PiCCS row packets in canonical child order. -/
def arithmeticRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth) :
    List Rows.CompiledRow :=
  statementBindingRows logicalWidth publicFits ++
    initialClaimRows logicalWidth publicFits ++
    sumcheckRows logicalWidth publicFits ++
    evalKRows logicalWidth publicFits ++
    evalARows logicalWidth publicFits ++
    ccsRows logicalWidth publicFits ++
    normRows logicalWidth publicFits ++
    finalIdentityRows logicalWidth publicFits

/-- Named physical ownership of the eight non-permutation child packets. -/
structure PacketHolds
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env) : Prop where
  statementBinding : R1CS.RowsHold env
    ((statementBindingRows logicalWidth publicFits).map
      Rows.CompiledRow.toR1CS)
  initialClaim : R1CS.RowsHold env
    ((initialClaimRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)
  sumcheck : R1CS.RowsHold env
    ((sumcheckRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)
  eval_K : R1CS.RowsHold env
    ((evalKRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)
  eval_A : R1CS.RowsHold env
    ((evalARows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)
  ccs : R1CS.RowsHold env
    ((ccsRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)
  norm : R1CS.RowsHold env
    ((normRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)
  finalIdentity : R1CS.RowsHold env
    ((finalIdentityRows logicalWidth publicFits).map Rows.CompiledRow.toR1CS)

/-- The one exported arithmetic list covers the eight named packets exactly
once and in canonical PiCCS child order. -/
theorem arithmeticRows_imply_packetHolds
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env)
    (holds : R1CS.RowsHold env
      ((arithmeticRows logicalWidth publicFits).map
        Rows.CompiledRow.toR1CS)) :
    PacketHolds logicalWidth publicFits env := by
  unfold arithmeticRows at holds
  simp only [List.map_append, R1CS.rowsHold_append] at holds
  rcases holds with
    ⟨⟨⟨⟨⟨⟨⟨statementBinding, initialClaim⟩, sumcheck⟩, eval_K⟩, eval_A⟩,
      ccs⟩, norm⟩, finalIdentity⟩
  exact ⟨statementBinding, initialClaim, sumcheck, eval_K, eval_A, ccs, norm,
    finalIdentity⟩

/-- Generic packet boundary used by each arithmetic child. The parent sees the
child contract and never unfolds the child's operations. -/
theorem compilePacket_implies_childSpec
    (child : FormalCircuit) (childOffset rowStart freshStart : Nat)
    (constraints : List Expr) (env : Env)
    (constraintsEq : constraints =
      flatConstraints (Circuit.ops child.main childOffset))
    (assumptions : child.assumptions childOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (packet : R1CS.RowsHold env
      ((compilePacket rowStart freshStart constraints).map
        Rows.CompiledRow.toR1CS)) :
    child.spec childOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply child.soundness
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) childOffset
    assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    (flatConstraints (Circuit.ops child.main childOffset))
  rw [← constraintsEq]
  exact compilePacket_sound rowStart freshStart constraints env packet

/-- The eight non-permutation conjuncts supplied to the canonical parent. -/
structure ArithmeticSpecs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Prop where
  statementBinding :
    (Formal.statementBindingCircuit
      (sharedInterface logicalWidth publicFits)).spec
      statementBindingLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  initialClaim :
    (Formal.initialClaimCircuit (sharedInterface logicalWidth publicFits)).spec
      initialClaimLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  sumcheck :
    (Formal.sumcheckCircuit (sharedInterface logicalWidth publicFits)).spec
      sumcheckLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  eval_K :
    (Formal.evalKCircuit (sharedInterface logicalWidth publicFits)).spec
      evalKLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  eval_A :
    (Formal.evalACircuit (sharedInterface logicalWidth publicFits)).spec
      evalALogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  ccs :
    (Formal.ccsCircuit relation
      (sharedInterface logicalWidth publicFits)).spec ccsLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  norm :
    (Formal.normCircuit relation
      (sharedInterface logicalWidth publicFits)).spec normLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  finalIdentity :
    (Formal.finalIdentityCircuit relation
      (sharedInterface logicalWidth publicFits)).spec finalIdentityLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)

theorem ArithmeticSpecs.statementBinding_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.statementBindingCircuit
      (sharedInterface logicalWidth publicFits)).spec
      (Formal.statementBindingOffset
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  exact specs.statementBinding

theorem ArithmeticSpecs.initialClaim_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.initialClaimCircuit (sharedInterface logicalWidth publicFits)).spec
      (Formal.initialClaimOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [← initialClaimLogicalStart_matches logicalWidth publicFits]
  exact specs.initialClaim

theorem ArithmeticSpecs.sumcheck_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.sumcheckCircuit (sharedInterface logicalWidth publicFits)).spec
      (Formal.sumcheckOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [← sumcheckLogicalStart_matches logicalWidth publicFits]
  exact specs.sumcheck

theorem ArithmeticSpecs.evalK_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.evalKCircuit (sharedInterface logicalWidth publicFits)).spec
      (Formal.evalKOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [← evalKLogicalStart_matches logicalWidth publicFits]
  exact specs.eval_K

theorem ArithmeticSpecs.evalA_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.evalACircuit (sharedInterface logicalWidth publicFits)).spec
      (Formal.evalAOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [← evalALogicalStart_matches logicalWidth publicFits]
  exact specs.eval_A

theorem ArithmeticSpecs.ccs_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.ccsCircuit relation (sharedInterface logicalWidth publicFits)).spec
      (Formal.ccsOffset (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [← ccsLogicalStart_matches logicalWidth publicFits]
  exact specs.ccs

theorem ArithmeticSpecs.norm_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.normCircuit relation (sharedInterface logicalWidth publicFits)).spec
      (Formal.normOffset relation (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [Formal.normOffset_eq_normRowOffset,
    ← normLogicalStart_matches logicalWidth publicFits]
  exact specs.norm

theorem ArithmeticSpecs.finalIdentity_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {env : Env} (specs : ArithmeticSpecs logicalWidth publicFits relation env) :
    (Formal.finalIdentityCircuit relation
      (sharedInterface logicalWidth publicFits)).spec
      (Formal.finalIdentityOffset relation
        (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  rw [Formal.finalIdentityOffset_eq_finalIdentityRowOffset,
    ← finalIdentityLogicalStart_matches logicalWidth publicFits]
  exact specs.finalIdentity

/-- The eight held packets imply the eight exact child predicates
under the canonical parent assumptions. -/
theorem packetHolds_imply_arithmeticSpecs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env)
    (assumptions : Formal.Assumptions relation
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (packets : PacketHolds logicalWidth publicFits env) :
    ArithmeticSpecs logicalWidth publicFits relation env := by
  have statementBindingAssumptions :
      (Formal.statementBindingCircuit
        (sharedInterface logicalWidth publicFits)).assumptions
        statementBindingLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    exact assumptions.statementBinding
  have initialClaimAssumptions :
      (Formal.initialClaimCircuit
        (sharedInterface logicalWidth publicFits)).assumptions
        initialClaimLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [initialClaimLogicalStart_matches logicalWidth publicFits]
    exact assumptions.initialClaim
  have sumcheckAssumptions :
      (Formal.sumcheckCircuit
        (sharedInterface logicalWidth publicFits)).assumptions
        sumcheckLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [sumcheckLogicalStart_matches logicalWidth publicFits]
    exact assumptions.sumcheck
  have evalKAssumptions :
      (Formal.evalKCircuit
        (sharedInterface logicalWidth publicFits)).assumptions
        evalKLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [evalKLogicalStart_matches logicalWidth publicFits]
    exact assumptions.eval_K
  have evalAAssumptions :
      (Formal.evalACircuit
        (sharedInterface logicalWidth publicFits)).assumptions
        evalALogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [evalALogicalStart_matches logicalWidth publicFits]
    exact assumptions.eval_A
  have ccsAssumptions :
      (Formal.ccsCircuit relation
        (sharedInterface logicalWidth publicFits)).assumptions
        ccsLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [ccsLogicalStart_matches logicalWidth publicFits]
    exact assumptions.ccs
  have normAssumptions :
      (Formal.normCircuit relation
        (sharedInterface logicalWidth publicFits)).assumptions
        normLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [normLogicalStart_matches logicalWidth publicFits,
      ← Formal.normOffset_eq_normRowOffset relation
        (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
    exact assumptions.norm
  have finalIdentityAssumptions :
      (Formal.finalIdentityCircuit relation
        (sharedInterface logicalWidth publicFits)).assumptions
        finalIdentityLogicalStart
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
    rw [finalIdentityLogicalStart_matches logicalWidth publicFits,
      ← Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation
        (parentInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
    exact assumptions.finalIdentity
  refine {
    statementBinding := compilePacket_implies_childSpec
      (Formal.statementBindingCircuit
        (sharedInterface logicalWidth publicFits))
      statementBindingLogicalStart statementBindingRowStart
      statementBindingFreshStart
      (statementBindingConstraints logicalWidth publicFits) env rfl
      statementBindingAssumptions packets.statementBinding
    initialClaim := compilePacket_implies_childSpec
      (Formal.initialClaimCircuit (sharedInterface logicalWidth publicFits))
      initialClaimLogicalStart initialClaimRowStart initialClaimFreshStart
      (initialClaimConstraints logicalWidth publicFits) env rfl
      initialClaimAssumptions packets.initialClaim
    sumcheck := compilePacket_implies_childSpec
      (Formal.sumcheckCircuit (sharedInterface logicalWidth publicFits))
      sumcheckLogicalStart sumcheckRowStart sumcheckFreshStart
      (sumcheckConstraints logicalWidth publicFits) env rfl
      sumcheckAssumptions packets.sumcheck
    eval_K := compilePacket_implies_childSpec
      (Formal.evalKCircuit (sharedInterface logicalWidth publicFits))
      evalKLogicalStart evalKRowStart evalKFreshStart
      (evalKConstraints logicalWidth publicFits) env rfl
      evalKAssumptions packets.eval_K
    eval_A := compilePacket_implies_childSpec
      (Formal.evalACircuit (sharedInterface logicalWidth publicFits))
      evalALogicalStart evalARowStart evalAFreshStart
      (evalAConstraints logicalWidth publicFits) env rfl
      evalAAssumptions packets.eval_A
    ccs := compilePacket_implies_childSpec
      (Formal.ccsCircuit relation (sharedInterface logicalWidth publicFits))
      ccsLogicalStart ccsRowStart ccsFreshStart
      (ccsConstraints logicalWidth publicFits) env (by
        unfold ccsConstraints mainConstraints
        rw [Formal.ccsCircuit_main_eq_rowMain relation
          (sharedInterface logicalWidth publicFits)])
      ccsAssumptions packets.ccs
    norm := compilePacket_implies_childSpec
      (Formal.normCircuit relation (sharedInterface logicalWidth publicFits))
      normLogicalStart normRowStart normFreshStart
      (normConstraints logicalWidth publicFits) env (by
        unfold normConstraints mainConstraints
        rw [Formal.normCircuit_main_eq_rowMain relation
          (sharedInterface logicalWidth publicFits)])
      normAssumptions packets.norm
    finalIdentity := compilePacket_implies_childSpec
      (Formal.finalIdentityCircuit relation
        (sharedInterface logicalWidth publicFits))
      finalIdentityLogicalStart finalIdentityRowStart finalIdentityFreshStart
      (finalIdentityConstraints logicalWidth publicFits) env (by
        unfold finalIdentityConstraints mainConstraints
        rw [Formal.finalIdentityCircuit_main_eq_rowMain relation
          (sharedInterface logicalWidth publicFits)])
      finalIdentityAssumptions packets.finalIdentity }

def inputShapes
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiCCS.v1_1.InputShapes relation
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset :=
  NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.inputShapes relation
    (parentInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits)

theorem statementBindingRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (statementBindingRows logicalWidth publicFits).length = 160 := by
  rw [statementBindingRows, compilePacket_length]
  unfold statementBindingConstraints
  exact
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding.physicalRowCount_eq
      (sharedInterface logicalWidth publicFits)
      (fun childOffset => by
        simpa [sharedInterface, PiCCSInvocations.sharedInterface,
          parentInterface, PiCCSInvocations.parentInterface] using
          (inputShapes logicalWidth publicFits relation).statementBinding
            childOffset)
      statementBindingLogicalStart

theorem initialClaimRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (initialClaimRows logicalWidth publicFits).length = 116631 := by
  rw [initialClaimRows, compilePacket_length]
  unfold initialClaimConstraints
  rw [initialClaimLogicalStart_matches logicalWidth publicFits]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim.physicalRowCount_eq
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).initialClaim
    (Formal.initialClaimOffset (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem sumcheckRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sumcheckRows logicalWidth publicFits).length = 393959 := by
  rw [sumcheckRows, compilePacket_length]
  unfold sumcheckConstraints
  rw [sumcheckLogicalStart_matches logicalWidth publicFits]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain.physicalRowCount_eq
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).sumcheck
    (Formal.sumcheckOffset (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem evalKRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (evalKRows logicalWidth publicFits).length = 8486 := by
  rw [evalKRows, compilePacket_length]
  unfold evalKConstraints
  rw [evalKLogicalStart_matches logicalWidth publicFits]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal.physicalRowCount_eq
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).eval_K
    (Formal.evalKOffset (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem evalARows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (evalARows logicalWidth publicFits).length = 109574 := by
  rw [evalARows, compilePacket_length]
  unfold evalAConstraints
  rw [evalALogicalStart_matches logicalWidth publicFits]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal.physicalRowCount_eq
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).eval_A
    (Formal.evalAOffset (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem ccsRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (ccsRows logicalWidth publicFits).length = 20794 := by
  rw [ccsRows, compilePacket_length]
  unfold ccsConstraints mainConstraints
  rw [ccsLogicalStart_matches logicalWidth publicFits]
  rw [← Formal.ccsCircuit_main_eq_rowMain relation
    (sharedInterface logicalWidth publicFits)]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal.physicalRowCount_eq relation
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).ccs
    (Formal.ccsOffset (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem normRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (normRows logicalWidth publicFits).length = 752 := by
  rw [normRows, compilePacket_length]
  unfold normConstraints mainConstraints
  rw [normLogicalStart_matches logicalWidth publicFits]
  rw [← Formal.normCircuit_main_eq_rowMain relation
      (sharedInterface logicalWidth publicFits),
    ← Formal.normOffset_eq_normRowOffset relation
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
  exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal.physicalRowCount_eq relation
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).norm
    (Formal.normOffset relation (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)

theorem finalIdentityRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (finalIdentityRows logicalWidth publicFits).length = 130447 := by
  rw [finalIdentityRows, compilePacket_length]
  unfold finalIdentityConstraints mainConstraints
  rw [finalIdentityLogicalStart_matches logicalWidth publicFits]
  rw [← Formal.finalIdentityCircuit_main_eq_rowMain relation
      (sharedInterface logicalWidth publicFits),
    ← Formal.finalIdentityOffset_eq_finalIdentityRowOffset relation
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset]
  have footprint :=
    NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity.physicalRowCount_eq relation
    (sharedInterface logicalWidth publicFits)
    (inputShapes logicalWidth publicFits relation).finalIdentity
    (Formal.finalIdentityOffset relation
      (parentInterface logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset)
  have terminal := NightstreamFPrime.Layout.PiCCS.v1_1.terminalRowCost_eq
    relation (parentInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    (inputShapes logicalWidth publicFits relation)
  have terminalPhysical :
      NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity.terminalPhysicalRowCount
          (Formal.finalIdentityInterface relation
            (sharedInterface logicalWidth publicFits))
          (Formal.finalIdentityOffset relation
            (parentInterface logicalWidth publicFits)
            NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset) = 5326 := by
    simpa [NightstreamFPrime.Layout.PiCCS.v1_1.terminalRowCost,
      sharedInterface, parentInterface,
      PiCCSInvocations.sharedInterface] using terminal
  rw [footprint, terminalPhysical]

theorem arithmeticRows_length
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (arithmeticRows logicalWidth publicFits).length = 780803 := by
  unfold arithmeticRows
  rw [List.length_append, List.length_append, List.length_append,
    List.length_append, List.length_append, List.length_append,
    List.length_append,
    statementBindingRows_length logicalWidth publicFits relation,
    initialClaimRows_length logicalWidth publicFits relation,
    sumcheckRows_length logicalWidth publicFits relation,
    evalKRows_length logicalWidth publicFits relation,
    evalARows_length logicalWidth publicFits relation,
    ccsRows_length logicalWidth publicFits relation,
    normRows_length logicalWidth publicFits relation,
    finalIdentityRows_length logicalWidth publicFits relation]

def witnessInstructions
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Rows.witnessInstructions
    (arithmeticRows logicalWidth publicFits)

def assertionRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Rows.assertionRows (arithmeticRows logicalWidth publicFits)

end NightstreamFPrime.Export.Stage1.PiCCSArithmetic
