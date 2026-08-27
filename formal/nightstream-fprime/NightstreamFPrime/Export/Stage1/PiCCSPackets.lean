import NightstreamFPrime.Export.Stage1.WitnessProgram

/-!
Owns executable packets for the seven PiCCS arithmetic children that produce
both witness batches and ordinary rows.

Each packet constructs its child's operation list once. The existing witness
and row definitions remain the proof authority. The projection theorems below
prove that the shared executable value is exactly those definitions.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPackets

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

structure Packet where
  batches : List WitnessBatch
  rows : List Rows.CompiledRow

def make (main : Circuit Unit) (logicalStart rowStart freshStart : Nat) :
    Packet :=
  let operations := Circuit.ops main logicalStart
  {
    batches :=
      (witnesses operations).map WitnessProgram.remapBatch
    rows :=
      PiCCSArithmetic.compilePacket rowStart freshStart
        (flatConstraints operations) }

@[simp] theorem make_batches (main : Circuit Unit)
    (logicalStart rowStart freshStart : Nat) :
    (make main logicalStart rowStart freshStart).batches =
      WitnessProgram.childBatches main logicalStart := by
  rfl

@[simp] theorem make_rows (main : Circuit Unit)
    (logicalStart rowStart freshStart : Nat) :
    (make main logicalStart rowStart freshStart).rows =
      PiCCSArithmetic.compilePacket rowStart freshStart
        (PiCCSArithmetic.mainConstraints main logicalStart) := by
  rfl

def initialClaim
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.initialClaimCircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.initialClaimLogicalStart
    PiCCSArithmetic.initialClaimRowStart
    PiCCSArithmetic.initialClaimFreshStart

def sumcheck
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.sumcheckCircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.sumcheckLogicalStart
    PiCCSArithmetic.sumcheckRowStart
    PiCCSArithmetic.sumcheckFreshStart

def evalK
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.evalKCircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.evalKLogicalStart
    PiCCSArithmetic.evalKRowStart
    PiCCSArithmetic.evalKFreshStart

def evalA
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.evalACircuit
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits)).main
    PiCCSArithmetic.evalALogicalStart
    PiCCSArithmetic.evalARowStart
    PiCCSArithmetic.evalAFreshStart

def ccs
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.ccsRowMain
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    PiCCSArithmetic.ccsLogicalStart
    PiCCSArithmetic.ccsRowStart
    PiCCSArithmetic.ccsFreshStart

def norm
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.normRowMain
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    PiCCSArithmetic.normLogicalStart
    PiCCSArithmetic.normRowStart
    PiCCSArithmetic.normFreshStart

def finalIdentity
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Packet :=
  make
    (Formal.finalIdentityRowMain
      (PiCCSArithmetic.sharedInterface logicalWidth publicFits))
    PiCCSArithmetic.finalIdentityLogicalStart
    PiCCSArithmetic.finalIdentityRowStart
    PiCCSArithmetic.finalIdentityFreshStart

theorem initialClaim_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (initialClaim logicalWidth publicFits).batches =
      WitnessProgram.initialClaimBatches logicalWidth publicFits := by
  simp only [initialClaim, make_batches,
    WitnessProgram.initialClaimBatches]

theorem initialClaim_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (initialClaim logicalWidth publicFits).rows =
      PiCCSArithmetic.initialClaimRows logicalWidth publicFits := by
  simp only [initialClaim, make_rows, PiCCSArithmetic.initialClaimRows,
    PiCCSArithmetic.initialClaimConstraints,
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
    PiCCSArithmetic.mainConstraints]

theorem sumcheck_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (sumcheck logicalWidth publicFits).batches =
      WitnessProgram.sumcheckBatches logicalWidth publicFits := by
  simp only [sumcheck, make_batches, WitnessProgram.sumcheckBatches]

theorem sumcheck_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (sumcheck logicalWidth publicFits).rows =
      PiCCSArithmetic.sumcheckRows logicalWidth publicFits := by
  simp only [sumcheck, make_rows, PiCCSArithmetic.sumcheckRows,
    PiCCSArithmetic.sumcheckConstraints,
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
    PiCCSArithmetic.mainConstraints]

theorem evalK_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (evalK logicalWidth publicFits).batches =
      WitnessProgram.evalKBatches logicalWidth publicFits := by
  simp only [evalK, make_batches, WitnessProgram.evalKBatches]

theorem evalK_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (evalK logicalWidth publicFits).rows =
      PiCCSArithmetic.evalKRows logicalWidth publicFits := by
  simp only [evalK, make_rows, PiCCSArithmetic.evalKRows,
    PiCCSArithmetic.evalKConstraints,
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
    PiCCSArithmetic.mainConstraints]

theorem evalA_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (evalA logicalWidth publicFits).batches =
      WitnessProgram.evalABatches logicalWidth publicFits := by
  simp only [evalA, make_batches, WitnessProgram.evalABatches]

theorem evalA_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (evalA logicalWidth publicFits).rows =
      PiCCSArithmetic.evalARows logicalWidth publicFits := by
  simp only [evalA, make_rows, PiCCSArithmetic.evalARows,
    PiCCSArithmetic.evalAConstraints,
    NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints,
    PiCCSArithmetic.mainConstraints]

theorem ccs_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (ccs logicalWidth publicFits).batches =
      WitnessProgram.ccsBatches logicalWidth publicFits := by
  simp only [ccs, make_batches, WitnessProgram.ccsBatches]

theorem ccs_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (ccs logicalWidth publicFits).rows =
      PiCCSArithmetic.ccsRows logicalWidth publicFits := by
  simp only [ccs, make_rows, PiCCSArithmetic.ccsRows,
    PiCCSArithmetic.ccsConstraints]

theorem norm_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (norm logicalWidth publicFits).batches =
      WitnessProgram.normBatches logicalWidth publicFits := by
  simp only [norm, make_batches, WitnessProgram.normBatches]

theorem norm_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (norm logicalWidth publicFits).rows =
      PiCCSArithmetic.normRows logicalWidth publicFits := by
  simp only [norm, make_rows, PiCCSArithmetic.normRows,
    PiCCSArithmetic.normConstraints]

theorem finalIdentity_batches (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (finalIdentity logicalWidth publicFits).batches =
      WitnessProgram.finalIdentityBatches logicalWidth publicFits := by
  simp only [finalIdentity, make_batches,
    WitnessProgram.finalIdentityBatches]

theorem finalIdentity_rows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (finalIdentity logicalWidth publicFits).rows =
      PiCCSArithmetic.finalIdentityRows logicalWidth publicFits := by
  simp only [finalIdentity, make_rows, PiCCSArithmetic.finalIdentityRows,
    PiCCSArithmetic.finalIdentityConstraints]

end NightstreamFPrime.Export.Stage1.PiCCSPackets
