import Nightstream.Implementation.Rust.PiCcsExecution.Receipt

/-!
Transport data for one deterministic production NIFS execution.

The receipt contains the complete public claim surfaces for `Pi_RLC` and
`Pi_DEC`, plus the exact `Pi_CCS` statement and SumCheck proof bytes.  The
generated fixture is data only.  The independent checkers own all acceptance
decisions.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.NifsProductionGolden

open Nightstream.Implementation.Rust.PiCcsExecution
open Nightstream.SuperNeo.Concrete

structure RawTranscriptSnapshot where
  lanes : List Nat
  absorbed : Nat
deriving Repr, DecidableEq

structure RawPermutationTrace where
  states : List (List Nat)
deriving Repr, DecidableEq

structure RawCommitment where
  degree : Nat
  verifierRows : Nat
  data : List Nat
deriving Repr, DecidableEq

/-- Rust `CeClaim` fields in canonical serialization order.  `evaluations`
contains four padded 64-lane rows in matrix-major order. -/
structure RawClaim where
  commitment : RawCommitment
  publicRows : Nat
  publicColumns : Nat
  publicInput : List Nat
  point : List RawK
  evaluations : List RawK
  constantTerms : List RawK
  publicWidth : Nat
  foldDigest : List Nat
  advPresent : Bool
deriving Repr, DecidableEq

structure ProductionReceipt where
  relationId : List Nat
  relationMatrices : List Nat
  fixtureAssignment : List Nat
  piCcsStatement : PiCcsCanonicalStatement
  piCcsProof : PiCcsExecutionProof
  poseidonPermutationTraces : List RawPermutationTrace
  piCcsPermutationCount : Nat
  rhoStartPermutationCount : Nat
  piCcsOutputsDigest : List Nat
  rhoStart : RawTranscriptSnapshot
  piRlcInputs : List RawClaim
  piRlcCombined : RawClaim
  piDecChildren : List RawClaim
  canonicalNifsProofByteCount : Nat
deriving Repr, DecidableEq

def canonicalFields (values : List Nat) : Bool :=
  values.all fun value => decide (value < goldilocksModulus)

def canonicalKValues (values : List RawK) : Bool :=
  values.all RawK.wellFormed

def rawKIsZero (value : RawK) : Bool :=
  decide (value.low = 0) && decide (value.high = 0)

def claimShapeCheck (claim : RawClaim) : Bool :=
  decide (claim.commitment.degree = 54) &&
    decide (claim.commitment.verifierRows = 18) &&
    decide (claim.commitment.data.length = 18 * 54) &&
    canonicalFields claim.commitment.data &&
    decide (claim.publicRows = 54) &&
    decide (claim.publicColumns = 1) &&
    decide (claim.publicInput.length = 54) &&
    canonicalFields claim.publicInput &&
    decide (claim.point.length = 6) &&
    canonicalKValues claim.point &&
    decide (claim.evaluations.length = 4 * 64) &&
    canonicalKValues claim.evaluations &&
    ((List.range 4).all fun matrix =>
      (List.range 10).all fun padding =>
        rawKIsZero
          (claim.evaluations.getD (matrix * 64 + 54 + padding) default)) &&
    decide (claim.constantTerms.length = 4) &&
    canonicalKValues claim.constantTerms &&
    ((List.range 4).all fun matrix =>
      decide (claim.constantTerms.getD matrix default =
        claim.evaluations.getD (matrix * 64) default)) &&
    decide (claim.publicWidth = 54) &&
    decide (claim.foldDigest.length = 4) &&
    canonicalFields claim.foldDigest &&
    decide (claim.advPresent = false)

def rhoStartShapeCheck (snapshot : RawTranscriptSnapshot) : Bool :=
  decide (snapshot.lanes.length = 8) &&
    canonicalFields snapshot.lanes &&
    decide (snapshot.absorbed <= 4)

/-- Fixed production relation and fixture fields. -/
def relationShapeCheck (receipt : ProductionReceipt) : Bool :=
  decide (receipt.relationId.length = 4) &&
    canonicalFields receipt.relationId &&
    decide (receipt.piCcsStatement.relationId = receipt.relationId) &&
    decide (receipt.relationMatrices.length = 3 * 54) &&
    canonicalFields receipt.relationMatrices &&
    decide (receipt.fixtureAssignment.length = 54) &&
    canonicalFields receipt.fixtureAssignment

def permutationTraceShapeCheck (trace : RawPermutationTrace) : Bool :=
  decide (trace.states.length = 31) &&
    trace.states.all fun state =>
      decide (state.length = 8) && canonicalFields state

def poseidonTraceShapeCheck (receipt : ProductionReceipt) : Bool :=
  decide (receipt.poseidonPermutationTraces.length = 56) &&
    decide (receipt.piCcsPermutationCount = 44) &&
    decide (receipt.rhoStartPermutationCount = 48) &&
    receipt.poseidonPermutationTraces.all permutationTraceShapeCheck

/-- Exact `Pi_CCS -> Pi_RLC` receipt surface. -/
def piRlcShapeCheck (receipt : ProductionReceipt) : Bool :=
  decide (receipt.piCcsOutputsDigest.length = 4) &&
    canonicalFields receipt.piCcsOutputsDigest &&
    rhoStartShapeCheck receipt.rhoStart &&
    decide (receipt.piRlcInputs.length = 1) &&
    receipt.piRlcInputs.all claimShapeCheck &&
    claimShapeCheck receipt.piRlcCombined &&
    (receipt.piRlcInputs.all fun input =>
      decide (input.foldDigest = receipt.piRlcCombined.foldDigest))

/-- Exact `Pi_RLC -> Pi_DEC` receipt surface. -/
def piDecShapeCheck (receipt : ProductionReceipt) : Bool :=
  claimShapeCheck receipt.piRlcCombined &&
    decide (receipt.piDecChildren.length = 14) &&
    receipt.piDecChildren.all claimShapeCheck &&
    (receipt.piDecChildren.all fun child =>
      decide (child.foldDigest = receipt.piRlcCombined.foldDigest))

/-- Cross-phase checks that do not duplicate a paper equation. -/
def receiptShapeCheck (receipt : ProductionReceipt) : Bool :=
  relationShapeCheck receipt &&
    poseidonTraceShapeCheck receipt &&
    piRlcShapeCheck receipt &&
    piDecShapeCheck receipt &&
    decide (receipt.canonicalNifsProofByteCount = 202205)

end Nightstream.Implementation.Rust.NifsProductionGolden
