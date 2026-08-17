/-!
Schema for the generated normalized PiRLC family-body row-owner ledger.

This file owns only inert artifact data types. It does not validate row
semantics, matrices, assignments, selectors, or lifecycle state.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema

def supportedSchemaVersion : Nat := 1

inductive RawFixedFamily where
  | selectorDomain
  | sharedDomain
  | armDomain
  | oneHot
  | publicPadding
  | privatePadding
  | ringPadding
  deriving DecidableEq, Repr

inductive RawRewriteKind where
  | poseidon2
  | shiftedTernaryCanonical
  | linearDefinition
  deriving DecidableEq, Repr

structure RawFixedRun where
  start : Nat
  length : Nat
  family : RawFixedFamily
  arm : Option Nat
  deriving DecidableEq, Repr

structure RawRetainedRun where
  arm : Nat
  sourceStart : Nat
  length : Nat
  emittedStart : Nat
  deriving DecidableEq, Repr

structure RawRewriteBatch where
  rewriteStart : Nat
  count : Nat
  rewriteStride : Nat
  arm : Nat
  kind : RawRewriteKind
  sourceStart : Nat
  sourceStride : Nat
  sourceWidth : Nat
  emittedStart : Nat
  emittedStride : Nat
  emittedWidth : Nat
  deriving DecidableEq, Repr

structure RawLedger where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  evenSourceRows : Nat
  oddSourceRows : Nat
  rewriteCount : Nat
  evenLinearDefinitionCount : Nat
  oddLinearDefinitionCount : Nat
  fixedRuns : List RawFixedRun
  retainedRuns : List RawRetainedRun
  rewriteBatches : List RawRewriteBatch
  deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema
