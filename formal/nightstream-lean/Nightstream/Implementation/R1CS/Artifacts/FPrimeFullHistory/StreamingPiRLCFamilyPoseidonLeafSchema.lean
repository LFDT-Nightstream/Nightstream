/-!
Wire schema for one relative production PiRLC Poseidon2 selective-row leaf.

Owns inert source-column roles, final-slot roles, source S-box expressions,
and thirteen relative final ports.

Does not own field semantics, row satisfaction, Rust conformance, replay-batch
coverage, recursive orchestration, or permission to remove constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

def supportedSchemaVersion : Nat := 1

inductive RawSourceColumn where
  | externalA (lane : Nat)
  | externalB (lane : Nat)
  | local (offset : Nat)
deriving DecidableEq, Repr

structure RawSourceTerm where
  column : RawSourceColumn
  coefficient : Nat
deriving DecidableEq, Repr

structure RawSourceLinearCombination where
  constant : Nat
  terms : List RawSourceTerm
deriving DecidableEq, Repr

inductive RawExplicitColumn where
  | one
  | selector
deriving DecidableEq, Repr

structure RawExplicitTerm where
  column : RawExplicitColumn
  coefficient : Nat
deriving DecidableEq, Repr

inductive RawSlot where
  | externalA (lane : Nat)
  | externalB (lane : Nat)
  | previousLocal (index : Nat)
  | local (index : Nat)
deriving DecidableEq, Repr

structure RawGeometricRun where
  slot : RawSlot
  initial : Nat
  ratio : Nat
deriving DecidableEq, Repr

structure RawPort where
  explicit : List RawExplicitTerm
  geometric : List RawGeometricRun
deriving DecidableEq, Repr

structure RawStep where
  rowOffset : Nat
  input : RawSourceLinearCombination
  output : RawSourceLinearCombination
deriving DecidableEq, Repr

structure RawRow where
  rowOffset : Nat
  ports : List RawPort
deriving DecidableEq, Repr

structure RawSourceImage where
  lane : Nat
  port : RawPort
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
