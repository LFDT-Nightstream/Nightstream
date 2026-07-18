import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Schema

/-!
Schema for the compact source-to-selective row-ownership artifact.

Owns: exact interval data joining each of the 14 source-R1CS stage leaves to
retained rows or named selective rewrite families.

Does not own: rewrite semantics, final matrix coefficients, column ownership,
selector truth, transcript authority, or row-removal authority.

Emits constraints: no.

| Artifact level | Physical meaning |
|---|---|
| `LoweredFragment` | one retained interval or complete compiler rewrite |
| `LoweredStageLeaf` | all fragments owned by one stable Rust stage path |
| `Artifact` | selected 5,724-source-row cross-branch bundle |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

inductive RewriteKind where
  | polynomialEvaluation
  | productSum
  | linearDefinition
deriving DecidableEq, Repr

inductive Disposition where
  | retained
  | rewrite (id : Nat) (kind : RewriteKind)
deriving DecidableEq, Repr

structure LoweredFragment where
  sourceRows : List RowBlock
  emittedRows : RowBlock
  disposition : Disposition
deriving DecidableEq, Repr

namespace LoweredFragment

def sourceRowCount (fragment : LoweredFragment) : Nat :=
  (fragment.sourceRows.map RowBlock.count).sum

def emittedRowCount (fragment : LoweredFragment) : Nat :=
  fragment.emittedRows.count

def sourceIndices (fragment : LoweredFragment) : List Nat :=
  fragment.sourceRows.flatMap RowBlock.indices

end LoweredFragment

structure LoweredStageLeaf where
  stagePath : String
  sourceRows : List RowBlock
  fragments : List LoweredFragment
deriving DecidableEq, Repr

namespace LoweredStageLeaf

def sourceRowCount (leaf : LoweredStageLeaf) : Nat :=
  (leaf.sourceRows.map RowBlock.count).sum

def emittedRowCount (leaf : LoweredStageLeaf) : Nat :=
  (leaf.fragments.map LoweredFragment.emittedRowCount).sum

def sourceIndices (leaf : LoweredStageLeaf) : List Nat :=
  leaf.sourceRows.flatMap RowBlock.indices

def fragmentSourceIndices (leaf : LoweredStageLeaf) : List Nat :=
  leaf.fragments.flatMap LoweredFragment.sourceIndices

end LoweredStageLeaf

structure Artifact where
  sourceArmRowCount : Nat
  finalRelationRowCount : Nat
  steadyArmRows : RowBlock
  leaves : List LoweredStageLeaf
deriving DecidableEq, Repr

namespace Artifact

def fragments (artifact : Artifact) : List LoweredFragment :=
  artifact.leaves.flatMap LoweredStageLeaf.fragments

def sourceRowCount (artifact : Artifact) : Nat :=
  (artifact.leaves.map LoweredStageLeaf.sourceRowCount).sum

def emittedRowCount (artifact : Artifact) : Nat :=
  (artifact.leaves.map LoweredStageLeaf.emittedRowCount).sum

def emittedIntervals (artifact : Artifact) : List RowBlock :=
  (artifact.fragments.map LoweredFragment.emittedRows).filter fun rows =>
    rows.count != 0

end Artifact

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective
