import Nightstream.Implementation.R1CS.CheckedProgram

/-!
Contract: total decoder for large generated checked programs.

The artifact exporter serializes natural-number tokens into comma-separated
decimal chunks. This module parses every token and reconstructs the exact
`Instruction` tree. Malformed tags, truncated payloads, invalid decimals, or
trailing tokens return `none`; no fallback program is accepted by a theorem
without a separate successful-decoding proof.
-/

namespace Nightstream.Implementation.R1CS.PackedProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

structure Reader where
  tokens : List Nat

def Reader.read : Reader → Option (Nat × Reader)
  | ⟨[]⟩ => none
  | ⟨head :: tail⟩ => some (head, ⟨tail⟩)

def parseChunk (chunk : String) : Option (List Nat) :=
  (chunk.splitOn ",").mapM String.toNat?

def parseChunks (chunks : List String) : Option (List Nat) := do
  let parsed ← chunks.mapM parseChunk
  pure parsed.flatten

def readTerms : Nat → Reader → Option (List (Nat × Nat) × Reader)
  | 0, reader => some ([], reader)
  | count + 1, reader => do
      let (column, reader) ← reader.read
      let (coefficient, reader) ← reader.read
      let (tail, reader) ← readTerms count reader
      pure ((column, coefficient) :: tail, reader)

def readTermList (reader : Reader) : Option (List (Nat × Nat) × Reader) := do
  let (count, reader) ← reader.read
  readTerms count reader

def readRhs (reader : Reader) : Option (Rhs × Reader) := do
  let (tag, reader) ← reader.read
  match tag with
  | 0 => do
      let (terms, reader) ← readTermList reader
      pure (.linear terms, reader)
  | 1 => do
      let (left, reader) ← readTermList reader
      let (right, reader) ← readTermList reader
      pure (.product left right, reader)
  | _ => none

def readRow (reader : Reader) : Option (Row × Reader) := do
  let (a, reader) ← readTermList reader
  let (b, reader) ← readTermList reader
  let (c, reader) ← readTermList reader
  pure (⟨a, b, c⟩, reader)

def readInstruction (reader : Reader) : Option (Instruction × Reader) := do
  let (tag, reader) ← reader.read
  match tag with
  | 0 => do
      let (output, reader) ← reader.read
      let (rhs, reader) ← readRhs reader
      pure (.define ⟨output, rhs⟩, reader)
  | 1 => do
      let (row, reader) ← readRow reader
      pure (.check row, reader)
  | _ => none

def readInstructions (count : Nat) (reader : Reader) :
    Option (List Instruction × Reader) := do
  let (reverse, reader) ← (List.range count).foldl
    (fun state _ => do
      let (reverse, reader) ← state
      let (instruction, reader) ← readInstruction reader
      pure (instruction :: reverse, reader))
    (some ([], reader))
  pure (reverse.reverse, reader)

def decode (chunks : List String) : Option (List Instruction) := do
  let tokens ← parseChunks chunks
  let (count, reader) ← Reader.read ⟨tokens⟩
  let (instructions, reader) ← readInstructions count reader
  if reader.tokens.isEmpty then some instructions else none

end Nightstream.Implementation.R1CS.PackedProgram
