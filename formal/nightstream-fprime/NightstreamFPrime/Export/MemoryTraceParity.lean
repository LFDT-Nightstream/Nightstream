import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Nebula.ReplayCheck

/-! Golden traces of the memory-event copy (`Spec.Nebula.MemoryTrace`) with the verdicts of the
copied definitions: the events of each trace, whether they replay (`replay?`), the final values
at some addresses, and whether the events of each row are bounded (`boundedMemoryEvents`). -/

namespace NightstreamFPrime.Export.MemoryTraceParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula.MemoryTrace

def schema : Nat := 1

/-- A golden trace: a catalog, rows as column values, initial cells set in order, and the
addresses whose final values the file reports. -/
structure Case where
  catalog : List MemorySpec
  rows : List (List ℕ)
  initial : List (ℕ × List ℕ × ℕ)
  queries : List (ℕ × List ℕ)

/-- A row from its column values: column `c` holds `values[c] mod q`, and 0 past the list
(Rocq `rowWitness`). -/
def rowWitness (values : List ℕ) : Witness :=
  ⟨fun c => ⟨((values.getD c 0 % fieldModulus : ℕ) : ℤ), Int.natCast_nonneg _,
    by exact_mod_cast Nat.mod_lt _ (Nat.pos_of_ne_zero (NeZero.ne fieldModulus))⟩⟩

/-- The initial state: no cell, then each cell set in order. -/
def stateOf (cells : List (ℕ × List ℕ × ℕ)) : MemoryState :=
  cells.foldl (fun state cell => update state cell.1 (cell.2.1.map Int.ofNat) cell.2.2)
    fun _ _ => none

private def row (entries : List (ℕ × ℕ)) : List ℕ :=
  (List.range 15).map fun c => (entries.lookup c).getD 0

private def word : ℕ := 2 ^ 32

/-- A RAM memory and a ROM memory with id 7. -/
private def catalogA : List MemorySpec :=
  [⟨0, .ram, [⟨[1], 2, .write (some 3), .when 10⟩, ⟨[1], 4, .read, .when 11⟩,
    ⟨[5, 6], 7, .write none, .unless 12⟩]⟩,
   ⟨7, .rom, [⟨[8], 9, .read, .always⟩, ⟨[], 13, .read, .when 14⟩]⟩]

/-- Two memories with one id. -/
private def catalogB : List MemorySpec :=
  [⟨3, .ram, [⟨[0], 1, .write none, .always⟩]⟩, ⟨3, .ram, [⟨[0], 2, .read, .always⟩]⟩]

/-- One column that activates a read when 0 and a write when 1. -/
private def catalogC : List MemorySpec :=
  [⟨2, .ram, [⟨[0], 1, .read, .«unless» 2⟩, ⟨[0], 1, .write none, .when 2⟩]⟩]

/-- The golden traces. Each one varies one rule: the order of events, an activation, a read, a
prior value, a fresh cell, a 32-bit bound, the reduction of a column value, an empty address,
two memories with one address, or two memories with one id. -/
def cases : List Case :=
  let base := [(8, 0), (9, 42), (12, 1)]
  [⟨catalogA, [row ([(1, 3), (2, 100), (3, 50), (10, 1), (4, 100), (11, 1)] ++ base)],
      [(0, [3], 50), (7, [0], 42)], [(0, [3]), (7, [0]), (0, [4])]⟩,
   ⟨catalogA, [row ([(1, 3), (2, 100), (3, 50), (10, 1), (4, 99), (11, 1)] ++ base)],
      [(0, [3], 50), (7, [0], 42)], [(0, [3])]⟩,
   ⟨catalogA, [row ([(1, 3), (2, 100), (3, 51), (10, 1)] ++ base)],
      [(0, [3], 50), (7, [0], 42)], [(0, [3])]⟩,
   ⟨catalogA, [row [(5, 2), (6, 3), (7, word), (10, 2), (8, 0), (9, 42)]],
      [(7, [0], 42)], [(0, [2, 3])]⟩,
   ⟨catalogA, [row [(12, 2), (8, 1), (9, 7)]], [(7, [1], 7)], [(7, [1])]⟩,
   ⟨catalogA, [row ([(1, 5), (2, 9), (3, 0), (10, 1)] ++ [(8, 0), (9, 1), (12, 1)]),
      row ([(1, 5), (4, 9), (11, 1)] ++ [(8, 0), (9, 1), (12, 1)])],
      [(0, [5], 0), (7, [0], 1)], [(0, [5])]⟩,
   ⟨catalogA, [row [(8, 3), (9, 5), (12, 1)]], [], [(7, [3])]⟩,
   ⟨catalogA, [row [(1, 6), (2, 1), (3, 0), (10, 1), (12, 1), (8, 0), (9, 0)]],
      [(7, [0], 0)], [(0, [6])]⟩,
   ⟨catalogA, [row [(8, 0), (9, fieldModulus - 1), (12, 1)]],
      [(7, [0], fieldModulus - 1)], [(7, [0])]⟩,
   ⟨catalogA, [row [(8, 0), (9, fieldModulus), (12, 1)]], [(7, [0], 0)], [(7, [0])]⟩,
   ⟨catalogA, [row [(14, 1), (13, 5), (12, 1), (8, 0), (9, 0)]],
      [(7, [], 5), (7, [0], 0)], [(7, [])]⟩,
   ⟨catalogA, [row [(1, 1), (2, word - 1), (3, word), (10, 1), (12, 1), (8, 0), (9, 0)],
      row [(1, 1), (2, 0), (3, word - 1), (10, 1), (12, 1), (8, 0), (9, 0)]],
      [(0, [1], word), (7, [0], 0)], [(0, [1])]⟩,
   ⟨catalogA, [row [(1, 0), (2, 5), (3, 0), (10, 1), (12, 1), (8, 0), (9, 3)]],
      [(0, [0], 0), (7, [0], 3)], [(0, [0]), (7, [0])]⟩,
   ⟨catalogB, [[4, 8, 8]], [], [(3, [4])]⟩,
   ⟨catalogB, [[4, 8, 7]], [], [(3, [4])]⟩,
   ⟨catalogC, [[1, 4, 0], [1, 6, 1], [1, 6, 0]], [(2, [1], 4)], [(2, [1])]⟩,
   ⟨catalogA, [], [], [(0, [0])]⟩]

private def kindCode : MemoryKind → ℕ
  | .rom => 0
  | .ram => 1

private def portKindValue : PortKind → Value
  | .read => .array [.atom 0]
  | .write none => .array [.atom 1]
  | .write (some c) => .array [.atom 2, .atom c]

private def activationValue : Activation → Value
  | .always => .array [.atom 0]
  | .when c => .array [.atom 1, .atom c]
  | .«unless» c => .array [.atom 2, .atom c]

private def portValue (port : PortSpec) : Value :=
  .array [.array (port.addressColumns.map .atom), .atom port.valueColumn,
    portKindValue port.portKind, activationValue port.activation]

private def memoryValue (m : MemorySpec) : Value :=
  .array [.atom m.memoryId, .atom (kindCode m.memoryKind), .array (m.ports.map portValue)]

/-- An integer as `[0, n]` for `n` and `[1, n]` for `-(n + 1)`. -/
private def int : ℤ → Value
  | .ofNat n => .array [.atom 0, .atom n]
  | .negSucc n => .array [.atom 1, .atom n]

private def addressValue (address : Address) : Value := .array (address.map int)

private def optionValue : Option ℤ → Value
  | none => .array [.atom 0]
  | some v => .array [.atom 1, int v]

private def eventValue : Event → Value
  | .read m address v => .array [.atom 0, .atom m, addressValue address, int v]
  | .write m address v prior => .array [.atom 1, .atom m, addressValue address, int v,
      optionValue prior]

private def cellValue (cell : ℕ × List ℕ × ℕ) : Value :=
  .array [.atom cell.1, .array (cell.2.1.map .atom), .atom cell.2.2]

private def queryValue (query : ℕ × List ℕ) : Value :=
  .array [.atom query.1, .array (query.2.map .atom)]

/-- The input of a trace, then its verdicts: the events, 1 when they replay, the final value at
each query when they replay, and 1 for each row whose events are bounded. -/
def caseValue (c : Case) : Value :=
  let rows := c.rows.map rowWitness
  let events := eventsForTrace c.catalog rows
  let replayed := replay? (stateOf c.initial) events
  .array [
    .array [.array (c.catalog.map memoryValue),
      .array (c.rows.map fun values => .array (values.map .atom)),
      .array (c.initial.map cellValue), .array (c.queries.map queryValue)],
    .array [.array (events.map eventValue),
      .atom (if replayed.isSome then 1 else 0),
      .array (match replayed with
        | some final => c.queries.map fun q => optionValue (final q.1 (q.2.map Int.ofNat))
        | none => []),
      .array (rows.map fun w =>
        .atom (if decide (boundedMemoryEvents (eventsForRow c.catalog w)) then 1 else 0))]]

def parityValue : Value := .array [.atom schema, .array (cases.map caseValue)]

end NightstreamFPrime.Export.MemoryTraceParity
