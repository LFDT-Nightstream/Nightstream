import Nightstream.Implementation.R1CS.Core.Program

/-!
Exact row schema for the bounded first-accepted tail of one recursive-profile
`Pi_RLC` coefficient sampler.

Owns: the six-row enough-accepts family, the checked zero prefix, the 54
position-indexed 11-candidate selection families, and their one-hot,
product, and binding subfamilies.

Does not own: proof that the tail returns the mathematical first 54 accepted
symbols, chunk classification, transcript candidates, production placement,
Rust conformance, aggregate product replacements, or whole-circuit costs.

Emits constraints: no. This file gives an existing template an auditable tree.

Authority boundary: selector witnesses are not authoritative. Their Boolean,
one-hot, accepted, prefix-count, and source-symbol bindings must all hold before
an output can refine the verifier-owned first-accepted function.

| Protocol | Phase | Constraint family | Multiplicity | Rows |
|---|---|---|---:|---:|
| `Pi_RLC` | sampler/acceptance bound | four-bit slack | 1 | 6 |
| `Pi_RLC` | sampler/selection init | zero prefix | 1 | 1 |
| `Pi_RLC` | sampler/selection | Boolean one-hot window | 54 | `11 + 1` each |
| `Pi_RLC` | sampler/selection | symbol/accept/prefix products | 54 | `11 * 3` each |
| `Pi_RLC` | sampler/selection | accept/prefix/symbol bindings | 54 | 3 each |
| `Pi_RLC` | sampler/tail | all families | 1 | 2,599 |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.SelectionRows

open Nightstream.Implementation.R1CS

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

def candidateCount : Nat := 64
def outputCount : Nat := 54
def selectionWindow : Nat := candidateCount - outputCount + 1

def acceptCol (candidate : Nat) : Nat := 1 + candidate
def symbolCol (candidate : Nat) : Nat := 65 + candidate
def cumulativeCol (candidate : Nat) : Nat := 129 + candidate

def slackCol : Nat := 193
def slackBitCol (offset : Nat) : Nat := 194 + offset
def zeroPrefixCol : Nat := 198

def selectionBase (position : Nat) : Nat := 199 + 45 * position
def selectorCol (position offset : Nat) : Nat := selectionBase position + offset
def symbolProductCol (position offset : Nat) : Nat :=
  selectionBase position + 11 + 3 * offset
def acceptProductCol (position offset : Nat) : Nat :=
  symbolProductCol position offset + 1
def prefixProductCol (position offset : Nat) : Nat :=
  symbolProductCol position offset + 2
def outputCol (position : Nat) : Nat := selectionBase position + 44

/-- Prefix count immediately before a candidate. -/
def prefixCol (candidate : Nat) : Nat :=
  if candidate = 0 then zeroPrefixCol else cumulativeCol (candidate - 1)

def zeroEqualityRow (terms : List (Nat × Nat)) : Row :=
  ⟨terms, [(0, 1)], []⟩

def slackRecompositionRow : Row :=
  zeroEqualityRow
    ([(slackCol, 1)] ++
      (List.range 4).map fun offset =>
        (slackBitCol offset, goldilocksP - 2 ^ offset))

def acceptedCountRow : Row :=
  zeroEqualityRow
    [(cumulativeCol 63, 1),
     (slackCol, goldilocksP - 1),
     (0, goldilocksP - outputCount)]

def acceptanceBoundRows : List Row :=
  (List.range 4).map (fun offset => bitRow (slackBitCol offset)) ++
    [slackRecompositionRow, acceptedCountRow]

def zeroPrefixRow : Row := zeroEqualityRow [(zeroPrefixCol, 1)]

def oneHotRows (position : Nat) : List Row :=
  (List.range selectionWindow).map
      (fun offset => bitRow (selectorCol position offset)) ++
    [zeroEqualityRow
      ((List.range selectionWindow).map
          (fun offset => (selectorCol position offset, 1)) ++
        [(0, goldilocksP - 1)])]

def productRowsAt (position offset : Nat) : List Row :=
  let candidate := position + offset
  [ ⟨[(selectorCol position offset, 1)],
      [(symbolCol candidate, 1)],
      [(symbolProductCol position offset, 1)]⟩,
    ⟨[(selectorCol position offset, 1)],
      [(acceptCol candidate, 1)],
      [(acceptProductCol position offset, 1)]⟩,
    ⟨[(selectorCol position offset, 1)],
      [(prefixCol candidate, 1)],
      [(prefixProductCol position offset, 1)]⟩ ]

def productRows (position : Nat) : List Row :=
  (List.range selectionWindow).flatMap (productRowsAt position)

def acceptBindingRow (position : Nat) : Row :=
  zeroEqualityRow
    ((List.range selectionWindow).map
        (fun offset => (acceptProductCol position offset, 1)) ++
      [(0, goldilocksP - 1)])

def prefixBindingRow (position : Nat) : Row :=
  zeroEqualityRow
    ((List.range selectionWindow).map
        (fun offset => (prefixProductCol position offset, 1)) ++
      if position = 0 then [] else [(0, goldilocksP - position)])

def symbolBindingRow (position : Nat) : Row :=
  zeroEqualityRow
    ([(outputCol position, 1)] ++
      (List.range selectionWindow).map fun offset =>
        (symbolProductCol position offset, goldilocksP - 1))

def bindingRows (position : Nat) : List Row :=
  [acceptBindingRow position, prefixBindingRow position,
    symbolBindingRow position]

/-- Constraint-family tree for one requested output position. -/
def selectionRows (position : Nat) : List Row :=
  oneHotRows position ++ productRows position ++ bindingRows position

/-- Phase tree for the complete fixed 54-of-64 tail. -/
def rows : List Row :=
  acceptanceBoundRows ++ [zeroPrefixRow] ++
    (List.range outputCount).flatMap selectionRows

theorem parameter_values :
    candidateCount = 64 /\ outputCount = 54 /\ selectionWindow = 11 := by
  decide

theorem acceptanceBoundRows_length : acceptanceBoundRows.length = 6 := by
  decide

theorem selectionRows_length (position : Nat) :
    (selectionRows position).length = 48 := by
  simp [selectionRows, oneHotRows, productRows, productRowsAt, bindingRows,
    selectionWindow, candidateCount, outputCount]
  decide

theorem rows_length : rows.length = 2599 := by
  decide

/-- The highest referenced local column is the final output wire. This also
records that the ownership map's larger allocation range is not itself a
constraint or a semantic obligation. -/
theorem final_output_column : outputCol 53 = 2628 := by
  decide

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.SelectionRows
