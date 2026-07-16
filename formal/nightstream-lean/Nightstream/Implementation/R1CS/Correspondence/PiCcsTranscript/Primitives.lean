import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachine

/-!
Concrete Poseidon2 transcript primitives used by the production-shaped
`Pi_CCS` verifier schedule.

Assurance tier: executable implementation semantics. This module is stated
without generated rows, columns, fixture values, or a Rust call graph.

Owns: arbitrary raw-field absorption including its length word and the native
eager full-rate normalization, raw field squeezing in four-lane blocks,
extension-field lane pairing, and the exact catch-up digest transition.

Does not own: which `Pi_CCS` messages are authoritative, the order of protocol
phases, SumCheck truth, equality with native Rust or `TranscriptGadget`, R1CS
lowering, cost totals, or permission to remove a row.

Emits constraints: no.

Authority boundary: every response is computed from the input state by the
independently extracted Poseidon2 permutation. No digest or challenge may be
supplied separately as evidence for this machine execution.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_CCS` | raw absorb | `appendRaw` | absorb the field-count word and payload, then eagerly permute an exactly full rate buffer |
| `Pi_CCS` | raw squeeze | `squeezeN` | absorb one, permute, and expose at most four rate lanes per block |
| `Pi_CCS` | extension packing | `pairFields` | pair consecutive base-field responses as `(c0,c1)` |
| `Pi_CCS` | catch-up | `catchup` | one exact digest transition jointly returns state and four lanes |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- One production extension-field value represented by its two base-field
coordinates. This type describes transcript transport, not extension-field
arithmetic. -/
structure Extension where
  c0 : Field
  c1 : Field
deriving DecidableEq

/-- Zero in the two-coordinate transport representation. -/
def Extension.zero : Extension :=
  { c0 := wordField 0, c1 := wordField 0 }

/-- Sequential overwrite absorption. Any necessary permutation is owned by
`absorbElem`, including the full-cursor case. -/
def absorbAll : State -> List Field -> State
  | state, [] => state
  | state, value :: rest => absorbAll (absorbElem state value) rest

/-- Native `absorb_slice` normal form. The production transcript eagerly
permutes when a slice finishes exactly on the full rate boundary. -/
def normalizeFull (state : State) : State :=
  if state.absorbed.val = rate then permute state else state

/-- Element-by-element raw absorption before native slice normalization. This
is named because the constant gadget path stops here, while the native and
variable-slice paths continue through `normalizeFull`. -/
def appendRawLazy (state : State) (fields : List Field) : State :=
  absorbAll state (wordField fields.length :: fields)

/-- Exact native `append_fields_raw(fields)` state semantics: the list length
is the first absorbed field, the payload is absorbed in order, and an exactly
full final buffer is eagerly permuted by `absorb_slice`. -/
def appendRaw (state : State) (fields : List Field) : State :=
  normalizeFull (appendRawLazy state fields)

/-- Canonical list view of a four-lane digest response. -/
def digestFields (lanes : Fin 4 -> Field) : List Field :=
  [lanes ⟨0, by decide⟩, lanes ⟨1, by decide⟩,
   lanes ⟨2, by decide⟩, lanes ⟨3, by decide⟩]

@[simp] theorem digestFields_length (lanes : Fin 4 -> Field) :
    (digestFields lanes).length = 4 := by
  rfl

/-- Execute a fixed number of raw squeeze blocks. Each block is one exact
`digest` transition and contributes all four response lanes. -/
def squeezeBlocks : State -> Nat -> State × List Field
  | state, 0 => (state, [])
  | state, blocks + 1 =>
      let first := digest state
      let rest := squeezeBlocks first.1 blocks
      (rest.1, digestFields first.2 ++ rest.2)

/-- Number of four-lane blocks required to expose `count` fields. -/
def blocksFor (count : Nat) : Nat :=
  (count + 3) / 4

/-- Exact raw `challenge_fields_raw(count)` transition. The final block may
expose fewer than four lanes, but the state still advances by the complete
permutation. -/
def squeezeN (state : State) (count : Nat) : State × List Field :=
  let execution := squeezeBlocks state (blocksFor count)
  (execution.1, execution.2.take count)

/-- Pair consecutive response fields as extension coordinates. An impossible
odd tail is deliberately ignored here; callers prove even response counts at
their typed schedule boundary. -/
def pairFields : List Field -> List Extension
  | first :: second :: rest =>
      { c0 := first, c1 := second } :: pairFields rest
  | _ => []

/-- Flatten extension coordinates in production `(c0,c1)` order. -/
def extensionFields (values : List Extension) : List Field :=
  values.flatMap fun value => [value.c0, value.c1]

/-- One extension response from a two-field squeeze result. The default branch
is unreachable for `squeezeN _ 2` and remains explicit rather than introducing
a prover-controlled option. -/
def firstExtension (fields : List Field) : Extension :=
  { c0 := fields.getD 0 (wordField 0)
    c1 := fields.getD 1 (wordField 0) }

/-- Exact verifier catch-up: one squeeze-domain field, one permutation, four
digest lanes, and their shared successor state. -/
def catchup (state : State) : State × (Fin 4 -> Field) :=
  digest state

@[simp] theorem absorbAll_nil (state : State) :
    absorbAll state [] = state := by
  rfl

@[simp] theorem absorbAll_cons (state : State) (value : Field)
    (rest : List Field) :
    absorbAll state (value :: rest) =
      absorbAll (absorbElem state value) rest := by
  rfl

@[simp] theorem appendRaw_eq_normalizeFull (state : State)
    (fields : List Field) :
    appendRaw state fields = normalizeFull (appendRawLazy state fields) := by
  rfl

@[simp] theorem squeezeBlocks_zero (state : State) :
    squeezeBlocks state 0 = (state, []) := by
  rfl

/-- Every complete raw squeeze block contributes exactly four response fields.
This is a schedule theorem, not a probabilistic statement. -/
theorem squeezeBlocks_fields_length (state : State) (blocks : Nat) :
    (squeezeBlocks state blocks).2.length = 4 * blocks := by
  induction blocks generalizing state with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      simp only [squeezeBlocks, List.length_append, digestFields_length]
      rw [inductionHypothesis]
      omega

@[simp] theorem pairFields_extensionFields (values : List Extension) :
    pairFields (extensionFields values) = values := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp only [extensionFields, List.flatMap_cons]
      change pairFields (value.c0 :: value.c1 :: extensionFields rest) =
        value :: rest
      simp [pairFields, inductionHypothesis]

/-- The catch-up response and successor are inseparable projections of one
deterministic permutation execution. -/
theorem catchup_eq_digest (state : State) :
    catchup state = digest state := by
  rfl

end Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
