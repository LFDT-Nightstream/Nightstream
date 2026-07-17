import Nightstream.SuperNeo.Concrete.Algebra

/-!
Diagnostic fixed-three-row message semantics for the `Pi_CCS` output digest
consumed before `Pi_RLC` challenge derivation.

Assurance tier: model-level (diagnostic profile); not the active 13-matrix
relation, not Rust-conformant, and not security-reduced. This module fixes the
legacy typed output surface and its field serialization without importing
generated R1CS rows, Rust emitters, profiler totals, seeded-matrix artifacts,
or carried digest columns.

Owns: both ASCII domain tags; seven-byte little-endian packing; the exact
15-output terminal shape; the three active `y_ring` rows; the active
`y_zcol` row; quadratic-extension limb order; and exact serialized lengths.

Does not own: whether production columns instantiate these messages; the SIS
map, its seed expansion, the final Poseidon2 envelope, transcript placement,
cryptographic binding, the active matrix-indexed output serialization, row
necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: an `OutputMessage` contains verifier-constrained
evaluation values. A digest becomes authoritative only after later theorems
prove both that accepted `Pi_CCS` rows determine these values and that the
complete SIS/Poseidon pipeline recomputes its four output fields. No active
module may infer a 13-row serialization from this three-row fixture.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_CCS` | output projection | `OutputMessage` | legacy three by 54 `K` evaluations plus one 54-element `K` sidecar |
| `Pi_CCS` | output serialization | outer tag/count | exact `pi_ccs_outputs_digest/v2` packing and output count |
| `Pi_CCS` | per-output serialization | inner tag/shapes | exact `pi_ccs_output_message_digest/v2`, row counts, and active widths |
| `Pi_CCS` | limb serialization | `K = F[u]/(u^2-7)` | `c0` then `c1`, with no hidden or padded lanes |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics

open Nightstream.SuperNeo.Concrete

/-- Production terminal output count: one output for each of the fifteen
fresh/running `Pi_CCS` inputs in the fixed terminal profile. -/
def outputCount : Nat := 15

/-- Number of prover-visible `y_ring` evaluation rows per output. -/
def yRingRows : Nat := 3

/-- Active cyclotomic coefficient width. -/
def activeWidth : Nat := ringDegree

/-- Exact outer domain bytes, stated independently of Rust string storage. -/
def outputsDomainBytes : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97,
   110, 47, 112, 105, 95, 99, 99, 115, 95, 111, 117, 116, 112,
   117, 116, 115, 95, 100, 105, 103, 101, 115, 116, 47, 118, 50]

/-- Exact per-output domain bytes, stated independently of Rust strings. -/
def outputMessageDomainBytes : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97,
   110, 47, 112, 105, 95, 99, 99, 115, 95, 111, 117, 116, 112,
   117, 116, 95, 109, 101, 115, 115, 97, 103, 101, 95, 100, 105,
   103, 101, 115, 116, 47, 118, 50]

/-- One zero-padded seven-byte little-endian word. -/
def packSevenAt (bytes : List Nat) (start : Nat) : Nat :=
  bytes.getD start 0 +
    256 * bytes.getD (start + 1) 0 +
    256 ^ 2 * bytes.getD (start + 2) 0 +
    256 ^ 3 * bytes.getD (start + 3) 0 +
    256 ^ 4 * bytes.getD (start + 4) 0 +
    256 ^ 5 * bytes.getD (start + 5) 0 +
    256 ^ 6 * bytes.getD (start + 6) 0

/-- Rust's `pack_bytes_as_fields`: byte length followed by seven-byte words. -/
def packBytesAsNats (bytes : List Nat) : List Nat :=
  bytes.length ::
    (List.range ((bytes.length + 6) / 7)).map
      (fun chunk => packSevenAt bytes (7 * chunk))

/-- Canonical conversion into the production Goldilocks base field. -/
def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def packBytesAsFields (bytes : List Nat) : List F :=
  (packBytesAsNats bytes).map fieldOfNat

/-- Closed checks of both independently written byte strings and packing. -/
theorem outputsDomainTag_eq :
    packBytesAsNats outputsDomainBytes =
      [39, 30521782141150574, 31069335676202596,
       32478900775383087, 32780223149076319,
       32481117145948019, 846606196] := by
  decide

theorem outputMessageDomainTag_eq :
    packBytesAsNats outputMessageDomainBytes =
      [46, 30521782141150574, 31069335676202596,
       32478900775383087, 32780223149076319,
       29099071086357855, 32481117145948005, 846606196] := by
  decide

/-- Only the active fields that remain prover-selected after the verifier has
reconstructed the rest of a `Pi_CCS` output CE claim. -/
structure OutputMessage where
  yRing : Fin yRingRows -> Fin activeWidth -> K
  yZcol : Fin activeWidth -> K

def extensionFields (value : K) : List F := [value.c0, value.c1]

/-- Length-prefixed active `K` vector, with `c0,c1` limb order. -/
def serializeKVector (values : Fin activeWidth -> K) : List F :=
  fieldOfNat activeWidth ::
    (List.ofFn values).flatMap extensionFields

@[simp] theorem serializeKVector_length (values : Fin activeWidth -> K) :
    (serializeKVector values).length = 109 := by
  simp [serializeKVector, extensionFields, activeWidth, ringDegree]

/-- Exact active projection of one output CE message. -/
def serializeOutput (output : OutputMessage) : List F :=
  packBytesAsFields outputMessageDomainBytes ++
    [fieldOfNat yRingRows] ++
    serializeKVector (output.yRing ⟨0, by decide⟩) ++
    serializeKVector (output.yRing ⟨1, by decide⟩) ++
    serializeKVector (output.yRing ⟨2, by decide⟩) ++
    serializeKVector output.yZcol

@[simp] theorem serializeOutput_length (output : OutputMessage) :
    (serializeOutput output).length = 445 := by
  rw [serializeOutput]
  simp [packBytesAsFields, outputMessageDomainTag_eq,
    yRingRows, serializeKVector_length]

/-- Full ordered `Pi_CCS` output-message preimage, before SIS compression. -/
def serializeOutputs (outputs : List OutputMessage) : List F :=
  packBytesAsFields outputsDomainBytes ++
    [fieldOfNat outputs.length] ++ outputs.flatMap serializeOutput

@[simp] theorem serializeOutputs_length (outputs : List OutputMessage) :
    (serializeOutputs outputs).length = 8 + 445 * outputs.length := by
  have constantSum : (outputs.map (fun _ => 445)).sum =
      445 * outputs.length := by
    induction outputs with
    | nil => simp
    | cons output rest inductionHypothesis =>
        simp [inductionHypothesis, Nat.mul_succ, Nat.add_comm]
  rw [serializeOutputs]
  simp [packBytesAsFields, outputsDomainTag_eq, List.length_flatMap,
    serializeOutput_length]
  rw [constantSum]
  omega

/-- Fixed terminal serialization, while retaining a typed finite output
index instead of accepting a caller-provided list length. -/
def serializeTerminalOutputs
    (outputs : Fin outputCount -> OutputMessage) : List F :=
  serializeOutputs (List.ofFn outputs)

theorem serializeTerminalOutputs_length
    (outputs : Fin outputCount -> OutputMessage) :
    (serializeTerminalOutputs outputs).length = 6683 := by
  simp [serializeTerminalOutputs, outputCount]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Semantics
