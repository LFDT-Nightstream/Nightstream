import Mathlib.Data.List.GetD
import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.Poseidon2
import NightstreamFPrime.Spec.Folding.Nifs

/-!
Owns the Stage 1 public-state binding: the canonical self-delimiting
serialization of the Construction-2 hash preimage `(tag, vk, i, z0, zi, U, pc)`
into field words, the state hash `XOut = Poseidon2(preimage)`, the fixed
public-instance encoding of a digest, and the paper default running instance.
Every function is computable (Rust parity surface, spec §11).

The sponge `Poseidon2.hash` does not absorb its input length, so two word
lists that differ only by trailing zeros collide. Every protocol preimage is
therefore a fixed-length tag followed by length-prefixed blocks: the
serialization is injective on well-formed preimages, and a list with extra
trailing zeros is not the serialization of any preimage.
-/

namespace NightstreamFPrime.Lifecycle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction (EvaluationFamily)
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Stage 1 carriers at the key's types. -/
abbrev Running := Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
  (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape
abbrev Fresh := Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
  (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape
abbrev Proof (degreeBound : Nat) :=
  Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape degreeBound
abbrev HashPreimage :=
  HyperNova.Construction2.Paper.HashPreimage KeyDigest AppState
    (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount

/-- Domain-separation tag `HyperNova/NIVC/state/v1`, one byte per word. -/
def stateDomainTag : List F :=
  ([72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 78, 73, 86, 67,
    47, 115, 116, 97, 116, 101, 47, 118, 49] : List Nat).map Poseidon2.ofNat

def natWord (n : Nat) : F := Poseidon2.ofNat n

/-- A length prefix makes every variable-length block self-delimiting. -/
def block (xs : List F) : List F := natWord xs.length :: xs

def serializeK (k : K) : List F := [k.c0, k.c1]
def serializeRingF (a : RingF) : List F := (List.finRange ringDegree).map fun i => a i
def serializeCommitment (c : PaperAlgebra.Commitment) : List F :=
  (List.finRange productionProfile.commitmentWidth).flatMap fun r => serializeRingF (c r)
def serializePublicInput
    (x : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange (FullShape logicalWidth publicFits).publicWidth).map fun j => x j
def serializePoint (p : CubePoint K cubeVariables) : List F :=
  p.coordinates.flatMap serializeK
def serializeEvaluations (e : EvaluationFamily K productionShape) : List F :=
  (List.finRange productionShape.coefficientCount).flatMap
      (fun l => serializeK (e.pad l)) ++
    (List.finRange productionShape.matrixCount).flatMap fun j =>
      (List.finRange productionShape.coefficientCount).flatMap fun l =>
        serializeK (e.matrix j l)

/-- The complete running vector: point, then for each of the 16 slots its
commitment, public input, and evaluation family, each as a block. -/
def serializeRunning (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  block (serializePoint u.point) ++
  ((List.finRange productionShape.runningCount).flatMap fun i =>
    block (serializeCommitment (u.commitments i)) ++
      block (serializePublicInput (publicFits := publicFits) (u.publicInputs i)) ++
      block (serializeEvaluations (u.evaluations i)))

/-- `hEnc(tag, vk, i, z0, zi, U, pc)`; `slotCount = 1`, so the key and running
vectors each have one entry. -/
def serializePreimage (p : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  stateDomainTag ++ block (p.verifierKeys functionIndex) ++ [natWord p.iteration] ++
    block p.z0 ++ block p.current ++ serializeRunning (publicFits := publicFits) (p.running functionIndex) ++
    [natWord p.pc]

theorem stateDomainTag_length : stateDomainTag.length = 23 := by
  simp [stateDomainTag]

@[simp] theorem block_length (xs : List F) :
    (block xs).length = xs.length + 1 := by
  simp [block]

@[simp] theorem serializeK_length (value : K) :
    (serializeK value).length = 2 := by
  rfl

@[simp] theorem serializeRingF_length (value : RingF) :
    (serializeRingF value).length = ringDegree := by
  simp [serializeRingF]

@[simp] theorem serializeCommitment_length (value : PaperAlgebra.Commitment) :
    (serializeCommitment value).length =
      productionProfile.commitmentWidth * ringDegree := by
  simp [serializeCommitment]

@[simp] theorem serializePublicInput_length
    (value : PaperAlgebra.PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits)) :
    (serializePublicInput (publicFits := publicFits) value).length =
      (FullShape logicalWidth publicFits).publicWidth := by
  simp [serializePublicInput]

@[simp] theorem serializePoint_length (value : CubePoint K cubeVariables) :
    (serializePoint value).length = cubeVariables * 2 := by
  simp [serializePoint, value.dimension]

@[simp] theorem serializeEvaluations_length
    (value : EvaluationFamily K productionShape) :
    (serializeEvaluations value).length =
      (productionShape.matrixCount + 1) *
        productionShape.coefficientCount * 2 := by
  simp [serializeEvaluations, Nat.add_mul, Nat.mul_assoc, Nat.add_comm]

theorem serializeRunning_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeRunning (publicFits := publicFits) value).length = 45897 := by
  simp [serializeRunning, productionShape, productionProfile, fullShape,
    publicRingColumns, ringDegree, cubeVariables,
    Phi81Relation.Shape.publicWidth, Phi81MatrixSource.phi81Shape]

/-- Exact static serialization cost. Only the verifier-key and two
application-state block lengths remain parameters. -/
theorem serializePreimage_length
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializePreimage (publicFits := publicFits) value).length =
      45925 + (value.verifierKeys functionIndex).length +
        value.z0.length + value.current.length := by
  simp [serializePreimage, stateDomainTag_length, serializeRunning_length]
  omega

/-- `stateHash`: the public output of one F′ step. -/
def stateHash (p : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) : Digest :=
  Poseidon2.hash (serializePreimage (publicFits := publicFits) p)

/-- One canonical little-endian bit of a Goldilocks word. -/
def digestBitWord (value : F) (bit : Nat) : F :=
  Poseidon2.ofNat ((value.val / 2 ^ bit) % 2)

/-- The four digest words serialized as 256 canonical little-endian bits. -/
def serializeDigestBits (digest : Digest) : List F :=
  (List.range 256).map fun index =>
    digestBitWord (digest.getD (index / 64) 0) (index % 64)

@[simp] theorem serializeDigestBits_length (digest : Digest) :
    (serializeDigestBits digest).length = 256 := by
  simp [serializeDigestBits]

/-- Public position of a bounded natural digest bit after the marker. -/
def digestBitIndexNat (word : Fin 4) (bit : Nat) :
    Fin (ringDegree * publicRingColumns) :=
  ⟨1 + word.val * 64 + bit % 64, by
    have wordBound := word.isLt
    have bitBound : bit % 64 < 64 := Nat.mod_lt _ (by decide)
    norm_num [ringDegree, publicRingColumns] at wordBound bitBound ⊢
    omega⟩

/-- Public position of one digest bit after the leading marker. -/
def digestBitIndex (word : Fin 4) (bit : Fin 64) :
    Fin (ringDegree * publicRingColumns) :=
  digestBitIndexNat (logicalWidth := logicalWidth) word bit.val

/-- Reconstruct one canonical digest word from the public bit encoding. -/
def decodeHashWord
    (input : PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (word : Fin 4) : F :=
  (List.range 64).foldl (fun value bit =>
    value + Poseidon2.ofNat (2 ^ bit) *
      input (digestBitIndexNat (logicalWidth := logicalWidth) word bit)) 0

/-- Recover the four field words used by the transcript from `encHash`. -/
def decodeHash
    (input : PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Digest :=
  List.ofFn (decodeHashWord input)

/-- Pure five-ring-column cell encoding, before it is viewed at one relation
shape. -/
def encodedHashCells (d : Digest) : Fin (ringDegree * publicRingColumns) → F :=
  fun j => (1 :: serializeDigestBits d).getD j.val 0

/-- Fixed public-instance encoding of a digest: `encHash(d) = enc_inst((⊥, d))`,
placed as `[1, bits(d₀) … bits(d₃), 0 …]` in five public ring columns. Its
length is independent of circuit padding (HyperNova Def. 12 prop. 6). -/
def encHash (d : Digest) :
    PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  encodedHashCells d

def encHashMarkerIndex : Fin (ringDegree * publicRingColumns) :=
  ⟨0, by norm_num [ringDegree, publicRingColumns]⟩

theorem encHash_marker (digest : Digest) :
    encHash (publicFits := publicFits) digest encHashMarkerIndex = 1 := by
  rfl

theorem encodedHashCells_marker (digest : Digest) :
    encodedHashCells digest encHashMarkerIndex = 1 := by
  rfl

theorem encodedHashCells_digestBitNat (digest : Digest) (word : Fin 4)
    (bit : Nat) (bounded : bit < 64) :
    encodedHashCells digest
        (digestBitIndexNat (logicalWidth := logicalWidth) word bit) =
      digestBitWord (digest.getD word.val 0) bit := by
  unfold encodedHashCells digestBitIndexNat serializeDigestBits
  simp only [Nat.mod_eq_of_lt bounded]
  have indexBound : word.val * 64 + bit < 256 := by
    have wordBound := word.isLt
    omega
  rw [show 1 + word.val * 64 + bit = (word.val * 64 + bit) + 1 by omega]
  rw [List.getD_cons_succ]
  rw [List.getD_eq_getElem (l := _) (d := 0) (by simpa using indexBound)]
  simp only [List.getElem_map, List.getElem_range]
  congr 2
  · omega
  · omega

theorem encHash_digestBitNat (digest : Digest) (word : Fin 4)
    (bit : Nat) (bounded : bit < 64) :
    encHash (publicFits := publicFits) digest
        (digestBitIndexNat (logicalWidth := logicalWidth) word bit) =
      digestBitWord (digest.getD word.val 0) bit := by
  exact encodedHashCells_digestBitNat digest word bit bounded

theorem encHash_digestBit (digest : Digest) (word : Fin 4) (bit : Fin 64) :
    encHash (publicFits := publicFits) digest
        (digestBitIndex (logicalWidth := logicalWidth) word bit) =
      digestBitWord (digest.getD word.val 0) bit.val := by
  exact encHash_digestBitNat digest word bit.val bit.isLt

theorem encodedHashCells_tail (digest : Digest)
    (index : Fin (ringDegree * publicRingColumns)) (tail : 257 ≤ index.val) :
    encodedHashCells digest index = 0 := by
  unfold encodedHashCells
  apply List.getD_eq_default
  simp only [serializeDigestBits_length, List.length_cons]
  exact tail

theorem encHash_tail (digest : Digest) (index : Fin (ringDegree * publicRingColumns))
    (tail : 257 ≤ index.val) :
    encHash (publicFits := publicFits) digest index = 0 := by
  exact encodedHashCells_tail digest index tail

private theorem digestBitWord_norm (value : F) (bit : Nat) :
    centeredMagnitude (digestBitWord value bit) < 2 := by
  have residueBound : (value.val / 2 ^ bit) % 2 < 2 :=
    Nat.mod_lt _ (by decide)
  have residueCases :
      (value.val / 2 ^ bit) % 2 = 0 ∨
        (value.val / 2 ^ bit) % 2 = 1 := by
    omega
  rcases residueCases with residue | residue
  · simp [digestBitWord, residue, Poseidon2.ofNat, centeredMagnitude]
  · simp [digestBitWord, residue, Poseidon2.ofNat, centeredMagnitude,
      goldilocksModulus]

/-- Every coordinate of the fixed recursive public input satisfies the exact
fresh-opening norm. -/
theorem encHash_norm (digest : Digest)
    (column : Fin (ringDegree * publicRingColumns)) :
    centeredMagnitude (encHash (publicFits := publicFits) digest column) < 2 := by
  obtain ⟨index, indexBound⟩ := column
  cases index with
  | zero =>
      change centeredMagnitude 1 < 2
      decide
  | succ index =>
      by_cases bitRegion : index < 256
      · unfold encHash encodedHashCells
        rw [List.getD_cons_succ]
        rw [List.getD_eq_getElem (l := serializeDigestBits digest) (d := 0)
          (by simpa using bitRegion)]
        simp only [serializeDigestBits, List.getElem_map, List.getElem_range]
        exact digestBitWord_norm _ _
      · let tailColumn : Fin (ringDegree * publicRingColumns) :=
          ⟨index + 1, indexBound⟩
        change centeredMagnitude
          (encHash (publicFits := publicFits) digest tailColumn) < 2
        rw [encHash_tail digest tailColumn (by
          change 257 ≤ index + 1
          omega)]
        decide

private def digestBitNat (value bit : Nat) : Nat :=
  value / 2 ^ bit % 2

private def reconstructBits (value count : Nat) : Nat :=
  (List.range count).foldl (fun total bit =>
    total + 2 ^ bit * digestBitNat value bit) 0

private theorem reconstructBits_succ (value count : Nat) :
    reconstructBits value (count + 1) =
      reconstructBits value count + 2 ^ count * digestBitNat value count := by
  simp [reconstructBits, List.range_succ, List.foldl_append]

private theorem reconstructBits_eq_mod (value : Nat) :
    ∀ count, reconstructBits value count = value % 2 ^ count
  | 0 => by simp [reconstructBits, Nat.mod_one]
  | count + 1 => by
      rw [reconstructBits_succ, reconstructBits_eq_mod value count,
        Nat.mod_pow_succ]
      simp [digestBitNat]

private theorem ofNat_add (left right : Nat) :
    Poseidon2.ofNat left + Poseidon2.ofNat right =
      Poseidon2.ofNat (left + right) := by
  apply Fin.eq_of_val_eq
  simp [Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

private theorem ofNat_mul (left right : Nat) :
    Poseidon2.ofNat left * Poseidon2.ofNat right =
      Poseidon2.ofNat (left * right) := by
  apply Fin.eq_of_val_eq
  simp [Poseidon2.ofNat, Fin.val_mul, Nat.mul_mod]

private def reconstructField (value : F) (count : Nat) : F :=
  (List.range count).foldl (fun total bit =>
    total + Poseidon2.ofNat (2 ^ bit) * digestBitWord value bit) 0

private theorem reconstructField_succ (value : F) (count : Nat) :
    reconstructField value (count + 1) =
      reconstructField value count +
        Poseidon2.ofNat (2 ^ count) * digestBitWord value count := by
  simp [reconstructField, List.range_succ, List.foldl_append]

private theorem reconstructField_eq (value : F) :
    ∀ count,
      reconstructField value count =
        Poseidon2.ofNat (reconstructBits value.val count)
  | 0 => by rfl
  | count + 1 => by
      rw [reconstructField_succ, reconstructBits_succ,
        reconstructField_eq value count]
      unfold digestBitWord digestBitNat
      rw [ofNat_mul, ofNat_add]

private theorem foldl_congr_mem
    {α β : Type} (items : List β) (left right : α → β → α)
    (initial : α)
    (equalStep : ∀ accumulator item, item ∈ items →
      left accumulator item = right accumulator item) :
    items.foldl left initial = items.foldl right initial := by
  induction items generalizing initial with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.foldl_cons, List.foldl_cons,
        equalStep initial item (by simp)]
      apply inductionHypothesis
      intro accumulator current member
      exact equalStep accumulator current (by simp [member])

theorem decodeHashWord_encHash (digest : Digest) (word : Fin 4) :
    decodeHashWord (publicFits := publicFits)
        (encHash (publicFits := publicFits) digest) word =
      digest.getD word.val 0 := by
  have encodedFold :
      decodeHashWord (publicFits := publicFits)
          (encHash (publicFits := publicFits) digest) word =
        reconstructField (digest.getD word.val 0) 64 := by
    unfold decodeHashWord reconstructField
    apply foldl_congr_mem
    intro accumulator bit member
    rw [encHash_digestBitNat digest word bit (List.mem_range.mp member)]
  rw [encodedFold, reconstructField_eq, reconstructBits_eq_mod]
  have valueBound : (digest.getD word.val 0).val < 2 ^ 64 := by
    exact lt_trans (digest.getD word.val 0).isLt (by
      norm_num [goldilocksModulus])
  rw [Nat.mod_eq_of_lt valueBound]
  apply Fin.eq_of_val_eq
  change (digest.getD word.val 0).val % goldilocksModulus =
    (digest.getD word.val 0).val
  exact Nat.mod_eq_of_lt (digest.getD word.val 0).isLt

theorem decodeHash_encHash (digest : Digest) (fixed : digest.length = 4) :
    decodeHash (publicFits := publicFits)
        (encHash (publicFits := publicFits) digest) = digest := by
  apply List.ext_get
  · simp [decodeHash, fixed]
  · intro index leftBound rightBound
    simp only [decodeHash]
    rw [List.get_ofFn]
    rw [decodeHashWord_encHash]
    exact List.getD_eq_getElem (l := digest) (d := 0) rightBound

/-- The fixed public-instance encoding is injective on canonical four-word
digests. -/
theorem encHash_injective_fixed {left right : Digest}
    (leftFixed : left.length = 4) (rightFixed : right.length = 4)
    (equal : encHash (publicFits := publicFits) left =
      encHash (publicFits := publicFits) right) :
    left = right := by
  rw [← decodeHash_encHash left leftFixed,
    ← decodeHash_encHash right rightFixed, equal]

/-- The paper default CE instance `u_⊥` (HyperNova H.2): the commitment of the
zero assignment with zero randomness, the zero public input (the projection of
the zero assignment), the zero point, and zero evaluations. -/
def zeroPoint : CubePoint K cubeVariables :=
  ⟨List.replicate cubeVariables K.zero, by simp⟩

def defaultRunning : Running (logicalWidth := logicalWidth) (publicFits := publicFits) where
  point := zeroPoint
  commitments := fun _ _ => ringFZero
  publicInputs := fun _ _ => 0
  evaluations := fun _ => {
    pad := fun _ => K.zero
    matrix := fun _ _ => K.zero
  }

end

end NightstreamFPrime.Lifecycle
