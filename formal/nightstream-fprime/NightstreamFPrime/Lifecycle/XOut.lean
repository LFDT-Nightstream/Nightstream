import Mathlib.Data.List.GetD
import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec.FlatMap
import NightstreamFPrime.Spec.Poseidon2
import NightstreamFPrime.Spec.Folding.Nifs
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-!
Owns the Stage 1 public-state binding: the Construction-2 hash preimage
`(vk, i, z0, zi, U)` as field words, the state hash
`XOut = Poseidon2(preimage)`, the fixed public-instance encoding of a digest,
and the paper default running instance. Every function is computable (Rust
parity surface, spec §11).

The preimage starts with one constant domain chunk. Then it lists the running
commitments, `Eval_K`, `Eval_A` and the point, each child-major, then the Π_DEC
parent public input once, packed three coordinates per word, then the tail
`vk, i, z0, zi`. All widths are fixed by the production profile, so the
preimage needs no length prefixes. It omits `pc`; Stage 1 has one function and
fixes `pc = 1`. The children's public inputs are the verifier-computed split of
the parent (SuperNeo Π_DEC verifier step 2), so the packed parent determines
them on every canonical state (`StateEncoding.serializePreimage_injective`).
The committed state columns are exactly these words.

The sponge `Poseidon2.hash` does not absorb its input length, so two word
lists that differ only by trailing zeros collide. Injectivity therefore holds
only on preimages of the fixed production width
(`PilotProduction.FixedPreimage`).
-/

namespace NightstreamFPrime.Lifecycle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction (EvaluationFamily)
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

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

/-- Domain-separation tag `HyperNova/NIVC/state/v2` as ASCII bytes. -/
def stateDomainBytes : List Nat :=
  [72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 78, 73, 86, 67,
    47, 115, 116, 97, 116, 101, 47, 118, 50]

/-- Little-endian bytes as one word. Eight ASCII bytes stay below `p`. -/
def packBytes (bytes : List Nat) : F :=
  Poseidon2.ofNat (bytes.foldr (fun byte rest => byte + 256 * rest) 0)

/-- The first sponge chunk: the domain tag, eight bytes per word, then zero
words up to one full rate chunk. Every word is a constant. -/
def stateDomainChunk : List F :=
  (List.range Poseidon2.rate).map fun word =>
    packBytes ((stateDomainBytes.drop (8 * word)).take 8)

def natWord (n : Nat) : F := Poseidon2.ofNat n

/-- A length prefix makes one variable-length transcript block
self-delimiting. The state layouts have fixed widths and use none. -/
def block (xs : List F) : List F := natWord xs.length :: xs

@[simp] theorem block_length (xs : List F) :
    (block xs).length = xs.length + 1 := by
  simp [block]

def serializeK (k : K) : List F := [k.c0, k.c1]
def serializeRingF (a : RingF) : List F := (List.finRange ringDegree).map fun i => a i
def serializeCommitment (c : PaperAlgebra.Commitment) : List F :=
  (List.finRange productionProfile.commitmentWidth).flatMap fun r => serializeRingF (c r)
def serializePublicInput
    (x : PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange (FullShape logicalWidth publicFits).publicWidth).map fun j => x j
def serializePoint (p : CubePoint K cubeVariables) : List F :=
  p.coordinates.flatMap serializeK

/-- The `Eval_K` words of one evaluation family. -/
def serializeEvalK (e : EvaluationFamily K productionShape) : List F :=
  (List.finRange productionShape.coefficientCount).flatMap fun l => serializeK (e.pad l)

/-- The `Eval_A` words of one evaluation family, matrix-major. -/
def serializeEvalA (e : EvaluationFamily K productionShape) : List F :=
  (List.finRange productionShape.matrixCount).flatMap fun j =>
    (List.finRange productionShape.coefficientCount).flatMap fun l =>
      serializeK (e.matrix j l)

/-- `Eval_K` then `Eval_A` of one family: the PiCCS output proof-input order. -/
def serializeEvaluations (e : EvaluationFamily K productionShape) : List F :=
  serializeEvalK e ++ serializeEvalA e

/-- Running sections, each child-major. -/
def serializeCommitments
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange productionShape.runningCount).flatMap fun source =>
    serializeCommitment (u.commitments source)

def serializeEvalKs
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange productionShape.runningCount).flatMap fun source =>
    serializeEvalK (u.evaluations source)

def serializeEvalAs
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange productionShape.runningCount).flatMap fun source =>
    serializeEvalA (u.evaluations source)

def serializeChildPublicInputs
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange productionShape.runningCount).flatMap fun source =>
    serializePublicInput (publicFits := publicFits) (u.publicInputs source)

theorem runningCount_eq_radixChildCount :
    productionShape.runningCount = productionGlobalParams.k := by
  rfl

/-- The Π_DEC parent public input `Σ_j 2^j x_j` of the sixteen children. -/
def parentPublic
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : F :=
  Spec.Phi81Relation.PiDECAlgebra.Radix.recomposeScalar fun child =>
    u.publicInputs (Fin.cast runningCount_eq_radixChildCount.symm child) column

/-- The sixteen child digits of one parent coordinate. -/
def childDigits
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Radix.ChildIndex → F :=
  fun child => u.publicInputs (Fin.cast runningCount_eq_radixChildCount.symm child) column

/-- Every parent coordinate splits into common-sign digits: the Π_DEC
accepted public-input language. -/
def ChildrenCanonical
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  ∀ column, ∃ sign, Radix.UniformSignedDigits.ConstraintPredicate sign (childDigits u column)

theorem ChildrenCanonical.accepted
    {u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (canonical : ChildrenCanonical u)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    ∃ sign, Radix.UniformSignedDigits.Accepted (parentPublic u column) sign
      (childDigits u column) := by
  rcases canonical column with ⟨sign, constraint⟩
  exact ⟨sign, constraint, rfl⟩

theorem ChildrenCanonical.parentBounded
    {u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (canonical : ChildrenCanonical u)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    centeredMagnitude (parentPublic u column) < 2 ^ 16 := by
  rcases canonical.accepted column with ⟨sign, accepted⟩
  have bounded := accepted.parentBounded
  rw [Radix.production_parameters.2.2] at bounded
  exact bounded

/-- Canonical children are the production split of their parent. -/
theorem ChildrenCanonical.childDigits_eq
    {u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (canonical : ChildrenCanonical u)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    childDigits u column = Radix.splitScalar (parentPublic u column) := by
  rcases canonical.accepted column with ⟨_, accepted⟩
  exact accepted.digits_eq_splitScalar

/-- Three parent coordinates share one word in radix `2^17`. Every accepted
coordinate has centered magnitude below `2^16`, so the radix is injective and
every packed word has magnitude below `2^51`. -/
def packRadix : F := Poseidon2.ofNat (2 ^ 17)

def packWord (low middle high : F) : F :=
  low + packRadix * middle + packRadix * packRadix * high

def packedParentWords : Nat := 90

theorem publicWidth_eq :
    (FullShape logicalWidth publicFits).publicWidth = 3 * packedParentWords := by
  rfl

/-- Parent coordinate `3 · word + lane`. -/
def packedColumn (word : Fin packedParentWords) (lane : Fin 3) :
    Fin (FullShape logicalWidth publicFits).publicWidth :=
  ⟨3 * word.val + lane.val, by
    rw [publicWidth_eq]
    have wordBound := word.isLt
    have laneBound := lane.isLt
    omega⟩

def serializeParentPublic
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  (List.finRange packedParentWords).map fun word =>
    packWord (parentPublic u (packedColumn word 0))
      (parentPublic u (packedColumn word 1))
      (parentPublic u (packedColumn word 2))

/-- The running fields that both layouts share: commitments, `Eval_K`,
`Eval_A`, then the point. -/
def serializeRunningFields
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  serializeCommitments u ++ serializeEvalKs u ++ serializeEvalAs u ++
    serializePoint u.point

/-- The running instance as hashed: the shared fields, then the packed parent
public input. -/
def serializeRunning
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  serializeRunningFields u ++ serializeParentPublic u

/-- `vk, i, z0, zi`; `slotCount = 1`, so the key and running vectors each
have one entry. -/
def serializeTail
    (p : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  p.verifierKeys functionIndex ++ [natWord p.iteration] ++ p.z0 ++ p.current

/-- The preimage `domain chunk, U, vk, i, z0, zi`. -/
def serializePreimage
    (p : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) : List F :=
  stateDomainChunk ++ serializeRunning (publicFits := publicFits) (p.running functionIndex) ++
    serializeTail p

theorem stateDomainChunk_length : stateDomainChunk.length = 12 := by
  simp [stateDomainChunk, Poseidon2.rate]

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

@[simp] theorem serializeEvalK_length (value : EvaluationFamily K productionShape) :
    (serializeEvalK value).length = productionShape.coefficientCount * 2 := by
  simp [serializeEvalK]

@[simp] theorem serializeEvalA_length (value : EvaluationFamily K productionShape) :
    (serializeEvalA value).length =
      productionShape.matrixCount * (productionShape.coefficientCount * 2) := by
  simp [serializeEvalA]

@[simp] theorem serializeEvaluations_length
    (value : EvaluationFamily K productionShape) :
    (serializeEvaluations value).length =
      (productionShape.matrixCount + 1) * productionShape.coefficientCount * 2 := by
  simp [serializeEvaluations, Nat.add_mul, Nat.mul_assoc, Nat.add_comm]

/-! The serializers are injective at their fixed widths. -/

theorem block_injective : Function.Injective block :=
  fun _ _ same => (List.cons.inj same).2

theorem serializeK_injective : Function.Injective serializeK := by
  intro left right same
  cases left
  cases right
  simp only [serializeK, List.cons.injEq, and_true] at same
  obtain ⟨rfl, rfl⟩ := same
  rfl

theorem serializeKs_injective {left right : List K}
    (same : left.flatMap serializeK = right.flatMap serializeK) : left = right := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons _ _ => simp [serializeK] at same
  | cons value values inductionHypothesis =>
      cases right with
      | nil => simp [serializeK] at same
      | cons other others =>
          simp only [List.flatMap_cons] at same
          obtain ⟨headEqual, tailEqual⟩ := List.append_inj same rfl
          rw [serializeK_injective headEqual, inductionHypothesis tailEqual]

theorem serializeRingF_injective : Function.Injective serializeRingF := by
  intro left right same
  funext coefficient
  exact List.map_inj_left.mp same coefficient (List.mem_finRange coefficient)

theorem serializeCommitment_injective : Function.Injective serializeCommitment := by
  intro left right same
  funext row
  exact serializeRingF_injective (flatMap_eq_of_lengths _ _ _ (fun _ _ => by simp) same row
    (List.mem_finRange row))

theorem serializePublicInput_injective :
    Function.Injective
      (serializePublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
  intro left right same
  funext column
  exact List.map_inj_left.mp same column (List.mem_finRange column)

theorem serializeEvaluations_injective : Function.Injective serializeEvaluations := by
  intro left right same
  obtain ⟨padWords, matrixWords⟩ := List.append_inj same (by simp)
  have pad : left.pad = right.pad := by
    funext coefficient
    exact serializeK_injective (flatMap_eq_of_lengths _
      (fun coefficient => serializeK (left.pad coefficient))
      (fun coefficient => serializeK (right.pad coefficient))
      (fun _ _ => rfl) padWords coefficient (List.mem_finRange _))
  have matrix : left.matrix = right.matrix := by
    funext matrix coefficient
    have matrices := flatMap_eq_of_lengths _
      (fun matrix => (List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => serializeK (left.matrix matrix coefficient))
      (fun matrix => (List.finRange productionShape.coefficientCount).flatMap
        fun coefficient => serializeK (right.matrix matrix coefficient))
      (fun _ _ => by simp) matrixWords matrix (List.mem_finRange _)
    exact serializeK_injective (flatMap_eq_of_lengths _
      (fun coefficient => serializeK (left.matrix matrix coefficient))
      (fun coefficient => serializeK (right.matrix matrix coefficient))
      (fun _ _ => rfl) matrices coefficient (List.mem_finRange _))
  cases left
  cases right
  simp only at pad matrix
  rw [pad, matrix]

theorem serializeCommitments_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeCommitments value).length = 19008 := by
  simp [serializeCommitments, productionShape, productionProfile, ringDegree,
    Phi81MatrixSource.phi81Shape]

theorem serializeEvalKs_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeEvalKs value).length = 1728 := by
  simp [serializeEvalKs, productionShape, productionProfile, ringDegree,
    Phi81MatrixSource.phi81Shape]

theorem serializeEvalAs_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeEvalAs value).length = 6912 := by
  simp [serializeEvalAs, productionShape, productionProfile, ringDegree,
    Phi81MatrixSource.phi81Shape]

theorem serializeChildPublicInputs_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeChildPublicInputs (publicFits := publicFits) value).length = 4320 := by
  simp [serializeChildPublicInputs, productionShape, productionProfile, FullShape,
    fullShape, Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree,
    Phi81MatrixSource.phi81Shape]

theorem serializeParentPublic_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeParentPublic value).length = packedParentWords := by
  simp [serializeParentPublic]

theorem serializeRunningFields_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeRunningFields value).length = 27704 := by
  simp only [serializeRunningFields, List.length_append, serializeCommitments_length,
    serializeEvalKs_length, serializeEvalAs_length, serializePoint_length]
  norm_num [cubeVariables, Phi81MatrixSource.phi81Shape]

theorem serializeRunning_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeRunning (publicFits := publicFits) value).length = 27794 := by
  simp only [serializeRunning, List.length_append, serializeRunningFields_length,
    serializeParentPublic_length]
  norm_num [packedParentWords]

theorem serializeTail_length
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeTail value).length =
      (value.verifierKeys functionIndex).length + 1 + value.z0.length +
        value.current.length := by
  simp [serializeTail]
  omega

/-- Exact preimage length. Only the verifier-key and two application-state
lengths remain parameters. -/
theorem serializePreimage_length
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializePreimage (publicFits := publicFits) value).length =
      27807 + (value.verifierKeys functionIndex).length +
        value.z0.length + value.current.length := by
  simp only [serializePreimage, List.length_append, stateDomainChunk_length,
    serializeRunning_length, serializeTail_length]
  omega

/-! ## Hash input from committed state words -/

/-- Concatenated fixed-width encodings: the word at `position·width + inner`. -/
theorem finRange_flatMap_getD
    {count width : Nat}
    (encode : Fin count → List F)
    (encodedLength : ∀ index, (encode index).length = width)
    (position : Fin count)
    (inner : Nat)
    (innerBound : inner < width) :
    ((List.finRange count).flatMap encode).getD
        (position.val * width + inner) 0 =
      (encode position).getD inner 0 := by
  induction count with
  | zero => exact Fin.elim0 position
  | succ count inductionHypothesis =>
      rw [List.finRange_succ, List.flatMap_cons]
      refine Fin.cases ?_ (fun tail => ?_) position
      · simp only [Fin.val_zero, Nat.zero_mul, Nat.zero_add]
        rw [List.getD_append]
        rw [encodedLength]
        exact innerBound
      · simp only [Fin.val_succ]
        have offset :
            (tail.val + 1) * width + inner =
              width + (tail.val * width + inner) := by
          simp only [Nat.add_mul, Nat.one_mul]
          omega
        rw [List.getD_append_right]
        · rw [encodedLength]
          rw [offset]
          simp only [Nat.add_sub_cancel_left]
          rw [List.flatMap_map]
          exact inductionHypothesis (fun index => encode index.succ)
            (fun index => encodedLength index.succ) tail
        · rw [encodedLength]
          rw [offset]
          omega

theorem finRange_map_getD
    {count : Nat} (encode : Fin count → F) (position : Fin count) :
    ((List.finRange count).map encode).getD position.val 0 =
      encode position := by
  rw [List.getD_eq_get _ _ ⟨position.val, by simp⟩]
  simp only [List.get_eq_getElem, List.getElem_map,
    List.getElem_finRange]
  apply congrArg encode
  exact Fin.ext rfl

theorem serializeChildPublicInputs_getD
    (u : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Fin productionShape.runningCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (serializeChildPublicInputs (publicFits := publicFits) u).getD
        (child.val * 270 + column.val) 0 =
      u.publicInputs child column := by
  rw [show child.val * 270 + column.val =
      child.val * (FullShape logicalWidth publicFits).publicWidth + column.val by rfl]
  unfold serializeChildPublicInputs
  rw [finRange_flatMap_getD _ (fun _ => serializePublicInput_length _) child
    column.val column.isLt]
  exact finRange_map_getD (fun index => u.publicInputs child index) column

/-! ## Packed parent decoding -/

def packOffset : F := Poseidon2.ofNat (2 ^ 16)

/-- `pack(2^16, 2^16, 2^16)`: the shift that makes every bounded lane a natural
number below `2^17`. -/
def packShift : F := packWord packOffset packOffset packOffset

/-- Radix-`2^17` digits of a shifted packed word. The high lane keeps every
remaining bit, so `packWord ∘ unpackWord` is the identity on every word. -/
def unpackWord (word : F) (lane : Fin 3) : F :=
  let shifted := (word + packShift).val
  match lane with
  | 0 => Poseidon2.ofNat (shifted % 2 ^ 17) - packOffset
  | 1 => Poseidon2.ofNat (shifted / 2 ^ 17 % 2 ^ 17) - packOffset
  | 2 => Poseidon2.ofNat (shifted / 2 ^ 34) - packOffset

/-- The parent coordinate stored in packed word `column / 3`. -/
def unpackParent (packed : Fin packedParentWords → F)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : F :=
  unpackWord (packed ⟨column.val / 3, by
      have bound : column.val < 3 * packedParentWords :=
        lt_of_lt_of_eq column.isLt publicWidth_eq
      omega⟩)
    ⟨column.val % 3, Nat.mod_lt _ (by decide)⟩

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

theorem ofNat_add (left right : Nat) :
    Poseidon2.ofNat left + Poseidon2.ofNat right =
      Poseidon2.ofNat (left + right) := by
  apply Fin.eq_of_val_eq
  simp [Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

theorem ofNat_mul (left right : Nat) :
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
