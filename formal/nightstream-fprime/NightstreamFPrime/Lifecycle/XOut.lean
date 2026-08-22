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
  (List.finRange productionShape.matrixCount).flatMap fun j =>
    (List.finRange productionShape.coefficientCount).flatMap fun l => serializeK (e j l)

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
      productionShape.matrixCount * productionShape.coefficientCount * 2 := by
  simp [serializeEvaluations, Nat.mul_assoc]

theorem serializeRunning_length
    (value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializeRunning (publicFits := publicFits) value).length = 40705 := by
  simp [serializeRunning, productionShape, productionProfile, fullShape,
    publicRingColumns, ringDegree, cubeVariables,
    Phi81Relation.Shape.publicWidth, Phi81MatrixSource.phi81Shape]

/-- Exact static serialization cost. Only the verifier-key and two
application-state block lengths remain parameters. -/
theorem serializePreimage_length
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (serializePreimage (publicFits := publicFits) value).length =
      40733 + (value.verifierKeys functionIndex).length +
        value.z0.length + value.current.length := by
  simp [serializePreimage, stateDomainTag_length, serializeRunning_length]
  omega

/-- `stateHash`: the public output of one F′ step. -/
def stateHash (p : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) : Digest :=
  Poseidon2.hash (serializePreimage (publicFits := publicFits) p)

/-- Fixed public-instance encoding of a digest: `encHash(d) = enc_inst((⊥, d))`,
placed in the one public ring column: `[1, d₀ … d₃, 0 …]`. Its length is the
public width, independent of any circuit padding (HyperNova Def. 12 prop. 6). -/
def encHash (d : Digest) :
    PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun j => (1 :: d).getD j.val 0

/-- The paper default CE instance `u_⊥` (HyperNova H.2): the commitment of the
zero assignment with zero randomness, the zero public input (the projection of
the zero assignment), the zero point, and zero evaluations. -/
def zeroPoint : CubePoint K cubeVariables :=
  ⟨List.replicate cubeVariables K.zero, by simp⟩

def defaultRunning : Running (logicalWidth := logicalWidth) (publicFits := publicFits) where
  point := zeroPoint
  commitments := fun _ _ => ringFZero
  publicInputs := fun _ _ => 0
  evaluations := fun _ _ _ => K.zero

end

end NightstreamFPrime.Lifecycle
