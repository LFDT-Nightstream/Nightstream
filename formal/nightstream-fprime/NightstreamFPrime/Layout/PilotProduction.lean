import NightstreamFPrime.Layout.Pilot
import NightstreamFPrime.Layout.PilotValues
import NightstreamFPrime.Layout.Poseidon2
import NightstreamFPrime.Layout.Range.CanonicalPublicU64

/-!
Owns the fixed Stage 1 pilot ABI. The production lifecycle carries the
verifier-key digest and both application-state boundaries as four Goldilocks
words. Prior-state columns come first, then its 270-cell encoded public input,
the output-state preimage, and the four-word output digest.
-/

namespace NightstreamFPrime.Layout.PilotProduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle

def digestWords : Nat := PilotValues.digestWords

/-- `45925 + |vk| + |z0| + |zi|` at the fixed four-word production ABI. -/
def stateHashWords : Nat :=
  PilotValues.stateHashBaseWords + digestWords + digestWords + digestWords

theorem stateHashWords_eq : stateHashWords = 45937 := by
  rfl

def priorPreimageStart : Nat := 0
def priorPublicInputStart : Nat := priorPreimageStart + stateHashWords
def outputPreimageStart : Nat :=
  priorPublicInputStart + PriorStateHash.publicWidth
def outputDigestStart : Nat := outputPreimageStart + stateHashWords

def externalColumnCount : Nat := outputDigestStart + digestWords

theorem externalColumnCount_eq : externalColumnCount = 92148 := by
  rfl

/-- Fixed physical schedule values derived from the v1.1 state-hash width. -/
def absorbCount : Nat :=
  (stateHashWords + NightstreamFPrime.Spec.Poseidon2.rate - 1) /
    NightstreamFPrime.Spec.Poseidon2.rate

def permutationRecipeCount : Nat := PilotValues.permutationRecipeCount

def hashWitnessCount : Nat :=
  (absorbCount + 1) * permutationRecipeCount

def hashRowCount : Nat := hashWitnessCount + digestWords

def priorHashRowStart : Nat := 0
def priorBindingRowCount : Nat := PilotValues.priorBindingRowCount
def priorBindingRowStart : Nat := priorHashRowStart + hashWitnessCount
def outputHashRowStart : Nat := priorBindingRowStart + priorBindingRowCount

def physicalRowCountValue : Nat :=
  hashWitnessCount + priorBindingRowCount + hashRowCount

theorem absorbCount_eq : absorbCount = 11485 := by
  norm_num [absorbCount, stateHashWords, digestWords,
    PilotValues.stateHashBaseWords, PilotValues.digestWords,
    NightstreamFPrime.Spec.Poseidon2.rate]

theorem hashWitnessCount_eq : hashWitnessCount = 6799712 := by
  norm_num [hashWitnessCount, absorbCount_eq, permutationRecipeCount,
    PilotValues.permutationRecipeCount]

theorem hashRowCount_eq : hashRowCount = 6799716 := by
  norm_num [hashRowCount, hashWitnessCount_eq, digestWords,
    PilotValues.digestWords]

theorem physicalRowCountValue_eq : physicalRowCountValue = 13600754 := by
  norm_num [physicalRowCountValue, hashWitnessCount_eq, hashRowCount_eq,
    priorBindingRowCount, PilotValues.priorBindingRowCount,
    PilotValues.priorExtraRowCount, PilotValues.priorCanonicalRowCount,
    PilotValues.priorFixedRowCount]

def variableExprs (start count : Nat) : List Expr :=
  List.ofFn fun index : Fin count => Expr.var (start + index.val)

@[simp] theorem variableExprs_length (start count : Nat) :
    (variableExprs start count).length = count := by
  simp [variableExprs]

theorem variableExprs_affine (start count : Nat) :
    Poseidon2.ListAffine (variableExprs start count) := by
  intro expression member
  rw [variableExprs, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  exact R1CS.isAffine_var _

theorem variableExprs_below (start count bound : Nat)
    (fits : start + count ≤ bound) :
    ∀ expression ∈ variableExprs start count, expression.VarsBelow bound := by
  intro expression member
  rw [variableExprs, List.mem_ofFn'] at member
  rcases member with ⟨index, rfl⟩
  simp only [Expr.VarsBelow]
  omega

def priorPreimage (_offset : Nat) : List Expr :=
  variableExprs priorPreimageStart stateHashWords

def priorPublicInput (_offset : Nat)
    (column : Fin PriorStateHash.publicWidth) : Expr :=
  Expr.var (priorPublicInputStart + column.val)

def outputPreimage (_offset : Nat) : List Expr :=
  variableExprs outputPreimageStart stateHashWords

def outputDigest (_offset : Nat) (lane : Fin 4) : Expr :=
  Expr.var (outputDigestStart + lane.val)

def makePriorInterface (preimage : Nat → List Expr)
    (publicInput : Nat → Fin PriorStateHash.publicWidth → Expr) :
    PriorStateHash.Interface where
  preimage := preimage
  publicInput := publicInput

@[simp] theorem makePriorInterface_preimage
    (preimage : Nat → List Expr)
    (publicInput : Nat → Fin PriorStateHash.publicWidth → Expr) :
    (makePriorInterface preimage publicInput).preimage = preimage := by
  rfl

@[simp] theorem makePriorInterface_publicInput
    (preimage : Nat → List Expr)
    (publicInput : Nat → Fin PriorStateHash.publicWidth → Expr) :
    (makePriorInterface preimage publicInput).publicInput = publicInput := by
  rfl

def makeOutputInterface (preimage : Nat → List Expr)
    (digest : Nat → Fin 4 → Expr) : OutputHash.Interface where
  preimage := preimage
  digest := digest

@[simp] theorem makeOutputInterface_preimage
    (preimage : Nat → List Expr) (digest : Nat → Fin 4 → Expr) :
    (makeOutputInterface preimage digest).preimage = preimage := by
  rfl

def priorInterface : PriorStateHash.Interface :=
  makePriorInterface priorPreimage priorPublicInput

def outputInterface : OutputHash.Interface :=
  makeOutputInterface outputPreimage outputDigest

@[simp] theorem priorInterface_preimage_apply (offset : Nat) :
    priorInterface.preimage offset = priorPreimage offset := by
  unfold priorInterface
  rw [makePriorInterface_preimage]

@[simp] theorem outputInterface_preimage_apply (offset : Nat) :
    outputInterface.preimage offset = outputPreimage offset := by
  unfold outputInterface
  rw [makeOutputInterface_preimage]

@[simp] theorem priorInterface_publicInput_apply (offset : Nat)
    (column : Fin PriorStateHash.publicWidth) :
    priorInterface.publicInput offset column =
      priorPublicInput offset column := by
  unfold priorInterface
  rw [makePriorInterface_publicInput]

def interface : Lifecycle.Pilot.Interface where
  prior := priorInterface
  output := outputInterface

/-- Logical witnesses begin after every verifier/public ABI column. -/
def witnessOffset : Nat := externalColumnCount

@[simp] theorem priorHashInterface_input (offset : Nat) :
    (PriorStateHash.hashInterface priorInterface).input offset =
      priorPreimage offset := by
  calc
    (PriorStateHash.hashInterface priorInterface).input offset =
        priorInterface.preimage offset :=
      PriorStateHash.hashInterface_input priorInterface offset
    _ = priorPreimage offset := priorInterface_preimage_apply offset

@[simp] theorem outputHashInterface_input (offset : Nat) :
    (OutputHash.hashInterface outputInterface).input offset =
      outputPreimage offset := by
  rw [OutputHash.hashInterface_input]
  unfold outputInterface
  rw [makeOutputInterface_preimage]

@[simp] theorem outputHashInterface_expected (offset : Nat) (lane : Fin 4) :
    (OutputHash.hashInterface outputInterface).expected offset lane =
      outputDigest offset lane := by
  rfl

theorem priorPreimage_affine (offset : Nat) :
    Poseidon2.ListAffine (priorPreimage offset) := by
  unfold priorPreimage
  exact variableExprs_affine _ _

theorem outputPreimage_affine (offset : Nat) :
    Poseidon2.ListAffine (outputPreimage offset) := by
  unfold outputPreimage
  exact variableExprs_affine _ _

theorem outputHash_affine :
    Poseidon2.HashInterfaceAffine
      (OutputHash.hashInterface outputInterface)
      (Pilot.outputOffset interface witnessOffset) := by
  unfold Poseidon2.HashInterfaceAffine
  constructor
  · rw [outputHashInterface_input]
    exact outputPreimage_affine _
  · intro lane
    rw [outputHashInterface_expected]
    unfold outputDigest
    exact R1CS.isAffine_var _

theorem stateHash_chunkCount (start : Nat) :
    (Hash.inputChunks (variableExprs start stateHashWords)).length = 11485 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range, variableExprs_length]
  norm_num [stateHashWords, digestWords,
    PilotValues.stateHashBaseWords, PilotValues.digestWords,
    NightstreamFPrime.Spec.Poseidon2.rate]

theorem priorPreimage_chunkCount (offset : Nat) :
    (Hash.inputChunks (priorPreimage offset)).length = 11485 := by
  unfold priorPreimage
  exact stateHash_chunkCount priorPreimageStart

theorem outputPreimage_chunkCount (offset : Nat) :
    (Hash.inputChunks (outputPreimage offset)).length = 11485 := by
  unfold outputPreimage
  exact stateHash_chunkCount outputPreimageStart

/-! ## Direct production digest wiring -/

def priorDigestStart : Nat :=
  witnessOffset + absorbCount * permutationRecipeCount +
    PilotValues.permutationOutputLocalStart

def fastPriorDigest (word : Fin 4) : Expr :=
  Expr.var (priorDigestStart + word.val)

theorem fastPriorDigest_eq (word : Fin 4) :
    fastPriorDigest word =
      RawFormal.digest (PriorStateHash.hashInterface priorInterface)
        witnessOffset word := by
  unfold fastPriorDigest priorDigestStart RawFormal.digest RawFormal.program
  rw [NightstreamFPrime.Layout.Poseidon2.hash_compile_output_eq,
    PriorStateHash.hashInterface_input, priorInterface_preimage_apply,
    priorPreimage_chunkCount]
  simp [Hash.digestE, Permutation.freshState, absorbCount_eq,
    permutationRecipeCount, PilotValues.permutationRecipeCount,
    PilotValues.permutationOutputLocalStart]

def fastPriorWordInterface (word : Fin 4) :
    NightstreamFPrime.Gadgets.Range.CanonicalPublicU64.Interface where
  source := fun _ => fastPriorDigest word
  bit := fun _ bit => priorInterface.publicInput witnessOffset
    (PriorStateHash.digestBitIndexNat word bit)

theorem fastPriorWordInterface_eq (word : Fin 4) :
    fastPriorWordInterface word =
      PriorStateHash.wordInterface priorInterface witnessOffset word := by
  unfold fastPriorWordInterface PriorStateHash.wordInterface
  have sourceEq : (fun _ : Nat => fastPriorDigest word) =
      (fun _ : Nat => RawFormal.digest
        (PriorStateHash.hashInterface priorInterface) witnessOffset word) := by
    funext offset
    exact fastPriorDigest_eq word
  rw [sourceEq]

def fastPriorWordCircuit (word : Fin 4) : FormalCircuit :=
  NightstreamFPrime.Gadgets.Range.CanonicalPublicU64.circuit
    (fastPriorWordInterface word)

def fastPriorWordOp (word : Fin 4) : Op :=
  Sequence.childOp (PriorStateHash.wordName word) (fastPriorWordCircuit word)
    (PriorStateHash.wordOffset priorInterface witnessOffset word)

theorem fastPriorWordOp_eq (word : Fin 4) :
    fastPriorWordOp word =
      PriorStateHash.wordOp priorInterface witnessOffset word := by
  unfold fastPriorWordOp fastPriorWordCircuit PriorStateHash.wordOp
    PriorStateHash.wordCircuit
  rw [fastPriorWordInterface_eq]

def fastPriorWordOps (_unit : Unit) : List Op :=
  List.ofFn fastPriorWordOp

theorem fastPriorWordOps_eq :
    fastPriorWordOps () =
      PriorStateHash.wordOps priorInterface witnessOffset := by
  unfold fastPriorWordOps PriorStateHash.wordOps
  apply congrArg List.ofFn
  funext word
  exact fastPriorWordOp_eq word

def priorRawConstraints (_unit : Unit) : List Expr :=
  flatConstraints [PriorStateHash.hashOp priorInterface witnessOffset]

theorem priorRawConstraints_eq :
    priorRawConstraints () = recipeConstraints witnessOffset
      (Hash.compile witnessOffset (priorPreimage witnessOffset)).recipes := by
  unfold priorRawConstraints
  rw [PriorStateHash.hashOp_flatConstraints_eq,
    priorInterface_preimage_apply]

theorem priorHash_freshCount :
    R1CS.totalFreshCount (priorRawConstraints ()) = 0 := by
  rw [priorRawConstraints_eq]
  exact Poseidon2.hash_recipeConstraints_freshCount witnessOffset
    (priorPreimage witnessOffset) (priorPreimage_affine witnessOffset)

theorem priorHash_rowCount :
    R1CS.totalRowCount (priorRawConstraints ()) = 6799712 := by
  rw [priorRawConstraints_eq,
    Poseidon2.hash_recipeConstraints_rowCount witnessOffset
      (priorPreimage witnessOffset) (priorPreimage_affine witnessOffset),
    priorPreimage_chunkCount]

def priorWordConstraints (word : Fin 4) : List Expr :=
  flatConstraints [PriorStateHash.wordOp priorInterface witnessOffset word]

theorem priorWordConstraints_eq (word : Fin 4) :
    priorWordConstraints word =
      NightstreamFPrime.Layout.Range.CanonicalPublicU64.logicalConstraints
        (PriorStateHash.wordInterface priorInterface witnessOffset word)
        (PriorStateHash.wordOffset priorInterface witnessOffset word) := by
  rfl

theorem priorWordInputsAffine (word : Fin 4) :
    NightstreamFPrime.Layout.Range.CanonicalPublicU64.InputsAffine
      (PriorStateHash.wordInterface priorInterface witnessOffset word)
      (PriorStateHash.wordOffset priorInterface witnessOffset word) := by
  constructor
  · rw [PriorStateHash.wordInterface_source]
    unfold RawFormal.digest RawFormal.program
    rw [Poseidon2.hash_compile_output_eq]
    exact R1CS.isAffine_var _
  · intro bit bounded
    rw [PriorStateHash.wordInterface_bit]
    rw [priorInterface_publicInput_apply]
    unfold priorPublicInput
    exact R1CS.isAffine_var _

theorem priorWord_freshCount (word : Fin 4) :
    R1CS.totalFreshCount (priorWordConstraints word) = 197 := by
  rw [priorWordConstraints_eq]
  exact NightstreamFPrime.Layout.Range.CanonicalPublicU64.totalFreshCount_eq
    _ _ (priorWordInputsAffine word)

theorem priorWord_rowCount (word : Fin 4) :
    R1CS.totalRowCount (priorWordConstraints word) = 328 := by
  rw [priorWordConstraints_eq]
  exact NightstreamFPrime.Layout.Range.CanonicalPublicU64.totalRowCount_eq
    _ _ (priorWordInputsAffine word)

def priorWordConstraintsAll (_unit : Unit) : List Expr :=
  flatConstraints (PriorStateHash.wordOps priorInterface witnessOffset)

private theorem priorWordOps_eq :
    PriorStateHash.wordOps priorInterface witnessOffset =
      [PriorStateHash.wordOp priorInterface witnessOffset 0,
       PriorStateHash.wordOp priorInterface witnessOffset 1,
       PriorStateHash.wordOp priorInterface witnessOffset 2,
       PriorStateHash.wordOp priorInterface witnessOffset 3] := by
  simp [PriorStateHash.wordOps, List.ofFn_succ]

private theorem priorWordConstraintsAll_eq :
    priorWordConstraintsAll () =
      priorWordConstraints 0 ++ priorWordConstraints 1 ++
        priorWordConstraints 2 ++ priorWordConstraints 3 := by
  unfold priorWordConstraintsAll
  rw [priorWordOps_eq]
  rw [show [PriorStateHash.wordOp priorInterface witnessOffset 0,
      PriorStateHash.wordOp priorInterface witnessOffset 1,
      PriorStateHash.wordOp priorInterface witnessOffset 2,
      PriorStateHash.wordOp priorInterface witnessOffset 3] =
    [PriorStateHash.wordOp priorInterface witnessOffset 0] ++
      [PriorStateHash.wordOp priorInterface witnessOffset 1] ++
      [PriorStateHash.wordOp priorInterface witnessOffset 2] ++
      [PriorStateHash.wordOp priorInterface witnessOffset 3] by rfl]
  rw [flatConstraints_append, flatConstraints_append,
    flatConstraints_append]
  rfl

theorem priorWordConstraints_freshCount :
    R1CS.totalFreshCount (priorWordConstraintsAll ()) = 788 := by
  rw [priorWordConstraintsAll_eq, R1CS.totalFreshCount_append,
    R1CS.totalFreshCount_append,
    R1CS.totalFreshCount_append, priorWord_freshCount,
    priorWord_freshCount, priorWord_freshCount, priorWord_freshCount]

theorem priorWordConstraints_rowCount :
    R1CS.totalRowCount (priorWordConstraintsAll ()) = 1312 := by
  rw [priorWordConstraintsAll_eq, R1CS.totalRowCount_append,
    R1CS.totalRowCount_append,
    R1CS.totalRowCount_append, priorWord_rowCount,
    priorWord_rowCount, priorWord_rowCount, priorWord_rowCount]

theorem outputHash_freshCount :
    R1CS.totalFreshCount
      (Poseidon2.hashConstraints
        (OutputHash.hashInterface outputInterface)
        (Pilot.outputOffset interface witnessOffset)) = 0 :=
  Poseidon2.hashConstraints_freshCount _ _ outputHash_affine

theorem outputHash_rowCount :
    R1CS.totalRowCount
      (Poseidon2.hashConstraints
        (OutputHash.hashInterface outputInterface)
        (Pilot.outputOffset interface witnessOffset)) = 6799716 := by
  rw [Poseidon2.hashConstraints_rowCount _ _ outputHash_affine,
    outputHashInterface_input,
    outputPreimage_chunkCount]

def priorBindingConstraints : List Expr :=
  flatConstraints
    (PriorStateHash.bindingAssertions priorInterface witnessOffset)

theorem priorBindingConstraints_length :
    priorBindingConstraints.length = 14 := by
  simp [priorBindingConstraints, PriorStateHash.bindingAssertions,
    flatConstraints, Op.flatConstraints]

theorem priorBindingConstraint_counts (expression : Expr)
    (member : expression ∈ priorBindingConstraints) :
    R1CS.constraintFreshCount expression = 0 ∧
      R1CS.constraintRowCount expression = 1 := by
  unfold priorBindingConstraints at member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [PriorStateHash.bindingAssertions, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    rw [priorInterface_publicInput_apply]
    unfold priorPublicInput
    exact ⟨rfl, rfl⟩
  · rw [List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    rw [priorInterface_publicInput_apply]
    unfold priorPublicInput
    exact ⟨rfl, rfl⟩

theorem priorBindingConstraints_noFresh :
    ∀ expression ∈ priorBindingConstraints,
      R1CS.constraintFreshCount expression = 0 := by
  intro expression member
  exact (priorBindingConstraint_counts expression member).1

theorem priorBindingConstraints_rowsOne :
    ∀ expression ∈ priorBindingConstraints,
      R1CS.constraintRowCount expression = 1 := by
  intro expression member
  exact (priorBindingConstraint_counts expression member).2

theorem priorBindingConstraints_freshCount :
    R1CS.totalFreshCount priorBindingConstraints = 0 :=
  R1CS.totalFreshCount_eq_zero_of_noFresh _
    priorBindingConstraints_noFresh

theorem priorBindingConstraints_rowCount :
    R1CS.totalRowCount priorBindingConstraints = 14 := by
  rw [R1CS.totalRowCount_eq_length_of_rowsOne _
    priorBindingConstraints_rowsOne, priorBindingConstraints_length]

theorem interface_prior : interface.prior = priorInterface := by
  rfl

theorem interface_output : interface.output = outputInterface := by
  rfl

theorem priorConstraints_decomposition :
    Pilot.priorConstraints interface witnessOffset =
      priorRawConstraints () ++ priorWordConstraintsAll () ++
      priorBindingConstraints := by
  rw [Pilot.priorConstraints_eq, interface_prior]
  rfl

theorem outputConstraints_decomposition :
    Pilot.outputConstraints interface witnessOffset =
      Poseidon2.hashConstraints
        (OutputHash.hashInterface outputInterface)
        (Pilot.outputOffset interface witnessOffset) := by
  rw [Pilot.outputConstraints_eq, interface_output]

theorem outputConstraints_noFresh :
    ∀ expression ∈ Pilot.outputConstraints interface witnessOffset,
      R1CS.constraintFreshCount expression = 0 := by
  rw [outputConstraints_decomposition]
  exact Poseidon2.hashConstraints_noFresh _ _ outputHash_affine

theorem outputConstraints_rowsOne :
    ∀ expression ∈ Pilot.outputConstraints interface witnessOffset,
      R1CS.constraintRowCount expression = 1 := by
  rw [outputConstraints_decomposition]
  exact Poseidon2.hashConstraints_rowsOne _ _ outputHash_affine

theorem priorConstraints_freshCount :
    R1CS.totalFreshCount
      (Pilot.priorConstraints interface witnessOffset) = 788 := by
  rw [priorConstraints_decomposition, R1CS.totalFreshCount_append,
    R1CS.totalFreshCount_append, priorHash_freshCount,
    priorWordConstraints_freshCount, priorBindingConstraints_freshCount]

theorem priorConstraints_rowCount :
    R1CS.totalRowCount
      (Pilot.priorConstraints interface witnessOffset) = 6801038 := by
  rw [priorConstraints_decomposition, R1CS.totalRowCount_append,
    R1CS.totalRowCount_append, priorHash_rowCount,
    priorWordConstraints_rowCount, priorBindingConstraints_rowCount]

theorem outputConstraints_freshCount :
    R1CS.totalFreshCount
      (Pilot.outputConstraints interface witnessOffset) = 0 := by
  rw [outputConstraints_decomposition, outputHash_freshCount]

theorem outputConstraints_rowCount :
    R1CS.totalRowCount
      (Pilot.outputConstraints interface witnessOffset) = 6799716 := by
  rw [outputConstraints_decomposition, outputHash_rowCount]

theorem logicalConstraints_freshCount :
    R1CS.totalFreshCount
      (Pilot.logicalConstraints interface witnessOffset) = 788 := by
  unfold Pilot.logicalConstraints
  rw [R1CS.totalFreshCount_append, priorConstraints_freshCount,
    outputConstraints_freshCount]

theorem logicalConstraints_rowCount :
    R1CS.totalRowCount
      (Pilot.logicalConstraints interface witnessOffset) = 13600754 := by
  unfold Pilot.logicalConstraints
  rw [R1CS.totalRowCount_append, priorConstraints_rowCount,
    outputConstraints_rowCount]

theorem physicalRowCount_eq :
    Pilot.physicalRowCount interface witnessOffset = 13600754 := by
  rw [Pilot.physicalRowCount_eq, logicalConstraints_rowCount]

theorem priorHashLogicalLength_eq :
    PriorStateHash.hashLength priorInterface witnessOffset = 6799712 := by
  rw [PriorStateHash.hashLength_eq, priorInterface_preimage_apply,
    priorPreimage_chunkCount]

theorem priorWitnessCount :
    PriorStateHash.logicalPrivateCount priorInterface witnessOffset =
      6799976 := by
  unfold PriorStateHash.logicalPrivateCount
  rw [priorHashLogicalLength_eq]

theorem outputWitnessCount :
    OutputHash.hashLength outputInterface
      (Pilot.outputOffset interface witnessOffset) = 6799712 := by
  unfold OutputHash.hashLength
  rw [outputInterface_preimage_apply, outputPreimage_chunkCount]

theorem outputOffset_eq :
    Pilot.outputOffset interface witnessOffset = 6892124 := by
  unfold Pilot.outputOffset Lifecycle.Pilot.outputOffset
  rw [interface_prior, priorWitnessCount]
  unfold witnessOffset
  rw [externalColumnCount_eq]

def lifecycleOutputOffset : Nat :=
  6892124

/-- The materialized executable offset is exactly the logical pilot offset. -/
theorem lifecycleOutputOffset_matches :
    lifecycleOutputOffset =
      Lifecycle.Pilot.outputOffset interface witnessOffset := by
  change lifecycleOutputOffset = Pilot.outputOffset interface witnessOffset
  rw [outputOffset_eq]
  rfl

theorem lifecycleOutputOffset_matches_layout :
    lifecycleOutputOffset = Pilot.outputOffset interface witnessOffset := by
  change lifecycleOutputOffset =
    Lifecycle.Pilot.outputOffset interface witnessOffset
  exact lifecycleOutputOffset_matches

theorem lifecycleOutputOffset_eq : lifecycleOutputOffset = 6892124 := by
  rfl

theorem witnessOffset_eq : witnessOffset = 92148 := by
  unfold witnessOffset
  exact externalColumnCount_eq

theorem witnessOffset_le_lifecycleOutputOffset :
    witnessOffset ≤ Lifecycle.Pilot.outputOffset interface witnessOffset := by
  change witnessOffset ≤ Pilot.outputOffset interface witnessOffset
  rw [outputOffset_eq, witnessOffset_eq]
  norm_num

theorem logicalColumnCount_eq :
    Pilot.logicalColumnCount interface witnessOffset = 13691836 := by
  unfold Pilot.logicalColumnCount
  rw [interface_output, outputWitnessCount, outputOffset_eq]

theorem physicalColumnCount_eq :
    Pilot.physicalColumnCount interface witnessOffset = 13692624 := by
  rw [Pilot.physicalColumnCount_eq, logicalConstraints_freshCount,
    logicalColumnCount_eq]

def jointDomain : Nat :=
  13692624

/-- The materialized executable domain is exactly the semantic pilot domain. -/
theorem jointDomain_matches :
    jointDomain =
      max (Pilot.physicalRowCount interface witnessOffset)
        (Pilot.physicalColumnCount interface witnessOffset) := by
  rw [physicalRowCount_eq, physicalColumnCount_eq]
  rfl

theorem jointDomain_eq : jointDomain = 13692624 := by
  rfl

/-- The complete pilot layout fits the fixed `2^28` production domain. -/
theorem jointDomain_le_twoPow28 : jointDomain ≤ 2 ^ 28 := by
  rw [jointDomain_eq]
  norm_num

theorem layoutAssumptions (env : Env) :
    Lifecycle.Pilot.Assumptions interface witnessOffset env := by
  constructor
  · constructor
    · rw [interface_prior]
      rw [priorInterface_preimage_apply]
      unfold priorPreimage
      exact variableExprs_below _ _ witnessOffset (by
        unfold witnessOffset externalColumnCount outputDigestStart
          outputPreimageStart priorPublicInputStart priorPreimageStart
        omega)
    · rw [interface_prior]
      intro column
      rw [priorInterface_publicInput_apply]
      unfold priorPublicInput
      simp only [Expr.VarsBelow]
      unfold witnessOffset externalColumnCount outputDigestStart
        outputPreimageStart
      omega
  · unfold OutputHash.Assumptions Formal.Assumptions
    constructor
    · intro expression member
      rw [OutputHash.hashInterface_input, interface_output] at member
      rw [outputInterface_preimage_apply] at member
      have belowWitness : expression.VarsBelow witnessOffset := by
        apply variableExprs_below outputPreimageStart stateHashWords
          witnessOffset (by
            unfold witnessOffset externalColumnCount outputDigestStart
              outputPreimageStart priorPublicInputStart
            omega) expression member
      exact Expr.VarsBelow.mono expression belowWitness
        witnessOffset_le_lifecycleOutputOffset
    · intro lane
      rw [OutputHash.hashInterface_expected, interface_output]
      have belowWitness :
          (outputInterface.digest lifecycleOutputOffset lane).VarsBelow
            witnessOffset := by
        unfold outputInterface makeOutputInterface outputDigest
        simp only [Expr.VarsBelow]
        unfold witnessOffset externalColumnCount outputDigestStart digestWords
        norm_num [PilotValues.digestWords]
      rw [← lifecycleOutputOffset_matches]
      have materializedBound : witnessOffset ≤ lifecycleOutputOffset := by
        rw [witnessOffset_eq, lifecycleOutputOffset_eq]
        norm_num
      exact Expr.VarsBelow.mono _ belowWitness
        materializedBound

theorem physical_complete (env : Env)
    (logical : ConstraintsHold env
      (Pilot.logicalConstraints interface witnessOffset)) :
    ∃ completed,
      AgreesOutside env completed
        (Pilot.logicalColumnCount interface witnessOffset) 788 ∧
      Pilot.PhysicalHolds interface witnessOffset completed := by
  rcases R1CS.lowerConstraints_complete env
      (Pilot.logicalConstraints interface witnessOffset)
      (Pilot.logicalColumnCount interface witnessOffset)
      (Pilot.logicalConstraints_varsBelow interface witnessOffset
        (layoutAssumptions env)) logical with
    ⟨completed, agrees, rows⟩
  rw [logicalConstraints_freshCount] at agrees
  exact ⟨completed, agrees, rows⟩

/-- Values supplied at the fixed verifier/parent ABI boundary. -/
structure ExternalValues where
  priorPreimage : Fin stateHashWords → F
  priorPublicInput : Fin PriorStateHash.publicWidth → F
  outputPreimage : Fin stateHashWords → F
  outputDigest : Fin digestWords → F

/-- Load the four disjoint external segments. All logical witness columns are
left at zero until the witness program executes. -/
def loadExternal (values : ExternalValues) : Env := fun index =>
  if prior : index < priorPublicInputStart then
    values.priorPreimage ⟨index, by
      simpa [priorPublicInputStart, priorPreimageStart] using prior⟩
  else if priorPublic : index < outputPreimageStart then
    values.priorPublicInput ⟨index - priorPublicInputStart, by
      unfold outputPreimageStart at priorPublic
      omega⟩
  else if output : index < outputDigestStart then
    values.outputPreimage ⟨index - outputPreimageStart, by
      unfold outputDigestStart at output
      omega⟩
  else if digest : index < externalColumnCount then
    values.outputDigest ⟨index - outputDigestStart, by
      unfold externalColumnCount at digest
      omega⟩
  else
    0

theorem eval_priorPreimage (values : ExternalValues) :
    Hash.evalList (loadExternal values) (priorPreimage witnessOffset) =
      List.ofFn values.priorPreimage := by
  simp only [Hash.evalList, priorPreimage, variableExprs, List.map_ofFn]
  apply congrArg List.ofFn
  funext index
  change loadExternal values (priorPreimageStart + index.val) =
    values.priorPreimage index
  have inPrior : priorPreimageStart + index.val < priorPublicInputStart := by
    unfold priorPreimageStart priorPublicInputStart
    omega
  have inPrior' : index.val < priorPublicInputStart := by
    simpa [priorPreimageStart] using inPrior
  simp [loadExternal, inPrior', priorPreimageStart]

theorem eval_priorPublicInput (values : ExternalValues) :
    (fun column =>
      (priorInterface.publicInput witnessOffset column).eval
        (loadExternal values)) = values.priorPublicInput := by
  funext column
  rw [priorInterface_publicInput_apply]
  simp [priorPublicInput, loadExternal, priorPublicInputStart,
    outputPreimageStart]

theorem eval_outputPreimage (values : ExternalValues) :
    Hash.evalList (loadExternal values)
      (outputPreimage (Pilot.outputOffset interface witnessOffset)) =
      List.ofFn values.outputPreimage := by
  simp only [Hash.evalList, outputPreimage, variableExprs, List.map_ofFn]
  apply congrArg List.ofFn
  funext index
  change loadExternal values (outputPreimageStart + index.val) =
    values.outputPreimage index
  have afterPrior : ¬ outputPreimageStart + index.val <
      priorPublicInputStart := by
    unfold outputPreimageStart
    omega
  have afterPriorPublic : ¬ outputPreimageStart + index.val <
      outputPreimageStart := by omega
  have inOutput : outputPreimageStart + index.val < outputDigestStart := by
    unfold outputDigestStart
    omega
  simp [loadExternal, afterPrior, afterPriorPublic, inOutput]

theorem eval_outputDigest (values : ExternalValues) :
    (fun lane =>
      (outputInterface.digest (Pilot.outputOffset interface witnessOffset) lane).eval
        (loadExternal values)) = values.outputDigest := by
  funext lane
  change loadExternal values (outputDigestStart + lane.val) =
    values.outputDigest lane
  have afterPrior : ¬ outputDigestStart + lane.val <
      priorPublicInputStart := by
    unfold outputDigestStart outputPreimageStart
    omega
  have afterPriorPublic : ¬ outputDigestStart + lane.val <
      outputPreimageStart := by
    unfold outputDigestStart
    omega
  have afterOutput : ¬ outputDigestStart + lane.val < outputDigestStart := by
    omega
  have inDigest : outputDigestStart + lane.val < externalColumnCount := by
    unfold externalColumnCount digestWords
    norm_num [PilotValues.digestWords]
  simp [loadExternal, afterPrior, afterPriorPublic, afterOutput, inDigest]

def fixedList {count : Nat} (values : List F)
    (lengthEquals : values.length = count) : Fin count → F :=
  fun index => values.get (Fin.cast lengthEquals.symm index)

theorem ofFn_fixedList {count : Nat} (values : List F)
    (lengthEquals : values.length = count) :
    List.ofFn (fixedList values lengthEquals) = values := by
  cases lengthEquals
  exact List.ofFn_get values

section Protocol

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def FixedPreimage
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits)) : Prop :=
  (preimage.verifierKeys functionIndex).length = digestWords ∧
    preimage.z0.length = digestWords ∧
    preimage.current.length = digestWords

theorem serializePreimage_length_fixed
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (fixed : FixedPreimage preimage) :
    (serializePreimage (publicFits := publicFits) preimage).length =
      stateHashWords := by
  rw [serializePreimage_length]
  rcases fixed with ⟨keyLength, z0Length, currentLength⟩
  rw [keyLength, z0Length, currentLength]
  rfl

def protocolValues
    (prior : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (priorPublic : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (output : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : FixedPreimage prior)
    (outputFixed : FixedPreimage output)
    (digestFixed : digest.length = digestWords) : ExternalValues where
  priorPreimage := fixedList
    (serializePreimage (publicFits := publicFits) prior)
    (serializePreimage_length_fixed prior priorFixed)
  priorPublicInput := priorPublic
  outputPreimage := fixedList
    (serializePreimage (publicFits := publicFits) output)
    (serializePreimage_length_fixed output outputFixed)
  outputDigest := fixedList digest digestFixed

def protocolEnv
    (prior : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (priorPublic : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (output : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : FixedPreimage prior)
    (outputFixed : FixedPreimage output)
    (digestFixed : digest.length = digestWords) : Env :=
  loadExternal (protocolValues prior priorPublic output digest
    priorFixed outputFixed digestFixed)

theorem protocolEnv_represents
    (prior : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (priorPublic : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (output : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : FixedPreimage prior)
    (outputFixed : FixedPreimage output)
    (digestFixed : digest.length = digestWords) :
    let env := protocolEnv prior priorPublic output digest
      priorFixed outputFixed digestFixed
    PriorStateHash.RepresentsPreimage priorInterface witnessOffset env prior ∧
      PriorStateHash.RepresentsPublicInput priorInterface witnessOffset env
        priorPublic ∧
      OutputHash.RepresentsPreimage outputInterface
        (Pilot.outputOffset interface witnessOffset) env output ∧
      OutputHash.RepresentsDigest outputInterface
        (Pilot.outputOffset interface witnessOffset) env digest := by
  let values := protocolValues prior priorPublic output digest
    priorFixed outputFixed digestFixed
  let env := loadExternal values
  change PriorStateHash.RepresentsPreimage priorInterface witnessOffset env prior ∧
    PriorStateHash.RepresentsPublicInput priorInterface witnessOffset env
      priorPublic ∧
    OutputHash.RepresentsPreimage outputInterface
      (Pilot.outputOffset interface witnessOffset) env output ∧
    OutputHash.RepresentsDigest outputInterface
      (Pilot.outputOffset interface witnessOffset) env digest
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold PriorStateHash.RepresentsPreimage
    rw [show priorInterface.preimage witnessOffset =
      priorPreimage witnessOffset by
        exact priorInterface_preimage_apply witnessOffset]
    rw [eval_priorPreimage]
    exact ofFn_fixedList
      (serializePreimage (publicFits := publicFits) prior)
      (serializePreimage_length_fixed prior priorFixed)
  · intro column
    have loaded := congrFun (eval_priorPublicInput values) column
    simpa [values, protocolValues, env] using loaded
  · unfold OutputHash.RepresentsPreimage
    rw [show outputInterface.preimage
      (Pilot.outputOffset interface witnessOffset) =
        outputPreimage (Pilot.outputOffset interface witnessOffset) by
          exact outputInterface_preimage_apply _]
    rw [eval_outputPreimage]
    exact ofFn_fixedList
      (serializePreimage (publicFits := publicFits) output)
      (serializePreimage_length_fixed output outputFixed)
  · unfold OutputHash.RepresentsDigest
    rw [eval_outputDigest]
    exact (ofFn_fixedList digest digestFixed).symm

theorem priorPreimage_belowWitness :
    ∀ expression ∈ priorInterface.preimage witnessOffset,
      expression.VarsBelow witnessOffset := by
  rw [priorInterface_preimage_apply]
  unfold priorPreimage
  exact variableExprs_below _ _ witnessOffset (by
    unfold witnessOffset externalColumnCount outputDigestStart
      outputPreimageStart priorPublicInputStart priorPreimageStart
    omega)

theorem priorPublicInput_belowWitness (column : Fin PriorStateHash.publicWidth) :
    (priorInterface.publicInput witnessOffset column).VarsBelow
      witnessOffset := by
  rw [priorInterface_publicInput_apply]
  unfold priorPublicInput
  simp only [Expr.VarsBelow]
  unfold witnessOffset externalColumnCount outputDigestStart
    outputPreimageStart
  omega

theorem outputPreimage_belowWitness :
    ∀ expression ∈ outputInterface.preimage lifecycleOutputOffset,
      expression.VarsBelow witnessOffset := by
  rw [outputInterface_preimage_apply]
  unfold outputPreimage
  exact variableExprs_below _ _ witnessOffset (by
    unfold witnessOffset externalColumnCount outputDigestStart
      outputPreimageStart priorPublicInputStart priorPreimageStart
    omega)

theorem outputDigest_belowWitness (lane : Fin 4) :
    (outputInterface.digest lifecycleOutputOffset lane).VarsBelow
      witnessOffset := by
  unfold outputInterface makeOutputInterface outputDigest
  simp only [Expr.VarsBelow]
  unfold witnessOffset externalColumnCount outputDigestStart digestWords
  norm_num [PilotValues.digestWords]

theorem assumptions (env : Env) :
    Lifecycle.Pilot.Assumptions interface witnessOffset env :=
  layoutAssumptions env

def AgreesBelow (left right : Env) (bound : Nat) : Prop :=
  ∀ index, index < bound → left index = right index

theorem evalList_eq_of_agreesBelow (left right : Env) (bound : Nat)
    (expressions : List Expr)
    (below : ∀ expression ∈ expressions, expression.VarsBelow bound)
    (agrees : AgreesBelow left right bound) :
    Hash.evalList left expressions = Hash.evalList right expressions := by
  unfold Hash.evalList
  apply List.map_congr_left
  intro expression member
  exact expression.eval_eq_of_agree_below bound left right
    (below expression member) agrees

theorem protocolEnv_represents_of_agreesBelow
    (prior : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (priorPublic : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (output : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : FixedPreimage prior)
    (outputFixed : FixedPreimage output)
    (digestFixed : digest.length = digestWords)
    (env : Env)
    (agrees : AgreesBelow env
      (protocolEnv prior priorPublic output digest
        priorFixed outputFixed digestFixed) witnessOffset) :
    PriorStateHash.RepresentsPreimage priorInterface witnessOffset env prior ∧
      PriorStateHash.RepresentsPublicInput priorInterface witnessOffset env
        priorPublic ∧
      OutputHash.RepresentsPreimage outputInterface
        (Pilot.outputOffset interface witnessOffset) env output ∧
      OutputHash.RepresentsDigest outputInterface
        (Pilot.outputOffset interface witnessOffset) env digest := by
  let base := protocolEnv prior priorPublic output digest
    priorFixed outputFixed digestFixed
  have represented := protocolEnv_represents prior priorPublic output digest
    priorFixed outputFixed digestFixed
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold PriorStateHash.RepresentsPreimage at represented ⊢
    calc
      Hash.evalList env (priorInterface.preimage witnessOffset) =
          Hash.evalList base (priorInterface.preimage witnessOffset) :=
        evalList_eq_of_agreesBelow env base witnessOffset _
          priorPreimage_belowWitness agrees
      _ = serializePreimage (publicFits := publicFits) prior := represented.1
  · intro column
    calc
      (priorInterface.publicInput witnessOffset column).eval env =
          (priorInterface.publicInput witnessOffset column).eval base :=
        (priorInterface.publicInput witnessOffset column).eval_eq_of_agree_below
          witnessOffset env base (priorPublicInput_belowWitness column) agrees
      _ = priorPublic column := represented.2.1 column
  · unfold OutputHash.RepresentsPreimage at represented ⊢
    calc
      Hash.evalList env (outputInterface.preimage
          (Pilot.outputOffset interface witnessOffset)) =
          Hash.evalList base (outputInterface.preimage
            (Pilot.outputOffset interface witnessOffset)) :=
        evalList_eq_of_agreesBelow env base witnessOffset _ (by
          rw [← lifecycleOutputOffset_matches_layout]
          exact outputPreimage_belowWitness) agrees
      _ = serializePreimage (publicFits := publicFits) output :=
        represented.2.2.1
  · unfold OutputHash.RepresentsDigest at represented ⊢
    calc
      digest = List.ofFn (fun lane =>
          (outputInterface.digest
            (Pilot.outputOffset interface witnessOffset) lane).eval base) :=
        represented.2.2.2
      _ = List.ofFn (fun lane =>
          (outputInterface.digest
            (Pilot.outputOffset interface witnessOffset) lane).eval env) := by
        congr 1
        funext lane
        exact ((outputInterface.digest
          (Pilot.outputOffset interface witnessOffset) lane).eval_eq_of_agree_below
            witnessOffset base env (Expr.VarsBelow.mono _
              (by
                rw [← lifecycleOutputOffset_matches_layout]
                exact outputDigest_belowWitness lane) (by omega))
            (fun index lower => (agrees index lower).symm))

/-- Production deterministic bridge for the two pilot slots. The concrete ABI
constructs every representation fact; none is supplied as a free hypothesis. -/
theorem physical_implies_recursive_hash_slots
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (priorFixed : FixedPreimage
      (priorHashPreimage (setup relation ajtai vk) input))
    (outputFixed : FixedPreimage
      (nextHashPreimage (setup relation ajtai vk) input output))
    (digestFixed : output.x.length = digestWords)
    (env : Env)
    (agrees : AgreesBelow env
      (protocolEnv
        (priorHashPreimage (setup relation ajtai vk) input)
        ((machine publicFits F).freshPublic input.fresh)
        (nextHashPreimage (setup relation ajtai vk) input output)
        output.x priorFixed outputFixed digestFixed) witnessOffset)
    (physical : Pilot.PhysicalHolds interface witnessOffset env) :
    (machine publicFits F).freshPublic input.fresh =
        (machine publicFits F).encodeInstance
          ((machine publicFits F).hash
            (priorHashPreimage (setup relation ajtai vk) input)) ∧
      OutputHolds (setup relation ajtai vk) (machine publicFits F)
        input output := by
  have specification := Pilot.physical_implies_spec interface witnessOffset env
    (assumptions env) physical
  have represented := protocolEnv_represents_of_agreesBelow
    (priorHashPreimage (setup relation ajtai vk) input)
    ((machine publicFits F).freshPublic input.fresh)
    (nextHashPreimage (setup relation ajtai vk) input output)
    output.x priorFixed outputFixed digestFixed env agrees
  exact Lifecycle.Pilot.builders_imply_hash_slots interface witnessOffset env
    relation ajtai vk F input output specification represented.1
    represented.2.1 represented.2.2.1 represented.2.2.2

end Protocol

end NightstreamFPrime.Layout.PilotProduction
