import NightstreamFPrime.Layout.Pilot
import NightstreamFPrime.Layout.Poseidon2

/-!
Owns the fixed Stage 1 pilot ABI. The production lifecycle carries the
verifier-key digest and both application-state boundaries as four Goldilocks
words. Prior-state columns come first, then its 54-cell encoded public input,
the output-state preimage, and the four-word output digest.
-/

namespace NightstreamFPrime.Layout.PilotProduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle

def digestWords : Nat := 4

/-- `40733 + |vk| + |z0| + |zi|` at the fixed four-word production ABI. -/
def stateHashWords : Nat := 40733 + digestWords + digestWords + digestWords

theorem stateHashWords_eq : stateHashWords = 40745 := by
  rfl

def priorPreimageStart : Nat := 0
def priorPublicInputStart : Nat := priorPreimageStart + stateHashWords
def outputPreimageStart : Nat :=
  priorPublicInputStart + PriorStateHash.publicWidth
def outputDigestStart : Nat := outputPreimageStart + stateHashWords

def externalColumnCount : Nat := outputDigestStart + digestWords

theorem externalColumnCount_eq : externalColumnCount = 81548 := by
  rfl

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
  rw [PriorStateHash.hashInterface_input]
  unfold priorInterface
  rw [makePriorInterface_preimage]

@[simp] theorem priorHashInterface_expected (offset : Nat) (lane : Fin 4) :
    (PriorStateHash.hashInterface priorInterface).expected offset lane =
      priorPublicInput offset (PriorStateHash.digestIndex lane) := by
  rfl

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

theorem priorHash_affine :
    Poseidon2.HashInterfaceAffine
      (PriorStateHash.hashInterface priorInterface) witnessOffset := by
  unfold Poseidon2.HashInterfaceAffine
  constructor
  · rw [priorHashInterface_input]
    exact priorPreimage_affine _
  · intro lane
    rw [priorHashInterface_expected]
    unfold priorPublicInput
    exact R1CS.isAffine_var _

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
    (Hash.inputChunks (variableExprs start stateHashWords)).length = 10187 := by
  unfold Hash.inputChunks
  rw [List.length_map, List.length_range, variableExprs_length]
  norm_num [stateHashWords, digestWords,
    NightstreamFPrime.Spec.Poseidon2.rate]

theorem priorPreimage_chunkCount (offset : Nat) :
    (Hash.inputChunks (priorPreimage offset)).length = 10187 := by
  unfold priorPreimage
  exact stateHash_chunkCount priorPreimageStart

theorem outputPreimage_chunkCount (offset : Nat) :
    (Hash.inputChunks (outputPreimage offset)).length = 10187 := by
  unfold outputPreimage
  exact stateHash_chunkCount outputPreimageStart

theorem priorHash_freshCount :
    R1CS.totalFreshCount
      (Poseidon2.hashConstraints
        (PriorStateHash.hashInterface priorInterface) witnessOffset) = 0 :=
  Poseidon2.hashConstraints_freshCount _ _ priorHash_affine

theorem priorHash_rowCount :
    R1CS.totalRowCount
      (Poseidon2.hashConstraints
        (PriorStateHash.hashInterface priorInterface) witnessOffset) =
      6031300 := by
  rw [Poseidon2.hashConstraints_rowCount _ _ priorHash_affine,
    priorHashInterface_input,
    priorPreimage_chunkCount]

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
        (Pilot.outputOffset interface witnessOffset)) = 6031300 := by
  rw [Poseidon2.hashConstraints_rowCount _ _ outputHash_affine,
    outputHashInterface_input,
    outputPreimage_chunkCount]

def priorBindingConstraints : List Expr :=
  flatConstraints
    (PriorStateHash.bindingAssertions priorInterface witnessOffset)

theorem priorBindingConstraints_length :
    priorBindingConstraints.length = 50 := by
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
    R1CS.totalRowCount priorBindingConstraints = 50 := by
  rw [R1CS.totalRowCount_eq_length_of_rowsOne _
    priorBindingConstraints_rowsOne, priorBindingConstraints_length]

theorem interface_prior : interface.prior = priorInterface := by
  rfl

theorem interface_output : interface.output = outputInterface := by
  rfl

theorem priorConstraints_decomposition :
    Pilot.priorConstraints interface witnessOffset =
      Poseidon2.hashConstraints
        (PriorStateHash.hashInterface priorInterface) witnessOffset ++
      priorBindingConstraints := by
  rw [Pilot.priorConstraints_eq, interface_prior]
  rfl

theorem outputConstraints_decomposition :
    Pilot.outputConstraints interface witnessOffset =
      Poseidon2.hashConstraints
        (OutputHash.hashInterface outputInterface)
        (Pilot.outputOffset interface witnessOffset) := by
  rw [Pilot.outputConstraints_eq, interface_output]

theorem priorConstraints_noFresh :
    ∀ expression ∈ Pilot.priorConstraints interface witnessOffset,
      R1CS.constraintFreshCount expression = 0 := by
  rw [priorConstraints_decomposition]
  intro expression member
  rcases List.mem_append.mp member with member | member
  · exact Poseidon2.hashConstraints_noFresh _ _ priorHash_affine
      expression member
  · exact priorBindingConstraints_noFresh expression member

theorem priorConstraints_rowsOne :
    ∀ expression ∈ Pilot.priorConstraints interface witnessOffset,
      R1CS.constraintRowCount expression = 1 := by
  rw [priorConstraints_decomposition]
  intro expression member
  rcases List.mem_append.mp member with member | member
  · exact Poseidon2.hashConstraints_rowsOne _ _ priorHash_affine
      expression member
  · exact priorBindingConstraints_rowsOne expression member

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
      (Pilot.priorConstraints interface witnessOffset) = 0 := by
  rw [priorConstraints_decomposition, R1CS.totalFreshCount_append,
    priorHash_freshCount, priorBindingConstraints_freshCount]

theorem priorConstraints_rowCount :
    R1CS.totalRowCount
      (Pilot.priorConstraints interface witnessOffset) = 6031350 := by
  rw [priorConstraints_decomposition, R1CS.totalRowCount_append,
    priorHash_rowCount, priorBindingConstraints_rowCount]

theorem outputConstraints_freshCount :
    R1CS.totalFreshCount
      (Pilot.outputConstraints interface witnessOffset) = 0 := by
  rw [outputConstraints_decomposition, outputHash_freshCount]

theorem outputConstraints_rowCount :
    R1CS.totalRowCount
      (Pilot.outputConstraints interface witnessOffset) = 6031300 := by
  rw [outputConstraints_decomposition, outputHash_rowCount]

theorem logicalConstraints_noFresh :
    ∀ expression ∈ Pilot.logicalConstraints interface witnessOffset,
      R1CS.constraintFreshCount expression = 0 := by
  intro expression member
  rcases List.mem_append.mp member with member | member
  · exact priorConstraints_noFresh expression member
  · exact outputConstraints_noFresh expression member

theorem logicalConstraints_rowsOne :
    ∀ expression ∈ Pilot.logicalConstraints interface witnessOffset,
      R1CS.constraintRowCount expression = 1 := by
  intro expression member
  rcases List.mem_append.mp member with member | member
  · exact priorConstraints_rowsOne expression member
  · exact outputConstraints_rowsOne expression member

theorem logicalConstraints_freshCount :
    R1CS.totalFreshCount
      (Pilot.logicalConstraints interface witnessOffset) = 0 := by
  unfold Pilot.logicalConstraints
  rw [R1CS.totalFreshCount_append, priorConstraints_freshCount,
    outputConstraints_freshCount]

theorem logicalConstraints_rowCount :
    R1CS.totalRowCount
      (Pilot.logicalConstraints interface witnessOffset) = 12062650 := by
  unfold Pilot.logicalConstraints
  rw [R1CS.totalRowCount_append, priorConstraints_rowCount,
    outputConstraints_rowCount]

theorem physicalRowCount_eq :
    Pilot.physicalRowCount interface witnessOffset = 12062650 := by
  rw [Pilot.physicalRowCount_eq, logicalConstraints_rowCount]

theorem physical_complete (env : Env)
    (logical : ConstraintsHold env
      (Pilot.logicalConstraints interface witnessOffset)) :
    Pilot.PhysicalHolds interface witnessOffset env :=
  R1CS.lowerConstraints_complete_of_noFresh env _
    (Pilot.logicalColumnCount interface witnessOffset)
    logicalConstraints_noFresh logical

theorem priorWitnessCount :
    localLength
      (Circuit.ops (Lifecycle.Pilot.priorCircuit interface).main
        witnessOffset) = 6031296 := by
  rw [Lifecycle.Pilot.priorCircuit_localLength, interface_prior,
    priorInterface_preimage_apply,
    Hash.compile_recipes_length, priorPreimage_chunkCount]

theorem outputWitnessCount :
    localLength
      (Circuit.ops (Lifecycle.Pilot.outputCircuit interface).main
        (Pilot.outputOffset interface witnessOffset)) = 6031296 := by
  rw [Lifecycle.Pilot.outputCircuit_localLength, interface_output,
    outputInterface_preimage_apply,
    Hash.compile_recipes_length, outputPreimage_chunkCount]

theorem outputOffset_eq :
    Pilot.outputOffset interface witnessOffset = 6112844 := by
  rw [Pilot.outputOffset_eq_add, priorWitnessCount]
  rfl

def lifecycleOutputOffset : Nat :=
  Lifecycle.Pilot.outputOffset interface witnessOffset

theorem lifecycleOutputOffset_eq : lifecycleOutputOffset = 6112844 := by
  change Pilot.outputOffset interface witnessOffset = 6112844
  exact outputOffset_eq

theorem witnessOffset_eq : witnessOffset = 81548 := by
  unfold witnessOffset
  exact externalColumnCount_eq

theorem witnessOffset_le_lifecycleOutputOffset :
    witnessOffset ≤ Lifecycle.Pilot.outputOffset interface witnessOffset := by
  change witnessOffset ≤ lifecycleOutputOffset
  rw [witnessOffset_eq, lifecycleOutputOffset_eq]
  norm_num

theorem logicalColumnCount_eq :
    Pilot.logicalColumnCount interface witnessOffset = 12144140 := by
  rw [Pilot.logicalColumnCount_eq_add]
  rw [outputWitnessCount, outputOffset_eq]

theorem physicalColumnCount_eq :
    Pilot.physicalColumnCount interface witnessOffset = 12144140 := by
  rw [Pilot.physicalColumnCount_eq, logicalConstraints_freshCount,
    logicalColumnCount_eq]

def jointDomain : Nat :=
  max (Pilot.physicalRowCount interface witnessOffset)
    (Pilot.physicalColumnCount interface witnessOffset)

theorem jointDomain_eq : jointDomain = 12144140 := by
  simp [jointDomain, physicalRowCount_eq, physicalColumnCount_eq]

/-- The complete pilot layout fits the fixed `2^24` production domain. -/
theorem jointDomain_le_twoPow24 : jointDomain ≤ 2 ^ 24 := by
  rw [jointDomain_eq]
  norm_num

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
    omega
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
  omega

theorem assumptions (env : Env) :
    Lifecycle.Pilot.Assumptions interface witnessOffset env := by
  constructor
  · constructor
    · rw [interface_prior]
      exact priorPreimage_belowWitness
    · rw [interface_prior]
      exact priorPublicInput_belowWitness
  · unfold OutputHash.Assumptions Formal.Assumptions
    constructor
    · intro expression member
      rw [OutputHash.hashInterface_input, interface_output] at member
      have small := outputPreimage_belowWitness expression (by
        simpa [lifecycleOutputOffset] using member)
      exact Expr.VarsBelow.mono expression small
        witnessOffset_le_lifecycleOutputOffset
    · intro lane
      rw [OutputHash.hashInterface_expected, interface_output]
      have small := outputDigest_belowWitness lane
      simpa [lifecycleOutputOffset] using
        Expr.VarsBelow.mono _ small witnessOffset_le_lifecycleOutputOffset

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
          simpa [lifecycleOutputOffset] using outputPreimage_belowWitness) agrees
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
              (by simpa [lifecycleOutputOffset] using
                outputDigest_belowWitness lane) (by omega))
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
