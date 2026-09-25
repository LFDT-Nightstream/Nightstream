import NightstreamFPrime.Export.Package

/-!
Owns the bounded-memory executable form of the canonical package identity.
The semantic authority remains `Package.relationIdentifierValue`.
-/

namespace NightstreamFPrime.Export.StreamingIdentity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec

structure Node where
  tag : Nat
  value : Nat

def Node.words (node : Node) : List F :=
  [Poseidon2.ofNat node.tag,
   Poseidon2.ofNat (node.value % Package.limbBase),
   Poseidon2.ofNat ((node.value / Package.limbBase) % Package.limbBase),
   Poseidon2.ofNat (node.value / (Package.limbBase * Package.limbBase))]

def nodes : Value → List Node
  | .atom value => [⟨0, value⟩]
  | .array values =>
      ⟨1, values.length⟩ :: values.flatMap nodes

def canonicalWords (value : Value) : List F :=
  (nodes value).flatMap Node.words

theorem valuePreimageRev_eq_reverse_canonicalWords (value : Value) :
    ∀ tail,
      Package.valuePreimageRev value tail =
        (canonicalWords value).reverse ++ tail :=
  Value.rec
    (motive_1 := fun value => ∀ tail,
      Package.valuePreimageRev value tail =
        (canonicalWords value).reverse ++ tail)
    (motive_2 := fun values => ∀ tail,
      values.foldl (fun state child =>
        Package.valuePreimageRev child state) tail =
          (values.flatMap canonicalWords).reverse ++ tail)
    (fun value tail => by
      simp [Package.valuePreimageRev, canonicalWords, nodes, Node.words,
        Package.prependIdentityNode])
    (fun values valuesInduction tail => by
      simp [Package.valuePreimageRev, canonicalWords, nodes, Node.words,
        Package.prependIdentityNode, valuesInduction, List.flatMap_assoc,
        List.append_assoc]
      change List.flatMap (fun child =>
          List.flatMap Node.words (nodes child)) values =
        List.flatMap (fun child =>
          List.flatMap Node.words (nodes child)) values
      rfl)
    (by simp)
    (fun head tail headInduction tailInduction initial => by
      simp [headInduction, tailInduction, List.reverse_append,
        List.append_assoc])
    value

theorem canonicalWords_eq_valuePreimage (value : Value) :
    canonicalWords value = Package.valuePreimage value := by
  rw [Package.valuePreimage,
    valuePreimageRev_eq_reverse_canonicalWords value []]
  simp

structure HashState where
  sponge : Poseidon2.State
  carry : F

def Node.block (node : Node) (carry : F) : List F :=
  [carry, node.words.getD 0 0, node.words.getD 1 0,
    node.words.getD 2 0]

def Node.nextCarry (node : Node) : F :=
  node.words.getD 3 0

def pushNode (state : HashState) (node : Node) : HashState :=
  {
    sponge := Poseidon2.absorbBlock state.sponge (node.block state.carry)
    carry := node.nextCarry }

def processNodes (state : HashState) (stream : List Node) : HashState :=
  stream.foldl pushNode state

@[specialize push] def processValueWith {State : Type}
    (push : State → Node → State) :
    Value → State → State
  | .atom value, state => push state ⟨0, value⟩
  | .array values, state =>
      values.foldl (fun current child => processValueWith push child current)
        (push state ⟨1, values.length⟩)

/-- Canonical value traversal specialized to the semantic Poseidon2 node
transition. -/
def processValue : Value → HashState → HashState :=
  processValueWith pushNode

/-- Count canonical codec nodes without constructing the encoded node list. -/
def countNode (count : Nat) (_node : Node) : Nat := count + 1

theorem processValueWith_countNode (value : Value) : ∀ initial,
    processValueWith countNode value initial =
      initial + (nodes value).length :=
  Value.rec
    (motive_1 := fun value => ∀ initial,
      processValueWith countNode value initial =
        initial + (nodes value).length)
    (motive_2 := fun values => ∀ initial,
      values.foldl
          (fun current child => processValueWith countNode child current)
          initial =
        initial + (values.flatMap nodes).length)
    (fun value initial => by
      simp [processValueWith, countNode, nodes])
    (fun values valuesInduction initial => by
      simp only [processValueWith, countNode, nodes, List.length_cons]
      rw [valuesInduction]
      omega)
    (by
      intro initial
      simp)
    (fun head tail headInduction tailInduction initial => by
      simp only [List.foldl_cons, List.flatMap_cons, List.length_append]
      rw [headInduction, tailInduction]
      omega)
    value

@[simp] theorem processValueWith_pushNode (value : Value)
    (state : HashState) :
    processValueWith pushNode value state = processValue value state := by
  rfl

/-- A pointwise simulation of node transitions lifts through the complete
canonical value traversal. This is the proof boundary for an alternative
state representation: the alternative transition must still prove every
node step against the semantic transition. -/
theorem processValueWith_simulates
    {SourceState TargetState : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (value : Value) : ∀ state,
    denote (processValueWith sourcePush value state) =
      processValueWith targetPush value (denote state) :=
  Value.rec
    (motive_1 := fun value => ∀ state,
      denote (processValueWith sourcePush value state) =
        processValueWith targetPush value (denote state))
    (motive_2 := fun values => ∀ state,
      denote (values.foldl (fun current child =>
        processValueWith sourcePush child current) state) =
      values.foldl (fun current child =>
        processValueWith targetPush child current) (denote state))
    (fun value state => by
      simp [processValueWith, pushSimulates])
    (fun values valuesInduction state => by
      simp [processValueWith, valuesInduction, pushSimulates])
    (by
      intro state
      rfl)
    (fun head tail headInduction tailInduction state => by
      simp [headInduction, tailInduction])
    value

/-! ## Typed, allocation-bounded traversals -/

/-- Process encoded items without an enclosing list node. The traversal
encodes only the current item and never constructs `values.map format.encode`.
This is the primitive used to join several item producers under one
caller-owned array header. -/
@[specialize push format] def processEncodedItemsWith {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) : List Alpha → State
  | [] => state
  | value :: rest =>
      processEncodedItemsWith push
        (processValueWith push (format.encode value) state) format rest

theorem processEncodedItemsWith_eq_mappedFoldl {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (values : List Alpha) :
    processEncodedItemsWith push state format values =
      (values.map format.encode).foldl
        (fun current value => processValueWith push value current) state := by
  induction values generalizing state with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [processEncodedItemsWith, inductionHypothesis]

theorem processEncodedItemsWith_append {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (left right : List Alpha) :
    processEncodedItemsWith push state format (left ++ right) =
      processEncodedItemsWith push
        (processEncodedItemsWith push state format left) format right := by
  induction left generalizing state with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [processEncodedItemsWith, inductionHypothesis]

theorem processEncodedItemsWith_simulates
    {SourceState TargetState Alpha : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (state : SourceState) (format : Format Alpha) (values : List Alpha) :
    denote (processEncodedItemsWith sourcePush state format values) =
      processEncodedItemsWith targetPush (denote state) format values := by
  induction values generalizing state with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [processEncodedItemsWith,
        processValueWith_simulates denote sourcePush targetPush pushSimulates,
        inductionHypothesis]

/-- Process one codec list with exactly one canonical array header. -/
@[specialize push format] def processEncodedListWith {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (values : List Alpha) : State :=
  processEncodedItemsWith push (push state ⟨1, values.length⟩)
    format values

theorem processEncodedListWith_eq_processValueWith {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (values : List Alpha) :
    processEncodedListWith push state format values =
      processValueWith push ((list format).encode values) state := by
  unfold processEncodedListWith
  rw [processEncodedItemsWith_eq_mappedFoldl]
  simp only [Codec.list, processValueWith, List.length_map]

theorem processEncodedListWith_simulates
    {SourceState TargetState Alpha : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (state : SourceState) (format : Format Alpha) (values : List Alpha) :
    denote (processEncodedListWith sourcePush state format values) =
      processEncodedListWith targetPush (denote state) format values := by
  calc
    denote (processEncodedListWith sourcePush state format values) =
        denote (processValueWith sourcePush
          ((list format).encode values) state) := by
      rw [processEncodedListWith_eq_processValueWith]
    _ = processValueWith targetPush ((list format).encode values)
        (denote state) :=
      processValueWith_simulates denote sourcePush targetPush
        pushSimulates _ state
    _ = processEncodedListWith targetPush (denote state) format values := by
      rw [processEncodedListWith_eq_processValueWith]

/-- Number of immediate list items across ordered segments. This recursive
form avoids allocating `segments.map List.length`. -/
def encodedSegmentsLength {Alpha : Type} : List (List Alpha) → Nat
  | [] => 0
  | values :: rest => values.length + encodedSegmentsLength rest

theorem encodedSegmentsLength_eq_flatten_length {Alpha : Type}
    (segments : List (List Alpha)) :
    encodedSegmentsLength segments = segments.flatten.length := by
  induction segments with
  | nil => rfl
  | cons values rest inductionHypothesis =>
      simp [encodedSegmentsLength, inductionHypothesis]

/-- Process all items from ordered list segments without an enclosing array
header and without constructing their concatenation. -/
@[specialize push format] def processEncodedSegmentsItemsWith
    {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) : List (List Alpha) → State
  | [] => state
  | values :: rest =>
      processEncodedSegmentsItemsWith push
        (processEncodedItemsWith push state format values) format rest

theorem processEncodedSegmentsItemsWith_eq_items {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (segments : List (List Alpha)) :
    processEncodedSegmentsItemsWith push state format segments =
      processEncodedItemsWith push state format segments.flatten := by
  induction segments generalizing state with
  | nil => rfl
  | cons values rest inductionHypothesis =>
      rw [processEncodedSegmentsItemsWith, List.flatten_cons,
        processEncodedItemsWith_append, inductionHypothesis]

theorem processEncodedSegmentsItemsWith_simulates
    {SourceState TargetState Alpha : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (state : SourceState) (format : Format Alpha)
    (segments : List (List Alpha)) :
    denote
        (processEncodedSegmentsItemsWith sourcePush state format segments) =
      processEncodedSegmentsItemsWith targetPush (denote state) format
        segments := by
  induction segments generalizing state with
  | nil => rfl
  | cons values rest inductionHypothesis =>
      simp [processEncodedSegmentsItemsWith,
        processEncodedItemsWith_simulates denote sourcePush targetPush
          pushSimulates,
        inductionHypothesis]

/-- Process ordered list segments beneath one canonical array header. -/
@[specialize push format] def processEncodedSegmentsWith {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (segments : List (List Alpha)) : State :=
  processEncodedSegmentsItemsWith push
    (push state ⟨1, encodedSegmentsLength segments⟩) format segments

theorem processEncodedSegmentsWith_eq_processValueWith {State Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (segments : List (List Alpha)) :
    processEncodedSegmentsWith push state format segments =
      processValueWith push ((list format).encode segments.flatten) state := by
  unfold processEncodedSegmentsWith
  rw [processEncodedSegmentsItemsWith_eq_items,
    encodedSegmentsLength_eq_flatten_length]
  exact processEncodedListWith_eq_processValueWith
    push state format segments.flatten

theorem processEncodedSegmentsWith_simulates
    {SourceState TargetState Alpha : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (state : SourceState) (format : Format Alpha)
    (segments : List (List Alpha)) :
    denote (processEncodedSegmentsWith sourcePush state format segments) =
      processEncodedSegmentsWith targetPush (denote state) format segments := by
  calc
    denote (processEncodedSegmentsWith sourcePush state format segments) =
        denote (processValueWith sourcePush
          ((list format).encode segments.flatten) state) := by
      rw [processEncodedSegmentsWith_eq_processValueWith]
    _ = processValueWith targetPush
        ((list format).encode segments.flatten) (denote state) :=
      processValueWith_simulates denote sourcePush targetPush
        pushSimulates _ state
    _ = processEncodedSegmentsWith targetPush (denote state) format
        segments := by
      rw [processEncodedSegmentsWith_eq_processValueWith]

/-- Number of immediate items in one ordered block expansion. The expansion
is evaluated one block at a time; the flattened list is not constructed. -/
@[specialize expand] def encodedFlatMapLength {Block Alpha : Type}
    (expand : Block → List Alpha) : List Block → Nat
  | [] => 0
  | block :: rest =>
      (expand block).length + encodedFlatMapLength expand rest

theorem encodedFlatMapLength_eq_flatMap_length {Block Alpha : Type}
    (expand : Block → List Alpha) (blocks : List Block) :
    encodedFlatMapLength expand blocks = (blocks.flatMap expand).length := by
  induction blocks with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      simp [encodedFlatMapLength, inductionHypothesis]

/-- Process block expansions without an enclosing array header and without
constructing `blocks.flatMap expand`. -/
@[specialize push format expand] def processEncodedFlatMapItemsWith
    {State Block Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (expand : Block → List Alpha) :
    List Block → State
  | [] => state
  | block :: rest =>
      processEncodedFlatMapItemsWith push
        (processEncodedItemsWith push state format (expand block))
        format expand rest

theorem processEncodedFlatMapItemsWith_eq_items
    {State Block Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (expand : Block → List Alpha)
    (blocks : List Block) :
    processEncodedFlatMapItemsWith push state format expand blocks =
      processEncodedItemsWith push state format (blocks.flatMap expand) := by
  induction blocks generalizing state with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      rw [processEncodedFlatMapItemsWith, List.flatMap_cons,
        processEncodedItemsWith_append, inductionHypothesis]

theorem processEncodedFlatMapItemsWith_simulates
    {SourceState TargetState Block Alpha : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (state : SourceState) (format : Format Alpha)
    (expand : Block → List Alpha) (blocks : List Block) :
    denote
        (processEncodedFlatMapItemsWith sourcePush state format expand blocks) =
      processEncodedFlatMapItemsWith targetPush (denote state) format expand
        blocks := by
  induction blocks generalizing state with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      simp [processEncodedFlatMapItemsWith,
        processEncodedItemsWith_simulates denote sourcePush targetPush
          pushSimulates,
        inductionHypothesis]

/-- Process an ordered block expansion beneath one canonical array header. -/
@[specialize push format expand] def processEncodedFlatMapWith
    {State Block Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (expand : Block → List Alpha)
    (blocks : List Block) : State :=
  processEncodedFlatMapItemsWith push
    (push state ⟨1, encodedFlatMapLength expand blocks⟩)
    format expand blocks

theorem processEncodedFlatMapWith_eq_processValueWith
    {State Block Alpha : Type}
    (push : State → Node → State) (state : State)
    (format : Format Alpha) (expand : Block → List Alpha)
    (blocks : List Block) :
    processEncodedFlatMapWith push state format expand blocks =
      processValueWith push
        ((list format).encode (blocks.flatMap expand)) state := by
  unfold processEncodedFlatMapWith
  rw [processEncodedFlatMapItemsWith_eq_items,
    encodedFlatMapLength_eq_flatMap_length]
  exact processEncodedListWith_eq_processValueWith
    push state format (blocks.flatMap expand)

theorem processEncodedFlatMapWith_simulates
    {SourceState TargetState Block Alpha : Type}
    (denote : SourceState → TargetState)
    (sourcePush : SourceState → Node → SourceState)
    (targetPush : TargetState → Node → TargetState)
    (pushSimulates : ∀ state node,
      denote (sourcePush state node) = targetPush (denote state) node)
    (state : SourceState) (format : Format Alpha)
    (expand : Block → List Alpha) (blocks : List Block) :
    denote (processEncodedFlatMapWith sourcePush state format expand blocks) =
      processEncodedFlatMapWith targetPush (denote state) format expand
        blocks := by
  calc
    denote
        (processEncodedFlatMapWith sourcePush state format expand blocks) =
      denote (processValueWith sourcePush
        ((list format).encode (blocks.flatMap expand)) state) := by
        rw [processEncodedFlatMapWith_eq_processValueWith]
    _ = processValueWith targetPush
        ((list format).encode (blocks.flatMap expand)) (denote state) :=
      processValueWith_simulates denote sourcePush targetPush
        pushSimulates _ state
    _ = processEncodedFlatMapWith targetPush (denote state) format expand
        blocks := by
      rw [processEncodedFlatMapWith_eq_processValueWith]

/-- Semantic specializations for callers that do not supply another proved
node transition. -/
def processEncodedItems {Alpha : Type} (state : HashState)
    (format : Format Alpha) (values : List Alpha) : HashState :=
  processEncodedItemsWith pushNode state format values

def processEncodedList {Alpha : Type} (state : HashState)
    (format : Format Alpha) (values : List Alpha) : HashState :=
  processEncodedListWith pushNode state format values

def processEncodedSegmentsItems {Alpha : Type} (state : HashState)
    (format : Format Alpha) (segments : List (List Alpha)) : HashState :=
  processEncodedSegmentsItemsWith pushNode state format segments

def processEncodedSegments {Alpha : Type} (state : HashState)
    (format : Format Alpha) (segments : List (List Alpha)) : HashState :=
  processEncodedSegmentsWith pushNode state format segments

def processEncodedFlatMapItems {Block Alpha : Type} (state : HashState)
    (format : Format Alpha) (expand : Block → List Alpha)
    (blocks : List Block) : HashState :=
  processEncodedFlatMapItemsWith pushNode state format expand blocks

def processEncodedFlatMap {Block Alpha : Type} (state : HashState)
    (format : Format Alpha) (expand : Block → List Alpha)
    (blocks : List Block) : HashState :=
  processEncodedFlatMapWith pushNode state format expand blocks

theorem processEncodedList_eq_processValue {Alpha : Type}
    (state : HashState) (format : Format Alpha) (values : List Alpha) :
    processEncodedList state format values =
      processValue ((list format).encode values) state := by
  exact processEncodedListWith_eq_processValueWith
    pushNode state format values

theorem processEncodedSegments_eq_processValue {Alpha : Type}
    (state : HashState) (format : Format Alpha)
    (segments : List (List Alpha)) :
    processEncodedSegments state format segments =
      processValue ((list format).encode segments.flatten) state := by
  exact processEncodedSegmentsWith_eq_processValueWith
    pushNode state format segments

theorem processEncodedFlatMap_eq_processValue {Block Alpha : Type}
    (state : HashState) (format : Format Alpha)
    (expand : Block → List Alpha) (blocks : List Block) :
    processEncodedFlatMap state format expand blocks =
      processValue ((list format).encode (blocks.flatMap expand)) state := by
  exact processEncodedFlatMapWith_eq_processValueWith
    pushNode state format expand blocks

theorem processValue_eq_processNodes (value : Value) :
    ∀ state, processValue value state = processNodes state (nodes value) :=
  Value.rec
    (motive_1 := fun value => ∀ state,
      processValue value state = processNodes state (nodes value))
    (motive_2 := fun values => ∀ state,
      values.foldl (fun current child => processValue child current) state =
        processNodes state (values.flatMap nodes))
    (fun value state => by
      unfold processValue
      rw [processValueWith, nodes]
      rfl)
    (fun values valuesInduction state => by
      unfold processValue
      rw [processValueWith, nodes]
      change
        values.foldl
            (fun current child => processValueWith pushNode child current)
            (pushNode state ⟨1, values.length⟩) =
          processNodes (pushNode state ⟨1, values.length⟩)
            (values.flatMap nodes)
      simpa only [processValueWith_pushNode] using
        valuesInduction (pushNode state ⟨1, values.length⟩))
    (by
      intro state
      rfl)
    (fun head tail headInduction tailInduction state => by
      simp [processNodes, headInduction, tailInduction,
        List.foldl_append])
    value

@[simp] theorem Node.words_length (node : Node) : node.words.length = 4 := by
  rfl

theorem canonicalWords_length (value : Value) :
    (canonicalWords value).length = (nodes value).length * 4 := by
  simp [canonicalWords]

theorem absorbBlocksFast_add (first second : Nat)
    (state : Poseidon2.State) (input : List F) :
    Poseidon2.absorbBlocksFast (first + second) state input =
      Poseidon2.absorbBlocksFast second
        (Poseidon2.absorbBlocksFast first state input)
        (input.drop (first * Poseidon2.rate)) := by
  induction first generalizing state input with
  | zero => simp [Poseidon2.absorbBlocksFast]
  | succ first inductionHypothesis =>
      rw [Nat.succ_add, Poseidon2.absorbBlocksFast,
        inductionHypothesis, Poseidon2.absorbBlocksFast]
      rw [List.drop_drop]
      congr 2
      simp [Nat.succ_mul, Nat.add_comm]

@[simp] theorem take_nodeStream (carry : F) (node : Node)
    (tail : List F) :
    (carry :: node.words ++ tail).take Poseidon2.rate = node.block carry := by
  rcases node with ⟨tag, value⟩
  simp [Node.words, Node.block, Poseidon2.rate]

@[simp] theorem drop_nodeStream (carry : F) (node : Node)
    (tail : List F) :
    (carry :: node.words ++ tail).drop Poseidon2.rate =
      node.nextCarry :: tail := by
  rcases node with ⟨tag, value⟩
  simp [Node.words, Node.nextCarry, Poseidon2.rate]

theorem processNodes_finalAbsorb (state : HashState)
    (stream : List Node) :
    Poseidon2.absorbBlocksFast (stream.length + 1) state.sponge
        (state.carry :: stream.flatMap Node.words) =
      Poseidon2.absorbBlock (processNodes state stream).sponge
        [(processNodes state stream).carry] := by
  induction stream generalizing state with
  | nil =>
      simp [Poseidon2.absorbBlocksFast, processNodes, Poseidon2.rate]
  | cons node rest inductionHypothesis =>
      rw [List.length_cons]
      change Poseidon2.absorbBlocksFast (rest.length + 1 + 1) state.sponge
          (state.carry :: node.words ++ rest.flatMap Node.words) = _
      rw [Poseidon2.absorbBlocksFast, take_nodeStream, drop_nodeStream]
      change Poseidon2.absorbBlocksFast (rest.length + 1)
          (pushNode state node).sponge
          ((pushNode state node).carry :: rest.flatMap Node.words) = _
      rw [inductionHypothesis]
      rfl

def initialState : HashState where
  sponge := Poseidon2.absorbBlocksFast 7 Poseidon2.zeroState
    Package.identityDomain
  carry := Package.identityDomain.getD 28 0

@[simp] theorem identityDomain_length : Package.identityDomain.length = 29 := by
  rfl

theorem identityCanonical_blockCount (value : Value) :
    ((Package.identityDomain ++ canonicalWords value).length +
        Poseidon2.rate - 1) / Poseidon2.rate =
      (nodes value).length + 8 := by
  rw [List.length_append, identityDomain_length, canonicalWords_length]
  norm_num [Poseidon2.rate]
  have aligned : 29 + (nodes value).length * 4 + 3 =
      ((nodes value).length + 8) * 4 := by
    omega
  rw [aligned]
  simp

theorem absorbFirstSeven (value : Value) :
    Poseidon2.absorbBlocksFast 7 Poseidon2.zeroState
        (Package.identityDomain ++ canonicalWords value) =
      initialState.sponge := by
  simp [initialState, Poseidon2.absorbBlocksFast, Poseidon2.rate,
    Package.identityDomain]

theorem dropFirstSeven (value : Value) :
    (Package.identityDomain ++ canonicalWords value).drop
        (7 * Poseidon2.rate) =
      initialState.carry :: canonicalWords value := by
  simp [initialState, Poseidon2.rate, Package.identityDomain]

theorem streamedAbsorbed_eq (value : Value) :
    Poseidon2.absorbBlocksFast
        (((Package.identityDomain ++ canonicalWords value).length +
          Poseidon2.rate - 1) / Poseidon2.rate)
        Poseidon2.zeroState
        (Package.identityDomain ++ canonicalWords value) =
      Poseidon2.absorbBlock (processValue value initialState).sponge
        [(processValue value initialState).carry] := by
  rw [identityCanonical_blockCount]
  calc
    Poseidon2.absorbBlocksFast ((nodes value).length + 8)
        Poseidon2.zeroState
        (Package.identityDomain ++ canonicalWords value) =
      Poseidon2.absorbBlocksFast (7 + ((nodes value).length + 1))
        Poseidon2.zeroState
        (Package.identityDomain ++ canonicalWords value) := by
          congr 1
          omega
    _ = Poseidon2.absorbBlocksFast ((nodes value).length + 1)
        (Poseidon2.absorbBlocksFast 7 Poseidon2.zeroState
          (Package.identityDomain ++ canonicalWords value))
        ((Package.identityDomain ++ canonicalWords value).drop
          (7 * Poseidon2.rate)) := by
          rw [absorbBlocksFast_add]
    _ = Poseidon2.absorbBlocksFast ((nodes value).length + 1)
        initialState.sponge
        (initialState.carry :: canonicalWords value) := by
          rw [absorbFirstSeven, dropFirstSeven]
    _ = Poseidon2.absorbBlock
        (processNodes initialState (nodes value)).sponge
        [(processNodes initialState (nodes value)).carry] := by
          exact processNodes_finalAbsorb initialState (nodes value)
    _ = Poseidon2.absorbBlock (processValue value initialState).sponge
        [(processValue value initialState).carry] := by
          rw [processValue_eq_processNodes]

def finalize (state : HashState) : List F :=
  let absorbed := Poseidon2.absorbBlock state.sponge [state.carry]
  let padded := Poseidon2.permute ((List.range Poseidon2.width).map fun lane =>
    if lane = 0 then absorbed.getD 0 0 + 1 else absorbed.getD lane 0)
  padded.take Poseidon2.digestLen

/-- Bounded-memory executable relation identifier for one canonical value. -/
def relationIdentifierValueFast (value : Value) : List F :=
  finalize (processValue value initialState)

theorem relationIdentifierValueFast_eq (value : Value) :
    relationIdentifierValueFast value =
      Package.relationIdentifierValue value := by
  rw [Package.relationIdentifierValue, Poseidon2.hash_eq_hashFast,
    ← canonicalWords_eq_valuePreimage]
  unfold relationIdentifierValueFast finalize Poseidon2.hashFast
  dsimp only
  rw [streamedAbsorbed_eq]

end NightstreamFPrime.Export.StreamingIdentity
