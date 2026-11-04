import NightstreamFPrime.Export.Package

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
  pending : List F

def pushNode (state : HashState) (node : Node) : HashState :=
  let buffer := state.pending ++ node.words
  if Poseidon2.rate ≤ buffer.length then
    { sponge := Poseidon2.absorbBlock state.sponge (buffer.take Poseidon2.rate)
      pending := buffer.drop Poseidon2.rate }
  else
    { sponge := state.sponge
      pending := buffer }

def processNodes (state : HashState) (stream : List Node) : HashState :=
  stream.foldl pushNode state

@[specialize push] def processValueWith {State : Type}
    (push : State → Node → State) :
    Value → State → State
  | .atom value, state => push state ⟨0, value⟩
  | .array values, state =>
      values.foldl (fun current child => processValueWith push child current)
        (push state ⟨1, values.length⟩)

def processValue : Value → HashState → HashState :=
  processValueWith pushNode

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

private theorem rate_eq : Poseidon2.rate = 12 := rfl

theorem processNodes_finalAbsorb (state : HashState) (stream : List Node)
    (odd : state.pending.length % 2 = 1)
    (short : state.pending.length < Poseidon2.rate) :
    Poseidon2.absorbBlocksFast
        ((state.pending.length + stream.length * 4 + Poseidon2.rate - 1) /
          Poseidon2.rate)
        state.sponge (state.pending ++ stream.flatMap Node.words) =
      Poseidon2.absorbBlock (processNodes state stream).sponge
        (processNodes state stream).pending := by
  rcases state with ⟨sponge, pending⟩
  dsimp only at odd short ⊢
  induction stream generalizing sponge pending with
  | nil =>
      have count : (pending.length + (List.length ([] : List Node)) * 4 +
          Poseidon2.rate - 1) / Poseidon2.rate = 0 + 1 := by
        simp only [List.length_nil, rate_eq] at short ⊢
        omega
      rw [count, Poseidon2.absorbBlocksFast, Poseidon2.absorbBlocksFast]
      simp [processNodes, List.take_of_length_le (Nat.le_of_lt short)]
  | cons node rest inductionHypothesis =>
      have words : node.words.length = 4 := Node.words_length node
      change Poseidon2.absorbBlocksFast _ sponge
          (pending ++ (node.words ++ rest.flatMap Node.words)) =
        Poseidon2.absorbBlock
          (processNodes (pushNode ⟨sponge, pending⟩ node) rest).sponge
          (processNodes (pushNode ⟨sponge, pending⟩ node) rest).pending
      rw [← List.append_assoc, List.length_cons]
      by_cases full : Poseidon2.rate ≤ pending.length + 4
      · have pushed : pushNode ⟨sponge, pending⟩ node =
            ⟨Poseidon2.absorbBlock sponge
                ((pending ++ node.words).take Poseidon2.rate),
              (pending ++ node.words).drop Poseidon2.rate⟩ := by
          simp [pushNode, words, full]
        have count : (pending.length + (rest.length + 1) * 4 +
              Poseidon2.rate - 1) / Poseidon2.rate =
            (((pending ++ node.words).drop Poseidon2.rate).length +
              rest.length * 4 + Poseidon2.rate - 1) / Poseidon2.rate + 1 := by
          simp only [List.length_drop, List.length_append, words, rate_eq]
            at full ⊢
          omega
        have long : Poseidon2.rate ≤ (pending ++ node.words).length := by
          simp only [List.length_append, words]
          exact full
        rw [count, Poseidon2.absorbBlocksFast,
          List.take_append_of_le_length long,
          List.drop_append_of_le_length long, pushed]
        apply inductionHypothesis
        · simp only [List.length_drop, List.length_append, words, rate_eq]
            at full short ⊢
          omega
        · simp only [List.length_drop, List.length_append, words, rate_eq]
            at full short ⊢
          omega
      · have pushed : pushNode ⟨sponge, pending⟩ node =
            ⟨sponge, pending ++ node.words⟩ := by
          simp [pushNode, words, full]
        have count : (pending.length + (rest.length + 1) * 4 +
              Poseidon2.rate - 1) / Poseidon2.rate =
            ((pending ++ node.words).length + rest.length * 4 +
              Poseidon2.rate - 1) / Poseidon2.rate := by
          simp only [List.length_append, words]
          congr 1
          omega
        rw [count, pushed]
        apply inductionHypothesis
        · simp only [List.length_append, words]
          omega
        · simp only [List.length_append, words, rate_eq] at full ⊢
          omega

def prefixState (blocks : Nat) (words : List F) : HashState where
  sponge := Poseidon2.absorbBlocksFast blocks Poseidon2.zeroState words
  pending := words.drop (blocks * Poseidon2.rate)

theorem absorbBlocksFast_append (blocks : Nat) (state : Poseidon2.State)
    (words tail : List F) (covered : blocks * Poseidon2.rate ≤ words.length) :
    Poseidon2.absorbBlocksFast blocks state (words ++ tail) =
      Poseidon2.absorbBlocksFast blocks state words := by
  induction blocks generalizing state words with
  | zero => rfl
  | succ blocks inductionHypothesis =>
      have long : Poseidon2.rate ≤ words.length := by
        rw [Nat.succ_mul] at covered
        omega
      rw [Poseidon2.absorbBlocksFast, Poseidon2.absorbBlocksFast,
        List.take_append_of_le_length long,
        List.drop_append_of_le_length long]
      apply inductionHypothesis
      rw [List.length_drop]
      rw [Nat.succ_mul] at covered
      omega

theorem prefixStreamed_eq (blocks : Nat) (words : List F) (value : Value)
    (covered : blocks * Poseidon2.rate ≤ words.length)
    (short : words.length < blocks * Poseidon2.rate + Poseidon2.rate)
    (odd : (words.length - blocks * Poseidon2.rate) % 2 = 1) :
    Poseidon2.absorbBlocksFast
        (((words ++ canonicalWords value).length + Poseidon2.rate - 1) /
          Poseidon2.rate)
        Poseidon2.zeroState (words ++ canonicalWords value) =
      Poseidon2.absorbBlock (processValue value (prefixState blocks words)).sponge
        (processValue value (prefixState blocks words)).pending := by
  have pendingLength :
      (prefixState blocks words).pending.length =
        words.length - blocks * Poseidon2.rate := by
    simp [prefixState]
  have count :
      ((words ++ canonicalWords value).length + Poseidon2.rate - 1) /
          Poseidon2.rate =
        blocks + ((prefixState blocks words).pending.length +
          (nodes value).length * 4 + Poseidon2.rate - 1) / Poseidon2.rate := by
    rw [pendingLength, List.length_append, canonicalWords_length]
    simp only [rate_eq] at covered short ⊢
    omega
  rw [count, absorbBlocksFast_add, absorbBlocksFast_append _ _ _ _ covered,
    List.drop_append_of_le_length covered, processValue_eq_processNodes]
  exact processNodes_finalAbsorb (prefixState blocks words) (nodes value)
    (by rw [pendingLength]; exact odd)
    (by rw [pendingLength]; omega)

def initialState : HashState := prefixState 2 Package.identityDomain

@[simp] theorem identityDomain_length : Package.identityDomain.length = 29 := by
  rfl

theorem streamedAbsorbed_eq (value : Value) :
    Poseidon2.absorbBlocksFast
        (((Package.identityDomain ++ canonicalWords value).length +
          Poseidon2.rate - 1) / Poseidon2.rate)
        Poseidon2.zeroState
        (Package.identityDomain ++ canonicalWords value) =
      Poseidon2.absorbBlock (processValue value initialState).sponge
        (processValue value initialState).pending := by
  apply prefixStreamed_eq <;> simp [rate_eq]

def finalize (state : HashState) : List F :=
  let absorbed := Poseidon2.absorbBlock state.sponge state.pending
  let padded := Poseidon2.permute ((List.range Poseidon2.width).map fun lane =>
    if lane = 0 then absorbed.getD 0 0 + 1 else absorbed.getD lane 0)
  padded.take Poseidon2.digestLen

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
