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

def processValue : Value → HashState → HashState
  | .atom value, state => pushNode state ⟨0, value⟩
  | .array values, state =>
      values.foldl (fun current child => processValue child current)
        (pushNode state ⟨1, values.length⟩)

theorem processValue_eq_processNodes (value : Value) :
    ∀ state, processValue value state = processNodes state (nodes value) :=
  Value.rec
    (motive_1 := fun value => ∀ state,
      processValue value state = processNodes state (nodes value))
    (motive_2 := fun values => ∀ state,
      values.foldl (fun current child => processValue child current) state =
        processNodes state (values.flatMap nodes))
    (fun value state => by
      simp [processValue, processNodes, nodes])
    (fun values valuesInduction state => by
      simp [processValue, nodes, processNodes, valuesInduction])
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
