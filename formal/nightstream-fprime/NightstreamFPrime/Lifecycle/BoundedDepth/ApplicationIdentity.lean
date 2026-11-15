import NightstreamFPrime.Lifecycle.RandomOracle.WordSchema
import NightstreamFPrime.Spec.Sponge

/-! The identity of the step packages with a Rust application (`SPEC.md` section 4.4): the
Poseidon2 hash of the word stream of Rust `BoundedDepthPackages::with_application`
(`crates/nightstream-fprime/src/bounded_depth/application.rs`).

The records here are a schema of the Rust records: counted lists, fixed tuples and tags, with
the names of the Rust fields. They do not say what a record means, and no theorem here relates a
record to a relation. An index or a count is a natural number, and a constant or a coefficient
is a field element, because Rust refuses a record constant at or above `q`.

`Application.words` is the word stream. It fixes the records (`Application.words_injective`),
also before any later words (`Application.body_prefixFree`). Rust hashes each word as a field
element (`Application.input`) and refuses a stream with a word at or above `q`
(`Application.Canonical`). So the hash input of a canonical stream has the words as its values
(`Application.input_val`) and fixes the records (`Application.input_injective`), and two
canonical applications with one identity are equal or give two distinct padded inputs with one
sponge digest (`Application.identity_binds`).

The stream does not hold the nodes of a recipe or the rows that give the output state. Rust
reads them only to make a witness. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.ApplicationIdentity

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.WordSchema

/-- The identity domain of Rust `IDENTITY_DOMAIN`. -/
def tagText : String := "Nightstream/OptionB/application/combined-native-experiment/v1"

/-- One word for each byte of the tag. -/
def tagWords : List ℕ := tagText.toList.map Char.toNat

/-- The tag as field elements. -/
def domain : List F := fieldWords tagWords

/-- A term: a variable or a column, then its coefficient. -/
def term : ℕ × F → List ℕ := pair word fieldWord

theorem term_prefixFree : PrefixFree term := word_prefixFree.pair fieldWord_prefixFree

/-- Rust `Linear`: a constant and terms. -/
structure Linear where
  constant : F
  terms : List (ℕ × F)

/-- The constant, the term count, then the terms. -/
def Linear.words (linear : Linear) : List ℕ :=
  fieldWord linear.constant ++ counted term linear.terms

theorem Linear.words_prefixFree : PrefixFree Linear.words :=
  (fieldWord_prefixFree.pair term_prefixFree.counted).comap
    (view := fun linear : Linear => (linear.constant, linear.terms))
    fun left right same => by cases left; cases right; simpa using same

/-- A row of the application records (Rust `ApplicationRowHeader` with its `ApplicationTerm`s):
the A, B and C forms. -/
structure RecordRow where
  a : Linear
  b : Linear
  c : Linear

/-- The header, a constant and a term count for each form, then the terms of the forms. -/
def RecordRow.words (row : RecordRow) : List ℕ :=
  [row.a.constant.val, row.a.terms.length, row.b.constant.val, row.b.terms.length,
    row.c.constant.val, row.c.terms.length] ++
    (sized term row.a.terms ++ (sized term row.b.terms ++ sized term row.c.terms))

theorem RecordRow.words_prefixFree : PrefixFree RecordRow.words := by
  rintro ⟨⟨constantA, termsA⟩, ⟨constantB, termsB⟩, ⟨constantC, termsC⟩⟩
    ⟨⟨constantA', termsA'⟩, ⟨constantB', termsB'⟩, ⟨constantC', termsC'⟩⟩ rest rest' same
  simp only [RecordRow.words, List.cons_append, List.nil_append, List.cons.injEq,
    List.append_assoc] at same
  obtain ⟨constantsA, lengthsA, constantsB, lengthsB, constantsC, lengthsC, tail⟩ := same
  obtain ⟨sameA, tailB⟩ := term_prefixFree.sized lengthsA tail
  obtain ⟨sameB, tailC⟩ := term_prefixFree.sized lengthsB tailB
  obtain ⟨sameC, rests⟩ := term_prefixFree.sized lengthsC tailC
  rw [Fin.ext constantsA, Fin.ext constantsB, Fin.ext constantsC, sameA, sameB, sameC]
  exact ⟨rfl, rests⟩

/-- Rust `NativeBlock`: the first local of a permutation and its 16 input lanes. -/
structure Block where
  firstLocal : ℕ
  lanes : List.Vector Linear 16

def Block.words (block : Block) : List ℕ :=
  word block.firstLocal ++ array Linear.words block.lanes

theorem Block.words_prefixFree : PrefixFree Block.words :=
  (word_prefixFree.pair Linear.words_prefixFree.array).comap
    (view := fun block : Block => (block.firstLocal, block.lanes))
    fun left right same => by cases left; cases right; simpa using same

/-- A row of a step package (Rust `Row`): each form with its own count. -/
structure Row where
  a : Linear
  b : Linear
  c : Linear

def Row.words (row : Row) : List ℕ := row.a.words ++ (row.b.words ++ row.c.words)

theorem Row.words_prefixFree : PrefixFree Row.words :=
  (Linear.words_prefixFree.pair (Linear.words_prefixFree.pair Linear.words_prefixFree)).comap
    (view := fun row : Row => (row.a, row.b, row.c))
    fun left right same => by cases left; cases right; simpa using same

/-- The index of a row of the first piece, then the row. -/
def indexedRow : ℕ × Row → List ℕ := pair word Row.words

theorem indexedRow_prefixFree : PrefixFree indexedRow := word_prefixFree.pair Row.words_prefixFree

/-- A group of Rust `GuardFusion`. -/
structure FusionGroup where
  sourceRecords : List (ℕ × ℕ)
  sources : List (ℕ × Row)
  certificates : List (ℕ × Row)
  row : Row

def FusionGroup.words (group : FusionGroup) : List ℕ :=
  counted (pair word word) group.sourceRecords ++
    (counted indexedRow group.sources ++ (counted indexedRow group.certificates ++ group.row.words))

theorem FusionGroup.words_prefixFree : PrefixFree FusionGroup.words :=
  ((word_prefixFree.pair word_prefixFree).counted.pair (indexedRow_prefixFree.counted.pair
    (indexedRow_prefixFree.counted.pair Row.words_prefixFree))).comap
    (view := fun group : FusionGroup =>
      (group.sourceRecords, group.sources, group.certificates, group.row))
    fun left right same => by cases left; cases right; simpa using same

/-- Rust `Bit` of a word packing: the constant 0, the constant 1, or a column. -/
inductive Bit where
  | zero
  | one
  | column (index : ℕ)

/-- The tag, then the column, which is 0 for a constant. -/
def Bit.words : Bit → List ℕ
  | .zero => [0, 0]
  | .one => [1, 0]
  | .column index => [2, index]

theorem Bit.words_prefixFree : PrefixFree Bit.words := by
  rintro (_ | _ | index) (_ | _ | index') rest rest' same <;>
    simp [Bit.words] at same <;> simp [same]

/-- A group of Rust `WordPacking`. -/
structure PackingGroup where
  sources : List (ℕ × Row)
  certificates : List (ℕ × Row)
  guard : Linear
  pairs : List (Bit × Bit)

def PackingGroup.words (group : PackingGroup) : List ℕ :=
  counted indexedRow group.sources ++ (counted indexedRow group.certificates ++
    (group.guard.words ++ counted (pair Bit.words Bit.words) group.pairs))

theorem PackingGroup.words_prefixFree : PrefixFree PackingGroup.words :=
  (indexedRow_prefixFree.counted.pair (indexedRow_prefixFree.counted.pair
    (Linear.words_prefixFree.pair (Bit.words_prefixFree.pair Bit.words_prefixFree).counted))).comap
    (view := fun group : PackingGroup =>
      (group.sources, group.certificates, group.guard, group.pairs))
    fun left right same => by cases left; cases right; simpa using same

/-- Rust `Product` of the packed Booleans. -/
structure Product where
  row : ℕ
  left : ℕ
  right : ℕ
  output : ℕ

def Product.words (product : Product) : List ℕ :=
  word product.row ++ (word product.left ++ (word product.right ++ word product.output))

theorem Product.words_prefixFree : PrefixFree Product.words :=
  (word_prefixFree.pair (word_prefixFree.pair (word_prefixFree.pair word_prefixFree))).comap
    (view := fun product : Product =>
      (product.row, product.left, product.right, product.output))
    fun left right same => by cases left; cases right; simpa using same

/-- Rust `ZeroProduct` of the packed Booleans: a row and two inputs, each a column and its
complement flag. -/
structure ZeroProduct where
  row : ℕ
  leftColumn : ℕ
  leftComplement : Bool
  rightColumn : ℕ
  rightComplement : Bool

def ZeroProduct.words (zero : ZeroProduct) : List ℕ :=
  word zero.row ++ (word zero.leftColumn ++ (flag zero.leftComplement ++
    (word zero.rightColumn ++ flag zero.rightComplement)))

theorem ZeroProduct.words_prefixFree : PrefixFree ZeroProduct.words :=
  (word_prefixFree.pair (word_prefixFree.pair (flag_prefixFree.pair
    (word_prefixFree.pair flag_prefixFree)))).comap
    (view := fun zero : ZeroProduct =>
      (zero.row, zero.leftColumn, zero.leftComplement, zero.rightColumn, zero.rightComplement))
    fun left right same => by cases left; cases right; simpa using same

/-- A slot run of a package: its first column, its count and its kind. -/
structure Slot where
  start : ℕ
  count : ℕ
  kind : ℕ

def Slot.words (slot : Slot) : List ℕ := word slot.start ++ (word slot.count ++ word slot.kind)

theorem Slot.words_prefixFree : PrefixFree Slot.words :=
  (word_prefixFree.pair (word_prefixFree.pair word_prefixFree)).comap
    (view := fun slot : Slot => (slot.start, slot.count, slot.kind))
    fun left right same => by cases left; cases right; simpa using same

/-- What Rust realizes for one step package with the application: the guard fusion, the word
packing, the packed Booleans (their first column, sources, removed rows, products and zero
products) and the slot runs. -/
structure Layout where
  fusion : List FusionGroup
  packing : List PackingGroup
  packedStart : ℕ
  packedSources : List ℕ
  removed : List ℕ
  products : List Product
  zeros : List ZeroProduct
  slots : List Slot

def Layout.words (layout : Layout) : List ℕ :=
  counted FusionGroup.words layout.fusion ++ (counted PackingGroup.words layout.packing ++
    (word layout.packedStart ++ (counted word layout.packedSources ++
      (counted word layout.removed ++ (counted Product.words layout.products ++
        (counted ZeroProduct.words layout.zeros ++ counted Slot.words layout.slots))))))

theorem Layout.words_prefixFree : PrefixFree Layout.words :=
  (FusionGroup.words_prefixFree.counted.pair (PackingGroup.words_prefixFree.counted.pair
    (word_prefixFree.pair (word_prefixFree.counted.pair (word_prefixFree.counted.pair
      (Product.words_prefixFree.counted.pair (ZeroProduct.words_prefixFree.counted.pair
        Slot.words_prefixFree.counted))))))).comap
    (view := fun layout : Layout =>
      (layout.fusion, layout.packing, layout.packedStart, layout.packedSources, layout.removed,
        layout.products, layout.zeros, layout.slots))
    fun left right same => by cases left; cases right; simpa using same

/-- What the identity binds: the identity of the Lean packages, the witness word count, the
rows, the row of each recipe, the blocks, and the layouts of the light and the refresh
package. -/
structure Application where
  package : List.Vector F 4
  witnessWords : ℕ
  rows : List RecordRow
  recipeRows : List ℕ
  blocks : List Block
  light : Layout
  refresh : Layout

namespace Application

/-- The words after the tag. -/
def body (application : Application) : List ℕ :=
  array fieldWord application.package ++ (word application.witnessWords ++
    (counted RecordRow.words application.rows ++ (counted word application.recipeRows ++
      (counted Block.words application.blocks ++
        (application.light.words ++ application.refresh.words)))))

/-- The words after the tag fix the records before any later words. -/
theorem body_prefixFree : PrefixFree body :=
  (fieldWord_prefixFree.array.pair (word_prefixFree.pair (RecordRow.words_prefixFree.counted.pair
    (word_prefixFree.counted.pair (Block.words_prefixFree.counted.pair
      (Layout.words_prefixFree.pair Layout.words_prefixFree)))))).comap
    (view := fun application : Application =>
      (application.package, application.witnessWords, application.rows, application.recipeRows,
        application.blocks, application.light, application.refresh))
    fun left right same => by cases left; cases right; simpa using same

/-- The word stream of Rust `BoundedDepthPackages::with_application`. -/
def words (application : Application) : List ℕ := tagWords ++ application.body

/-- The word stream fixes the records. -/
theorem words_injective : Function.Injective words := fun _ _ same =>
  body_prefixFree.injective (List.append_cancel_left same)

/-- Each word of the tag is a byte, so it is below `q`. -/
theorem tagWords_canonical : WordSchema.Canonical tagWords := by decide

/-- No word of the stream is at or above `q`. Rust refuses the application otherwise. -/
def Canonical (application : Application) : Prop := WordSchema.Canonical application.words

instance (application : Application) : Decidable application.Canonical := by
  unfold Canonical
  infer_instance

/-- The hash input: each word as a field element. -/
def input (application : Application) : List F := fieldWords application.words

theorem domain_prefix (application : Application) : domain <+: application.input := by
  simp only [input, words, domain, fieldWords, List.map_append]
  exact List.prefix_append _ _

/-- The hash input of a canonical stream has the words as its values: no word is reduced. -/
theorem input_val {application : Application} (canonical : application.Canonical) :
    application.input.map Fin.val = application.words :=
  fieldWords_val canonical

/-- The hash input of a canonical stream fixes the records. -/
theorem input_injective {left right : Application} (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical) (same : left.input = right.input) : left = right :=
  words_injective (fieldWords_injective leftCanonical rightCanonical same)

/-- The identity of the packages with the application. -/
def identity (application : Application) : List F := Poseidon2.hash application.input

theorem identity_length (application : Application) : application.identity.length = 4 :=
  Poseidon2.hash_length _

/-- Two canonical applications with one identity have the same records, or their padded hash
inputs are two distinct block lists with one sponge digest. -/
theorem identity_binds {left right : Application} (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical) (same : left.identity = right.identity) :
    left = right ∨
      (Spec.Sponge.padding.pad left.input ≠ Spec.Sponge.padding.pad right.input ∧
        Spec.Sponge.digest (Spec.Sponge.padding.pad left.input) =
          Spec.Sponge.digest (Spec.Sponge.padding.pad right.input)) := by
  by_cases inputs : left.input = right.input
  · exact Or.inl (input_injective leftCanonical rightCanonical inputs)
  · exact Or.inr (Spec.Sponge.collision_blocks inputs same)

end Application

end NightstreamFPrime.Lifecycle.BoundedDepth.ApplicationIdentity
