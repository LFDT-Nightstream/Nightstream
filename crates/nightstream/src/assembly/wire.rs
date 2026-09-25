//! Named views of the existing numeric-array ABI in Export/Package.lean,
//! Stage1/ApplicationPackage.lean and Stage1/PerApplicationAssignmentTransport.lean.

use serde::{ser::SerializeTuple, Deserialize, Serialize, Serializer};
use serde_json::Value;

macro_rules! wire_tuple {
    ($name:ident { $($field:ident: $ty:ty),+ $(,)? }) => {
        #[derive(Clone, Debug, Deserialize, PartialEq)]
        pub(super) struct $name { $(pub $field: $ty),+ }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut tuple = serializer.serialize_tuple([$(stringify!($field)),+].len())?;
                $(tuple.serialize_element(&self.$field)?;)+
                tuple.end()
            }
        }
    };
}

wire_tuple!(Envelope {
    schema: usize, source: SourcePackage, matrix: Vec<Value>, application: ApplicationPlan,
    assignment: Assignment, next_preimage: [usize; 2], logical_public: usize,
});

wire_tuple!(SourcePackage {
    schema: usize, profile: Value, poseidon: Value, layout: Layout, relation: Relation,
    permutation: Value, hash_chains: Vec<HashChain>, permutation_invocations: Vec<PermutationInvocation>,
    compact_templates: Vec<Value>, compact_invocations: Vec<CompactInvocation>,
    batches: Vec<Batch>, instructions: Vec<Instruction>, rows: Vec<Row>, terminal: Value,
});

wire_tuple!(Layout {
    rows: usize, private: usize, constant: usize, public: usize, total: usize,
    private_segments: Vec<Segment>, public_segments: Vec<Segment>,
});
wire_tuple!(Segment {
    role: usize,
    start: usize,
    length: usize
});
wire_tuple!(Relation {
    rows: usize, columns: usize, cube_variables: usize, matrices: Vec<usize>, degree: usize, polynomial: Value,
});
wire_tuple!(HashChain {
    phase: usize,
    row_start: usize,
    row_count: usize,
    input_start: usize,
    input_length: usize,
    witness_start: usize,
    witness_length: usize,
    absorb_count: usize,
    digest_length: usize,
    digest_start: usize,
});
wire_tuple!(PermutationInvocation { phase: usize, row_start: usize, witness_start: usize, inputs: Vec<Combination> });
wire_tuple!(CompactInvocation {
    phase: usize, template: usize, row_start: usize, local_start: usize, inputs: Vec<InputRange>,
});
wire_tuple!(InputRange {
    input_start: usize,
    input_count: usize,
    column_start: usize,
    column_stride: usize
});
wire_tuple!(Batch { start: usize, recipes: Vec<Value>, hints: Vec<Value> });
wire_tuple!(Combination { constant: u64, terms: Vec<(usize, u64)> });
wire_tuple!(Instruction {
    row: usize,
    target: usize,
    a: Combination,
    b: Combination
});
wire_tuple!(Row {
    index: usize,
    a: Combination,
    b: Combination,
    c: Combination
});
wire_tuple!(ApplicationPlan {
    schema: usize, witness_count: usize, input_columns: Vec<usize>, witness_columns: Vec<usize>,
    output_columns: Vec<usize>, private_start: usize, private_count: usize, row_start: usize, row_count: usize,
    hash_chains: Vec<Value>, permutations: Vec<Value>, compact_templates: Vec<Value>, compact_invocations: Vec<Value>,
    batches: Vec<Batch>, instructions: Vec<Instruction>, rows: Vec<Row>,
});
wire_tuple!(Assignment {
    schema: usize, blocks: Vec<AssignmentBlock>, phi81: Value, first54: Value,
    digest_block: usize, digest_expressions: Vec<Value>,
});
wire_tuple!(AssignmentBlock {
    opcode: usize, slot_kind: usize, slot_count: usize, source_domain: usize, runs: Vec<[usize; 3]>,
});
