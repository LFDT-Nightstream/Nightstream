use serde_json::{json, Value};

const P: u64 = 0xffff_ffff_0000_0001;

pub fn program() -> Value {
    json!([[
        [0, 3, 0, 2],
        [
            5,
            [
                [7, [[100, 2], [101, 3], [100, P - 1]]],
                [11, [[101, 4]]],
                [13, [[100, P - 1], [100, 1]]],
                [17, [[100, 5]]]
            ],
            [[[100, 2, [0, 2, 4], 0]], []],
            [0, 0, 1],
            0,
            2
        ]
    ]])
}

pub fn malformed() -> Vec<(&'static str, Value)> {
    let mut constant = program();
    constant[0][1][1][0][0] = json!(P);
    let mut coefficient = program();
    coefficient[0][1][1][0][1][0][1] = json!(P);
    let mut word_shape = program();
    word_shape[0][1][1][0] = json!([7]);
    let mut missing_source = program();
    missing_source[0][1][2] = json!([[], []]);
    let mut overlap = program();
    let range = overlap[0][1][2][0][0].clone();
    overlap[0][1][2][0].as_array_mut().unwrap().push(range);
    let mut missing_table = program();
    missing_table[0][1][1] = json!([]);
    let mut missing_tag = program();
    missing_tag[0][1][3] = json!([]);
    vec![
        ("noncanonical constant", constant),
        ("noncanonical coefficient", coefficient),
        ("invalid word shape", word_shape),
        ("missing source", missing_source),
        ("overlapping source", overlap),
        ("missing word", missing_table),
        ("missing tag", missing_tag),
    ]
}
