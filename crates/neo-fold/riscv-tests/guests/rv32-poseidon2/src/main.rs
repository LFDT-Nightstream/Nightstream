#![no_std]
#![no_main]

#[derive(nightstream_sdk::NeoAbi)]
struct PoseidonInput {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

#[derive(nightstream_sdk::NeoAbi)]
struct PoseidonDigest {
    d0: u64,
    d1: u64,
    d2: u64,
    d3: u64,
}

#[nightstream_sdk::provable]
fn poseidon2_example(input: PoseidonInput) -> PoseidonDigest {
    let digest = nightstream_sdk::poseidon2::poseidon2_hash(&[input.a, input.b, input.c, input.d]);
    PoseidonDigest {
        d0: digest[0],
        d1: digest[1],
        d2: digest[2],
        d3: digest[3],
    }
}
