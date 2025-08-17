use criterion::{criterion_group, criterion_main, Criterion};
use neo_modint::ModInt;
use neo_ring::RingElement;

fn bench_ring_mul(c: &mut Criterion) {
    let mut rng = rand::rng();
    let n = 64;
    let a = RingElement::<ModInt>::random_uniform(&mut rng, n);
    let b = RingElement::<ModInt>::random_uniform(&mut rng, n);
    c.bench_function("ring_mul_n64", |bencher| bencher.iter(|| a.clone() * b.clone()));
}

fn bench_decompose_coeffs(c: &mut Criterion) {
    let mut rng = rand::rng();
    let n = 64;
    let re = RingElement::<ModInt>::random_uniform(&mut rng, n);
    let b = 3u64;
    let k = 45;
    c.bench_function("decompose_coeffs_n64", |bencher| bencher.iter(|| re.decompose_coeffs(b, k)));
}

fn bench_ring_mul_large(c: &mut Criterion) {
    let mut rng = rand::rng();
    let n = 1024;
    let a = RingElement::<ModInt>::random_uniform(&mut rng, n);
    let b = RingElement::<ModInt>::random_uniform(&mut rng, n);
    c.bench_function("ring_mul_n1024", |bencher| bencher.iter(|| a.clone() * b.clone()));
}

criterion_group!(benches, bench_ring_mul, bench_decompose_coeffs, bench_ring_mul_large);
criterion_main!(benches);
