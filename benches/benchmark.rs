#[macro_use]
extern crate criterion;
extern crate gridiron;
extern crate rand;
extern crate num_traits;

use criterion::Criterion;
use gridiron::fp_256;
use rand::{ThreadRng, RngCore};
use num_traits::{Inv, Pow};
use std::ops::Neg;


fn criterion_benchmark(c: &mut Criterion) {

    fn gen_rand_limbs(rng : &mut ThreadRng) -> [u64; fp_256::NUMLIMBS] {
        let mut limbs = [0u64; fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64();
        }
        limbs
    }

    fn gen_rand_double_limbs(rng : &mut ThreadRng) -> [u64; 2 * fp_256::NUMLIMBS] {
        let mut limbs = [0u64; 2 * fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64();
        }
        limbs
    }

    fn gen_rand_fp256(rng : &mut ThreadRng) -> fp_256::Fp256 {
        fp_256::Fp256::new(gen_rand_limbs(rng))
    }

    c.bench_function("normalize 256 bits to Fp256", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                gen_rand_fp256(&mut rng)
            },
            |val_to_norm| {
                val_to_norm.normalize(0)
            }
        );
    });

    c.bench_function("normalize 512 bits to Fp256", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                gen_rand_double_limbs(&mut rng)
            },
            |val_to_norm| {
                fp_256::Fp256::reduce_barrett(&val_to_norm)
            }
        );
    });

    c.bench_function("add two Fp256s", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng))
            },
            |(a, b)| {
                a + b
            }
        );
    });

    c.bench_function("subtract two Fp256s", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng))
            },
            |(a, b)| {
                a - b
            }
        );
    });

    c.bench_function("multiply two Fp256s", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng))
            },
            |(a, b)| {
                a * b
            }
        );
    });

    c.bench_function("divide two Fp256s", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng))
            },
            |(a, b)| {
                a / b
            }
        );
    });

    c.bench_function("negate an Fp256", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                gen_rand_fp256(&mut rng)
            },
            |a| {
                a.neg()
            }
        );
    });

    c.bench_function("invert an Fp256", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                gen_rand_fp256(&mut rng)
            },
            |a| {
                a.inv()
            }
        );
    });

    c.bench_function("exponentiate an Fp256", |b| {
        let mut rng = rand::thread_rng();
        b.iter_with_setup(
            || {
                let val = gen_rand_fp256(&mut rng);
                let exp = rng.next_u64();
                (val, exp)
            },
            |(a, exp)| {
                a.pow(exp)
            }
        );
    });

}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = criterion_benchmark
}
criterion_main!(benches);
