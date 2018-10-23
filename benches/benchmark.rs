#[macro_use]
extern crate criterion;
extern crate gridiron;
extern crate num_traits;
extern crate rand;

use criterion::{black_box, Criterion};
use gridiron::fp_256;
use num_traits::{Inv, Pow};
use rand::{RngCore, ThreadRng};
use std::ops::Neg;

fn criterion_benchmark(c: &mut Criterion) {
    fn gen_rand_limbs(rng: &mut ThreadRng) -> [u64; fp_256::NUMLIMBS] {
        let mut limbs = [0u64; fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64();
        }
        limbs
    }

    fn gen_rand_double_limbs(rng: &mut ThreadRng) -> [u64; 2 * fp_256::NUMLIMBS] {
        let mut limbs = [0u64; 2 * fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64();
        }
        limbs
    }

    fn gen_rand_fp256_raw(rng: &mut ThreadRng) -> fp_256::Fp256 {
        fp_256::Fp256::new(gen_rand_limbs(rng))
    }

    fn gen_rand_fp256(rng: &mut ThreadRng) -> fp_256::Fp256 {
        (gen_rand_fp256_raw(rng)).normalize(0)
    }

    c.bench_function("normalize 256 bits to Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_fp256_raw(&mut rng),
            |val_to_norm| {
                for _ in 0..100 {
                    black_box(val_to_norm.normalize(0));
                }
            },
        );
    });

    c.bench_function("normalize 512 bits to Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_double_limbs(&mut rng),
            |val_to_norm| {
                for _ in 0..100 {
                    black_box(fp_256::Fp256::reduce_barrett(&val_to_norm));
                }
            },
        );
    });

    c.bench_function("add two Fp256s 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a = black_box(a + b);
                }
            },
        );
    });

    c.bench_function("add an Fp256 into another Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a += b;
                }
            },
        );
    });

    c.bench_function("subtract two Fp256s 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a = black_box(a - b);
                }
            },
        );
    });

    c.bench_function("subtract an Fp256 from another Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a -= b;
                }
            },
        );
    });

    c.bench_function("multiply two Fp256s 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a = black_box(a * b);
                }
            },
        );
    });

    c.bench_function("multiply an Fp256 into another Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a *= b;
                }
            },
        );
    });

    c.bench_function("bitwise AND two Fp256s 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(mut a, b)| {
                for _ in 0..100 {
                    a = black_box(a & b);
                }
            },
        );
    });

    c.bench_function(
        "bitwise AND an Fp256 into another Fp256 100 times",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a &= b;
                    }
                },
            );
        },
    );

    c.bench_function("bitwise AND an Fp256 and a u64 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), rng.next_u64()),
            |(mut a, b)| {
                for _ in 0..100 {
                    a = black_box(a & b);
                }
            },
        );
    });

    c.bench_function("bitwise AND a u64 into an Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), rng.next_u64()),
            |(mut a, b)| {
                for _ in 0..100 {
                    a &= b;
                }
            },
        );
    });

    c.bench_function("divide two Fp256s", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(a, b)| {
                black_box(a / b);
            },
        );
    });

    c.bench_function("negate an Fp256 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_fp256(&mut rng),
            |mut a| {
                for _ in 0..100 {
                    a = a.neg();
                }
            },
        );
    });

    c.bench_function("invert an Fp256", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(|| gen_rand_fp256(&mut rng), |a| a.inv());
    });

    c.bench_function("square an Fp256", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(|| gen_rand_fp256(&mut rng), |a| a.square());
    });

    c.bench_function("exponentiate an Fp256 by an Fp256", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(a, exp)| a.pow(exp),
        );
    });

    c.bench_function("exponentiate an Fp256 by a u64", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), rng.next_u64()),
            |(a, exp)| a.pow(exp),
        );
    });

    c.bench_function("roundtrip to and from byte array 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_fp256(&mut rng),
            |a| {
                for _ in 0..100 {
                    let byte_array = a.to_bytes_array();
                    fp_256::Fp256::from(byte_array);
                }
            },
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = criterion_benchmark
}
criterion_main!(benches);
