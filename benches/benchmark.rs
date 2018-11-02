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
    fn gen_rand_limbs(rng: &mut ThreadRng) -> [u32; fp_256::NUMLIMBS] {
        let mut limbs = [0u32; fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u32();
        }
        limbs
    }

    fn gen_rand_double_limbs(rng: &mut ThreadRng) -> [u32; 2 * fp_256::NUMLIMBS] {
        let mut limbs = [0u32; 2 * fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u32();
        }
        limbs
    }

    fn gen_rand_fp256_raw(rng: &mut ThreadRng) -> fp_256::Fp256 {
        fp_256::Fp256::new(gen_rand_limbs(rng))
    }

    fn gen_rand_fp256(rng: &mut ThreadRng) -> fp_256::Fp256 {
        (gen_rand_fp256_raw(rng)).normalize_big(0)
    }

    c.bench_function(
        "Fp256 - normalize_big (256 bits to Fp256 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || gen_rand_fp256_raw(&mut rng),
                |val_to_norm| {
                    for _ in 0..100 {
                        black_box(val_to_norm.normalize_big(0));
                    }
                },
            );
        },
    );

    c.bench_function(
        "Fp256 - reduce_barrett (normalize 512 bits to Fp256 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || gen_rand_double_limbs(&mut rng),
                |val_to_norm| {
                    for _ in 0..100 {
                        black_box(fp_256::Fp256::reduce_barrett(&val_to_norm));
                    }
                },
            );
        },
    );

    c.bench_function("Fp256 - add (add two Fp256s 100 times)", |bench| {
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

    c.bench_function(
        "Fp256 - add_assign (add an Fp256 into another Fp256 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a += b;
                    }
                },
            );
        },
    );

    c.bench_function("Fp256 - sub (subtract two Fp256s 100 times)", |bench| {
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

    c.bench_function(
        "Fp256 - sub_assign (subtract an Fp256 from another Fp256 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a -= b;
                    }
                },
            );
        },
    );

    c.bench_function("Fp256 - mul (two Fp256s 100 times)", |bench| {
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

    c.bench_function(
        "Fp256 - mul_assign (an Fp256 into another Fp256 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a *= b;
                    }
                },
            );
        },
    );

    

    // c.bench_function("Fp256 - div (divide two Fp256s)", |bench| {
    //     let mut rng = rand::thread_rng();
    //     bench.iter_with_setup(
    //         || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
    //         |(a, b)| {
    //             black_box(a / b);
    //         },
    //     );
    // });

    c.bench_function("Fp256 - neg (negate an Fp256 100 times)", |bench| {
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

    // c.bench_function("Fp256 - inv (invert an Fp256)", |bench| {
    //     let mut rng = rand::thread_rng();
    //     bench.iter_with_setup(|| gen_rand_fp256(&mut rng), |a| a.inv());
    // });

    c.bench_function("Fp256 - square (square an Fp256)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(|| gen_rand_fp256(&mut rng), |a| a.square());
    });

    c.bench_function("Fp256 - pow (exponentiate an Fp256 by an Fp256)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(a, exp)| a.pow(exp),
        );
    });

    // c.bench_function("Fp256 - pow (exponentiate an Fp256 by a u64)", |bench| {
    //     let mut rng = rand::thread_rng();
    //     bench.iter_with_setup(
    //         || (gen_rand_fp256(&mut rng), rng.next_u64()),
    //         |(a, exp)| a.pow(exp),
    //     );
    // });

    c.bench_function(
        "Fp256 - from, to_bytes_array (roundtrip to and from byte array 100 times)",
        |bench| {
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
        },
    );

    c.bench_function("Fp256 - Monty - Mul 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || {
                (
                    gen_rand_fp256(&mut rng).to_monty(),
                    gen_rand_fp256(&mut rng).to_monty(),
                )
            },
            |(mut a, b)| {
                for _ in 0..100 {
                    a = black_box(a * b);
                }
            },
        );
    });

    c.bench_function("Fp256 - Monty - to_monty 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_fp256(&mut rng),
            |a| {
                for _ in 0..100 {
                    black_box(a.to_monty());
                }
            },
        );
    });

    c.bench_function("Fp256 - Monty - to_norm 100 times", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_fp256(&mut rng).to_monty(),
            |a| {
                for _ in 0..100 {
                    black_box(a.to_norm());
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
