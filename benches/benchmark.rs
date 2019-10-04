#[macro_use]
extern crate criterion;
extern crate gridiron;
extern crate num_traits;
extern crate rand;

use criterion::{black_box, Criterion};
use gridiron::fp_256;
use gridiron::fp_480;
use num_traits::{Inv, Pow};
use rand::{RngCore};
use rand::rngs::ThreadRng;
use core::ops::Neg;

fn criterion_benchmark(c: &mut Criterion) {
    fn gen_rand_limbs(rng: &mut ThreadRng) -> [u32; fp_256::NUMLIMBS] {
        let mut limbs = [0u32; fp_256::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u32();
        }
        limbs[fp_256::NUMLIMBS - 1] &= 0xFF; //Ensure the last limb isn't too big
        limbs
    }

    fn gen_rand_sixty_four_bytes(rng: &mut ThreadRng) -> [u8; 64] {
        let mut limbs = [0u8; 64];
        rng.fill_bytes(&mut limbs[..]);
        limbs
    }

    fn gen_rand_fp256_raw(rng: &mut ThreadRng) -> fp_256::Fp256 {
        fp_256::Fp256::new(gen_rand_limbs(rng))
    }

    fn gen_rand_fp256(rng: &mut ThreadRng) -> fp_256::Fp256 {
        gen_rand_fp256_raw(rng).normalize_little()
    }

    fn gen_rand_480_limbs(rng: &mut ThreadRng) -> [u32; fp_480::NUMLIMBS] {
        let mut limbs = [0u32; fp_480::NUMLIMBS];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u32();
        }
        limbs[fp_480::NUMLIMBS - 1] &= 0xFEFF;
        limbs
    }

    fn gen_rand_fp480_raw(rng: &mut ThreadRng) -> fp_480::Fp480 {
        fp_480::Fp480::new(gen_rand_480_limbs(rng))
    }

    fn gen_rand_fp480(rng: &mut ThreadRng) -> fp_480::Fp480 {
        (gen_rand_fp480_raw(rng)).normalize_little()
    }

    c.bench_function(
        "Fp256 - normalize_little (256 bits to Fp256 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || gen_rand_fp256_raw(&mut rng),
                |val_to_norm| {
                    for _ in 0..100 {
                        black_box(val_to_norm.normalize_little());
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

    c.bench_function("Fp256 - div (divide two Fp256s)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(a, b)| {
                black_box(a / b);
            },
        );
    });

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

    c.bench_function("Fp256 - inv (invert an Fp256)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(|| gen_rand_fp256(&mut rng), |a| a.inv());
    });

    c.bench_function("Fp256 - pow (exponentiate an Fp256 by an Fp256)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), gen_rand_fp256(&mut rng)),
            |(a, exp)| a.pow(exp),
        );
    });

    c.bench_function("Fp256 - pow (exponentiate an Fp256 by a u32)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || (gen_rand_fp256(&mut rng), rng.next_u32()),
            |(a, exp)| a.pow(exp),
        );
    });

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

    c.bench_function("Fp256 - from 64 bytes", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_sixty_four_bytes(&mut rng),
            |bytes| fp_256::Fp256::from(bytes),
        );
    });

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

    c.bench_function(
        "Fp480 - normalize_little (480 bits to Fp480 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || gen_rand_fp480_raw(&mut rng),
                |val_to_norm| {
                    for _ in 0..100 {
                        black_box(val_to_norm.normalize_little());
                    }
                },
            );
        },
    );

    c.bench_function(
        "Fp480 - add_assign (add an Fp480 into another Fp480 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp480(&mut rng), gen_rand_fp480(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a += b;
                    }
                },
            );
        },
    );

    c.bench_function(
        "Fp480 - sub_assign (subtract an Fp480 from another Fp480 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp480(&mut rng), gen_rand_fp480(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a -= b;
                    }
                },
            );
        },
    );

    c.bench_function(
        "Fp480 - mul_assign (an Fp480 into another Fp480 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || (gen_rand_fp480(&mut rng), gen_rand_fp480(&mut rng)),
                |(mut a, b)| {
                    for _ in 0..100 {
                        a *= b;
                    }
                },
            );
        },
    );

    c.bench_function("Fp480 - inv (invert an Fp480)", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(|| gen_rand_fp480(&mut rng), |a| a.inv());
    });

    c.bench_function(
        "Fp480 - from, to_bytes_array (roundtrip to and from byte array 100 times)",
        |bench| {
            let mut rng = rand::thread_rng();
            bench.iter_with_setup(
                || gen_rand_fp480(&mut rng),
                |a| {
                    for _ in 0..100 {
                        let byte_array = a.to_bytes_array();
                        fp_480::Fp480::from(byte_array);
                    }
                },
            );
        },
    );

    c.bench_function("Fp480 - from 64 bytes", |bench| {
        let mut rng = rand::thread_rng();
        bench.iter_with_setup(
            || gen_rand_sixty_four_bytes(&mut rng),
            |bytes| fp_480::Fp480::from(bytes),
        );
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = criterion_benchmark
}
criterion_main!(benches);
