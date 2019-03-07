Gridiron [![](https://img.shields.io/crates/v/gridiron.svg)](https://crates.io/crates/gridiron) [![](https://docs.rs/gridiron/badge.svg)](https://docs.rs/gridiron) [![](https://travis-ci.com/IronCoreLabs/gridiron.svg?branch=master)](https://travis-ci.com/IronCoreLabs/gridiron?branch=master)
====================

This library is a work in progress. To use it, you can either use one of the provided finite fields, or you can call the macro to create your own. The two that are included are:

* `fp_480::Fp480`
* `fp_256::Fp256`

These were created like so:

    // p = 65000549695646603732796438742359905742825358107623003571877145026864184071783
    fp31!(
        fp_256, // Name of mod
        Fp256,  // Name of class
        256,    // Number of bits for prime
        9,      // Number of limbs (ceil(bits/31))
        [
            // prime number in limbs, least sig first
            // get this from sage with p.digits(2^31)
            1577621095, 817453272, 47634040, 1927038601, 407749150, 1308464908, 685899370, 1518399909,
            143
        ],
        // barrett reduction for reducing values up to twice
        // the number of prime bits (double limbs):
        // floor(2^(31*numlimbs*2)/p)
        [
            618474456, 1306750627, 1454330209, 2032300189, 1138536719, 1905629153, 1016481908,
            1139000707, 1048853973, 14943480
        ],
        // montgomery R = 2^(W*N) where W = word size and N = limbs
        //            R = 2^(9*31) = 2^279
        // montgomery R^-1 mod p
        // 41128241662407537452081084990737892697811449013582128001435272241165411523443
        [
            1126407027, 1409097648, 718270744, 92148126, 1120340506, 1733383256, 1472506103,
            1994474164, 90
        ],
        // montgomery R^2 mod p
        // 26753832205083639112203412356185740914827891884263043594389452794758614404120
        [
            1687342104, 733402836, 182672516, 801641709, 2122695487, 1290522951, 66525586, 319877849,
            59
        ],
        // -p[0]^-1
        // in sage: m = p.digits(2^31)[0]
        //          (-m).inverse_mod(2^31)
        2132269737
    );


To use it, you'll need to import headers for the math operations you want. So, for example:

    use std::ops::Add;
    let one = fp_256::Fp256::one();
    let two = one + one;

This is a work in progress and we hope to make it more performant. All operations are constant time except:

`Mul<u64>`, `Pow<u64>` - If you need a constant time version of those, you can lift them into an Fp type and use `Mul<Fp>` and `Pow<Fp>`. 
The will be much slower and typically the u64s are not secret values so it's ok for them to be non constant time.

# Code Audit

NCC Group has conducted an audit of this library - release [0.6.0](https://github.com/IronCoreLabs/gridiron/releases/tag/0.6.0) contains all of the audited code, including updates that were created to resolve issues that were discovered during the audit. The NCC Group audit found that the chosen pairing and elliptic curve are cryptographically sound, and that the Rust implementation is a faithful and correct embodiment of the target protocol. In addition, the audit specifically looked for but did not find any leak of secret information via timing or memory access pattern side-channel attacks.
