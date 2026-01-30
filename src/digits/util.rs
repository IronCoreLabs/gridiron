use num_traits::{One, Zero};

///Convert the src into the limbs. This _does not_ mod off the value. This will take the first
///len bytes and split them into 31 bit limbs.
///Note that this will _not_ check anything about the length of limbs and could be unsafe... BE CAREFUL!
///
///If your limbs cannot hold the src content when it's converted this will reference the limbs slice out of bounds. If the src slice
///is shorter than `len` this will also reference the src slice out of bounds.
///
///For more safe versions of this, check the convert_bytes_to_limbs in ff31::$classname.
#[inline]
pub fn unsafe_convert_bytes_to_limbs_mut(src: &[u8], limbs: &mut [u32], len: usize) {
    let mut acc = 0u32;
    let mut acc_len = 0i32;
    let mut v = 0;
    for b in src.iter().rev().take(len) {
        let b_u32 = *b as u32;
        acc |= b_u32 << acc_len;
        acc_len += 8;
        if acc_len >= 31 {
            limbs[v] = acc & 0x7FFFFFFFu32;
            v += 1;
            acc_len -= 31;
            //Note that because we're adding 8 each time through the loop
            //and check that acc_len >= 31 that 8 - acc_len can _never_ be negative.
            acc = b_u32 >> (8 - acc_len);
        }
    }
    if acc_len != 0 {
        limbs[v] = acc;
    }
}

///This function assumes that the buf pointer has at least 4 spaces starting at the beginning of the
///slice. You need to assure this before calling.
#[inline]
pub fn u32_to_bytes_big_endian(x: u32, buf: &mut [u8]) {
    debug_assert!(buf.len() >= 4);
    buf[0] = (x >> 24) as u8;
    buf[1] = (x >> 16) as u8;
    buf[2] = (x >> 8) as u8;
    buf[3] = x as u8;
}

///This function assumes that the buf pointer has at least 8 spaces starting at the beginning of the
///slice. You need to ensure this before calling.
#[inline]
pub fn u64_to_bytes_big_endian(x: u64, buf: &mut [u8]) {
    debug_assert!(buf.len() >= 8);
    buf[0] = (x >> 56) as u8;
    buf[1] = (x >> 48) as u8;
    buf[2] = (x >> 40) as u8;
    buf[3] = (x >> 32) as u8;
    buf[4] = (x >> 24) as u8;
    buf[5] = (x >> 16) as u8;
    buf[6] = (x >> 8) as u8;
    buf[7] = x as u8;
}

///Sum t n times. Reveals the value of n.
#[inline]
pub fn sum_n<T: Zero + Copy>(mut t: T, n: u32) -> T {
    if n == 0 {
        Zero::zero()
    } else if n == 1 {
        t
    } else {
        let mut extra = t;
        let mut k = n - 1;
        while k != 1 {
            let x = if (k & 1) == 1 { t + extra } else { extra };
            t = t + t;
            k >>= 1;
            extra = x;
        }
        t + extra
    }
}

///This reveals the exponent so it should not be called with secret values.
#[inline]
pub fn exp_by_squaring<T: One + Copy>(orig_x: T, mut n: u32) -> T {
    if n == 0 {
        T::one()
    } else {
        let mut y = T::one();
        let mut x = orig_x;
        while n > 1 {
            if (n & 1) == 0 {
                x = x * x;
                n /= 2;
            } else {
                y = x * y;
                x = x * x;
                n = (n - 1) / 2;
            }
        }
        y * x
    }
}

#[inline]
pub fn mul_add(a: u32, b: u32, c: u32) -> u64 {
    a as u64 * b as u64 + c as u64
}

/// Returns array with least sig in pos 0 and carry in pos 2
#[inline]
pub fn split_u64_to_31b_array(i: u64) -> [u32; 3] {
    let mut res = [0u32; 3];
    res[0] = (i & 0x7FFFFFFF) as u32;
    res[1] = ((i >> 31) & 0x7FFFFFFF) as u32;
    res[2] = (i >> 62) as u32;
    res
}

/// Returns (high, low) where high uses extra bit for carry
/// and low has a cleared 32nd bit
#[inline]
pub fn split_u64_to_31b(i: u64) -> (u32, u32) {
    ((i >> 31) as u32, (i & 0x7FFFFFFF) as u32)
}

// 62-bit limb utility functions

///Convert the src into 62-bit limbs. This _does not_ mod off the value. This will take the first
///len bytes and split them into 62 bit limbs.
///Note that this will _not_ check anything about the length of limbs and could be unsafe... BE CAREFUL!
///
///If your limbs cannot hold the src content when it's converted this will reference the limbs slice out of bounds. If the src slice
///is shorter than `len` this will also reference the src slice out of bounds.
///
///For more safe versions of this, check the convert_bytes_to_limbs in ff62::$classname.
#[inline]
pub fn unsafe_convert_bytes_to_limbs_mut_62(src: &[u8], limbs: &mut [u64], len: usize) {
    let mut acc = 0u64;
    let mut acc_len = 0i32;
    let mut v = 0;
    for b in src.iter().rev().take(len) {
        let b_u64 = *b as u64;
        acc |= b_u64 << acc_len;
        acc_len += 8;
        if acc_len >= 62 {
            limbs[v] = acc & 0x3FFFFFFFFFFFFFFF;
            v += 1;
            acc_len -= 62;
            //Note that because we're adding 8 each time through the loop
            //and check that acc_len >= 62 that 8 - acc_len can _never_ be negative.
            acc = b_u64 >> (8 - acc_len);
        }
    }
    if acc_len != 0 {
        limbs[v] = acc;
    }
}

#[inline]
pub fn mul_add_62(a: u64, b: u64, c: u64) -> u128 {
    a as u128 * b as u128 + c as u128
}

/// Returns array with least sig in pos 0 and carry in pos 2
#[inline]
pub fn split_u128_to_62b_array(i: u128) -> [u64; 3] {
    let mut res = [0u64; 3];
    res[0] = (i & 0x3FFFFFFFFFFFFFFF) as u64;
    res[1] = ((i >> 62) & 0x3FFFFFFFFFFFFFFF) as u64;
    res[2] = (i >> 124) as u64;
    res
}

/// Returns (high, low) where high uses extra bit for carry
/// and low has a cleared 63rd bit
#[inline]
pub fn split_u128_to_62b(i: u128) -> (u64, u64) {
    ((i >> 62) as u64, (i & 0x3FFFFFFFFFFFFFFF) as u64)
}

/// WARNING: This macro is always exported, but will generate code that depends on proptest. Don't call it
/// in a location where that won't compile.
///
/// Shared proptest strategy for generating arbitrary field elements with edge case coverage.
///
/// This macro generates an `arb_fp` function that produces field elements with:
/// - ~25% edge cases: 0, 1, -1 (p-1), 2, -2 (p-2)
/// - ~75% uniformly distributed random values
///
/// The random generation properly normalizes values that may exceed 2p by looping
/// until the value is fully reduced, satisfying `normalize_little`'s precondition.
///
/// # Parameters
/// - `$classname`: The field element type (e.g., `Fp256`)
/// - `$limbsizebits`: Bits per limb (31 or 62)
/// - `$limbtype`: Limb type (`u32` or `u64`)
/// - `$limb_mask`: Mask for valid limb bits (`0x7FFFFFFFu32` or `0x3FFFFFFFFFFFFFFFu64`)
///
/// Note: This macro must NOT be gated by `#[cfg(test)]` because it's referenced via
/// `$crate::define_arb_fp!` from within the `fp31!`/`fp62!` macros' test modules.
/// When a downstream crate runs tests, `$crate` refers to gridiron compiled as a
/// dependency (not in test mode), so the macro must always be exported.
#[macro_export]
macro_rules! define_arb_fp {
    ($classname:ident, $limbsizebits:expr, $limbtype:ty, $limb_mask:expr) => {
        prop_compose! {
            fn arb_fp()(
                choice in 0u8..20,
                limb_values in any::<[$limbtype; NUMLIMBS]>()
            ) -> $classname {
                match choice {
                    0 => $classname::zero(),
                    1 => $classname::one(),
                    2 => -$classname::one(),           // p - 1
                    3 => $classname::from(2u8),
                    4 => -$classname::from(2u8),       // p - 2
                    _ => {
                        let mut limbs = [0 as $limbtype; NUMLIMBS];
                        for (i, &val) in limb_values.iter().enumerate() {
                            limbs[i] = val & $limb_mask;
                        }
                        // This ensures that the top limb isn't using more bits than it should be
                        // for the given PRIMEBITS. The `+1` and subsequent `-1` are to guard against
                        // PRIMEBITS being divisible by $limbsizebits.
                        const TOP_LIMB_BITS: usize = ((PRIMEBITS - 1) % $limbsizebits) + 1;
                        limbs[NUMLIMBS - 1] &= ((1 as $limbtype) << TOP_LIMB_BITS) - 1;

                        // Fully normalize - may need multiple subtractions if value >= 2p
                        let mut result = $classname { limbs };
                        while result.limbs.const_ge(PRIME).0 == 1 {
                            result.normalize_assign_little();
                        }
                        result
                    }
                }
            }
        }
    };
}
