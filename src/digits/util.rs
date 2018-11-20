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
        // v += 1
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

///Sum t n times. Reveals the value of n.
#[inline]
pub(crate) fn sum_n<T: Zero + Copy>(mut t: T, n: u64) -> T {
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
            k = k >> 1;
            extra = x;
        }
        t + extra
    }
}

///This reveals the exponent so it should not be called with secret values.
#[inline]
pub(crate) fn exp_by_squaring<T: One + Copy>(orig_x: T, nn: u64) -> T {
    if nn == 0 {
        T::one()
    } else {
        let mut y = T::one();
        let mut x = orig_x;
        let mut n = nn;
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
pub(crate) fn mul_add(a: u32, b: u32, c: u32) -> u64 {
    a as u64 * b as u64 + c as u64
}

/// Returns array with least sig in pos 0 and carry in pos 2
#[inline]
pub fn split_u64_to_31b_array(i: u64) -> [u32; 3] {
    let mut res = [032; 3];
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
