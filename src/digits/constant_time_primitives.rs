/*
 * Contents of this file heavily borrowed from or influenced by BearSSL by Thomas Pornin
 * https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/inner.h
 */

use crate::digits::constant_bool::*;
use num_traits::{NumOps, One, Zero};
use std::cmp::Ordering;
use std::mem::size_of;
use std::num::Wrapping;
use std::ops::{BitAnd, BitOr, BitXor, Neg, Not};

///
///Values which support swapping the values in place.
///
pub trait ConstantSwap {
    ///Swapping the values if the swap was true. Note that this should be done in a constant
    ///time way to support constant time algorithms.
    fn swap_if(&mut self, other: &mut Self, swap: ConstantBool<u32>);
}

pub trait ConstantUnsignedPrimitives
where
    Wrapping<Self>: Neg<Output = Wrapping<Self>> + BitOr<Output = Wrapping<Self>>,
    Self: NumOps + Copy + BitAnd<Output = Self> + BitXor<Output = Self> + One + Zero + PartialEq,
{
    const SIZE: u32;
    fn not(self) -> Self;
    ///This chooses the first value if self is 1, chooses the 2nd value if the value is 0.
    ///This is only well defined for 0 or 1. If your value can be anything except those 2, do not use this.
    fn mux(self, x: Self, y: Self) -> Self;
    fn const_eq(self, y: Self) -> ConstantBool<Self>;
    fn const_eq0(self) -> ConstantBool<Self>;
    fn const_neq(self, y: Self) -> ConstantBool<Self>;
    fn const_gt(self, y: Self) -> ConstantBool<Self>;
    fn const_ge(self, y: Self) -> ConstantBool<Self>;
    fn const_lt(self, y: Self) -> ConstantBool<Self>;
    fn const_le(self, y: Self) -> ConstantBool<Self>;
    ///Removes the high bit if it's set, otherwise leaves number as is.
    fn const_abs(self) -> Self;
    fn min(self, y: Self) -> Self;
    fn max(self, y: Self) -> Self;
}

macro_rules! constant_unsigned { ($($T:ty),*) => { $(
impl ConstantUnsignedPrimitives for $T {
    const SIZE: u32 = (size_of::<$T>() * 8) as u32;

    #[inline]
    fn not(self) -> Self {
        self ^ 1
    }
    #[inline]
    fn mux(self, x: Self, y: Self) -> Self {
        y ^ (self.wrapping_neg() & (x ^ y))
    }
    #[inline]
    fn const_eq(self, y: Self) -> ConstantBool<Self> {
        let q = self ^ y;
        ConstantBool((q | q.wrapping_neg()) >> (Self::SIZE - 1)).not()
    }
    #[inline]
    fn const_eq0(self) -> ConstantBool<Self> {
        let q = self;
        ConstantBool((q | q.wrapping_neg()) >> (Self::SIZE - 1)).not()
    }
    #[inline]
    fn const_neq(self, y: Self) -> ConstantBool<Self> {
        let q = self ^ y;
        ConstantBool((q | q.wrapping_neg()) >> (Self::SIZE - 1))
    }
    #[inline]
    fn const_gt(self, y: Self) -> ConstantBool<Self> {
        let z = y.wrapping_sub(self);
        ConstantBool((z ^ ((self ^ y) & (self ^ z))) >> (Self::SIZE - 1))
    }
    #[inline]
    fn const_ge(self, y: Self) -> ConstantBool<Self> {
        y.const_gt(self).not()
    }
    #[inline]
    fn const_lt(self, y: Self) -> ConstantBool<Self> {
        y.const_gt(self)
    }
    #[inline]
    fn const_le(self, y: Self) -> ConstantBool<Self> {
        self.const_gt(y).not()
    }
    #[inline]
    fn const_abs(self) -> Self{
        let high_bit_is_set = ConstantBool(self >> (Self::SIZE - 1));
        high_bit_is_set.mux(self.wrapping_neg(), self)
    }
    #[inline]
    fn min(self, y: Self) -> Self {
        self.const_gt(y).mux(y, self)
    }
    #[inline]
    fn max(self, y: Self) -> Self {
        self.const_gt(y).mux(self, y)
    }
}
)+ }}
constant_unsigned! { u64, u32 }

pub trait ConstantUnsignedArray31 {
    fn const_eq(self, y: Self) -> ConstantBool<u32>;
    fn const_eq0(self) -> ConstantBool<u32>;
    fn const_neq0(self) -> ConstantBool<u32>;
    fn const_neq(self, y: Self) -> ConstantBool<u32>;
    fn const_gt(self, y: Self) -> ConstantBool<u32>;
    fn const_ge(self, y: Self) -> ConstantBool<u32>;
    fn const_lt(self, y: Self) -> ConstantBool<u32>;
    fn const_le(self, y: Self) -> ConstantBool<u32>;
    fn const_copy_if(&mut self, src: &Self, ctl: ConstantBool<u32>);
    fn const_ordering(&self, y: &Self) -> Option<Ordering>;
}
macro_rules! constant_unsigned_array31 { ($($N:expr),*) => { $(
/// Must have maximum of 31-bits used per limb
impl ConstantUnsignedArray31 for [u32; $N] {

    #[inline]
    fn const_eq(self, y: Self) -> ConstantBool<u32> {
        self.const_neq(y).not()
    }

    #[inline]
    fn const_eq0(self) -> ConstantBool<u32> {
        let mut accum = 0u32;
        self.iter().for_each(|l| { accum |= *l });
        ConstantBool::is_zero(accum)
    }

    #[inline]
    fn const_neq0(self) -> ConstantBool<u32> {
        self.const_eq0().not()
    }

    #[inline]
    fn const_neq(self, y: Self) -> ConstantBool<u32> {
        let q = self.iter().zip(y.iter()).fold(0u32, |acc, (xlimb, ylimb)| {
            acc | ((xlimb & 0x7FFFFFFF) ^ (ylimb & 0x7FFFFFFF))
        });
        ConstantBool((q | q.wrapping_neg()) >> 31)
    }

    #[inline]
    fn const_lt(self, y: Self) -> ConstantBool<u32> {
        let mut borrow = 0u32;
        self.iter().zip(y.iter()).for_each ( |(a, b)| {
            let diff = (*a).wrapping_sub(*b).wrapping_sub(borrow);
            borrow = diff >> 31;
        });
        ConstantBool(borrow)
    }

    #[inline]
    fn const_le(self, y: Self) -> ConstantBool<u32> {
        y.const_lt(self).not()
    }

    #[inline]
    fn const_gt(self, y: Self) -> ConstantBool<u32> {
        y.const_lt(self)
    }

    #[inline]
    fn const_ge(self, y: Self) -> ConstantBool<u32> {
        self.const_lt(y).not()
    }

    fn const_ordering(&self, y:&Self) -> Option<Ordering> {
        let mut res = 0u64;
        self.iter().zip(y.iter()).rev().for_each(|(l, r)| {
            let limbcmp = (l.const_gt(*r).0 as u64) | ((r.const_gt(*l).0 as u64).wrapping_neg());
            res = res.const_abs().mux(res, limbcmp);
        });
        match res as i64 {
            -1 => Some(Ordering::Less),
            0 => Some(Ordering::Equal),
            1 => Some(Ordering::Greater),
            _ => None
        }
    }

    #[inline]
    fn const_copy_if(&mut self, src: &Self, ctl: ConstantBool<u32>) {
        for (s, d) in src.iter().zip(self.iter_mut()) {
            *d = ctl.mux(*s, *d);
        }
    }
}
)+ }}
constant_unsigned_array31! { 9, 16 }

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn const_eq() {
        assert_eq!([1u32; 9].const_eq([1u32; 9]).0, 1u32);
        assert_eq!([1u32; 9].const_eq([0u32; 9]).0, 0u32);
        assert_eq!([0x7FFFFFFFu32; 9].const_eq([1u32; 9]).0, 0u32);
        assert_eq!([0x7FFFFFFFu32; 9].const_eq([0x7FFFFFFFu32; 9]).0, 1u32);
        let left: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(left.const_eq(left).0, 1u32);
        let right: [u32; 9] = [1, 2, 3, 3, 5, 6, 7, 8, 9];
        assert_eq!(left.const_eq(right).0, 0u32);
    }
    #[test]
    fn const_gt() {
        assert_eq!([1u32; 9].const_gt([1u32; 9]).0, 0u32);
        // test little minus big
        let max = [0x7FFFFFFFu32; 9];
        let big = [
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFEu32, // max - 2^31
            0x7FFFFFFFu32,
        ];
        let little = [4u32; 9];
        let zero = [0u32; 9];

        assert_eq!(max.const_gt(zero).0, 1u32);
        assert_eq!(zero.const_gt(max).0, 0u32);
        assert_eq!(max.const_gt(big).0, 1u32);
        assert_eq!(big.const_gt(max).0, 0u32);
        assert_eq!(max.const_gt(little).0, 1u32);
        assert_eq!(little.const_gt(max).0, 0u32);
        assert_eq!(little.const_gt(big).0, 0u32);
        assert_eq!(little.const_gt(little).0, 0u32);
        assert_eq!(max.const_gt(max).0, 0u32);
    }

    #[test]
    fn const_lt() {
        assert_eq!([1u32; 9].const_lt([1u32; 9]).0, 0u32);
        // test little minus big
        let max = [0x7FFFFFFFu32; 9];
        let big = [
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFEu32, // max - 2^31
            0x7FFFFFFFu32,
        ];
        let little = [4u32; 9];

        assert_eq!(max.const_lt(big).0, 0u32);
        assert_eq!(big.const_lt(max).0, 1u32);
        assert_eq!(max.const_lt(little).0, 0u32);
        assert_eq!(little.const_lt(max).0, 1u32);
        assert_eq!(little.const_lt(big).0, 1u32);
        assert_eq!(little.const_lt(little).0, 0u32);
        assert_eq!(max.const_lt(max).0, 0u32);
    }

    #[test]
    fn const_le() {
        assert_eq!([1u32; 9].const_le([1u32; 9]).0, 1u32);
        // test little minus big
        let max = [0x7FFFFFFFu32; 9];
        let big = [
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFEu32, // max - 2^31
            0x7FFFFFFFu32,
        ];
        let little = [4u32; 9];

        assert_eq!(max.const_le(big).0, 0u32);
        assert_eq!(big.const_le(max).0, 1u32);
        assert_eq!(max.const_le(little).0, 0u32);
        assert_eq!(little.const_le(max).0, 1u32);
        assert_eq!(little.const_le(big).0, 1u32);
        assert_eq!(little.const_le(little).0, 1u32);
        assert_eq!(max.const_le(max).0, 1u32);
    }

    #[test]
    fn const_ge() {
        assert_eq!([1u32; 9].const_ge([1u32; 9]).0, 1u32);
        // test little minus big
        let max = [0x7FFFFFFFu32; 9];
        let big = [
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFEu32, // max - 2^31
            0x7FFFFFFFu32,
        ];
        let little = [4u32; 9];

        assert_eq!(max.const_ge(big).0, 1u32);
        assert_eq!(big.const_ge(max).0, 0u32);
        assert_eq!(max.const_ge(little).0, 1u32);
        assert_eq!(little.const_ge(max).0, 0u32);
        assert_eq!(little.const_ge(big).0, 0u32);
        assert_eq!(little.const_ge(little).0, 1u32);
        assert_eq!(max.const_ge(max).0, 1u32);
    }

    #[test]
    fn const_ordering() {
        let max = [0x7FFFFFFFu32; 9];
        let big = [
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFFu32,
            0x7FFFFFFEu32, // max - 2^31
            0x7FFFFFFFu32,
        ];
        let little = [4u32; 9];

        assert_eq!(max.const_ordering(&big).unwrap(), Ordering::Greater);
        assert_eq!(big.const_ordering(&max).unwrap(), Ordering::Less);
        assert_eq!(max.const_ordering(&little).unwrap(), Ordering::Greater);
        assert_eq!(little.const_ordering(&max).unwrap(), Ordering::Less);
        assert_eq!(max.const_ordering(&little).unwrap(), Ordering::Greater);
        assert_eq!(little.const_ordering(&big).unwrap(), Ordering::Less);
        assert_eq!(little.const_ordering(&little).unwrap(), Ordering::Equal);
        assert_eq!(max.const_ordering(&max).unwrap(), Ordering::Equal);
    }

    #[test]
    fn u64_const_eq0() {
        assert_eq!(std::u64::MAX.const_eq0().0, ConstantBool::new_false().0);

        let zero: u64 = 0;
        assert_eq!(zero.const_eq0().0, ConstantBool::new_true().0);

        let one: u64 = 1;
        assert_eq!(one.const_eq0().0, ConstantBool::new_false().0);
    }

    proptest! {
        #[test]
        fn u64_const_eq(a in any::<u64>(), b in any::<u64>()) {
            let result = a.const_eq(b);
            if a == b{
                prop_assert_eq!(result.0, ConstantBool::new_true().0);
            } else{
                prop_assert_eq!(result.0, ConstantBool::new_false().0);
            }
        }

        #[test]
        fn u64_const_gt(a in any::<u64>(), b in any::<u64>()) {
            let result = a.const_gt(b);
            if a > b{
                prop_assert_eq!(result.0, ConstantBool::new_true().0);
            } else{
                prop_assert_eq!(result.0, ConstantBool::new_false().0);
            }
        }

        #[test]
        fn u64_const_gte(a in any::<u64>(), b in any::<u64>()) {
            let result = a.const_ge(b);
            if a >= b{
                prop_assert_eq!(result.0, ConstantBool::new_true().0);
            } else{
                prop_assert_eq!(result.0, ConstantBool::new_false().0);
            }
        }

        #[test]
        fn u64_const_lt(a in any::<u64>(), b in any::<u64>()) {
            let result = a.const_lt(b);
            if a < b{
                prop_assert_eq!(result.0, ConstantBool::new_true().0);
            } else{
                prop_assert_eq!(result.0, ConstantBool::new_false().0);
            }
        }

        #[test]
        fn u64_const_lte(a in any::<u64>(), b in any::<u64>()) {
            let result = a.const_le(b);
            if a <= b{
                prop_assert_eq!(result.0, ConstantBool::new_true().0);
            } else{
                prop_assert_eq!(result.0, ConstantBool::new_false().0);
            }
        }

        #[test]
        fn u64_const_abs(a in any::<i64>()) {
            let result = (a as u64).const_abs();
            prop_assert_eq!(a.abs() as u64, result);
        }
    }
}
