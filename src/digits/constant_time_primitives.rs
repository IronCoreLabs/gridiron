/*
 * Contents of this file heavily borrowed from or influenced by BearSSL by Thomas Pornin
 * https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/inner.h
 */

use digits::constant_bool::*;
// use $crate::digits::constant_bool::*;
use num_traits::{NumOps, One, Zero};
use std::convert::From;
use std::mem::size_of;
use std::num::Wrapping;
use std::ops::{BitAnd, BitOr, BitXor, Neg, Not};

pub trait ConstantUnsignedPrimitives
where
    Wrapping<Self>: Neg<Output = Wrapping<Self>> + BitOr<Output = Wrapping<Self>>,
    Self: NumOps + Copy + BitAnd<Output = Self> + BitXor<Output = Self> + One + Zero + PartialEq,
{
    const SIZE: u32;
    fn not(self) -> Self;
    fn mux(self, x: Self, y: Self) -> Self;
    fn const_eq(self, y: Self) -> ConstantBool<Self>;
    fn const_neq(self, y: Self) -> ConstantBool<Self>;
    fn const_gt(self, y: Self) -> ConstantBool<Self>;
    fn const_ge(self, y: Self) -> ConstantBool<Self>;
    fn const_lt(self, y: Self) -> ConstantBool<Self>;
    fn const_le(self, y: Self) -> ConstantBool<Self>;
    fn min(self, y: Self) -> Self;
    fn max(self, y: Self) -> Self;
}
pub trait ConstantSignedPrimitives {
    fn not(self) -> Self;
    fn mux(self, x: Self, y: Self) -> Self;
    fn const_eq0(self) -> Self;
    fn const_gt0(self) -> Self;
    fn const_ge0(self) -> Self;
    fn const_lt0(self) -> Self;
    fn const_le0(self) -> Self;
    fn const_eq(self, y: Self) -> Self;
    fn const_neq(self, y: Self) -> Self;
    fn const_gt(self, y: Self) -> Self;
    fn const_ge(self, y: Self) -> Self;
    fn const_lt(self, y: Self) -> Self;
    fn const_le(self, y: Self) -> Self;
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
impl ConstantSignedPrimitives for i64 {
    #[inline]
    fn not(self) -> Self {
        self ^ 1
    }
    #[inline]
    fn mux(self, x: Self, y: Self) -> Self {
        y ^ (self.wrapping_neg() & (x ^ y))
    }
    #[inline]
    fn const_eq0(self) -> Self {
        let q = self as u64;
        (!(q | q.wrapping_neg()) >> 63) as i64
    }
    #[inline]
    fn const_gt0(self) -> Self {
        let q = self as u64;
        ((!q & q.wrapping_neg()) >> 63) as i64
    }
    #[inline]
    fn const_ge0(self) -> Self {
        let x = self as u64;
        (!x >> 63) as i64
    }
    #[inline]
    fn const_lt0(self) -> Self {
        let x = self as u64;
        (x >> 63) as i64
    }
    #[inline]
    fn const_le0(self) -> Self {
        /*
         * ~-x has its high bit set if and only if -x is nonnegative (as
         * a signed int), i.e. x is in the -(2^31-1) to 0 range. We must
         * do an OR with x itself to account for x = -2^31.
         */
        let q = self as u64;
        ((q | !q.wrapping_neg()) >> 63) as i64
    }
    #[inline]
    fn const_eq(self, y: Self) -> Self {
        let q = self ^ y;
        ConstantSignedPrimitives::not((q | q.wrapping_neg()) >> 63)
    }
    #[inline]
    fn const_neq(self, y: Self) -> Self {
        let q = self ^ y;
        (q | q.wrapping_neg()) >> 63
    }
    #[inline]
    fn const_gt(self, y: Self) -> Self {
        let z = y.wrapping_sub(self);
        (z ^ ((self ^ y) & (self ^ z))) >> 63
    }
    #[inline]
    fn const_ge(self, y: Self) -> Self {
        ConstantSignedPrimitives::not(y.const_gt(self))
    }
    #[inline]
    fn const_lt(self, y: Self) -> Self {
        y.const_gt(self)
    }
    #[inline]
    fn const_le(self, y: Self) -> Self {
        ConstantSignedPrimitives::not(self.const_gt(y))
    }
}

pub trait ConstantUnsignedArray31 {
    fn const_not(self) -> Self;
    fn const_eq(self, y: Self) -> ConstantBool<u32>;
    fn const_neq(self, y: Self) -> ConstantBool<u32>;
    fn const_gt(self, y: Self) -> ConstantBool<u32>;
    fn const_ge(self, y: Self) -> ConstantBool<u32>;
    fn const_lt(self, y: Self) -> ConstantBool<u32>;
    fn const_le(self, y: Self) -> ConstantBool<u32>;
}
macro_rules! constant_unsigned_array31 { ($($N:expr),*) => { $(
/// Must have maximum of 31-bits used per limb
impl ConstantUnsignedArray31 for [u32; $N] {
    #[inline]
    fn const_not(mut self) -> Self {
        self.iter_mut().for_each(|l| *l ^= 1);
        self
    }

    #[inline]
    fn const_eq(self, y: Self) -> ConstantBool<u32> {
        self.const_neq(y).not()
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
}
)+ }}
constant_unsigned_array31! { 9, 16 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn const_not() {
        assert_eq!([0u32; 9].const_not(), [1u32; 9]);
    }

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

}
