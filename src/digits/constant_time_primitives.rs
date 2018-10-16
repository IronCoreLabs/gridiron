/*
 * Contents of this file heavily borrowed from or influenced by BearSSL by Thomas Pornin
 * https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/inner.h
 */

use std::mem::size_of;

pub trait ConstantUnsignedPrimitives {
    const SIZE: u32;
    fn not(self) -> Self;
    fn mux(self, x: Self, y: Self) -> Self;
    fn const_eq(self, y: Self) -> Self;
    fn const_neq(self, y: Self) -> Self;
    fn const_gt(self, y: Self) -> Self;
    fn const_ge(self, y: Self) -> Self;
    fn const_lt(self, y: Self) -> Self;
    fn const_le(self, y: Self) -> Self;
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
    fn const_eq(self, y: Self) -> Self {
        let q = self ^ y;
        return ((q | q.wrapping_neg()) >> (Self::SIZE - 1)).not();
    }
    #[inline]
    fn const_neq(self, y: Self) -> Self {
        let q = self ^ y;
        (q | q.wrapping_neg()) >> (Self::SIZE - 1)
    }
    #[inline]
    fn const_gt(self, y: Self) -> Self {
        let z = y.wrapping_sub(self);
        (z ^ ((self ^ y) & (self ^ z))) >> (Self::SIZE - 1)
    }
    #[inline]
    fn const_ge(self, y: Self) -> Self {
        y.const_gt(self).not()
    }
    #[inline]
    fn const_lt(self, y: Self) -> Self {
        y.const_gt(self)
    }
    #[inline]
    fn const_le(self, y: Self) -> Self {
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
constant_unsigned! { u64, u32, u8, usize }
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
        return ((q | q.wrapping_neg()) >> 63).not();
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
        y.const_gt(self).not()
    }
    #[inline]
    fn const_lt(self, y: Self) -> Self {
        y.const_gt(self)
    }
    #[inline]
    fn const_le(self, y: Self) -> Self {
        self.const_gt(y).not()
    }
}
