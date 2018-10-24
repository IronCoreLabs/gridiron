use num_traits::{NumOps, One, Zero};
use std::convert::From;
use std::num::Wrapping;
use std::ops::{BitAnd, BitOr, BitXor, Neg, Not};

/*
 * Contents of this file heavily borrowed from or influenced by BearSSL by Thomas Pornin
 * https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/inner.h
 */

#[derive(Copy, Clone)]
pub struct ConstantBool<T: NumOps + Copy>(pub T);

macro_rules! constantbool_from_impl {
    ($($t:ty)*) => ($(
        impl From<bool> for ConstantBool<$t> {
            fn from(b: bool) -> ConstantBool<$t> {
                ConstantBool(b as $t)
            }
        }
    )*)
}
constantbool_from_impl! { i32 i64 u32 u64 }

impl<T: NumOps + Copy + BitXor<Output = T> + One> Not for ConstantBool<T> {
    type Output = Self;

    /// Performs the unary `!` operation.
    #[inline]
    fn not(self) -> ConstantBool<T> {
        ConstantBool(self.0 ^ <T>::one())
    }
}

// impl BitAndAssign
// impl BitOrAssign
// impl BitXOrAssign
//
impl<T: NumOps + Copy + Zero> Default for ConstantBool<T> {
    #[inline]
    fn default() -> ConstantBool<T> {
        ConstantBool(<T>::zero())
    }
}

impl<T: NumOps + Copy + BitAnd<Output = T>> BitAnd for ConstantBool<T> {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        ConstantBool(self.0 & rhs.0)
    }
}

impl<T: NumOps + Copy + BitOr<Output = T>> BitOr for ConstantBool<T> {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ConstantBool(self.0 | rhs.0)
    }
}
//Neg<Output = Wrapping<T>>
impl<T> ConstantBool<T>
where
    u64: From<T>,
    Wrapping<T>: Neg<Output = Wrapping<T>>,
    T: NumOps
        + Copy
        + Neg<Output = T>
        + BitAnd<Output = T>
        + BitXor<Output = T>
        + From<u64>
        + One
        + Zero
        + PartialEq,
{
    pub fn mux(self, x: T, y: T) -> T {
        y ^ (Wrapping(self.0).neg().0 & (x ^ y))
    }
    fn is_zero(i: T) -> Self {
        // let q = i as u64;
        let q = u64::from(i);
        ConstantBool(T::from(!(q | q.wrapping_neg()) >> 63))
    }
    fn not_zero(i: T) -> Self {
        ConstantBool(Self::is_zero(i).0 ^ <T>::one())
    }
}
