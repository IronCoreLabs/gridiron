use num_traits::{NumOps, One, Zero};
use std::convert::From;
use std::mem::size_of;
use std::num::Wrapping;
use std::ops::Shr;
use std::ops::{BitAnd, BitOr, BitOrAssign, BitXor, Neg, Not};

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

impl<T: NumOps + Copy + BitOr<Output = T>> BitOrAssign for ConstantBool<T> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

//Neg<Output = Wrapping<T>>
impl<T> ConstantBool<T>
where
    // u64: From<T>,
    Wrapping<T>: Neg<Output = Wrapping<T>> + BitOr<Output = Wrapping<T>>,
    T: NumOps
        + Copy
        + BitAnd<Output = T>
        + BitXor<Output = T>
        + BitOr<Output = T>
        + Shr<usize, Output = T>
        + One
        + Zero
        + PartialEq,
{
    #[inline]
    pub fn mux(self, x: T, y: T) -> T {
        y ^ (Wrapping(self.0).neg().0 & (x ^ y))
    }
    #[inline]
    pub fn is_zero(i: T) -> Self {
        // let q = i as u64;
        // let q = u64::from(i);
        let q = Wrapping(i);
        let shift_amount = size_of::<T>() * 8 - 1;
        ConstantBool((q | q.neg()).0 >> shift_amount).not()
    }
    #[inline]
    pub fn not_zero(i: T) -> Self {
        ConstantBool(Self::is_zero(i).0 ^ <T>::one())
    }

    pub fn new_true() -> Self {
        ConstantBool(<T>::one())
    }
}
