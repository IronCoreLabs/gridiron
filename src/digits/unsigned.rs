use digits::util::*;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Rem, Shr, ShrAssign, Sub, SubAssign};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnsignedDigitsArray<T: DigitsArray + Clone>(T);

impl<T: DigitsArray + Clone> UnsignedDigitsArray<T> {
    #[inline]
    #[allow(dead_code)]
    fn shift_right_digits(mut self, shifts: usize) -> Self {
        self.0.shift_right_digits_assign(shifts);
        self
    }
}

impl<T: DigitsArray + Clone> Deref for UnsignedDigitsArray<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: DigitsArray + Clone> DerefMut for UnsignedDigitsArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: DigitsArray + Clone> UnsignedDigitsArray<T> {
    #[inline]
    pub fn new(limbs: T) -> Self {
        UnsignedDigitsArray(limbs)
    }
}

impl<T: DigitsArray + Clone> Shr<usize> for UnsignedDigitsArray<T> {
    type Output = UnsignedDigitsArray<T>;
    #[inline]
    fn shr(mut self, rhs: usize) -> UnsignedDigitsArray<T> {
        self >>= rhs;
        self
    }
}

impl<T: DigitsArray + Clone> ShrAssign<usize> for UnsignedDigitsArray<T> {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        self.0.shift_right_bits_assign(rhs);
    }
}

impl<T: DigitsArray + Clone> AddAssign for UnsignedDigitsArray<T> {
    #[inline]
    fn add_assign(&mut self, other: UnsignedDigitsArray<T>) {
        self.0.add_assign_equiv(other.0);
    }
}

impl<T: DigitsArray + Clone> Add for UnsignedDigitsArray<T> {
    type Output = UnsignedDigitsArray<T>;
    #[inline]
    fn add(mut self, other: UnsignedDigitsArray<T>) -> UnsignedDigitsArray<T> {
        self.add_assign_equiv(other.0);
        self
    }
}

impl<T: DigitsArray + Clone> Sub for UnsignedDigitsArray<T> {
    type Output = UnsignedDigitsArray<T>;
    #[inline]
    fn sub(mut self, other: UnsignedDigitsArray<T>) -> UnsignedDigitsArray<T> {
        self -= other;
        self
    }
}

impl<T: DigitsArray + Clone> SubAssign for UnsignedDigitsArray<T> {
    #[inline]
    fn sub_assign(&mut self, other: UnsignedDigitsArray<T>) {
        self.0.sub_assign_equiv(other.0);
    }
}

impl<T: DigitsArray + Clone> Rem<u64> for UnsignedDigitsArray<T> {
    type Output = u64;
    #[inline]
    fn rem(self, modulus: u64) -> u64 {
        self.0.rem_1(modulus)
    }
}

impl<T: DigitsArray + Clone + PartialEq + Debug> PartialOrd for UnsignedDigitsArray<T> {
    #[inline]
    fn partial_cmp(&self, other: &UnsignedDigitsArray<T>) -> Option<Ordering> {
        self.0.cmp(&other.0)
    }
}

impl<T: DigitsArray + Clone> Mul for UnsignedDigitsArray<T>
where
    <T as DigitsArray>::TARRAYTIMESTWO: DigitsArray,
    <T as DigitsArray>::TARRAYTIMESTWO: Clone,
{
    type Output = UnsignedDigitsArray<T::TARRAYTIMESTWO>;

    #[inline]
    fn mul(self, rhs: UnsignedDigitsArray<T>) -> UnsignedDigitsArray<T::TARRAYTIMESTWO> {
        UnsignedDigitsArray(self.0.mul_classic_equiv(rhs.0))
    }
}
