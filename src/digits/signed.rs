use digits::util::*;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg, Rem, Shr, ShrAssign, Sub, SubAssign};

/// This is a signed-magnitude calculation needed for the division algorithm
/// and possibly some other implementations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignedDigitsArray<T: DigitsArray + Clone> {
    pub negative: bool,
    pub limbs: T,
}

impl<T: DigitsArray + Clone> SignedDigitsArray<T> {
    #[inline]
    pub fn new(is_negative: bool, limbs: T) -> SignedDigitsArray<T> {
        SignedDigitsArray {
            negative: is_negative,
            limbs: limbs,
        }
    }
    #[inline]
    pub fn new_pos(limbs: T) -> SignedDigitsArray<T> {
        SignedDigitsArray {
            negative: false,
            limbs: limbs,
        }
    }

    #[inline]
    pub fn new_neg(limbs: T) -> SignedDigitsArray<T> {
        SignedDigitsArray {
            negative: true,
            limbs: limbs,
        }
    }

    #[inline]
    pub fn is_even(&self) -> bool {
        self.limbs.is_even()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        !self.negative && self.limbs.is_one()
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_zero()
    }

    #[inline]
    pub fn is_neg(&self) -> bool {
        self.negative
    }
}

impl<T: DigitsArray + Clone> Shr<usize> for SignedDigitsArray<T> {
    type Output = SignedDigitsArray<T>;
    #[inline]
    fn shr(mut self, rhs: usize) -> SignedDigitsArray<T> {
        self >>= rhs;
        self
    }
}

impl<T: DigitsArray + Clone> ShrAssign<usize> for SignedDigitsArray<T> {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        self.limbs.shift_right_bits_assign(rhs);
        if self.is_zero() {
            self.negative = false;
        }
    }
}

impl<T: DigitsArray + Clone> Neg for SignedDigitsArray<T> {
    type Output = SignedDigitsArray<T>;
    #[inline]
    fn neg(mut self) -> SignedDigitsArray<T> {
        if self.is_zero() {
            self.negative = false;
        } else {
            self.negative = !self.negative;
        }
        self
    }
}

impl<T: DigitsArray + Clone> AddAssign for SignedDigitsArray<T> {
    #[inline]
    fn add_assign(&mut self, other: SignedDigitsArray<T>) {
        let newsign = self
            .limbs
            .add_assign_signed(self.negative, other.limbs, other.negative);
        if self.is_zero() {
            self.negative = false;
        } else {
            self.negative = newsign;
        }
    }
}

impl<T: DigitsArray + Clone> Add for SignedDigitsArray<T> {
    type Output = SignedDigitsArray<T>;
    #[inline]
    fn add(mut self, other: SignedDigitsArray<T>) -> SignedDigitsArray<T> {
        self += other;
        self
    }
}

impl<T: DigitsArray + Clone> Sub for SignedDigitsArray<T> {
    type Output = SignedDigitsArray<T>;
    #[inline]
    fn sub(mut self, other: SignedDigitsArray<T>) -> SignedDigitsArray<T> {
        self -= other;
        self
    }
}

impl<T: DigitsArray + Clone> Rem<u64> for SignedDigitsArray<T> {
    type Output = u64;
    #[inline]
    fn rem(self, modulus: u64) -> u64 {
        self.limbs.rem_1(modulus)
    }
}

impl<T: DigitsArray + Clone> SubAssign for SignedDigitsArray<T> {
    #[inline]
    fn sub_assign(&mut self, other: SignedDigitsArray<T>) {
        let newsign = self
            .limbs
            .sub_assign_signed(self.negative, other.limbs, other.negative);
        if self.is_zero() {
            self.negative = false;
        } else {
            self.negative = newsign;
        }
    }
}

impl<T: DigitsArray + Clone> SubAssign<i64> for SignedDigitsArray<T> {
    #[inline]
    fn sub_assign(&mut self, other: i64) {
        if other < 0 {
            self.sub_assign(Self::new_neg(T::from_u64((other * -1) as u64)));
        } else {
            self.sub_assign(Self::new_pos(T::from_u64(other as u64)));
        }
    }
}

impl<T: DigitsArray + Clone + PartialEq + Debug> PartialOrd for SignedDigitsArray<T> {
    #[inline]
    fn partial_cmp(&self, other: &SignedDigitsArray<T>) -> Option<Ordering> {
        match (self.negative, other.negative) {
            (false, true) => Some(Ordering::Greater),       // pos > neg
            (true, false) => Some(Ordering::Less),          // neg < pos
            (false, false) => self.limbs.cmp(&other.limbs), // positive so bigger is greater
            (true, true) => other.limbs.cmp(&self.limbs),   // negative so smaller is greater
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_subtraction() {
        let three = [3, 0];
        let two = [2, 0];
        // 3 - 2 = +1
        let mut x = three.clone();
        assert_eq!(x.sub_assign_signed(false, two, false), false); // positive result
        assert_eq!(x, [1, 0]);

        // 3 - -2 = +5
        let mut x = three.clone();
        assert_eq!(x.sub_assign_signed(false, two, true), false); // positive result
        assert_eq!(x, [5, 0]);

        // -3 - 2 = -5
        let mut x = three.clone();
        assert_eq!(x.sub_assign_signed(true, two, false), true); // negative result
        assert_eq!(x, [5, 0]);

        // -3 - -2 = -1
        let mut x = three.clone();
        assert_eq!(x.sub_assign_signed(true, two, true), true); // negative result
        assert_eq!(x, [1, 0]);

        // 2 - 3 = -1
        let mut x = two.clone();
        assert_eq!(x.sub_assign_signed(false, three, false), true); // negative result
        assert_eq!(x, [1, 0]);

        // 2 - -3 = +5
        let mut x = two.clone();
        assert_eq!(x.sub_assign_signed(false, three, true), false); // positive result
        assert_eq!(x, [5, 0]);

        // -2 - 3 = -5
        let mut x = two.clone();
        assert_eq!(x.sub_assign_signed(true, three, false), true); // negative result
        assert_eq!(x, [5, 0]);

        // -2 - -3 = 1
        let mut x = two.clone();
        assert_eq!(x.sub_assign_signed(true, three, true), false); // positive result
        assert_eq!(x, [1, 0]);
    }

    #[test]
    fn signed_addition() {
        let three = [3, 0];
        let threed = SignedDigitsArray {
            negative: false,
            limbs: [3, 0],
        };
        let two = [2, 0];
        let twod = SignedDigitsArray {
            negative: true,
            limbs: [2, 0],
        };

        let oned = SignedDigitsArray {
            negative: false,
            limbs: [1, 0],
        };

        assert_eq!(threed + twod, oned); // three plus neg two

        // 3 + 2 = +5
        let mut x = three.clone();
        assert_eq!(x.add_assign_signed(false, two, false), false); // positive result
        assert_eq!(x, [5, 0]);

        // 3 + -2 = +1
        let mut x = three.clone();
        assert_eq!(x.add_assign_signed(false, two, true), false); // positive result
        assert_eq!(x, [1, 0]);

        // -3 + 2 = -1
        let mut x = three.clone();
        assert_eq!(x.add_assign_signed(true, two, false), true); // negative result
        assert_eq!(x, [1, 0]);

        // -3 + -2 = -5
        let mut x = three.clone();
        assert_eq!(x.add_assign_signed(true, two, true), true); // negative result
        assert_eq!(x, [5, 0]);

        // 2 + 3 = +5
        let mut x = two.clone();
        assert_eq!(x.add_assign_signed(false, three, false), false); // positive result
        assert_eq!(x, [5, 0]);

        // 2 + -3 = -1
        let mut x = two.clone();
        assert_eq!(x.add_assign_signed(false, three, true), true); // negative result
        assert_eq!(x, [1, 0]);

        // -2 + 3 = 1
        let mut x = two.clone();
        assert_eq!(x.add_assign_signed(true, three, false), false); // positive result
        assert_eq!(x, [1, 0]);

        // -2 + -3 = -5
        let mut x = two.clone();
        assert_eq!(x.add_assign_signed(true, three, true), true); // negative result
        assert_eq!(x, [5, 0]);
    }

    #[test]
    fn signed_comparison() {
        let three = SignedDigitsArray {
            negative: false,
            limbs: [3, 0],
        };
        let two = SignedDigitsArray {
            negative: false,
            limbs: [2, 0],
        };
        let zero = -three - -three;

        assert!(three > two); // pos, pos
        assert!(-three < two); // neg, pos
        assert!(-three < -two); // neg, neg
        assert!(zero < two); // zero, pos
        assert!(zero > -two); // zero, neg
        assert!(-zero > -two); // zero, neg (w/ neg zero)
        assert!(three != -three);
        assert!(zero == -zero);
    }
}
