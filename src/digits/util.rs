// use digits::constant_time_primitives::*;
use digits::constant_time_primitives::*;
use std::cmp::Ordering;
use std::num::Wrapping;

/// Decorate an array of T (u64 by default) with a bunch of handy
/// multi-precision math on the stack with fixed array sizes
pub trait DigitsArray<T = u64> {
    type TARRAYPLUSONE;
    type TARRAYMINUSONE;
    type TARRAYTIMESTWO;
    type TARRAYCARRY;
    type TARRAYPAIR;
    type TARRAYT;
    fn const_all<F>(&self, f: F) -> bool
    where
        Self: Sized,
        F: Fn(T) -> bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u64(d: u64) -> Self;
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_even(&self) -> bool;
    fn mul_by_digit(&self, y: T) -> Self::TARRAYPLUSONE;
    fn mul_classic(&self, b: &[T]) -> Self::TARRAYTIMESTWO;
    fn mul_classic_equiv(&self, b: Self) -> Self::TARRAYTIMESTWO;
    fn mul_add_by_digit(&self, y: T, z: T) -> Self::TARRAYPLUSONE;
    fn sub(&self, b: &[T]) -> Self::TARRAYCARRY;
    fn sub_assign(&mut self, b: &[T]) -> bool;
    fn sub_assign_equiv(&mut self, b: Self) -> bool;
    fn sub_ignore_carry(&self, b: &[T]) -> Self;
    fn sub_assign_signed(&mut self, self_is_neg: bool, other: Self, other_is_neg: bool) -> bool;
    fn add_assign_signed(&mut self, self_is_neg: bool, other: Self, other_is_neg: bool) -> bool;
    fn add(&self, b: &[T]) -> Self::TARRAYCARRY;
    fn add_assign(&mut self, b: &[T]) -> bool;
    fn add_assign_equiv(&mut self, b: Self) -> bool;
    fn add_ignore_carry(&self, b: &[T]) -> Self;
    // fn div_rem(&self, y: &[T]) -> Self::TARRAYPAIR;
    fn rem_1(&self, y: T) -> T;
    fn div_rem_1(&self, y: T) -> Self::TARRAYT;
    // fn div_rem_assign(&mut self, y: &[T]) -> Self;
    fn div2(&self) -> Self;
    fn div2_assign(&mut self);
    fn expand_one(&self) -> Self::TARRAYPLUSONE;
    fn contract_one(&self) -> Self::TARRAYMINUSONE;
    fn shift_left_digits(&self, shifts: usize) -> Self;
    fn shift_left_digits_assign(&mut self, shifts: usize);
    fn shift_right_digits(&self, shifts: usize) -> Self;
    fn shift_right_digits_assign(&mut self, shifts: usize);
    fn shift_right_bits(&self, bits: usize) -> Self;
    fn shift_right_bits_assign(&mut self, bits: usize);
    fn shift_left_bits(&self, bits: usize) -> Self;
    fn shift_left_bits_assign(&mut self, bits: usize);
    // fn div_short(&self, y: T) -> TARRAYT;
    fn copy(&self) -> Self;
    fn populate_padded_from_slice(y: &[u64]) -> Self;
    fn populate_padded_mostsig_from_slice(y: &[u64]) -> Self;
    fn b64_to_b32(input: Self) -> Self::TARRAYTIMESTWO;
    fn cmp(&self, y: &(Self)) -> Option<Ordering>;
    fn cmpi(&self, y: &(Self)) -> i64;
    fn greater_or_equal(&self, y: &(Self)) -> bool;
    fn less_or_equal(&self, y: &(Self)) -> bool;
    fn greater(&self, y: &(Self)) -> bool;
    fn less(&self, y: &(Self)) -> bool;
}

/// There are a few operations that we want to do often
/// without a fixed size and on a slice
pub trait DigitsSlice<T = u64> {
    fn add_assign(&mut self, b: &[T]) -> bool;
    fn sub_assign(&mut self, b: &[T]) -> bool;
    fn greater_or_equal(&self, y: &[T]) -> bool;
    fn less_or_equal(&self, y: &[T]) -> bool;
    fn greater(&self, y: &[T]) -> bool;
    fn less(&self, y: &[T]) -> bool;
}

impl<'a> DigitsSlice<u64> for &'a mut [u64] {
    #[inline]
    fn add_assign(&mut self, other: &[u64]) -> bool {
        debug_assert!(other.len() <= self.len());

        let mut carry = false;
        for (i, a) in self.iter_mut().enumerate() {
            // constant time means we dynamically pad b if it's shorter
            // rather than shortcutting the loop
            let ge = (i as u64).const_ge(other.len() as u64);
            let dummyindex = ge.mux(0, i as u64) as usize;
            let b = ge.mux(0, other[dummyindex]);
            let sum = a.wrapping_add(b);
            let olda = *a;
            *a = sum.wrapping_add(carry as u64);
            carry = (sum < olda) || (*a < sum);
        }
        carry
    }

    #[inline]
    fn sub_assign(&mut self, other: &[u64]) -> bool {
        debug_assert!(other.len() <= self.len());

        let mut borrow = false;
        for (i, a) in self.iter_mut().enumerate() {
            // constant time means we dynamically pad b if it's shorter
            // rather than shortcutting the loop
            let ge = (i as u64).const_ge(other.len() as u64);
            let dummyindex = ge.mux(0, i as u64) as usize;
            let b = ge.mux(0, other[dummyindex]);
            let diff = a.wrapping_sub(b);
            let olda = *a;
            *a = diff.wrapping_sub(borrow as u64);
            borrow = (diff > olda) || (*a > diff);
        }
        borrow
    }
    #[inline]
    fn greater_or_equal(&self, y: &[u64]) -> bool {
        cmpslice(&self[..], &y[..]) != Some(Ordering::Less)
    }
    #[inline]
    fn less_or_equal(&self, y: &[u64]) -> bool {
        cmpslice(&self[..], &y[..]) != Some(Ordering::Greater)
    }
    #[inline]
    fn greater(&self, y: &[u64]) -> bool {
        cmpslice(&self[..], &y[..]) == Some(Ordering::Greater)
    }
    #[inline]
    fn less(&self, y: &[u64]) -> bool {
        cmpslice(&self[..], &y[..]) == Some(Ordering::Less)
    }
}

/// Can't be generic with fixed size arrays, so we make
/// an implementation of our multi-precision math for
/// every array size that we might use.
macro_rules! digits_u64_impls { ($($M:ident $N:expr),+) => {
        $(
            impl DigitsArray<u64> for [u64; $N] {
                type TARRAYPLUSONE = [u64; $N+1];
                type TARRAYMINUSONE = [u64; $N-1];
                type TARRAYTIMESTWO = [u64; $N*2];
                type TARRAYCARRY = ([u64; $N], bool);
                type TARRAYPAIR = ([u64; $N], [u64; $N]);
                type TARRAYT = ([u64; $N], u64);

                #[inline]
                fn expand_one(&self) -> Self::TARRAYPLUSONE {
                    let mut ret = [0u64; $N+1];
                    for (dst, src) in ret.iter_mut().zip(self.iter()) {
                        *dst = *src;
                    }
                    ret
                }

                #[inline]
                fn contract_one(&self) -> Self::TARRAYMINUSONE {
                    let mut ret = [0u64; $N-1];
                    ret.copy_from_slice(&self[0..$N-1]);
                    ret
                }

                /// The general `.iter().all()` function short circuits. This does not.
                #[inline]
                fn const_all<F>(&self, f: F) -> bool where Self: Sized, F: Fn(u64) -> bool {
                    let mut ret = true;
                    self.iter().for_each(|x| ret &= f(*x));
                    ret
                }

                #[inline]
                fn zero() -> Self {
                    [0u64; $N]
                }

                #[inline]
                fn one() -> Self {
                    let mut ret = [0u64; $N];
                    ret[0] = 1;
                    ret
                }

                #[inline]
                fn from_u64(d: u64) -> Self {
                    let mut ret = Self::zero();
                    ret[0] = d;
                    ret
                }

                #[inline]
                fn is_zero(&self) -> bool {
                    self.const_all(|limb| limb == 0u64)
                }

                #[inline]
                fn is_one(&self) -> bool {
                    let mut ret = true;
                    self.iter().skip(1).for_each(|limb| ret &= limb == &0u64);
                    self[0] == 1u64 && ret
                }

                #[inline]
                fn is_even(&self) -> bool {
                    self[0] & 1u64 == 0u64
                }

                #[inline]
                fn b64_to_b32(input: Self) -> Self::TARRAYTIMESTWO {
                    let mut result = [0u64; $N*2];
                    for (i, a) in input.iter().enumerate() {
                        let (a_hi, a_lo) = split_u64_as_u64(*a);
                        result[i * 2] = a_lo;
                        result[i * 2 + 1] = a_hi;
                    }
                    result
                }

                #[inline]
                fn mul_by_digit(&self, y: u64) -> Self::TARRAYPLUSONE {
                    let mut ret = [0u64; $N + 1];

                    for (i, biglimb) in self.iter().take($N - 1).enumerate() {
                        let (hi, lo) = mul_1_limb_by_1_limb(*biglimb, y);
                        ret[i] = ret[i].wrapping_add(lo);
                            ret[i + 1] = ret[i+1].wrapping_add(hi).wrapping_add(ret[i].const_lt(lo));
                    }
                    let (_, lo) = mul_1_limb_by_1_limb(self[$N-1], y);
                    ret[$N - 1] = ret[$N - 1].wrapping_add(lo);
                    ret
                }

                /// mul_classic takes a fixed array times a slice provided the slice
                /// is the same or fewer limbs than the array so we can have a fixed
                /// size return of double the size of the current array.
                #[inline]
                fn mul_classic(&self, b: &[u64]) -> Self::TARRAYTIMESTWO {
                    debug_assert!(b.len() <= $N); // fixed return is 2N so b must be N or smaller

                    let mut res = [0u64; $N*2];
                    for i in 0..b.len() {
                        let mut c = 0;
                        for j in 0..$N {
                            let (mut u, mut v) = mul_1_limb_by_1_limb(self[j], b[i]);
                            v = add_accum_1by1(v, c, &mut u);
                            v = add_accum_1by1(v, res[i + j], &mut u);
                            res[i + j] = v;
                            c = u;
                        }
                        res[i + $N] = c;
                    }
                    res
                }

                /// mul_classic_equiv is basically a duplicate of mul_classic
                /// but the rhs is a fixed size array of the same size as the
                /// current array
                /// TODO: use a macro to DRY this
                #[inline]
                fn mul_classic_equiv(&self, b: [u64; $N]) -> [u64; $N*2] {
                    let mut res = [0u64; $N*2];
                    for i in 0..b.len() {
                        let mut c = 0;
                        for j in 0..$N {
                            let (mut u, mut v) = mul_1_limb_by_1_limb(self[j], b[i]);
                            v = add_accum_1by1(v, c, &mut u);
                            v = add_accum_1by1(v, res[i + j], &mut u);
                            res[i + j] = v;
                            c = u;
                        }
                        res[i + $N] = c;
                    }
                    res
                }

                /// self * y + z
                #[inline]
                fn mul_add_by_digit(&self, y: u64, z: u64) -> [u64; $N + 1] {
                    let mut ret = self.mul_by_digit(y);
                    let mut carry = 0u64;
                    ret[0] = add_accum_1by1(ret[0], z, &mut carry);
                    for a in ret.iter_mut().skip(1) {
                        let b = carry;
                        carry = 0;
                        *a = add_accum_1by1(*a, b, &mut carry);
                    }
                    ret
                }


                #[inline]
                fn sub(&self, other: &[u64]) -> Self::TARRAYCARRY {
                    let mut ret = self.copy();
                    let borrow = ret.sub_assign(other);
                    (ret, borrow)
                }

                #[inline]
                fn sub_assign(&mut self, other: &[u64]) -> bool {
                    debug_assert!(other.len() <= $N);

                    let mut borrow = false;
                    for (i, a) in self.iter_mut().enumerate() {
                        // constant time means we dynamically pad b if it's shorter
                        // rather than shortcutting the loop
                        // let b = if i >= other.len() { 0u64 } else { other[i] };
                        let ge = (i as u64).const_ge(other.len() as u64);
                        let dummyindex = ge.mux(0, i as u64) as usize;
                        let b = ge.mux(0, other[dummyindex]);
                        let diff = a.wrapping_sub(b);
                        let olda = *a;
                        *a = diff.wrapping_sub(borrow as u64);
                        borrow =  (diff > olda) || (*a > diff);
                    }
                    borrow
                }

                /// sub_assign_equiv is basically a duplicate of sub_assign
                /// but the rhs is a fixed size array of the same size as the
                /// current array
                /// TODO: use a macro to DRY this
                #[inline]
                fn sub_assign_equiv(&mut self, other: [u64; $N]) -> bool {
                    let mut borrow = false;
                    for (i, a) in self.iter_mut().enumerate() {
                        // constant time means we dynamically pad b if it's shorter
                        // rather than shortcutting the loop
                        let ge = (i as u64).const_ge(other.len() as u64);
                        let dummyindex = ge.mux(0, i as u64) as usize;
                        let b = ge.mux(0, other[dummyindex]);
                        let diff = a.wrapping_sub(b);
                        let olda = *a;
                        *a = diff.wrapping_sub(borrow as u64);
                        borrow =  (diff > olda) || (*a > diff);
                    }
                    borrow
                }

                #[inline]
                fn sub_ignore_carry(&self, other: &[u64]) -> Self {
                    debug_assert!(other.len() <= $N);
                    let mut ret = self.copy();
                    ret.sub_assign(other);
                    ret
                }

                /// returns true if result is negative
                #[inline]
                fn add_assign_signed(&mut self, self_is_neg: bool, other: [u64; $N], other_is_neg: bool) -> bool {
                    // subtracting a negative is same as adding, so flip other_is_neg
                    self.sub_assign_signed(self_is_neg, other, !other_is_neg)
                }

                /// returns true if result is negative
                // TODO: not constant
                fn sub_assign_signed(&mut self, self_is_neg: bool, other: [u64; $N], other_is_neg: bool) -> bool {
                    match (self_is_neg, other_is_neg, DigitsArray::cmp(&*self, &other)) {
                        (false, false, Some(Ordering::Less)) => {
                            // self - other -> neg; other > self
                            *self = other.sub_ignore_carry(&*self);
                            true
                        },
                        (false, false, _) => {
                            // self - other -> pos; self > other
                            self.sub_assign(&other);
                            false
                        },
                        (true, false, _) => {
                            // -self - other; self stays neg
                            self.add_assign(&other);
                            true
                        },
                        (false, true, _) => {
                            // self - -other = self + other; self stays pos
                            self.add_assign(&other);
                            false
                        },
                        (true, true, Some(Ordering::Less)) => {
                            // -self - -other = -self + other; self < other
                            *self = other.sub_ignore_carry(&*self);
                            false
                        },
                        (true, true, _) => {
                            // -self - -other = -self + other; self >= other
                            // other - self would give neg, so we go self - other and set self to neg
                            self.sub_assign(&other);
                            true
                        }
                    }
                }


                #[inline]
                fn add(&self, other: &[u64]) -> Self::TARRAYCARRY {
                    let mut ret = self.copy();
                    let carry = ret.add_assign(other);
                    (ret, carry)
                }

                #[inline]
                fn add_assign(&mut self, other: &[u64]) -> bool {
                    debug_assert!(other.len() <= $N);

                    let mut carry = false;
                    for (i, a) in self.iter_mut().enumerate() {
                        // constant time means we dynamically pad b if it's shorter
                        // rather than shortcutting the loop
                        let ge = (i as u64).const_ge(other.len() as u64);
                        let dummyindex = ge.mux(0, i as u64) as usize;
                        let b = ge.mux(0, other[dummyindex]);
                        let sum = a.wrapping_add(b);
                        let olda = *a;
                        *a = sum.wrapping_add(carry as u64);
                        carry =  (sum < olda) || (*a < sum);
                    }
                    carry
                }

                /// This is a copy of add_assign but takes a fixed size array,.
                /// TODO: use a macro or some such to make this DRY
                #[inline]
                fn add_assign_equiv(&mut self, other: [u64; $N]) -> bool {
                    let mut carry = false;
                    for (i, a) in self.iter_mut().enumerate() {
                        // constant time means we dynamically pad b if it's shorter
                        // rather than shortcutting the loop
                        let ge = (i as u64).const_ge(other.len() as u64);
                        let dummyindex = ge.mux(0, i as u64) as usize;
                        let b = ge.mux(0, other[dummyindex]);
                        let sum = a.wrapping_add(b);
                        let olda = *a;
                        *a = sum.wrapping_add(carry as u64);
                        carry = (sum < olda) || (*a < sum);
                    }
                    carry
                }

                #[inline]
                fn add_ignore_carry(&self, other: &[u64]) -> Self {
                    let (ret, _) = self.add(other);
                    ret
                }

                #[inline]
                fn copy(&self) -> [u64; $N] {
                    *self
                }

                #[inline]
                fn populate_padded_from_slice(y: &[u64]) -> [u64; $N] {
                    let mut new = [0u64; $N];
                    for (from, to) in y.iter().zip(new.iter_mut()) {
                        *to = *from;
                    }
                    new
                }

                #[inline]
                fn populate_padded_mostsig_from_slice(y: &[u64]) -> [u64; $N] {
                    let mut new = [0u64; $N];
                    for (from, to) in y.iter().rev().zip(new.iter_mut().rev()) {
                        *to = *from;
                    }
                    new
                }



                #[inline]
                fn rem_1(&self, y: u64) -> u64 {
                    let (_, rem) = self.div_rem_1(y);
                    rem
                }

                #[inline]
                fn div_rem_1(&self, y: u64) -> ([u64; $N], u64) {
                    debug_assert!(y != 0u64); // don't divide by zero

                    // println!("  div_rem_1({:?}, {:?})", self, y);

                    let mut quotient = [0u64; $N];
                    let mut r = 0;

                    // Knuth 4.3.1 Algo D, exercise 16 in Semi-numerical Algorithms
                    // div_short(u, v) where u is of length n
                    // r = 0
                    // for j = n - 1 to 0
                    //   w[j] = floor((rb+u[j])/v)
                    //   r = (rb + u[j]) % v
                    for (w, u) in quotient.iter_mut().rev().zip(self.iter().rev()) {
                        let (_, quotlo, rem) = div_2_limbs_by_1_limb(r, *u, y);
                        *w = quotlo;
                        r = rem;
                    }
                    (quotient, r)
                }

                #[inline]
                fn div2(&self) -> [u64; $N] {
                    self.shift_right_bits(1)
                }

                #[inline]
                fn div2_assign(&mut self) {
                    self.shift_right_bits_assign(1);
                }

                #[inline]
                fn shift_left_digits(&self, shifts: usize) -> Self {
                    debug_assert!(shifts <= $N);
                    let mut ret = self.copy();
                    ret.shift_left_digits_assign(shifts);
                    ret
                }

                #[inline]
                fn shift_left_digits_assign(&mut self, shifts: usize) {
                    debug_assert!(shifts <= $N);
                    for i in (0 .. $N).rev() {
                        let ge = (i as u64).const_ge(shifts as u64);
                        let dummyindex = ge.mux((i as u64).wrapping_sub(shifts as u64), 0) as usize;
                        self[i] = (i as u64).const_ge(shifts as u64).mux(self[dummyindex], 0);
                    }
                }

                #[inline]
                fn shift_right_digits(&self, shifts: usize) -> Self {
                    debug_assert!(shifts <= $N);
                    let mut ret = self.copy();
                    ret.shift_right_digits_assign(shifts);
                    ret
                }

                #[inline]
                fn shift_right_digits_assign(&mut self, shifts: usize) {
                    debug_assert!(shifts <= $N);
                    for i in 0 .. $N {
                        // if i + shifts < $N {
                        //     self[i] = self[i + shifts];
                        // } else {
                        //     self[i] = 0;
                        // }
                        let lt = ((i+shifts) as u64).const_lt($N);
                        let dummyindex = lt.mux((i+shifts) as u64, 0) as usize;
                        self[i] = lt.mux(self[dummyindex], 0);
                    }
                }

                #[inline]
                fn shift_right_bits(&self, bits: usize) -> [u64; $N] {
                    let mut res = self.copy();
                    res.shift_right_bits_assign(bits);
                    res
                }

                #[inline]
                fn shift_right_bits_assign(&mut self, bits: usize) {
                    let limbshift = bits / 64;
                    let newshift = bits % 64;
                    self.shift_right_digits_assign(limbshift); // constant time and works okay with zero
                    let mut lowbits = 0u64;
                    for d in self.iter_mut().rev() {
                        let tmp = *d << (64 - newshift);
                        *d >>= newshift;
                        *d |= lowbits;
                        lowbits = tmp;
                    }
                }

                #[inline]
                fn shift_left_bits(&self, bits: usize) -> [u64; $N] {
                    let mut res = self.copy();
                    res.shift_left_bits_assign(bits);
                    res
                }

                #[inline]
                fn shift_left_bits_assign(&mut self, bits: usize) {
                    let limbs = bits / 64;
                    let newshift = bits % 64;
                    self.shift_left_digits_assign(limbs);
                    let mut highbits = 0u64;
                    for d in self.iter_mut() {
                        let tmp = *d >> (64 - newshift);
                        *d <<= newshift;
                        *d |= highbits;
                        highbits = tmp;
                    }
                }

                #[inline]
                fn cmp(&self, y: &(Self)) -> Option<Ordering> {
                    match self.cmpi(y) {
                        -1 => Some(Ordering::Less),
                        0 => Some(Ordering::Equal),
                        1 => Some(Ordering::Greater),
                        _ => None
                    }
                }

                #[inline]
                fn cmpi(&self, y: &(Self)) -> i64 {
                    let mut res = 0i64;
                    self.iter().zip(y.iter()).rev().for_each(|(l, r)| {
                        let limbcmp = (l.const_gt(*r) as i64) | -(r.const_gt(*l) as i64);
                        res = res.abs().mux(res, limbcmp);
                    });
                    res
                }

                #[inline]
                fn greater_or_equal(&self, y: &(Self)) -> bool {
                    DigitsArray::cmp(self, &y) != Some(Ordering::Less)
                }
                #[inline]
                fn less_or_equal(&self, y: &(Self)) -> bool {
                    DigitsArray::cmp(self, &y) != Some(Ordering::Greater)
                }
                #[inline]
                fn greater(&self, y: &(Self)) -> bool {
                    DigitsArray::cmp(self, &y) == Some(Ordering::Greater)
                }
                #[inline]
                fn less(&self, y: &(Self)) -> bool {
                    DigitsArray::cmp(self, &y) == Some(Ordering::Less)
                }
            }

            #[cfg(test)]
            mod $M {
                use super::*;
                // use limb_math;
                use proptest::prelude::*;
                use rand::{OsRng, Rng};

                prop_compose! {
                    fn arb_limbs()(seed in any::<u64>()) -> [u64; $N] {
                        if seed == 0 {
                            [0u64; $N]
                        } else if seed == 1 {
                            let mut ret = [0u64; $N];
                            ret[0] = 1;
                            ret
                        } else {
                            let mut rng = OsRng::new().expect("Failed to get random number");
                            let mut limbs = [0u64; $N];
                            for limb in limbs.iter_mut() {
                                *limb = rng.next_u64();
                            }
                            limbs
                        }
                    }
                }

                proptest! {
                    #[test]
                    fn divrem_random(ref a in arb_limbs(), b in any::<u64>()) {
                        prop_assume!(b != 0);
                        let (quotient, remainder) = a.div_rem_1(b);
                        let a_back = quotient.mul_add_by_digit(b, remainder);
                        assert_eq!(&(a)[..], &a_back[..$N]);
                    }

                    #[test]
                    fn identity(a in arb_limbs()) {
                        prop_assert_eq!(a.mul_by_digit(1), a.expand_one());

                        // prop_assert_eq!(a.mul_classic(&<[u64; $N]>::one()), <[u64; $N*2]>::from(a));
                        prop_assert_eq!(a.mul_add_by_digit(1, 0), a.expand_one());
                        prop_assert_eq!(a.add_ignore_carry(&[0]), a);
                        prop_assert_eq!(a.add_ignore_carry(&<[u64; $N]>::zero()), a);

                        prop_assert_eq!(a.sub_ignore_carry(&<[u64; $N]>::zero()), a);

                        // prop_assert_eq!(a / a, $classname::one());
                        // prop_assert_eq!(a.pow(0), $classname::one());
                        // prop_assert_eq!(a.pow(1), a);
                    }


                    #[test]
                    fn zero(a in arb_limbs()) {
                        assert_eq!(a.mul_by_digit(0), [0u64; $N+1]);
                        // assert_eq!(a.mul_classic(&<[u64; $N]>::zero()), [0u64; $N*2]);
                        assert_eq!(a.sub(&a), (<[u64; $N]>::zero(), false));
                    }

                    #[test]
                    fn commutative(a in arb_limbs(), b in arb_limbs()) {
                        prop_assert_eq!(a.add(&b), b.add(&a));
                        // prop_assert_eq!(a.mul_classic_equiv(b), b.mul_classic_equiv * a);
                    }

                    #[test]
                    fn associative(a in arb_limbs(), b in arb_limbs(), c in arb_limbs()) {
                        prop_assert_eq!(a.add_ignore_carry(&b).add_ignore_carry(&c), a.add_ignore_carry(&b.add_ignore_carry(&c)));
                        // prop_assert_eq!((a * b) * c , a * (b * c));
                    }

                    #[test]
                    #[ignore]
                    fn distributive(a in arb_limbs(), b in arb_limbs(), c in arb_limbs()) {
                        // prop_assert_eq!(a * (b + c) , a * b + a * c);
                        // prop_assert_eq!((a + b) * c , a * c + b * c);
                        // prop_assert_eq!((a - b) * c , a * c - b * c);
                    }

                }

            }
        )+
}}

digits_u64_impls! {
    a1 1, a2 2, a3 3, a4 4, a5 5, a6 6, a7 7, a8 8, a9 9, a10 10, a11 11, a12 12, a13 13, a14 14, a15 15, a16 16, a17 17, a18 18, a19 19
}

// TODO: not constant time (variable size limbs makes this tricky)
#[inline]
pub fn cmpslice(x: &[u64], y: &[u64]) -> Option<Ordering> {
    let x_len = x.iter().rev().skip_while(|limb| **limb == 0u64).count();
    let y_len = y.iter().rev().skip_while(|limb| **limb == 0u64).count();
    match x_len.cmp(&y_len) {
        Ordering::Equal => match x[..x_len]
            .iter()
            .zip(y[..y_len].iter())
            .rev()
            .find(|&(lhs, rhs)| lhs.cmp(rhs) != Ordering::Equal)
        {
            Some((lhs, rhs)) => lhs.partial_cmp(rhs),
            None => Some(Ordering::Equal),
        },
        Ordering::Less => Some(Ordering::Less),
        Ordering::Greater => Some(Ordering::Greater),
    }
}

#[inline]
pub fn mul_slice_by_1_assign_carry(x: &mut [u64], y: u64) -> u64 {
    let mut carry = 0u64;
    for biglimb in x.iter_mut() {
        let (hi, lo) = mul_1_limb_by_1_limb(*biglimb, y);
        *biglimb = lo.wrapping_add(carry);
        carry = hi.wrapping_add((lo < carry) as u64);
    }
    carry
}

/* x * y + z */
#[inline]
pub fn mul_add_3_limbs(x: u64, y: u64, z: u64) -> (u64, u64) {
    let (hi, mut lo) = mul_1_limb_by_1_limb(x, y);
    let mut carry = 0u64;
    lo = add_accum_1by1(lo, z, &mut carry);
    (hi + carry, lo)
}

/* Adapted from https://github.com/Aatch/ramp/blob/master/src/ll/limb.rs
 * Apache License
 */
#[inline]
pub fn mul_1_limb_by_1_limb(u: u64, v: u64) -> (u64, u64) {
    // see http://www.hackersdelight.org/hdcodetxt/muldwu.c.txt

    const BITS: usize = 32;
    const LO_MASK: Wrapping<u64> = Wrapping((1u64 << BITS) - 1);

    let u = Wrapping(u);
    let v = Wrapping(v);

    let u0 = u >> BITS;
    let u1 = u & LO_MASK;
    let v0 = v >> BITS;
    let v1 = v & LO_MASK;

    let t = u1 * v1;
    let w3 = t & LO_MASK;
    let k = t >> BITS;

    let t = u0 * v1 + k;
    let w2 = t & LO_MASK;
    let w1 = t >> BITS;

    let t = u1 * v0 + w2;
    let k = t >> BITS;

    ((u0 * v0 + w1 + k).0, ((t << BITS) + w3).0)
}

pub fn mul_1_limb_by_1_limb_array(u: u64, v: u64) -> [u64; 2] {
    let temp = mul_1_limb_by_1_limb(u, v); // high, low
    [temp.1, temp.0] // [low, high]
}

#[inline]
// from bearssl https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/int/i32_div32.c
// takes high (u1), low (u0), divisor (v)
// returns (quotient, remainder)
pub fn xdiv_2_limbs_by_1_limb(u1: u64, u0: u64, v: u64) -> (u64, u64) {
    let mut q = 0u64;

    let mut hi = u1.const_eq(v).mux(0, u1);
    let mut lo = u0;
    for k in (1..64).rev() {
        let j: usize = 64 - k;
        let w = (hi << j) | (lo >> k);
        // let ctl = (if w >= v { 1 } else { 0 }) | (hi >> k);
        let ctl = w.const_ge(v) | (hi >> k);
        let hi2 = w.wrapping_sub(v) >> j;
        let lo2 = lo.wrapping_sub(v << k);
        hi = ctl.mux(hi2, hi);
        lo = ctl.mux(lo2, lo);
        q |= ctl << k;
    }
    // let cf = (if lo >= v { 1 } else { 0 }) | hi;
    let cf = lo.const_ge(v) | hi;
    q |= cf;
    let r = cf.mux(lo.wrapping_sub(v), lo);
    // println!("{} {} / {} = {} rem {}", u1, u0, v, q, r);
    (q, r)
}

pub fn div_2_limbs_by_1_limb(u1: u64, u0: u64, v: u64) -> (u64, u64, u64) {
    // let mut q1 = 0u64;
    let q0: u64;
    let r: u64;

    let ctl = v.const_le(u1);
    let q1 = ctl.mux(u1 / v, 0);
    let k = ctl.mux(u1 % v, u1);
    let (tmpq0, tmpr) = xdiv_2_limbs_by_1_limb(k, u0, v);
    q0 = tmpq0;
    r = tmpr;
    (q1, q0, r)
}

// only need to return quotient for this helper func
// assume everything is already normalized so y[1] has high bit set
// TODO: not constant time
#[inline]
pub fn div_3_limbs_by_2_limbs(u: [u64; 3], v: [u64; 2]) -> [u64; 2] {
    // debug_assert!(v[1].leading_zeros() == 0);
    debug_assert!(u[2] > 0 && v[1] > 0);

    //                 q1 q0
    //           ___________
    //     v1 v0)   u2 u1 u0
    //
    let (q1, q0, mut rest) = div_2_limbs_by_1_limb(u[2], u[1], v[1]);
    let mut qest = [q0, q1];
    // let mut restoverflow = false;
    while let Some(Ordering::Greater) = cmpslice(&(qest.mul_by_digit(v[0])), &[u[0], rest]) {
        qest.sub_assign(&[1]);
        let (newrest, overflow) = rest.overflowing_add(v[1]);
        rest = newrest;
        // restoverflow = overflow;
        if overflow {
            break;
        }
    }
    qest
}

#[inline]
pub fn add_accum_1by1(a: u64, b: u64, acc: &mut u64) -> u64 {
    // acc is the carry
    let (sum1, carry1) = a.overflowing_add(b);
    // let (sum2, carry2) = sum1.overflowing_add(*acc);
    *acc += carry1 as u64; // + carry2 as u64;
    sum1
}

/* returns the hi and lo results and a carry flag */
#[inline]
pub fn add_2by1(a: (u64, u64), b: u64) -> (u64, u64, u64) {
    let a_lo = a.1;
    let a_hi = a.0;
    let (sum_lo, carry) = a_lo.overflowing_add(b);
    let (sum_hi, carry2) = a_hi.overflowing_add(carry as u64);
    (sum_hi, sum_lo, carry2 as u64)
}

/* takes two limbs, adds them, and returns two limbs (hi, lo) where hi is 0 or 1 */
#[inline]
pub fn add_1by1(a: u64, b: u64) -> (u64, u64) {
    let (l, c) = a.overflowing_add(b);
    (c as u64, l)
}

// Subtract with borrow:
#[inline]
pub fn sub_accum(a: u64, b: u64, acc: &mut u64) -> u64 {
    // acc is the borrow
    let (sub1, borrow1) = a.overflowing_sub(b);
    *acc += borrow1 as u64;
    let (sub2, borrow2) = sub1.overflowing_sub(*acc);
    *acc = borrow2 as u64;
    sub2
}

#[inline]
pub fn split_u64(i: u64) -> (u32, u32) {
    (high_32(i), low_32(i))
}

#[inline]
pub fn high_32(i: u64) -> u32 {
    (i >> 32) as u32
}

#[inline]
pub fn low_32(i: u64) -> u32 {
    (i & 0xFFFFFFFF) as u32
}

#[inline]
pub fn split_u64_as_u64(i: u64) -> (u64, u64) {
    (i >> 32, i & 0xFFFFFFFF)
}

#[inline]
#[allow(dead_code)]
fn combine_u64(hi: u64, lo: u64) -> u64 {
    (hi << 32) | lo
}

#[inline]
pub fn combine_u32(hi: u32, lo: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    // use limb_math;
    use proptest::prelude::*;
    use rand::OsRng;

    prop_compose! {
        fn arb_limbs8()(seed in any::<u64>()) -> [u64; 8] {
            if seed == 0 {
                [0u64; 8]
            } else if seed == 1 {
                let mut ret = [0u64; 8];
                ret[0] = 1;
                ret
            } else {
                let mut rng = OsRng::new().expect("Failed to get random number");
                let mut limbs = [0u64; 8];
                for limb in limbs.iter_mut() {
                    *limb = rng.next_u64();
                }
                limbs
            }
        }
    }

    proptest!{}

    #[test]
    fn limbs_compare() {
        assert_eq!(cmpslice(&[0, 0], &[1, 0]), Some(Ordering::Less));
        assert_eq!(cmpslice(&[1, 0], &[1, 0]), Some(Ordering::Equal));
        assert_eq!(cmpslice(&[1, 0], &[0, 0]), Some(Ordering::Greater));
        assert_eq!(cmpslice(&[0, 0, 1], &[0, 0]), Some(Ordering::Greater));
        assert_eq!(cmpslice(&[1, 0, 0], &[1]), Some(Ordering::Equal)); // 001 == 1
        assert_eq!(cmpslice(&[2, 0, 0], &[1]), Some(Ordering::Greater)); // 2 > 1
        assert_eq!(
            cmpslice(&[0xFFFFFFFFFFFFFFFF, 0, 0], &[0, 1]),
            Some(Ordering::Less)
        ); // 2 > 1
    }

    #[test]
    fn limbs_compare_constant_equal() {
        assert_eq!(DigitsArray::cmp(&[0, 0], &[1, 0]), Some(Ordering::Less));
        assert_eq!(DigitsArray::cmp(&[1, 0], &[1, 0]), Some(Ordering::Equal));
        assert_eq!(DigitsArray::cmp(&[1, 0], &[0, 0]), Some(Ordering::Greater));
        assert_eq!(
            DigitsArray::cmp(&[0, 0, 1], &[0, 0, 0]),
            Some(Ordering::Greater)
        );
        assert_eq!(
            DigitsArray::cmp(&[0xFFFFFFFFFFFFFFFF, 0, 0], &[0, 1, 0]),
            Some(Ordering::Less)
        ); // 2 > 1
    }

    #[test]
    fn mul_add_precalc() {
        let mut carry = 0u64;
        let sum = add_accum_1by1(0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, &mut carry);
        assert_eq!(sum, 0xfffffffffffffffeu64);
        assert_eq!(carry, 1u64);

        let (h, l) = mul_1_limb_by_1_limb(0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64);
        assert_eq!(0xFFFFFFFFFFFFFFFEu64, h);
        assert_eq!(1u64, l);

        let (h, l) = mul_add_3_limbs(
            // x * y + z
            0xFFFFFFFFFFFFFFFFu64,
            0xFFFFFFFFFFFFFFFFu64,
            0xFFFFFFFFFFFFFFFFu64,
        );
        assert_eq!(0xFFFFFFFFFFFFFFFFu64, h);
        assert_eq!(0u64, l);
    }

    #[test]
    fn shifts() {
        let highbit = 1u64 << 63;
        let a = [1u64; 4];
        assert_eq!(a.shift_left_bits(1), [2u64; 4]);
        assert_eq!(a.shift_right_bits(1), [highbit, highbit, highbit, 0]);
        // b as a number is little endian, so 1u64 is least sig
        //  so in a left shift we have 4, 3, 2, 1 << 2 = 2, 1, 0, 0
        //                                   = 0, 0, 1, 2
        let b = [1u64, 2u64, 3u64, 4u64];
        assert_eq!(b.shift_left_digits(2), [0, 0, 1, 2]);
        assert_eq!(b.shift_left_digits(3), [0, 0, 0, 1]);
        assert_eq!(b.shift_left_digits(4), [0, 0, 0, 0]);

        assert_eq!(b.shift_right_digits(2), [3, 4, 0, 0]);
        assert_eq!(b.shift_right_digits(3), [4, 0, 0, 0]);
        assert_eq!(b.shift_right_digits(4), [0, 0, 0, 0]);
    }

    #[test]
    fn divrem1() {
        let mut limbs = [0u64; 8];
        limbs[1] = 10u64; // 184467440737095516160
        let (quotient, remainder) = limbs.div_rem_1(3u64);
        assert_eq!(remainder, 1u64);
        assert_eq!(quotient[1], 3u64);
        assert_eq!(quotient[0], 6148914691236517205u64);
    }

    #[test]
    fn div2by1() {
        let (q1, q0, _) = div_2_limbs_by_1_limb(
            18038829716013023058,
            8200750983511227423,
            10003326117277521608,
        );
        assert_eq!(q0, 14817959211909354406);
        assert_eq!(q1, 1);
    }

    #[test]
    fn div3by2() {
        let a = [1u64, 1, 1];
        let b = [2u64, 2];
        assert_eq!(div_3_limbs_by_2_limbs(a, b), [9223372036854775808, 0]);
    }

}
