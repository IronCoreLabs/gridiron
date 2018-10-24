#[macro_export]
macro_rules! from_unsigned31 { ($classname: ident; $($T:ty),*) => { $(
    impl From<$T> for $classname {
        fn from(other: $T) -> $classname {
            let mut ret = $classname::zero();
            ret.limbs[0] = other as u32;
            ret
        }
    }
)+ }}

#[macro_export]
macro_rules! from_signed31 { ($classname: ident; $($T:ty),*) => { $(
    impl From<$T> for $classname {
        // TODO: not constant time
        fn from(other: $T) -> $classname {
            unimplemented!();
            // let mut ret = $classname::zero();
            // if other < 0 {
            //   ret.limbs[0] = (other * -1) as u32;
            //   $classname { limbs: PRIME.sub_ignore_carry(&ret.limbs) }
            // } else{
            //   ret.limbs[0] = other as u32;
            //   ret
            // }
        }
    }
)+ }}

/// Create an Fp type given the following parameters:
/// - modname - the name of the module you want the Fp type in.
/// - classname - the name of the Fp struct
/// - bits - How many bits the prime is.
/// - limbs - Number of limbs (ceil(bits/64))
/// - prime - prime number in limbs, least significant digit first. (Note you can get this from `sage` using `num.digits(2 ^ 64)`).
/// - barrett - barrett reduction for reducing values up to twice the number of prime bits (double limbs). This is `floor(2^(64*numlimbs*2)/prime)`.
#[macro_export]
macro_rules! fp31 { ($modname: ident, $classname: ident, $bits: tt, $limbs: tt, $prime: expr, $barrettmu: expr, $montgomery_r_inv: expr, $montgomery_r_squared: expr, $montgomery_m0_inv: expr) => { pub mod $modname {
    use $crate::digits::util::*;
    use $crate::digits::signed::*;
    use $crate::digits::unsigned::*;
    use $crate::digits::constant_time_primitives::*;
    use std::cmp::Ordering;
    use std::fmt;
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign, BitAnd, BitAndAssign};
    use num_traits::{One, Zero, Inv, Pow};
    use std::convert::From;
    use std::option::Option;

    pub const LIMBSIZEBITS: usize = 31;
    pub const BITSPERBYTE: usize = 8;
    pub const PRIME: [u32; NUMLIMBS] = $prime;
    pub const PRIMEBITS: usize = $bits;
    pub const PRIMEBYTES: usize = PRIMEBITS / BITSPERBYTE;
    pub const NUMLIMBS: usize = $limbs;
    pub const NUMDOUBLELIMBS: usize = $limbs * 2;
    pub const BARRETTMU: [u32; NUMLIMBS + 1] = $barrettmu;
    pub const MONTRINV: [u32; NUMLIMBS] = $montgomery_r_inv;
    pub const MONTRSQUARED: [u32; NUMLIMBS] = $montgomery_r_squared;
    pub const MONTM0INV: u32 = $montgomery_m0_inv;

    #[derive(PartialEq, Eq, Ord, Clone, Copy)]
    pub struct $classname {
        pub(crate) limbs: [u32; NUMLIMBS],
    }

    #[derive(PartialEq, Eq, Ord, Clone, Copy)]
    pub struct Monty {
        pub(crate) limbs: [u32; NUMLIMBS],
    }

    impl fmt::Debug for $classname {
        fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
            write!(f, "{}(", stringify!($classname))?;
            let x: Vec<String> = self.limbs.iter().map(|x| format!("{:#x}", x)).collect();
            write!(f, "{}", x.join(", "))?;
            write!(f, ")")?;
            Ok(())
        }
    }

    impl fmt::Display for $classname {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", self.to_str_decimal())?;
            Ok(())
        }
    }

    /// Prints the hex value of the number in big endian (most significant
    /// digit on the left and least on the right) to make debugging easier.
    impl fmt::LowerHex for $classname {
        fn fmt(&self, fmtr: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            fmtr.write_fmt(format_args!("0x{}", self.to_str_hex()))
        }
    }

    impl PartialOrd for $classname {
        #[inline]
        fn partial_cmp(&self, other: &$classname) -> Option<Ordering> {
            unimplemented!();
        }
    }

    impl Zero for $classname {
        #[inline]
        fn zero() -> Self {
            $classname {
                limbs: [0u32; NUMLIMBS],
            }
        }

        #[inline]
        fn is_zero(&self) -> bool {
            self.limbs.iter().all(|limb| limb == &0u32)
        }
    }

    impl One for $classname {
        #[inline]
        fn one() -> Self {
            let mut ret = $classname::zero();
            ret.limbs[0] = 1u32;
            ret
        }

        #[inline]
        fn is_one(&self) -> bool {
            self.limbs[0] == 1u32 && self.limbs.iter().skip(1).all(|limb| limb == &0u32)
        }
    }

    impl Add for $classname {
        type Output = $classname;
        #[inline]
        fn add(mut self, other: $classname) -> $classname {
            self += other;
            self
        }
    }

    impl AddAssign for $classname {
        #[inline]
        fn add_assign(&mut self, other: $classname) {
            let a = &mut self.limbs;
            let mut ctl = $classname::add_limbs(a, other.limbs, 1);
            ctl |= $classname::sub_limbs(a, PRIME, 0).not();
            $classname::sub_limbs(a, PRIME, ctl);
        }
    }

    impl Sub for $classname {
        type Output = $classname;
        #[inline]
        fn sub(mut self, other: $classname) -> $classname {
            self -= other;
            self
        }
    }

    impl SubAssign for $classname {
        #[inline]
        fn sub_assign(&mut self, other: $classname) {
            let a = &mut self.limbs;
            let needs_add = $classname::sub_limbs(a, other.limbs, 1);
            $classname::add_limbs(a, PRIME, needs_add);
        }
    }

    impl Mul for $classname {
        type Output = $classname;
        #[inline]
        fn mul(mut self, rhs: $classname) -> $classname {
            self *= rhs;
            self
        }
    }

    impl Mul<u32> for $classname {
        type Output = $classname;
        #[inline]
        fn mul(mut self, rhs: u32) -> $classname {
            self *= $classname::new_from_u32(rhs);
            self
        }
    }

    impl MulAssign for $classname {
        #[inline]
        fn mul_assign(&mut self, rhs: $classname) {
            let doublesize = $classname::mul_limbs_classic(&self.limbs, &rhs.limbs);
            self.limbs = $classname::reduce_barrett(&doublesize);
        }
    }

    impl Inv for $classname {
        type Output = $classname;
        #[inline]
        fn inv(self) -> $classname {
            $classname::one().div(self)
        }
    }

    impl Pow<u8> for $classname {
        type Output = $classname;
        #[inline]
        fn pow(self, rhs: u8) -> $classname {
            unimplemented!();
        }
    }

    impl Pow<$classname> for $classname {
        type Output = $classname;
        #[inline]
        fn pow(self, rhs: $classname) -> $classname {
            unimplemented!();
        }
    }

    impl Div for $classname {
        type Output = $classname;
        fn div(self, rhs: $classname) -> $classname {
            unimplemented!();
        }
    }

    impl Neg for $classname {
        type Output = $classname;
        #[inline]
        fn neg(mut self) -> $classname {
            $classname::cond_negate(&mut self.limbs, 1);
            self
        }
    }


    from_unsigned31! { $classname; u32, u8 }
    from_signed31! { $classname; i32, i8 }

    /// Assume element zero is most sig
    impl From<[u8; PRIMEBYTES]> for $classname {
        fn from(src: [u8; PRIMEBYTES]) -> Self {
            unimplemented!();
        }
    }

    impl Default for $classname {
        #[inline]
        fn default() -> Self {
            Zero::zero()
        }
    }

    impl Monty {
        pub fn to_norm(self) -> $classname {
            let mut one = [0u32; NUMLIMBS];
            one[0] = 1;
            $classname { limbs: (self * Monty{limbs: one}).limbs }
        }

        #[inline]
        pub fn normalize_assign_little(&mut self) {
            let new_limbs = $classname::normalize_little_limbs(self.limbs);
            self.limbs = new_limbs;
        }

        pub (crate) fn new(limbs:[u32; NUMLIMBS]) -> Monty{
            Monty{limbs}
        }
    }

    impl Mul<Monty> for Monty {
        type Output = Monty;

        #[inline]
        fn mul(self, rhs: Monty) -> Monty {
            // Constant time montgomery mult from https://www.bearssl.org/bigint.html
            let a = self.limbs;
            let b = rhs.limbs;
            let mut d = [0u32; NUMLIMBS]; // result
            let mut dh = 0u32; // can be up to 2W
            for i in 0 .. NUMLIMBS {
                // f←(d[0]+a[i]b[0])g mod W
                // g is MONTM0INV, W is word size
                // This might not be right, and certainly isn't optimal. Ideally we'd only calculate the low 31 bits
                // MUL31_lo((d[1] + MUL31_lo(x[u + 1], y[1])), m0i);
                let f: u32 = $classname::mul_31_lo(d[0] + $classname::mul_31_lo(a[i], b[0]), MONTM0INV);
                let mut z: u64; // can be up to 2W^2
                let mut r = 0u32; // can be up to 2W
                let ai = a[i];
                for j in 0 .. NUMLIMBS {
                    // z ← d[j]+a[i]b[j]+fm[j]+c
                    z = (ai as u64 * b[j] as u64) + (d[j] as u64) + (f as u64 * PRIME[j] as u64) + (r as u64);
                    r = (z >> 31) as u32;
                    // If j>0, set: d[j−1] ← z mod W
                    if j > 0 {
                        d[j-1] = (z as u32) & 0x7FFFFFFF;
                    }
                }
                // z ← dh+c
                z = (dh + r) as u64;
                // d[N−1] ← z mod W
                d[NUMLIMBS - 1] = (z as u32) & 0x7FFFFFFF;
                // dh ← ⌊z/W⌋
                dh = (z >> 31) as u32;
            }

            // if dh≠0 or d≥m, set: d←d−m
            // d.sub_assign_if(m, dh.const_neq(0) | d.const_gt(PRIME));
            Monty { limbs: d }
        }
    }

    impl Mul<$classname> for Monty {
        type Output = $classname;

        #[inline]
        fn mul(self, rhs: $classname) -> $classname {
            $classname::new((self * Monty::new(rhs.limbs)).limbs)
        }
    }

    impl Mul<Monty> for $classname {
        type Output = $classname;

        #[inline]
        fn mul(self, rhs: Monty) -> $classname {
            $classname::new((Monty::new(self.limbs) * rhs).limbs)
        }
    }

    impl Add<Monty> for Monty {
        type Output = Monty;
        #[inline]
        fn add(mut self, rhs: Monty) -> Monty {
            self += rhs;
            self
        }
    }

    impl AddAssign for Monty {
        #[inline]
        fn add_assign(&mut self, other: Monty) {
            unimplemented!();
        }
    }

    impl Sub<Monty> for Monty {
        type Output = Monty;
        #[inline]
        fn sub(mut self, rhs: Monty) -> Monty {
            self -= rhs;
            self
        }
    }

    impl SubAssign for Monty {
        #[inline]
        fn sub_assign(&mut self, other: Monty) {
            unimplemented!();
        }
    }

    impl PartialOrd for Monty {
        #[inline]
        fn partial_cmp(&self, other: &Monty) -> Option<Ordering> {
            unimplemented!();
        }
    }

    impl $classname {
        pub fn to_monty(self) -> Monty {
            Monty{limbs:self.limbs} * Monty{limbs:MONTRSQUARED}
        }

        #[inline]
        pub fn normalize_assign_little(&mut self) {
            let new_limbs = $classname::normalize_little_limbs(self.limbs);
            self.limbs = new_limbs;
        }

        ///Take the extra limb and incorporate that into the existing value by modding by the prime.
        /// This normalize should only be used when the input is at most
        /// 2*p-1. Anything that might be bigger should use the normalize_big
        /// options, which use barrett.
        #[inline]
        pub fn normalize_little_limbs(mut limbs:[u32; NUMLIMBS]) -> [u32; NUMLIMBS] {
            let needs_sub = $classname::sub_limbs(&mut limbs, PRIME, 0);
            $classname::sub_limbs(&mut limbs, PRIME, needs_sub);
            limbs
        }

        ///Take the extra limb and incorporate that into the existing value by modding by the prime.
        #[inline]
        pub fn normalize_little(mut self) -> Self {
            self.normalize_assign_little();
            self
        }

        #[inline]
        pub fn normalize_big(mut self, extra_limb: u32) -> Self {
            self.normalize_assign_big(extra_limb);
            self
        }

        #[inline]
        pub fn normalize_assign_big(&mut self, extra_limb: u32) {
            unimplemented!();
        }

        ///Convert the value to a byte array which is `PRIMEBYTES` long.
        pub fn to_bytes_array(&self) -> [u8; PRIMEBYTES] {
            unimplemented!();
        }

        ///Create a new instance given the raw limbs form. Note that this is least significant bit first.
        #[allow(dead_code)]
        pub fn new(digits: [u32; NUMLIMBS]) -> $classname {
            $classname {
                limbs: digits
            }
        }

        ///Convenience function to create a value from a single limb.
        pub fn new_from_u32(x: u32) -> $classname {
            let mut ret = $classname::zero();
            ret.limbs[0] = x;
            ret
        }

        ///Write out the value in decimal form.
        pub fn to_str_decimal(mut self) -> String {
            unimplemented!();
        }

        pub fn to_str_hex(&self) -> String {
            unimplemented!();
        }

        // From Handbook of Applied Crypto algo 14.12
        #[inline]
        fn mul_limbs_classic(a: &[u32; NUMLIMBS], b: &[u32; NUMLIMBS]) -> [u32; NUMDOUBLELIMBS] {
            unimplemented!();
            // let mut res = [0u32; NUMDOUBLELIMBS];
            // for i in 0..NUMLIMBS {
            //     let mut c = 0;
            //     for j in 0..NUMLIMBS {
            //         let (mut u, mut v) = mul_1_limb_by_1_limb(a[j], b[i]);
            //         v = add_accum_1by1(v, c, &mut u);
            //         v = add_accum_1by1(v, res[i + j], &mut u);
            //         res[i + j] = v;
            //         c = u;
            //     }
            //     res[i + NUMLIMBS] = c;
            // }
            // res
        }

    // From Handbook of Applied Cryptography 14.42
        // INPUT: positive integers x = (x2k−1 · · · x1x0)b, m = (mk−1 · · · m1m0)b (with mk−1 ̸= 0), and μ = ⌊b2k/m⌋.
        // OUTPUT: r = x mod m.
        // 1. q1←⌊x/bk−1⌋, q2←q1 · μ, q3←⌊q2/bk+1⌋.
        // 2. r1←x mod bk+1, r2←q3 · m mod bk+1, r←r1 − r2. 3. Ifr<0thenr←r+bk+1.
        // 4. Whiler≥mdo:r←r−m.
        // 5. Return(r).
    // Also helpful: https://www.everything2.com/title/Barrett+Reduction
    #[inline]
    pub fn reduce_barrett(a: &[u32; NUMDOUBLELIMBS]) -> [u32; NUMLIMBS] {
        unimplemented!();
        // // In this case, k = NUMLIMBS
        // // let mut q1 = [0u64; NUMLIMBS];
        // // q1.copy_from_slice(&a[NUMLIMBS - 1..NUMDOUBLELIMBS-1]);
        // let q1 = a.shift_right_digits(NUMLIMBS - 1);

        // // q2 = q1 * mu
        // // let q2 = BARRETTMU.mul_classic(&q1);
        // let q2 = q1.mul_classic(&BARRETTMU[..]);

        // let mut q3 = [0u64; NUMLIMBS];
        // q3.copy_from_slice(&q2[NUMLIMBS + 1..NUMDOUBLELIMBS + 1]);

        // let mut r1 = [0u64; NUMLIMBS + 2];
        // r1.copy_from_slice(&a[..NUMLIMBS+2]);

        // let r2 = &q3.mul_classic(&PRIME)[..NUMLIMBS + 1];

        // // r = r1 - r2
        // let (r3, _) = r1.expand_one().sub(&r2);
        // let mut r = [0u64; NUMLIMBS]; // need to chop off extra limb
        // r.copy_from_slice(&r3[..NUMLIMBS]);

        // // at most two subtractions with p
        // for _i in 0..2 {
        //     if DigitsArray::cmp(&r, &PRIME) != Some(Ordering::Less) {
        //         r.sub_assign(&PRIME);
        //     } else {
        //         // this branch is for constant time
        //         r.sub_assign(&[0u64; NUMLIMBS]);
        //     }
        // }
        // debug_assert!(DigitsArray::cmp(&r, &PRIME) == Some(Ordering::Less));
        // r
    }

    #[inline]
    fn add_limbs(a: &mut [u32; NUMLIMBS], b: [u32; NUMLIMBS], ctl: u32) -> u32 {
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw.wrapping_add(bw).wrapping_add(cc);
            cc = naw >> 31;
            *aa = ctl.mux(naw & 0x7FFFFFFF, aw)
        }
        cc
    }

    #[inline]
    fn sub_limbs(a: &mut [u32; NUMLIMBS], b: [u32; NUMLIMBS], ctl: u32) -> u32 {
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw.wrapping_sub(bw).wrapping_sub(cc);
            cc = naw >> 31;
            *aa = ctl.mux(naw & 0x7FFFFFFF, aw)
        }
        cc
    }

    #[inline]
    fn mul_31_lo(x:u32, y:u32) -> u32 {
        (x as u64 * y as u64) as u32 & 0x7FFFFFFFu32
    }

    fn cond_negate(a: &mut [u32; NUMLIMBS], ctl: u32) {
        let mut cc = ctl;
        let xm = ctl.wrapping_neg() >> 1;
        for mut ai in a.iter_mut() {
            let mut aw = *ai;
            aw = (aw ^ xm) + cc;
            *ai = aw & 0x7FFFFFFF;
            cc = aw >> 31;
        }
        println!("{:?}", a);
    }
}


    #[cfg(test)]
    mod tests {
        use super::*;
        // use limb_math;
        use proptest::prelude::*;
        use rand::OsRng;

        #[test]
        fn default_is_zero() {
            assert_eq!($classname::zero(), $classname::default())
        }

        prop_compose! {
            fn arb_fp()(seed in any::<u32>()) -> $classname {
                if seed == 0 {
                    $classname::zero()
                } else if seed == 1 {
                    $classname::one()
                } else {
                    let mut rng = OsRng::new().expect("Failed to get random number");
                    let mut limbs = [0u32; NUMLIMBS];
                    for limb in limbs.iter_mut() {
                        *limb = rng.next_u32();
                    }
                    limbs[NUMLIMBS - 1] &= (1u32 << (PRIMEBITS % 32)) - 1;
                    $classname {
                        limbs: limbs
                    }.normalize_little()
                }
            }
        }

        proptest! {
            #[test]
            fn identity(a in arb_fp()) {
                prop_assert_eq!(a * 1, a);

                prop_assert_eq!(a * $classname::one(), a);
                prop_assert_eq!($classname::one() * a, a);

                prop_assert_eq!(a + $classname::zero(), a);
                prop_assert_eq!($classname::zero() + a, a);

                prop_assert_eq!(a - $classname::zero(), a);
                prop_assert_eq!($classname::zero() - a, -a);

                prop_assert_eq!(a / a, $classname::one());
                prop_assert_eq!(a.pow(0), $classname::one());
                prop_assert_eq!(a.pow(1), a);
            }


            #[test]
            fn zero(a in arb_fp()) {
                assert_eq!(a * $classname::zero(), $classname::zero());
                assert_eq!($classname::zero() * a, $classname::zero());
                assert_eq!(a - a, $classname::zero());
            }

            #[test]
            fn commutative(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a + b , b + a);
                prop_assert_eq!(a * b , b * a);
            }

            #[test]
            fn associative(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
                prop_assert_eq!((a + b) + c , a + (b + c));
                prop_assert_eq!((a * b) * c , a * (b * c));
            }

            #[test]
            fn distributive(a in arb_fp(), b in arb_fp(), c in arb_fp()) {
                prop_assert_eq!(a * (b + c) , a * b + a * c);
                prop_assert_eq!((a + b) * c , a * c + b * c);
                prop_assert_eq!((a - b) * c , a * c - b * c);
            }

            #[test]
            fn add_equals_mult(a in arb_fp()) {
                prop_assert_eq!(a + a, a * 2);
                prop_assert_eq!(a + a + a, a * 3);
            }

            #[test]
            fn mul_equals_div(a in arb_fp(), b in arb_fp()) {
                prop_assume!(!a.is_zero() && !b.is_zero());
                let c = a * b;
                prop_assert_eq!(c / a, b);
                prop_assert_eq!(c / b, a);
            }

            #[test]
            fn mul_equals_div_numerator_can_be_zero(a in arb_fp(), b in arb_fp()) {
                prop_assume!(!b.is_zero());
                let c = a * b;
                prop_assert_eq!(c / b, a);
            }

            #[test]
            #[should_panic]
            fn div_by_zero_should_panic(a in arb_fp()) {
                a / $classname::zero()
            }

            #[test]
            fn div_zero_by_anything_should_be_zero(a in arb_fp()) {
                prop_assume!(!a.is_zero());
                let result = $classname::zero()/a;
                assert!(result.is_zero())
            }

            #[test]
            fn pow_equals_mult(a in arb_fp()) {
                prop_assert_eq!(a * a, a.pow(2));
                prop_assert_eq!(a * a * a, a.pow(3));
            }

            #[test]
            fn neg(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(-(-a), a);
                // prop_assert_eq!(a - b, a + -b);
                // prop_assert_eq!(-(a * b), -a * b);
                // prop_assert_eq!(-a * b, a * -b);
                // prop_assert_eq!(a + -a, $classname::zero());
            }

            #[test]
            fn from_bytes_roundtrip(a in arb_fp()) {
                let bytes = a.to_bytes_array();
                prop_assert_eq!($classname::from(bytes), a);
            }

            #[test]
            fn from_signed_ints(a in any::<i32>()) {
                if a < 0 {
                    prop_assert_eq!($classname::from(a), $classname { limbs: PRIME } - $classname::new_from_u32(a.abs() as u32));
                } else {
                    prop_assert_eq!($classname::from(a), $classname::new_from_u32(a as u32));
                }
            }

            #[test]
            fn to_monty_roundtrip_to_norm(a in arb_fp()) {
                prop_assert_eq!(a, a.to_monty().to_norm());
            }

            #[test]
            fn monty_mult_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, (a.to_monty() * b.to_monty()).to_norm());
            }

            #[test]
            fn mont_times_normal_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, a.to_monty()* b);
            }

            #[test]
            fn normal_times_mont_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, a* b.to_monty());
            }

            #[test]
            fn mont_add_works(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a + b, (a.to_monty() + b.to_monty()).to_norm())
            }

            #[test]
            fn mont_sub_works(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a - b, (a.to_monty() - b.to_monty()).to_norm())
            }

        }
    }
}};}
