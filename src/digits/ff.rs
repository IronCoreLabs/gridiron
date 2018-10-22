#[macro_export]
macro_rules! from_unsigned { ($classname: ident; $($T:ty),*) => { $(
    impl From<$T> for $classname {
        fn from(other: $T) -> $classname {
            let mut ret = $classname::zero();
            ret.limbs[0] = other as u64;
            ret
        }
    }
)+ }}

#[macro_export]
macro_rules! from_signed { ($classname: ident; $($T:ty),*) => { $(
    impl From<$T> for $classname {
        // TODO: not constant time
        fn from(other: $T) -> $classname {
            let mut ret = $classname::zero();
            if other < 0 {
              ret.limbs[0] = (other * -1) as u64;
              $classname { limbs: PRIME.sub_ignore_carry(&ret.limbs) }
            } else{
              ret.limbs[0] = other as u64;
              ret
            }
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
macro_rules! fp { ($modname: ident, $classname: ident, $bits: tt, $limbs: tt, $prime: expr, $barrettmu: expr, $montgomery_r_inv: expr, $montgomery_r_squared: expr, $montgomery_m0_inv: expr) => { pub mod $modname {
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
    use std::ops::Deref;

    pub const LIMBSIZEBYTES: usize = 8;
    pub const LIMBSIZEBITS: usize = 64;
    pub const PRIME: [u64; NUMLIMBS] = $prime;
    pub const PRIMEBITS: usize = $bits;
    pub const NUMBYTES: usize = PRIMEBITS / LIMBSIZEBYTES;
    pub const NUMLIMBS: usize = $limbs;
    pub const NUMDOUBLELIMBS: usize = $limbs * 2;
    pub const BARRETTMU: [u64; NUMLIMBS + 1] = $barrettmu;
    pub const BITSPERBYTE: usize = 8;
    pub const MONTRINV: [u64; NUMLIMBS] = $montgomery_r_inv;
    pub const MONTRSQUARED: [u64; NUMLIMBS] = $montgomery_r_squared;
    pub const MONTM0INV: u64 = $montgomery_m0_inv;

    #[derive(PartialEq, Eq, Ord, Clone, Copy)]
    pub struct $classname {
        pub(crate) limbs: [u64; NUMLIMBS],
    }

    #[derive(PartialEq, Eq, Ord, Clone, Copy)]
    pub struct Mont{
        pub(crate) limbs: [u64; NUMLIMBS],
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

    impl Deref for $classname {
        type Target = [u64; NUMLIMBS];
        fn deref(&self) -> &[u64; NUMLIMBS] {
            &self.limbs
        }
    }

    impl PartialOrd for $classname {
        #[inline]
        fn partial_cmp(&self, other: &$classname) -> Option<Ordering> {
            DigitsArray::cmp(&self.limbs, &other.limbs)
        }
    }

    impl Zero for $classname {
        #[inline]
        fn zero() -> Self {
            $classname {
                limbs: [0u64; NUMLIMBS],
            }
        }

        #[inline]
        fn is_zero(&self) -> bool {
            self.limbs.iter().all(|limb| limb == &0u64)
        }
    }

    impl One for $classname {
        #[inline]
        fn one() -> Self {
            let mut ret = $classname::zero();
            ret.limbs[0] = 1u64;
            ret
        }

        #[inline]
        fn is_one(&self) -> bool {
            self.limbs[0] == 1u64 && self.limbs.iter().skip(1).all(|limb| limb == &0u64)
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
            let carry = self.limbs.add_assign(&other.limbs);
            self.normalize_assign_little(carry as u64);
            debug_assert!(&self.limbs.less(&PRIME));
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
            let borrow = self.limbs.sub_assign(&other.limbs);
            if borrow {
                self.limbs.add_assign(&PRIME);
            } else {
                // this is here to keep this constant time
                self.limbs.add_assign(&[0u64; NUMLIMBS]);
            }
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

    impl Mul<u64> for $classname {
        type Output = $classname;
        #[inline]
        fn mul(mut self, rhs: u64) -> $classname {
            self *= $classname::new_from_u64(rhs);
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

    impl Pow<u64> for $classname {
        type Output = $classname;
        #[inline]
        fn pow(self, rhs: u64) -> $classname {
            $classname::exp_by_squaring($classname::one(), &self, &$classname::new_from_u64(rhs))
        }
    }

    impl Pow<$classname> for $classname {
        type Output = $classname;
        #[inline]
        fn pow(self, rhs: $classname) -> $classname {
            // TODO: use montgomery fast exponentiation
            $classname::exp_by_squaring($classname::one(), &self, &rhs)
        }
    }

    impl Div for $classname {
        type Output = $classname;
        // TODO: not constant time
        fn div(self, rhs: $classname) -> $classname {
            assert!(!rhs.is_zero(), "You cannot divide by zero.");
            // debug_assert!(self < PRIME);
            // debug_assert!(rhs < PRIME);

            // This is algorithm 2.22 from Guide to Elliptic Curve Crypto
            // by Hankerson, Menezes and Vanstone
            // http://math.boisestate.edu/~liljanab/MATH508/GuideEllipticCurveCryptography.PDF

            // INPUT: Prime p and a ∈ [1, p - 1]
            // OUTPUT: a^-1 mod p (actually, returns b/a -- simple inversion is just b=1)
            // 1. u ← a, v ← p
            // 2. x1 ← b, x2 ← 0
            // 3. While (u!=1 and v!=1) do
            //    3.1 While u is even do
            //        u ← u/2
            //        If x1 is even then x1 ← x1/2; else x1 ← (x1 + p)/2
            //    3.2 While v is even do
            //        v ← v/2.
            //        If x2 is even then x2 ← x2/2; else x2 ← (x2 + p)/2
            //    3.3 If u >= v then: u ← u − v, x1 ← x1 − x2;
            //        Else: v ← v − u, x2 ← x2 − x1
            // 4. If u = 1 then return(x1 mod p); else return(x2 mod p)
            let p = SignedDigitsArray::new_pos(PRIME.expand_one());
            let mut u = SignedDigitsArray::new_pos(rhs.limbs.expand_one()); // rhs is denom
            let mut v = SignedDigitsArray::new_pos(PRIME.expand_one());
            let mut x1 = SignedDigitsArray::new_pos(self.limbs.expand_one());
            let mut x2 = SignedDigitsArray::new_pos($classname::zero().limbs.expand_one());

            // let mut i = 0;
            while !u.is_one() && !v.is_one() {
                while u.is_even() {
                    u >>= 1; // u = u / 2
                            // if even, `x1 / 2` else `(x1 + p) / 2`
                    if !x1.is_even() {
                        x1 += p;
                    }
                    x1 >>= 1;
                }
                while v.is_even() {
                    v >>= 1; // v = v / 2
                            // if even, `x2 / 2` else `(x2 + p) / 2`
                    if !x2.is_even() {
                        x2 += p;
                    }
                    x2 >>= 1;
                }
                // if u >= v
                if u >= v {
                    u -= v;
                    // x1 ← x1 − x2
                    x1 -= x2;
                } else {
                    v -= u;
                    // x2 ← x2 − x1
                    x2 -= x1;
                }
            }
            if u.is_one() {
                while x1.is_neg() {
                    x1 += p;
                }
                let extra_limb = x1.limbs[NUMLIMBS];
                $classname { limbs: x1.limbs.contract_one() }.normalize_little(extra_limb)
            } else {
                while x2.is_neg() {
                    x2 += p;
                }
                let extra_limb = x2.limbs[NUMLIMBS];
                ($classname { limbs: x2.limbs.contract_one() }).normalize_little(extra_limb)
            }
        }
    }

    impl Neg for $classname {
        type Output = $classname;
        #[inline]
        fn neg(self) -> $classname {
            $classname { limbs: PRIME.sub_ignore_carry(&self.limbs[..])}.normalize_little(0) // normalize is really just for the self == 0 case
        }
    }

    impl BitAnd<$classname> for $classname {
        type Output = $classname;
        fn bitand(mut self, rhs: $classname) -> $classname {
            self.bitand_assign(rhs);
            self
        }
    }

    impl BitAndAssign<$classname> for $classname {
        fn bitand_assign(&mut self, rhs: $classname) {
            rhs.limbs.iter().zip(self.limbs.iter_mut()).for_each(|(src, dst)| {
                *dst &= *src;
            });
        }
    }

    impl BitAnd<u64> for $classname {
        type Output = $classname;
        fn bitand(mut self, rhs: u64) -> $classname {
            self.bitand_assign(rhs);
            self
        }
    }

    impl BitAndAssign<u64> for $classname {
        fn bitand_assign(&mut self, rhs: u64) {
            self.limbs[0] &= rhs;
            self.limbs.iter_mut().skip(1).for_each(|x| *x = 0);
        }
    }


    from_unsigned! { $classname; u64, u32, u8 }
    from_signed! { $classname; i64, i32, i8 }

    /// Assume element zero is most sig
    // TODO: not constant time
    impl From<[u8; NUMBYTES]> for $classname {
        fn from(src: [u8; NUMBYTES]) -> Self {
            /*
              input: [0 1 2 3 ... NUMBYTES-1]
                     [most sig ->  least sig]

              output: [[NUMBYTES-1 NUMBYTES-2 ... NUMBYTES-7] ... [8 9 10 11 12 13 14 15] [0 1 2 3 4 5 6 7]]
                      [least sig limb                                                         most sig limb]
                       [most sig byte    ->   least sig byte]
            */
            let mut ret = $classname::zero();
            for (i, limb) in ret.limbs.iter_mut().enumerate() {
                for j in (0..LIMBSIZEBYTES).rev() {
                    let idx = i*LIMBSIZEBYTES + j;
                    if idx < NUMBYTES {
                        *limb <<= BITSPERBYTE;
                        *limb |= src[NUMBYTES - idx - 1] as u64;
                    }
                }
            }
            ret.normalize_big(0)
        }
    }

    impl Default for $classname {
        #[inline]
        fn default() -> Self {
            Zero::zero()
        }
    }

    impl Mont {
        pub fn to_norm(self) -> $classname {
            let mut one = [0u64; NUMLIMBS];
            one[0] = 1;
            $classname { limbs: (self * Mont{limbs: one}).limbs }
        }

        #[inline]
        pub fn normalize_assign_little(&mut self, extra_limb: u64) {
            let new_limbs = $classname::normalize_little_limbs(self.limbs, extra_limb);
            self.limbs = new_limbs;
        }

        pub (crate) fn new(limbs:[u64; NUMLIMBS]) -> Mont{
            Mont{limbs}
        }
    }

    impl Mul<Mont> for Mont {
        type Output = Mont;

        #[inline]
        fn mul(self, rhs: Mont) -> Mont {
            // Constant time montgomery mult from https://www.bearssl.org/bigint.html
            let a = self.limbs;
            let b = rhs.limbs;
            let mut d = [0u64; NUMLIMBS]; // result
            let mut dh = [0u64; 2]; // can be up to 2W
            for i in 0 .. NUMLIMBS {
                // f←(d[0]+a[i]b[0])g mod W
                // g is MONTM0INV, W is word size
                let f = {
                    let x = mul_add_3_limbs_array(a[i], b[0], d[0]).mul_by_digit(MONTM0INV);
                    x[0]
                };
                let mut z: [u64; 3]; // can be up to 2W^2
                let mut c = [0u64; 2]; // can be up to 2W
                for j in 0 .. NUMLIMBS {
                    // z ← d[j]+a[i]b[j]+fm[j]+c
                    z = {
                        let z1 = mul_add_3_limbs_array(a[i], b[j], d[j]);
                        let z2 = mul_1_limb_by_1_limb_array(f, PRIME[j]);
                        let (z3, carry1) = z2.add(&c);
                        let (sum, carry2) = z1.add(&z3);
                        let carry = carry1 as u64 + carry2 as u64;
                        // c ← ⌊z/W⌋
                        c = [sum[1], carry];
                        [sum[0], sum[1], carry]
                    };

                    // If j>0, set: d[j−1] ← z mod W
                    if j > 0 {
                        d[j-1] = z[0];
                    }
                }
                // z ← dh+c
                z = {
                    let (sum, carry) = dh.add(&c);
                    [sum[0], sum[1], carry as u64]
                };
                // d[N−1] ← z mod W
                d[NUMLIMBS - 1] = z[0];
                // dh ← ⌊z/W⌋
                dh = [z[1], z[2]];
            }

            // if dh≠0 or d≥m, set: d←d−m
            if dh != [0u64; 2] || d.greater_or_equal(&PRIME) {
                d.sub_assign(&PRIME);
            } else{
                d.sub_assign(&[0u64; NUMLIMBS]);
            }
            Mont { limbs: d }
        }
    }

    impl Mul<$classname> for Mont {
        type Output = $classname;

        #[inline]
        fn mul(self, rhs: $classname) -> $classname {
            $classname::new((self * Mont::new(rhs.limbs)).limbs)
        }
    }

    impl Mul<Mont> for $classname {
        type Output = $classname;

        #[inline]
        fn mul(self, rhs: Mont) -> $classname {
            $classname::new((Mont::new(self.limbs) * rhs).limbs)
        }
    }

    impl Add<Mont> for Mont {
        type Output = Mont;
        #[inline]
        fn add(mut self, rhs: Mont) -> Mont {
            self += rhs;
            self
        }
    }

    impl AddAssign for Mont {
        #[inline]
        fn add_assign(&mut self, other: Mont) {
            let carry = self.limbs.add_assign(&other.limbs);
//TODO this is duplicate.
            let mut r = self.limbs.expand_one();
            r[NUMLIMBS] = carry as u64;

            if r.greater_or_equal(&PRIME.expand_one()) {
                r.sub_assign(&PRIME[..]);
            } else {
                r.sub_assign(&[0u64; NUMLIMBS]);
            }
            self.limbs = r.contract_one();
            debug_assert!(&self.limbs.less(&PRIME));
        }
    }

    impl Sub<Mont> for Mont {
        type Output = Mont;
        #[inline]
        fn sub(mut self, rhs: Mont) -> Mont {
            self -= rhs;
            self
        }
    }

    impl SubAssign for Mont {
        #[inline]
        fn sub_assign(&mut self, other: Mont) {
            let borrow = self.limbs.sub_assign(&other.limbs);
            if borrow {
                self.limbs.add_assign(&PRIME);
            } else {
                // this is here to keep this constant time
                self.limbs.add_assign(&[0u64; NUMLIMBS]);
            }
        }
    }

    impl PartialOrd for Mont {
        #[inline]
        fn partial_cmp(&self, other: &Mont) -> Option<Ordering> {
            DigitsArray::cmp(&self.limbs, &other.limbs)
        }
    }

    impl $classname {
        pub fn to_mont(self) -> Mont {
            Mont{limbs:self.limbs} * Mont{limbs:MONTRSQUARED}
        }

        #[inline]
        pub fn normalize_assign_little(&mut self, extra_limb: u64) {
            let new_limbs = $classname::normalize_little_limbs(self.limbs, extra_limb);
            self.limbs = new_limbs;
        }

        ///Take the extra limb and incorporate that into the existing value by modding by the prime.
        /// This normalize should only be used when the input is at most
        /// 2*p-1. Anything that might be bigger should use the normalize_big
        /// options, which use barrett.
        #[inline]
        pub fn normalize_little_limbs(limbs:[u64; NUMLIMBS] , extra_limb: u64) -> [u64; NUMLIMBS] {
            let mut r = limbs.expand_one();
            r[NUMLIMBS] = extra_limb;

            if r.greater_or_equal(&PRIME.expand_one()) {
                r.sub_assign(&PRIME[..]);
            } else {
                r.sub_assign(&[0u64; NUMLIMBS]);
            }
            r.contract_one()
        }

        ///Take the extra limb and incorporate that into the existing value by modding by the prime.
        #[inline]
        pub fn normalize_little(mut self, extra_limb: u64) -> Self {
            self.normalize_assign_little(extra_limb);
            self
        }

        #[inline]
        pub fn normalize_big(mut self, extra_limb: u64) -> Self {
            self.normalize_assign_big(extra_limb);
            self
        }

        #[inline]
        pub fn normalize_assign_big(&mut self, extra_limb: u64) {
            let mut ret = [0u64; NUMLIMBS*2];
            for (dst, src) in ret.iter_mut().zip(self.iter()) {
                *dst = *src;
            }
            ret[NUMLIMBS] = extra_limb;
            self.limbs = $classname::reduce_barrett(&ret);
        }

        // TODO: not constant time
        ///Convert the value to a byte array which is `NUMBYTES` long.
        pub fn to_bytes_array(&self) -> [u8; NUMBYTES] {
            let mut ret = [0u8; NUMBYTES];

            for (i, limb) in self.limbs.iter().enumerate() {
                for j in 0..LIMBSIZEBYTES {
                    let idx = i*LIMBSIZEBYTES + j;
                    if idx < NUMBYTES {
                        ret[NUMBYTES - idx - 1] = (limb >> j*BITSPERBYTE) as u8;
                    }
                }
            }
            ret
        }

        #[inline]
        // TODO: not constant time
        pub fn exp_by_squaring(y: $classname, x: &$classname, n: &$classname) -> $classname {
            if n.is_zero() {
                y
            } else if n.is_one() {
                x.mul(y)
            } else if n.is_even() {
                $classname::exp_by_squaring(y, &x.square(), &n.div2())
            } else {
                $classname::exp_by_squaring(x.mul(y), &x.square(), &(*n - $classname::one()).div2())
            }
        }

        ///Divide the value by 2.
        #[inline]
        pub fn div2(&self) -> $classname {
            $classname {
                limbs: self.limbs.shift_right_bits(1),
            }
        }

        ///Square the value. Same as a value times itself, but slightly more performant.
        #[inline]
        pub fn square(&self) -> $classname {
            let doublesize = $classname::mul_limbs_classic(&self.limbs, &self.limbs);
            $classname {
                limbs: $classname::reduce_barrett(&doublesize),
            }
        }

        ///Check to see if the value is even.
        #[inline]
        pub fn is_even(&self) -> bool {
            self.limbs.is_even()
        }

        ///Create a new instance given the raw limbs form. Note that this is least significant bit first.
        #[allow(dead_code)]
        pub fn new(digits: [u64; NUMLIMBS]) -> $classname {
            $classname {
                limbs: digits
            }
        }

        ///Convenience function to create a value from a single limb.
        pub fn new_from_u64(x: u64) -> $classname {
            let mut ret = $classname::zero();
            ret.limbs[0] = x;
            ret
        }

        // TODO: not constant time
        ///Write out the value in decimal form.
        pub fn to_str_decimal(mut self) -> String {
            // largest 10-base digit in a u64 is 10^19. For i64, 10^18. We've precalculated this for speed.
            const MAX_BASE_10: u64 = 1000000000000000000; //10^18
            let mut retstr = String::with_capacity((PRIMEBITS / BITSPERBYTE) * 3); // three chars for every byte
            let mut ret: Vec<String> = vec![];

            while !self.limbs.iter().all(|limb| limb == &0u64) {
                let (tmp_new, rem_new) = self.limbs.div_rem_1(MAX_BASE_10);
                let decimal = format!("{:018}", rem_new);
                ret.push(decimal);
                self.limbs = tmp_new;
            }
            // strip leading zeros of the most significant digit
            if let Some(last) = ret.last_mut() {
                *last = last.as_str().trim_left_matches('0').to_string();
            }
            ret.iter().rev().for_each(|s| retstr.push_str(s));
            retstr
        }

        ///Write out the value in hex.
        #[allow(dead_code)]
        // TODO: not sure if this is constant time
        pub fn to_str_hex(&self) -> String {
            let mut ret = String::with_capacity((PRIMEBITS / BITSPERBYTE) * 2); // two chars for every byte
            self.to_bytes_array().iter().for_each(|byte| ret.push_str(&format!("{:02x}", byte)));
            ret
        }

        // TODO: build unit tests for this
        // TODO: not constant time
        /// Create a Non-Adjacent form of the value.
        /// return - Vector which represents the NAF of value.
        pub fn create_naf(&self) -> Vec<i8> {
            // TODO: build unit tests for this
            // Mutable collection which has all 0s in it.
            let mut naf = vec![0; PRIMEBITS + 1];
            let mut i = 0;
            let mut n = SignedDigitsArray::new_pos(self.limbs);
            let zero = SignedDigitsArray::new_pos([0u64; NUMLIMBS]);

            while n > zero {
                if !n.is_even() {
                    let diff = (n % 4) as i8;
                    naf[i] = 2i8 - diff;
                    n -= naf[i] as i64;
                } else {
                    n >>= 1;
                    i += 1;
                }
            }
            naf
        }

        // From Handbook of Applied Crypto algo 14.12
        #[inline]
        fn mul_limbs_classic(a: &[u64; NUMLIMBS], b: &[u64; NUMLIMBS]) -> [u64; NUMDOUBLELIMBS] {
            let mut res = [0u64; NUMDOUBLELIMBS];
            for i in 0..NUMLIMBS {
                let mut c = 0;
                for j in 0..NUMLIMBS {
                    let (mut u, mut v) = mul_1_limb_by_1_limb(a[j], b[i]);
                    v = add_accum_1by1(v, c, &mut u);
                    v = add_accum_1by1(v, res[i + j], &mut u);
                    res[i + j] = v;
                    c = u;
                }
                res[i + NUMLIMBS] = c;
            }
            res
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
    pub fn reduce_barrett(a: &[u64; NUMDOUBLELIMBS]) -> [u64; NUMLIMBS] {
        // In this case, k = NUMLIMBS
        // let mut q1 = [0u64; NUMLIMBS];
        // q1.copy_from_slice(&a[NUMLIMBS - 1..NUMDOUBLELIMBS-1]);
        let q1 = a.shift_right_digits(NUMLIMBS - 1);

        // q2 = q1 * mu
        // let q2 = BARRETTMU.mul_classic(&q1);
        let q2 = q1.mul_classic(&BARRETTMU[..]);

        let mut q3 = [0u64; NUMLIMBS];
        q3.copy_from_slice(&q2[NUMLIMBS + 1..NUMDOUBLELIMBS + 1]);

        let mut r1 = [0u64; NUMLIMBS + 2];
        r1.copy_from_slice(&a[..NUMLIMBS+2]);

        let r2 = &q3.mul_classic(&PRIME)[..NUMLIMBS + 1];

        // r = r1 - r2
        let (r3, _) = r1.expand_one().sub(&r2);
        let mut r = [0u64; NUMLIMBS]; // need to chop off extra limb
        r.copy_from_slice(&r3[..NUMLIMBS]);

        // at most two subtractions with p
        for _i in 0..2 {
            if DigitsArray::cmp(&r, &PRIME) != Some(Ordering::Less) {
                r.sub_assign(&PRIME);
            } else {
                // this branch is for constant time
                r.sub_assign(&[0u64; NUMLIMBS]);
            }
        }
        debug_assert!(DigitsArray::cmp(&r, &PRIME) == Some(Ordering::Less));
        r
    }}


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
            fn arb_fp()(seed in any::<u64>()) -> $classname {
                if seed == 0 {
                    $classname::zero()
                } else if seed == 1 {
                    $classname::one()
                } else {
                    let mut rng = OsRng::new().expect("Failed to get random number");
                    let mut limbs = [0u64; NUMLIMBS];
                    for limb in limbs.iter_mut() {
                        *limb = rng.next_u64();
                    }
                    limbs[NUMLIMBS - 1] &= (1u64 << (PRIMEBITS % 64)) - 1;
                    $classname {
                        limbs: limbs
                    }.normalize_little(0)
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
            fn bitand_agrees_is_even(a in arb_fp()) {
                let bit_and_result = a & 1;
                prop_assert_eq!(!a.is_even(), bit_and_result == $classname::one());
                prop_assert_eq!(a.is_even(), bit_and_result == $classname::zero());
            }

            #[test]
            fn bitand_lifted_into_class_is_same(a in arb_fp()) {
                prop_assert_eq!(a & $classname::from(1000), a & 1000);
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
                prop_assert_eq!(a - b, a + -b);
                prop_assert_eq!(-(a * b), -a * b);
                prop_assert_eq!(-a * b, a * -b);
                prop_assert_eq!(a + -a, $classname::zero());
            }

            #[test]
            fn from_bytes_roundtrip(a in arb_fp()) {
                let bytes = a.to_bytes_array();
                prop_assert_eq!($classname::from(bytes), a);
            }

            #[test]
            fn square_same_as_pow_2(a in arb_fp()) {
                prop_assert_eq!(a.square(), a.pow(2));
            }

            #[test]
            fn from_signed_ints(a in any::<i64>()) {
                if a < 0 {
                    prop_assert_eq!($classname::from(a), $classname { limbs: PRIME } - $classname::new_from_u64(a.abs() as u64));
                } else {
                    prop_assert_eq!($classname::from(a), $classname::new_from_u64(a as u64));
                }
            }

            #[test]
            fn to_mont_roundtrip_to_norm(a in arb_fp()) {
                prop_assert_eq!(a, a.to_mont().to_norm());
            }

            #[test]
            fn mont_mult_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, (a.to_mont() * b.to_mont()).to_norm());
            }

            #[test]
            fn mont_times_normal_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, a.to_mont()* b);
            }

            #[test]
            fn normal_times_mont_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, a* b.to_mont());
            }

            #[test]
            fn mont_add_works(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a + b, (a.to_mont() + b.to_mont()).to_norm())
            }

            #[test]
            fn mont_sub_works(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a - b, (a.to_mont() - b.to_mont()).to_norm())
            }

        }
    }
}};}
