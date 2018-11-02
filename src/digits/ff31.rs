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

/// Create an Fp type given the following parameters:
/// - modname - the name of the module you want the Fp type in.
/// - classname - the name of the Fp struct
/// - bits - How many bits the prime is.
/// - limbs - Number of limbs (ceil(bits/64))
/// - prime - prime number in limbs, least significant digit first. (Note you can get this from `sage` using `num.digits(2 ^ 64)`).
/// - barrett - barrett reduction for reducing values up to twice the number of prime bits (double limbs). This is `floor(2^(64*numlimbs*2)/prime)`.
#[macro_export]
macro_rules! fp31 { ($modname: ident, $classname: ident, $bits: tt, $limbs: tt, $prime: expr, $barrettmu: expr, $montgomery_r_inv: expr, $montgomery_r_squared: expr, $montgomery_m0_inv: expr) => { pub mod $modname {
    // use $crate::digits::util::*;
    // use $crate::digits::signed::*;
    // use $crate::digits::unsigned::*;
    use $crate::digits::constant_time_primitives::*;
    use $crate::digits::constant_bool::*;
    use std::cmp::Ordering;
    use std::fmt;
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Not, Neg, Sub, SubAssign, BitAnd, BitAndAssign};
    use num_traits::{One, Zero, Inv, Pow};
    use std::convert::From;
    use std::option::Option;
    use std::marker::{Copy, Send, Sync, Sized, self};

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

    pub struct FpBitIter<'a, $classname: 'a> {
        p: *const $classname,
        index: usize,
        endindex: usize,
        _marker: marker::PhantomData<&'a $classname>
    }

    impl<'a> Iterator for FpBitIter<'a, $classname> {
        type Item=ConstantBool<u32>;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.index += 1;
            let limbs = unsafe {
                (*self.p).limbs
            };
            if self.index <= self.endindex {
                Some($classname::test_bit(&limbs, self.index - 1))
            } else {
                None
            }
        }
    }

    impl<'a> DoubleEndedIterator for FpBitIter<'a, $classname> {
        #[inline]
        fn next_back(&mut self) -> Option<ConstantBool<u32>> {
            self.endindex -= 1;
            let limbs = unsafe {
                (*self.p).limbs
            };
            if self.index <= self.endindex {
                Some($classname::test_bit(&limbs, self.endindex))
            } else {
                None
            }
        }
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
            let mut res = 0i64;
            self.limbs.iter().zip(other.limbs.iter()).rev().for_each(|(l, r)| {
                let limbcmp = (l.const_gt(*r).0 as i64) | -(r.const_gt(*l).0 as i64);
                res = res.abs().mux(res, limbcmp);
            });
            match res {
                -1 => Some(Ordering::Less),
                0 => Some(Ordering::Equal),
                1 => Some(Ordering::Greater),
                _ => None
            }
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
            self.limbs.const_eq0().0 == 1
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
            self.limbs.const_eq(Self::one().limbs).0 == 1
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
            let mut ctl = $classname::add_assign_limbs_if(a, other.limbs, ConstantBool::new_true());
            ctl |= a.const_ge(PRIME);
            $classname::sub_assign_limbs_if(a, PRIME, ctl);
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
            let needs_add = $classname::sub_assign_limbs_if(a, other.limbs, ConstantBool(1));
            $classname::add_assign_limbs_if(a, PRIME, needs_add);
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
            self.pow($classname::from(rhs))
        }
    }

    impl Pow<$classname> for $classname {
        type Output = $classname;
        /// 14.94 Algorithm Montgomery exponentiation in Handbook of Applied Crypto
        /// INPUT:m=(ml−1···m0)b,R=bl,m′ =−m−1 modb,e=(et···e0)2 withet =1, and an integer x, 1 ≤ x < m.
        /// OUTPUT: xe mod m.
        /// 1. x􏰁← Mont(x,R2 mod m), A←R mod m. (R mod m and R2 mod m may be pro-ided as inputs.)
        /// 2. For i from t down to 0 do the following: 2.1 A←Mont(A,A).
        /// 2.2 If ei = 1 then A← Mont(A, x􏰁).
        /// 3. A←Mont(A,1).
        /// 4. Return(A).
        #[inline]
        fn pow(self, rhs: $classname) -> $classname {
            let mut t1 = self.to_monty();
            let mut x = Monty::one();
            let mut t2: Monty; // = Monty::zero();
            // count up to bitlength of exponent
            // not certain but just going to use prime bits
            // for k in 0 .. PRIMEBITS {
            for bit in rhs.iter_bit() {
                t2 = x * t1;
                x.limbs.const_copy_if(&t2.limbs, bit); // copy if bit is set
                t2 = t1 * t1;
                t1 = t2;
            }
            $classname{limbs: x.limbs}
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
            $classname::cond_negate_mod_prime(&mut self.limbs, ConstantBool::new_true());
            self
        }
    }

    from_unsigned31! { $classname; u32, u8 }

    /// Assume element zero is most sig
    impl From<[u8; PRIMEBYTES]> for $classname {
        fn from(src: [u8; PRIMEBYTES]) -> Self {
            let limbs = $classname::convert_bytes_to_limbs(src, PRIMEBYTES);
            $classname::normalize_little_limbs(limbs);
            $classname::new(limbs)
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
            let mut dh = 0u64; // can be up to 2W
            for i in 0 .. NUMLIMBS {
                // f←(d[0]+a[i]b[0])g mod W
                // g is MONTM0INV, W is word size
                // This might not be right, and certainly isn't optimal. Ideally we'd only calculate the low 31 bits
                // MUL31_lo((d[1] + MUL31_lo(x[u + 1], y[1])), m0i);
                let f: u32 = $classname::mul_31_lo(d[0] + $classname::mul_31_lo(a[i], b[0]), MONTM0INV);
                // println!("i={:?} f=({:?}+{:?}*{:?})*{:?}={:?}", i, d[0], a[i], b[0], MONTM0INV, f);
                let mut z: u64; // can be up to 2W^2
                let mut c: u64; // can be up to 2W
                let ai = a[i];

                z = (ai as u64 * b[0] as u64) + (d[0] as u64) + (f as u64 * PRIME[0] as u64);
                c = z >> 31;
                for j in 1 .. NUMLIMBS {
                    // z ← d[j]+a[i]b[j]+fm[j]+c
                    z = (ai as u64 * b[j] as u64) + (d[j] as u64) + (f as u64 * PRIME[j] as u64) + c;
                    // c ← ⌊z/W⌋
                    c = z >> 31;
                    // If j>0, set: d[j−1] ← z mod W
                    d[j-1] = (z & 0x7FFFFFFF) as u32;
                }
                // z ← dh+c
                z = dh + c;
                // d[N−1] ← z mod W
                d[NUMLIMBS - 1] = (z & 0x7FFFFFFF) as u32;
                // dh ← ⌊z/W⌋
                dh = z >> 31;
            }

            // if dh≠0 or d≥m, set: d←d−m
            // println!("answer before normalizing: {:?}", d);
            let dosub = ConstantBool(dh.const_neq(0).0 as u32) | d.const_gt(PRIME);
            $classname::sub_assign_limbs_if(&mut d, PRIME, dosub);
            // println!("answer aftter normalizing: {:?}", d);
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
            let a = &mut self.limbs;
            let mut ctl = $classname::add_assign_limbs_if(a, other.limbs, ConstantBool::new_true());
            ctl |= a.const_ge(PRIME);
            $classname::sub_assign_limbs_if(a, PRIME, ctl);
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
            let a = &mut self.limbs;
            let needs_add = $classname::sub_assign_limbs_if(a, other.limbs, ConstantBool(1));
            $classname::add_assign_limbs_if(a, PRIME, needs_add);
        }
    }

    impl PartialOrd for Monty {
        #[inline]
        fn partial_cmp(&self, other: &Monty) -> Option<Ordering> {
            unimplemented!();
        }
    }

    impl Zero for Monty {
        #[inline]
        fn zero() -> Self {
            Monty {
                limbs: [0u32; NUMLIMBS],
            }
        }

        #[inline]
        fn is_zero(&self) -> bool {
            self.limbs.const_eq0().0 == 1
        }
    }

    impl One for Monty {
        #[inline]
        fn one() -> Self {
            let mut ret = Monty { limbs: [0u32; NUMLIMBS] };
            ret.limbs[0] = 1u32;
            ret
        }

        #[inline]
        fn is_one(&self) -> bool {
            self.limbs.const_eq(Self::one().limbs).0 == 1
        }
    }

    impl $classname {

        ///Square the value. Same as a value times itself, but slightly more performant.
        #[inline]
        pub fn square(&self) -> $classname {
            let doublesize = $classname::mul_limbs_classic(&self.limbs, &self.limbs);
            $classname {
                limbs: $classname::reduce_barrett(&doublesize),
            }
        }
        #[inline]
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
            let needs_sub = limbs.const_gt(PRIME);
            $classname::sub_assign_limbs_if(&mut limbs, PRIME, needs_sub);
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
            let mut ret = [0u32; NUMLIMBS*2];
            for (dst, src) in ret.iter_mut().zip(self.limbs.iter()) {
                *dst = *src;
            }
            ret[NUMLIMBS] = extra_limb;
            self.limbs = $classname::reduce_barrett(&ret);
        }

        #[inline]
        fn u32_to_bytes_big_endian(x: u32, buf: &mut[u8]){
            buf[0] = (x >> 24) as u8;
            buf[1] = (x >> 16) as u8;
            buf[2] = (x >> 8) as u8;
            buf[3] = x as u8;
        }

        ///Convert the value to a byte array which is `PRIMEBYTES` long.
        #[inline]
        pub fn to_bytes_array(&self) -> [u8; PRIMEBYTES] {
            let mut k:usize = 0;
            let mut acc = 0u32;
            // How many bits have we accumulated.
            let mut acc_len = 0u32;
            // How many bytes are left.
            let mut len = PRIMEBYTES;
            let mut output: [u8;PRIMEBYTES] = [0u8; PRIMEBYTES];
            let mut current_output_index = len;
            while len != 0 {
                let current_limb = self.limbs[k];
                k += 1;
                if acc_len == 0{
                    acc = current_limb;
                    acc_len = 31;
                } else{
                    let z = acc | (current_limb << acc_len);
                    acc_len -= 1;
                    acc = current_limb >> (31 - acc_len);
                    if len >= 4 { //Pull off 4 bytes and put them into the output buffer.
                        current_output_index -= 4;
                        len -= 4;
                        $classname::u32_to_bytes_big_endian(z, &mut output[current_output_index..(current_output_index + 4)])
                    } else { //If we have less than 4 bytes left, manually pull off all 3 in succession.
                        if len == 3 {
                            output[current_output_index - len] = (z >> 16) as u8;
                            len -= 1;
                        }

                        if len == 2 {
                            output[current_output_index - len] = (z >> 8) as u8;
                            len -= 1;
                        }

                        if len == 1 {
                            output[current_output_index - len] = z as u8;
                        }
                        break;
                    }
                }
            }
            output
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
            let mut res = [0u32; NUMDOUBLELIMBS];
            for i in 0..NUMLIMBS {
                let mut c = 0u32;
                for j in 0..NUMLIMBS {
                    // Compute (uv)b = wi+j + xj · yi + c, and set wi+j ←v, c←u
                    let (u, v) = Self::split_u64_to_31b(Self::mul_add(a[j], b[i], res[i+j]) + c as u64);
                    res[i + j] = v;
                    c = u;
                }
                res[i + NUMLIMBS] = c;
            }
            res
        }

        fn test_bit(a: &[u32; NUMLIMBS], idx: usize) -> ConstantBool<u32> {
            let limb_idx = idx / LIMBSIZEBITS;
            let limb_bit_idx = idx - limb_idx * LIMBSIZEBITS;
            ConstantBool((a[limb_idx] >> limb_bit_idx) & 1)
        }

        fn as_ptr(&self) -> *const $classname {
            self as *const $classname
        }

        fn iter_bit(&self) -> FpBitIter<$classname> {
            FpBitIter {
                p: self.as_ptr(),
                index: 0,
                endindex: PRIMEBITS,
                _marker: marker::PhantomData
            }

        }

        #[inline]
        fn mul_add(a: u32, b: u32, c: u32) -> u64 {
            a as u64 * b as u64 + c as u64
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
        pub fn split_u64_to_32b(i: u64) -> (u32, u32) {
            (Self::high_32(i), Self::low_32(i))
        }

        /// Returns array with least sig in pos 0 and carry in pos 2
        #[inline]
        pub fn split_u64_to_31b_array(i: u64) -> [u32; 3] {
            let mut res = [032; 3];
            res[0] = (i & 0x7FFFFFFF) as u32;
            res[1] = ((i >> 31) & 0x7FFFFFFF) as u32;
            res[2] = (i >> 62) as u32;
            res
        }

        /// Returns (high, low) where high uses extra bit for carry
        /// and low has a cleared 32nd bit
        #[inline]
        pub fn split_u64_to_31b(i: u64) -> (u32, u32) {
            ((i >> 31) as u32, (i & 0x7FFFFFFF) as u32)
        }

    // From Handbook of Applied Cryptography 14.42
        // INPUT: positive integers x = (x2k−1 · · · x1x0)b, m = (mk−1 · · · m1m0)b (with mk−1 ̸= 0), and μ = ⌊b2k/m⌋.
        // OUTPUT: r = x mod m.
        // 1. q1←⌊x/bk−1⌋, q2←q1 · μ, q3←⌊q2/bk+1⌋.
        // 2. r1←x mod bk+1, r2←q3 · m mod bk+1, r←r1 − r2. 3. Ifr<0thenr←r+bk+1.
        // 4. While r≥m do:r←r−m.
        // 5. Return(r).
    // Also helpful: https://www.everything2.com/title/Barrett+Reduction
    #[inline]
    pub fn reduce_barrett(a: &[u32; NUMDOUBLELIMBS]) -> [u32; NUMLIMBS] {
        // q1←⌊x/bk−1⌋
        let mut q1= [0u32; NUMLIMBS + 1];
        q1.copy_from_slice(&a[NUMLIMBS - 1..]);

        // q2←q1 · μ
        // q1 * BARRETTMU
        // BARRETTMU is NUMLIMBS + 1
        let mut q2 = [0u32; 2*NUMLIMBS + 2];
        for i in 0..NUMLIMBS+1 {
            let mut c = 0u32;
            for j in 0..NUMLIMBS+1 {
                // Compute (uv)b = wi+j + xj · yi + c, and set wi+j ←v, c←u
                let (u, v) = Self::split_u64_to_31b(Self::mul_add(q1[j], BARRETTMU[i], q2[i+j]) + c as u64);
                q2[i + j] = v;
                c = u;
            }
            q2[i + NUMLIMBS + 1] = c;
        }

        // q3←⌊q2/bk+1⌋
        let mut q3= [0u32; NUMLIMBS];
        q3.copy_from_slice(&q2[NUMLIMBS + 1..NUMDOUBLELIMBS + 1]);

        // r1←x mod bk+1
        let mut r1 = [0u32; NUMLIMBS + 2];
        r1.copy_from_slice(&a[..NUMLIMBS+2]);

        // r2←q3 · m mod bk+1
        // let r2 = &q3.mul_classic(&PRIME)[..NUMLIMBS + 1];
        let mut r2 = [0u32; NUMLIMBS * 2];
        for i in 0..NUMLIMBS {
            let mut c = 0u32;
            for j in 0..NUMLIMBS {
                // Compute (uv)b = wi+j + xj · yi + c, and set wi+j ←v, c←u
                let (u, v) = Self::split_u64_to_31b(Self::mul_add(q3[j], PRIME[i], r2[i+j]) + c as u64);
                r2[i + j] = v;
                c = u;
            }
            r2[i + NUMLIMBS] = c;
        }


        // r←r1 − r2
        // r1 = r1 - r2
        let mut r= [0u32; NUMLIMBS];
        Self::sub_assign_limbs_slice(&mut r1[..NUMLIMBS], &r2[..NUMLIMBS]);
        r.copy_from_slice(&r1[..NUMLIMBS]);

        // If r<0 then r←r+bk+1
        // at most two subtractions with p
        let dosub = r.const_gt(PRIME);
        Self::sub_assign_limbs_if(&mut r, PRIME, dosub);

        r
    }

    ///Convert the src into the limbs. This _does not_ mod off the value. This will take the first
    ///len bytes and split them into 31 bit limbs.
    #[inline]
    fn convert_bytes_to_limbs(src: [u8; PRIMEBYTES], len: usize) -> [u32; NUMLIMBS]{
        let mut limbs = [0u32; NUMLIMBS];
        let mut acc = 0u32;
        let mut acc_len = 0i32;
        let mut v = 0;
        for b in src.iter().rev().take(len){
            let b_u32 = *b as u32;
            acc |= b_u32 << acc_len;
            acc_len += 8;
            if acc_len >= 31 {
                limbs[v] = acc & 0x7FFFFFFFu32;
                v += 1;
                acc_len -= 31;
                //Note that because we're adding 8 each time through the loop
                //and check that acc_len >= 31 that 8 - acc_len can _never_ be negative.
                acc = b_u32 >> (8- acc_len);
            }
        }
        if acc_len != 0{
            limbs[v] = acc;
            // v += 1
        }
        limbs
    }

    #[inline]
    fn add_assign_limbs_if(a: &mut [u32; NUMLIMBS], b: [u32; NUMLIMBS], ctl: ConstantBool<u32>) -> ConstantBool<u32> {
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw.wrapping_add(bw).wrapping_add(cc);
            cc = naw >> 31;
            *aa = ctl.mux(naw & 0x7FFFFFFF, aw)
        }
        ConstantBool(cc)
    }

    #[inline]
    fn sub_assign_limbs_if(a: &mut [u32; NUMLIMBS], b: [u32; NUMLIMBS], ctl: ConstantBool<u32>) -> ConstantBool<u32> {
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw.wrapping_sub(bw).wrapping_sub(cc);
            cc = naw >> 31;
            *aa = ctl.mux(naw & 0x7FFFFFFF, aw);
        }
        ConstantBool(cc)
    }

    #[inline]
    fn sub_assign_limbs_slice(a: &mut [u32], b: &[u32]) -> ConstantBool<u32> {
        debug_assert!(a.len()==b.len());
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw.wrapping_sub(bw).wrapping_sub(cc);
            cc = naw >> 31;
            *aa = naw & 0x7FFFFFFF;
        }
        ConstantBool(cc)
    }

    #[inline]
    fn mul_31_lo(x:u32, y:u32) -> u32 {
        (x as u64 * y as u64) as u32 & 0x7FFFFFFFu32
    }

    fn cond_negate_mod_prime(a: &mut [u32; NUMLIMBS], ctl: ConstantBool<u32>) {
        let mut foo = PRIME;
        $classname::sub_assign_limbs_if(&mut foo, *a, ctl);
        *a = $classname::normalize_little_limbs(foo);
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
                        *limb = rng.next_u32() & 0x7FFFFFFFu32;
                    }
                    limbs[NUMLIMBS - 1] &= (1u32 << (PRIMEBITS % 31)) - 1;
                    $classname {
                        limbs: limbs
                    }.normalize_little()
                }
            }
        }

        proptest! {
            #[test]
            fn identity(a in arb_fp()) {
                // prop_assert_eq!(a * 1, a);

                prop_assert_eq!(a * $classname::one(), a);
                prop_assert_eq!($classname::one() * a, a);

                prop_assert_eq!(a + $classname::zero(), a);
                prop_assert_eq!($classname::zero() + a, a);

                prop_assert_eq!(a - $classname::zero(), a);
                prop_assert_eq!($classname::zero() - a, -a);

                // prop_assert_eq!(a / a, $classname::one());
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
            #[ignore]
            fn mul_equals_div(a in arb_fp(), b in arb_fp()) {
                prop_assume!(!a.is_zero() && !b.is_zero());
                let c = a * b;
                prop_assert_eq!(c / a, b);
                prop_assert_eq!(c / b, a);
            }

            #[test]
            #[ignore]
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
            #[ignore]
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
            fn to_monty_roundtrip_to_norm(a in arb_fp()) {
                prop_assert_eq!(a, a.to_monty().to_norm());
            }

            #[test]
            fn monty_mult_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, (a.to_monty() * b.to_monty()).to_norm());
            }

            #[test]
            fn monty_times_normal_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, a.to_monty()* b);
            }

            #[test]
            fn normal_times_monty_equals_normal(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a * b, a * b.to_monty());
            }

            #[test]
            fn monty_add_works(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a + b, (a.to_monty() + b.to_monty()).to_norm())
            }

            #[test]
            fn monty_sub_works(a in arb_fp(), b in arb_fp()) {
                prop_assert_eq!(a - b, (a.to_monty() - b.to_monty()).to_norm())
            }

            #[test]
            fn monty_add_assign_works(a in arb_fp(), b in arb_fp()) {
                let mut aa = a.to_monty();
                aa += b.to_monty();
                prop_assert_eq!(a + b, aa.to_norm())
            }

            #[test]
            fn monty_sub_assign_works(a in arb_fp(), b in arb_fp()) {
                let mut aa = a.to_monty();
                aa -= b.to_monty();
                prop_assert_eq!(a - b, aa.to_norm())
            }
        }
    }
}};}
