use std::ops::Mul;
use digits::constant_time_primitives::*;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use digits::util::*;
#[derive(Debug, Copy, Clone)]
pub struct Fp256_32 {
    pub(crate) limbs: [u32; 9],
}

impl Fp256_32 {
    const PRIME: [u32; 9] = [
        1577621095, 817453272, 47634040, 1927038601, 407749150, 1308464908, 685899370, 1518399909,
        143,
    ];

    #[inline]
    fn add_limbs(a: &mut [u32; 9], b: [u32; 9], ctl: u32) -> u32 {
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw + bw + cc;
            cc = naw >> 31;
            *aa = ctl.mux(naw & 0x7FFFFFFF, aw)
        }
        cc
    }

    #[inline]
    fn sub_limbs(a: &mut [u32; 9], b: [u32; 9], ctl: u32) -> u32 {
        let mut cc = 0u32;
        for (mut aa, bb) in a.iter_mut().zip(b.iter()) {
            let aw = *aa;
            let bw = *bb;
            let naw = aw - bw - cc;
            cc = naw >> 31;
            *aa = ctl.mux(naw & 0x7FFFFFFF, aw)
        }
        cc
    }
}

impl Add<Fp256_32> for Fp256_32 {
    type Output = Fp256_32;
    #[inline]
    fn add(mut self, other: Fp256_32) -> Fp256_32 {
        self += other;
        self
    }
}

impl AddAssign<Fp256_32> for Fp256_32 {
    #[inline]
    fn add_assign(&mut self, other: Fp256_32) {
        let mut a = self.limbs;
        let mut ctl = Fp256_32::add_limbs(&mut a, other.limbs, 1);
        ctl |= Fp256_32::sub_limbs(&mut a, Fp256_32::PRIME, 0).not();
        Fp256_32::sub_limbs(&mut a, Fp256_32::PRIME, ctl);
    }
}

impl Sub<Fp256_32> for Fp256_32 {
    type Output = Fp256_32;
    #[inline]
    fn sub(mut self, other: Fp256_32) -> Fp256_32 {
        self += other;
        self
    }
}

impl SubAssign<Fp256_32> for Fp256_32 {
    #[inline]
    fn sub_assign(&mut self, other: Fp256_32) {
        let mut a = self.limbs;
        let needs_add = Fp256_32::sub_limbs(&mut a, other.limbs, 1);
        Fp256_32::add_limbs(&mut a, Fp256_32::PRIME, needs_add);
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Mont {
    pub(crate) limbs: [u64; 9],
}

impl Mul<Mont> for Mont {
    type Output = Mont;

    #[inline]
    fn mul(self, rhs: Mont) -> Mont {
        let MONTM0INV: u32 = 0u32;
        let a = self.limbs;
        let b = rhs.limbs;
        let mut d = [0u32; 9]; // result
        let mut dh = 0u64; // can be up to 2W
        for i in 0 .. 9 {
            // f←(d[0]+a[i]b[0])g mod W
            // g is MONTM0INV, W is word size
            // This might not be right, and certainly isn't optimal. Ideally we'd only calculate the low 31 bits
            let f: u32 = (a[i] as u64 * b[0] as u64 + d[0] as u64 * MONTM0INV as u64) as u32;
            let mut z: u64; // can be up to 2W^2
            let mut r = 0u64; // can be up to 2W
            for j in 0 .. 9 {
                // z ← d[j]+a[i]b[j]+fm[j]+c
                let xx: u64 = f as u64 * Fp256_32::PRIME[j] as u64;
                z = a[i] * b[j] + (d[j] as u64) + xx + (r as u64);
                         // z = (uint64_t)d[v + 1] + MUL31(xu, y[v + 1])
                         //         + MUL31(f, m[v + 1]) + r;
                         // r = z >> 31;
                         // d[v + 0] = (uint32_t)z & 0x7FFFFFFF;

                // If j>0, set: d[j−1] ← z mod W
                if j > 0 {
                    d[j-1] = (z as u32) & 0x7FFFFFFF;
                }
            }
            // z ← dh+c
            let zh = dh.add(&r);
            // d[N−1] ← z mod W
            d[9 - 1] = (zh as u32) & 0x7FFFFFFF;
            // dh ← ⌊z/W⌋
            dh = zh >> 31
        }

        // if dh≠0 or d≥m, set: d←d−m
        //  br_i31_sub(d, m, NEQ(dh, 0) | NOT(br_i31_sub(d, m, 0)));
        // if dh != 0u64 || d.greater_or_equal(&Fp256_32::PRIME) {
        //     d.sub_assign(&Fp256_32::PRIME);
        // } else{
        //     d.sub_assign(&[0u64; 9]);
        // }
        Mont { limbs: d }
    }
}
