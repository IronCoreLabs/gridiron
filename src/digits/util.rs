///Convert the src into the limbs. This _does not_ mod off the value. This will take the first
///len bytes and split them into 31 bit limbs.
///Note that this will _not_ check anything about the length of limbs and could be unsafe... BE CAREFUL!
///For more safe versions of this, check the convert_bytes_to_limbs in each $classname.
#[inline]
pub fn convert_bytes_to_limbs_mut(src: &[u8], limbs: &mut [u32], len: usize) {
    let mut acc = 0u32;
    let mut acc_len = 0i32;
    let mut v = 0;
    for b in src.iter().rev().take(len) {
        let b_u32 = *b as u32;
        acc |= b_u32 << acc_len;
        acc_len += 8;
        if acc_len >= 31 {
            limbs[v] = acc & 0x7FFFFFFFu32;
            v += 1;
            acc_len -= 31;
            //Note that because we're adding 8 each time through the loop
            //and check that acc_len >= 31 that 8 - acc_len can _never_ be negative.
            acc = b_u32 >> (8 - acc_len);
        }
    }
    if acc_len != 0 {
        limbs[v] = acc;
        // v += 1
    }
}

///This function assumes that the buf pointer has at least 4 spaces starting at the beginning of the
///slice. You need to assure this before calling.
#[inline]
pub fn u32_to_bytes_big_endian(x: u32, buf: &mut [u8]) {
    buf[0] = (x >> 24) as u8;
    buf[1] = (x >> 16) as u8;
    buf[2] = (x >> 8) as u8;
    buf[3] = x as u8;
}
