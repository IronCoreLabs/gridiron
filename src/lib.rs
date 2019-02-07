use crate::digits::util::unsafe_convert_bytes_to_limbs_mut;
extern crate arrayref;
extern crate num_traits;
#[cfg(test)]
extern crate rand;
#[cfg(test)]
#[macro_use]
extern crate proptest;

#[macro_use]
pub mod digits {
    #[macro_use]
    pub mod ff31;
    pub mod constant_bool;
    pub mod constant_time_primitives;
    pub mod util;
}

// p = 3121577065842246806003085452055281276803074876175537384188619957989004527066410274868798956582915008874704066849018213144375771284425395508176023
//   =
fp31!(
    fp_480, // Name of mod
    Fp480,  // Name of class
    480,    // Number of bits for prime
    16,     // Number of limbs (ceil(bits/31))
    [
        // prime number in limbs, least sig first
        // get this from sage with p.digits(2^31)
        1055483031, 1386897616, 898494285, 1391857335, 488544832, 1799384686, 193115992, 565079768,
        190358044, 1260077487, 1583277252, 222489098, 760385720, 330553579, 429458313, 32766
    ],
    //2^(31*(2*16-1)) mod p
    //1260953731944968926163185575789985373882767326957187433125548064552888900134320111509075687974556690810534580956522126321850117682987897396142693
    [
        52699749, 1788553808, 415039679, 2144920511, 546601702, 1042558412, 1066366637, 1687141834,
        285383806, 438033468, 619177062, 1199772911, 174285372, 1142848565, 1781567804, 13235
    ],
    // Montgomery One is R mod p
    // montgomery R = 2^(W*N) where W = word size and N = limbs
    //            R = 2^(16*31) = 2^496
    // one = 204586912993508866875824356051724947013540127877691549342705710506008362275292159680204380770369009821930417757972504438076078534117837065833032974336 mod p
    // 1873675273853457188138609473867413143403568023004720367747079366994575886696675578165954881343055291187920754699766806834859907881069622684602939
    [
        1588384315, 657481659, 1879608514, 2019977405, 241404753, 1339062904, 639566708, 740072562,
        1004131918, 1560224833, 2014075, 1848411426, 1733309265, 1811487384, 799788540, 19667
    ],
    // montgomery R^2 mod p
    // 457845372202231092221045514406304715517609899600516288088351276206864288839367561156406646278891945147846188034105187428603489846554823930520200
    [
        197589901, 1933752831, 580428568, 527417626, 249573438, 264164054, 609560334, 32358085,
        944568904, 1556682934, 1807973447, 1881920392, 10254137, 588677610, 1214264513, 6960
    ],
    // -p[0]^-1
    // in sage: m = p.digits(2^31)[0]
    //          (-m).inverse_mod(2^31)
    1345299673
);

// p = 65000549695646603732796438742359905742825358107623003571877145026864184071783
fp31!(
    fp_256, // Name of mod
    Fp256,  // Name of class
    256,    // Number of bits for prime
    9,      // Number of limbs (ceil(bits/31))
    [
        // prime number in limbs, least sig first
        // get this from sage with p.digits(2^31)
        1577621095, 817453272, 47634040, 1927038601, 407749150, 1308464908, 685899370, 1518399909,
        143
    ],
    //2^(31*(2*9-1)) mod p
    //18720133062205198694473358766232514389181011437180088121195238904893577296491
    [
        395508331, 432982901, 1116925886, 2092368399, 1335764116, 408528395, 1940570321, 832316282,
        41
    ],
    // Montgomery One is R mod p
    // montgomery R = 2^(W*N) where W = word size and N = limbs
    //            R = 2^(9*31) = 2^279
    // one = 971334446112864535459730953411759453321203419526069760625906204869452142602604249088 mod p
    // 31746963425510762026994079049051407537151967559209631525703407745209596424248
    [
        1368961080, 1174866893, 1632604085, 2004383869, 1511972380, 1964912876, 1176826515,
        403865604, 70
    ],
    // montgomery R^2 mod p
    // 26753832205083639112203412356185740914827891884263043594389452794758614404120
    [
        1687342104, 733402836, 182672516, 801641709, 2122695487, 1290522951, 66525586, 319877849,
        59
    ],
    // -p[0]^-1
    // in sage: m = p.digits(2^31)[0]
    //          (-m).inverse_mod(2^31)
    2132269737
);

impl From<[u8; 64]> for fp_256::Fp256 {
    fn from(src: [u8; 64]) -> Self {
        //In order to reduce a arbitrary integer we can break it up into pieces which are at most NUMLIMBS - 1 long and multiply it by REDUCTION_CONST using the following
        // formula. x0 + (x1 * REDUCTION_CONST) + (x2 * REDUCTION_CONST^2). In order to do this using only the one precomputed REDUCTION_CONST we can use Horner's method to evaluate
        // the polynomial to make it (x2 * REDUCTION_CONST + x1) * REDUCTION_CONST + x0. Note that this implementation is specific for 64 bytes, but the idea has no limit on the length
        // of the incoming number.
        let limbs = from_sixty_four_bytes(src);
        //Create fixed size views which are at most NUMLIMBS -1 in length.
        let (x0_view, x1_view, x2_view) =
            arrayref::array_refs![&limbs, fp_256::NUMLIMBS - 1, fp_256::NUMLIMBS - 1, 1];
        //Create 0 padded values that match the above views.
        let (mut x0, mut x1, mut x2) = (
            [0u32; fp_256::NUMLIMBS],
            [0u32; fp_256::NUMLIMBS],
            [0u32; fp_256::NUMLIMBS],
        );
        //This stinks, but I can't find a better way. We copy the views into the front of each of the limbs, leaving them padded to the right with 0s.
        x0[..fp_256::NUMLIMBS - 1].copy_from_slice(&x0_view[..]);
        x1[..fp_256::NUMLIMBS - 1].copy_from_slice(&x1_view[..]);
        x2[..1].copy_from_slice(&x2_view[..]);

        //We take x0 + (x1 * REDUCTION_CONST) + (x2 * REDUCTION_CONST^2) and use horner's method to reduce it to (x2 * REDUCTION_CONST + x1) * REDUCTION_CONST + x0
        (fp_256::Fp256::new(x2) * fp_256::REDUCTION_CONST + fp_256::Fp256::new(x1))
            * fp_256::REDUCTION_CONST
            + fp_256::Fp256::new(x0)
    }
}

impl From<[u8; 64]> for fp_256::Monty {
    fn from(src: [u8; 64]) -> Self {
        fp_256::Fp256::from(src).to_monty()
    }
}

impl From<[u8; 64]> for fp_480::Fp480 {
    fn from(src: [u8; 64]) -> Self {
        //See the 256 version for a play by play of this function.
        let limbs = from_sixty_four_bytes(src);
        let (x0_view, x1_view) = arrayref::array_refs![&limbs, fp_480::NUMLIMBS - 1, 2];
        let (mut x0, mut x1) = ([0u32; 16], [0u32; 16]);
        x0[..fp_480::NUMLIMBS - 1].copy_from_slice(&x0_view[..]);
        x1[..2].copy_from_slice(&x1_view[..]);

        fp_480::Fp480::new(x1) * fp_480::REDUCTION_CONST + fp_480::Fp480::new(x0)
    }
}

impl From<[u8; 64]> for fp_480::Monty {
    fn from(src: [u8; 64]) -> Self {
        fp_480::Fp480::from(src).to_monty()
    }
}

pub fn from_sixty_four_bytes(src: [u8; 64]) -> [u32; 17] {
    let mut limbs = [0u32; 17];
    unsafe_convert_bytes_to_limbs_mut(&src, &mut limbs, 64);
    limbs
}

#[cfg(test)]
mod lib {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn mont_mult1() {
        // 95268205315236501484672006935066056413858283446892086784168052156537964209835102730449048569806878637400128131440203902086374553015554146305
        let a = fp_480::Fp480::new([1u32; fp_480::NUMLIMBS]);
        // a * a % fp_480::PRIME =
        // 205669314559023345249322393444938088201822776871146042137485986789672375071531284450979897790335457986807231101745728970499097028834583423134417
        let expected = fp_480::Fp480::new([
            116566737, 258320304, 899113910, 662693571, 1878328939, 137325967, 973027057,
            1096098811, 1800707178, 257433595, 567863213, 586185298, 1453955551, 666215613,
            1815208656, 2158,
        ]);
        assert_eq!((a.to_monty() * a.to_monty()).to_norm(), expected);
    }

    #[test]
    fn mont_mult2() {
        // 452312848793890971808518248247112008541969316111895757139568199407784427521
        let a = fp_256::Fp256::new([1u32; fp_256::NUMLIMBS]);
        // a * R % fp_256::PRIME = 27935760211609813813226455184238240888269395514922035446130060411072102193610
        let expected = fp_256::Fp256::new([
            1001314762, 222542809, 1966841077, 1532144542, 1509311353, 1324885496, 689426205,
            1636449281, 61,
        ]);
        assert_eq!(a.to_monty().limbs, expected.limbs);
    }

    #[test]
    fn static_add_31_bit() {
        //41389210591178563197866013531977652355280622370776165812970320099896695112225
        let expected = fp_256::Fp256::new([
            1687077409, 1547669063, 1685320481, 1036948901, 4206667, 1832642533, 59073627,
            1086014588, 91,
        ]);
        //53194880143412583465331226137168779049052990239199584692423732563380439592004
        let a = fp_256::Fp256::new([
            558607428, 108819344, 866477261, 408251927, 1279719733, 496811896, 1446228323,
            1302207248, 117,
        ]);
        assert_eq!(a + a, expected);
    }

    #[test]
    fn static_div_31_bit() {
        //32500274847823301866398219371179952871412679053811501785938572513432092035892
        let result = fp_256::Fp256::new([
            788810548, 408726636, 1097558844, 963519300, 203874575, 654232454, 1416691509,
            1832941778, 71,
        ]);
        //41389210591178563197866013531977652355280622370776165812970320099896695112225
        let b = fp_256::Fp256::new([
            1687077409, 1547669063, 1685320481, 1036948901, 4206667, 1832642533, 59073627,
            1086014588, 91,
        ]);
        //53194880143412583465331226137168779049052990239199584692423732563380439592004
        let a = fp_256::Fp256::new([
            558607428, 108819344, 866477261, 408251927, 1279719733, 496811896, 1446228323,
            1302207248, 117,
        ]);
        assert_eq!(a / b, result);
        assert_eq!(result * b, a);
    }

    #[test]
    fn static_co_reduce_256_bit() {
        let a_result = [
            2102762755, 340721811, 1526670465, 1233221938, 1621045422, 3878, 0, 0, 0,
        ];
        let b_result = [
            496048871, 1583721686, 351053136, 72635571, 14163922, 1245, 0, 0, 0,
        ];
        let mut a = [
            2003540029, 1136642599, 2013451521, 1081750855, 2108178975, 1491192821, 4, 0, 0,
        ];
        let mut b = [
            2089475485, 1450247307, 1692152066, 1263335112, 856386648, 2075289019, 25, 0, 0,
        ];
        let pa = 6648347;
        let pb = -1201787;
        let qa = -12242368;
        let qb = 2213312;
        fp_256::Fp256::co_reduce(&mut a, &mut b, pa, pb, qa, qb);
        assert_eq!(a, a_result);
        assert_eq!(b, b_result);
    }

    #[test]
    fn fp_256_31_normalize_prime_minus_1() {
        let a = fp_256::Fp256::new([
            1577621094, 817453272, 47634040, 1927038601, 407749150, 1308464908, 685899370,
            1518399909, 143,
        ]);
        let result = a.normalize_little();
        assert_eq!(a, result);
    }

    #[test]
    fn fp_256_31_normalize_prime_plus_1() {
        let a = fp_256::Fp256::new([
            1577621096, 817453272, 47634040, 1927038601, 407749150, 1308464908, 685899370,
            1518399909, 143,
        ]);
        let result = a.normalize_little();
        assert_eq!(result, fp_256::Fp256::one());
    }

    #[test]
    fn hex_dec_print() {
        let p = fp_480::Fp480::new(fp_480::PRIME);
        // assert_eq!(p.to_str_decimal().as_str(),  "3121577065842246806003085452055281276803074876175537384188619957989004527066410274868798956582915008874704066849018213144375771284425395508176023");
        assert_eq!(p.to_str_hex().as_str(),  "fffc66640e249d9ec75ad5290b81a85d415797b931258da0d78b58a21c435cddb02e0add635a037371d1e9a40a5ec1d6ed637bd3695530683ee96497");

        let p = fp_256::Fp256::new(fp_256::PRIME);
        assert_eq!(
            p.to_str_hex().as_str(),
            "8fb501e34aa387f9aa6fecb86184dc21ee5b88d120b5b59e185cac6c5e089667"
        );
    }

    #[test]
    fn zero1() {
        let a = fp_480::Fp480::new([1u32; fp_480::NUMLIMBS]);
        assert_eq!(a - a, fp_480::Fp480::zero());
        assert_eq!(a + fp_480::Fp480::zero(), a);
        assert_eq!(a * fp_480::Fp480::zero(), fp_480::Fp480::zero());
    }

    #[test]
    fn mul_precalc() {
        // a = 95268205315236501484672006935066056413858283446892086784168052156537964209835102730449048569806878637400128131440203902086374553015554146305
        let a = fp_480::Fp480::new([1u32; fp_480::NUMLIMBS]);
        // a * a % fp_480::PRIME =
        // 205669314559023345249322393444938088201822776871146042137485986789672375071531284450979897790335457986807231101745728970499097028834583423134417
        let expected = fp_480::Fp480::new([
            116566737, 258320304, 899113910, 662693571, 1878328939, 137325967, 973027057,
            1096098811, 1800707178, 257433595, 567863213, 586185298, 1453955551, 666215613,
            1815208656, 2158,
        ]);
        assert_eq!(a * a, expected);
    }

    #[test]
    fn debug_hex_output_test256() {
        // 0x00000000000000000000000000000000000000003fffffffc000000000000000
        let other = fp_256::Fp256::new([0, 0, 0x00FFFFFFFFu32, 0, 0, 0, 0, 0, 0]);
        let str = format!("hex: {:x}", other);
        assert_eq!(
            &str.replace(" ", ""),
            "hex:0x00000000000000000000000000000000000000003fffffffc000000000000000"
        );

        // Note: we cap at PRIMEBYTES in length, discarding any higher bits
        // 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        let other = fp_256::Fp256::new([0x7FFFFFFF; 9]);
        let str = format!("hex: {:x}", other);
        assert_eq!(
            &str.replace(" ", ""),
            "hex:0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
    }

    #[test]
    fn neg_test256() {
        let a = fp_256::Fp256::one();
        let b = fp_256::Fp256::new([
            136300585, 707444127, 807555021, 1811877557, 2098044538, 317321736, 1206406714, 25, 0,
        ]);
        assert_eq!(a * b, b);
        assert_eq!(-a * b, -b);
    }

    #[test]
    fn test_from_sha_static() {
        let x = [1u8; 64];
        let expected = fp_256::Fp256::new([
            943682914, 296735281, 102601666, 655105971, 441508414, 1938904809, 1433209327,
            308023271, 117,
        ]);
        assert_eq!(fp_256::Fp256::from(x), expected);

        let mut x = [0u8; 64];
        x[16..32].iter_mut().for_each(|i| *i = 1);
        x[48..64].iter_mut().for_each(|i| *i = 1);
        let expected = fp_256::Fp256::new([
            967511966, 1307044956, 1229633257, 566771625, 922104236, 1401873859, 1287751493,
            1191577462, 120,
        ]);
        assert_eq!(fp_256::Fp256::from(x), expected);
    }

    #[test]
    fn test_from_sha_static_480() {
        let x = [1u8; 64];
        let expected = fp_480::Fp480::new([
            197889999, 570994369, 28975468, 902663725, 1105020808, 268027837, 176577716, 908958290,
            1600447047, 1231221665, 545584028, 1481371629, 67452331, 1668714925, 51469794, 9111,
        ]);
        assert_eq!(fp_480::Fp480::from(x), expected);
    }

    #[test]
    fn fp256_to_bytes_known_good_value() {
        use crate::fp_256::Fp256;
        let fp = Fp256::from(255u32);
        let bytes = fp.to_bytes_array();
        let expected_result = {
            let mut array = [0u8; 32];
            array[31] = 255;
            array
        };
        assert_eq!(bytes, expected_result);
    }

    #[test]
    fn fp256_from_bytes_should_mod() {
        use crate::fp_256::Fp256;
        let max_bytes = Fp256::from([255u8; 32]);
        let expected_result = Fp256::new([
            569862552, 1330030375, 2099849607, 220445046, 1739734497, 839018739, 1461584277,
            629083738, 112,
        ]);
        assert_eq!(max_bytes, expected_result);
        let to_bytes_result = max_bytes.to_bytes_array();
        assert_eq!(
            to_bytes_result,
            [
                112, 74, 254, 28, 181, 92, 120, 6, 85, 144, 19, 71, 158, 123, 35, 222, 17, 164,
                119, 46, 223, 74, 74, 97, 231, 163, 83, 147, 161, 247, 105, 152
            ]
        );
    }
}
