#!/usr/bin/env python3
"""
Compute Montgomery constants for gridiron finite field implementations.

This script computes all necessary constants for creating custom finite fields
with either 31-bit or 62-bit limb representations, and outputs the complete
Rust macro invocation ready to paste into your code.

Usage:
    # Using a known prime (Fp256 or Fp480)
    ./compute_constants.py --limb-size 31 --field fp256
    ./compute_constants.py --limb-size 62 --field fp480

    # Using a custom prime
    ./compute_constants.py --limb-size 31 --prime 123456789... --module my_field --classname MyField --bits 256

    # Quick mode (just outputs Rust macro)
    ./compute_constants.py --limb-size 62 --field fp256 --quiet
"""

import argparse
import sys
from typing import List, Tuple

# Known primes
KNOWN_PRIMES = {
    'fp256': {
        'prime': 65000549695646603732796438742359905742825358107623003571877145026864184071783,
        'bits': 256,
        'module': 'fp_256',
        'classname': 'Fp256',
    },
    'fp480': {
        'prime': 3121577065842246806003085452055281276803074876175537384188619957989004527066410274868798956582915008874704066849018213144375771284425395508176023,
        'bits': 480,
        'module': 'fp_480',
        'classname': 'Fp480',
    },
}


def to_limbs(n: int, limb_size: int, num_limbs: int) -> List[int]:
    """Convert integer to limb representation"""
    mask = (1 << limb_size) - 1
    limbs = []
    for _ in range(num_limbs):
        limbs.append(n & mask)
        n >>= limb_size
    return limbs


def from_limbs(limbs: List[int], limb_size: int) -> int:
    """Convert limb representation to integer"""
    result = 0
    for i, limb in enumerate(limbs):
        result |= limb << (limb_size * i)
    return result


def mod_inverse(a: int, m: int) -> int:
    """Compute modular inverse of a mod m using extended Euclidean algorithm"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    _, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m


def compute_num_limbs(bits: int, limb_size: int) -> int:
    """Compute number of limbs needed for given bit count and limb size"""
    return (bits + limb_size - 1) // limb_size


def compute_constants(prime: int, limb_size: int, bits: int) -> Tuple[List[int], List[int], List[int], List[int], int, int]:
    """
    Compute all Montgomery constants for a given prime.

    Returns:
        (prime_limbs, reduction_limbs, montgomery_one_limbs, montgomery_r2_limbs, montm0inv, num_limbs)
    """
    num_limbs = compute_num_limbs(bits, limb_size)

    # Convert prime to limbs
    prime_limbs = to_limbs(prime, limb_size, num_limbs)

    # Compute montm0inv: (-p[0])^-1 mod 2^limb_size
    modulus = 1 << limb_size
    montm0inv = mod_inverse(-prime_limbs[0], modulus)

    # R = 2^(limb_size * num_limbs)
    R = 1 << (limb_size * num_limbs)

    # Montgomery One: R mod p
    montgomery_one = R % prime
    montgomery_one_limbs = to_limbs(montgomery_one, limb_size, num_limbs)

    # Montgomery R^2: R^2 mod p
    montgomery_r2 = (R * R) % prime
    montgomery_r2_limbs = to_limbs(montgomery_r2, limb_size, num_limbs)

    # Reduction constant: 2^(limb_size * (2*num_limbs - 1)) mod p
    # This is used for Barrett reduction
    reduction_const = (1 << (limb_size * (2 * num_limbs - 1))) % prime
    reduction_limbs = to_limbs(reduction_const, limb_size, num_limbs)

    return prime_limbs, reduction_limbs, montgomery_one_limbs, montgomery_r2_limbs, montm0inv, num_limbs


def format_limb_array(limbs: List[int], indent: int = 8) -> str:
    """Format limb array for Rust code with proper line wrapping"""
    indent_str = ' ' * indent
    # Wrap at ~100 characters
    lines = []
    current_line = []
    current_length = 0

    for i, limb in enumerate(limbs):
        limb_str = str(limb)
        if i < len(limbs) - 1:
            limb_str += ','

        # Check if adding this limb would exceed line length
        test_length = current_length + len(limb_str) + (2 if current_line else 0)
        if current_line and test_length > 100:
            lines.append(indent_str + ' '.join(current_line))
            current_line = [limb_str]
            current_length = len(limb_str)
        else:
            current_line.append(limb_str)
            current_length = test_length

    if current_line:
        lines.append(indent_str + ' '.join(current_line))

    return '\n'.join(lines)


def generate_macro_invocation(module: str, classname: str, bits: int, limb_size: int,
                               prime_limbs: List[int], reduction_limbs: List[int],
                               montgomery_one_limbs: List[int], montgomery_r2_limbs: List[int],
                               montm0inv: int, num_limbs: int) -> str:
    """Generate the complete Rust macro invocation"""

    macro_name = f"fp{limb_size}"

    return f"""{macro_name}!(
    {module}, // Name of mod
    {classname},  // Name of class
    {bits},    // Number of bits for prime
    {num_limbs},      // Number of limbs (ceil(bits/{limb_size}))
    [
        // prime number in limbs, least significant first
        // get this from sage with p.digits(2^{limb_size})
{format_limb_array(prime_limbs)}
    ],
    [
        // Barrett reduction constant for reducing values up to twice
        // the number of prime bits (double limbs):
        // 2^({limb_size}*{num_limbs}*2 - {limb_size}) mod p = 2^{limb_size * num_limbs * 2 - limb_size} mod p
{format_limb_array(reduction_limbs)}
    ],
    [
        // Montgomery R = 2^(W*N) where W = word size and N = limbs
        //            R = 2^({num_limbs}*{limb_size}) = 2^{num_limbs * limb_size}
        // Montgomery R mod p
{format_limb_array(montgomery_one_limbs)}
    ],
    [
        // Montgomery R^2 mod p
{format_limb_array(montgomery_r2_limbs)}
    ],
    // -p[0]^-1 mod 2^{limb_size}
    // in sage: m = p.digits(2^{limb_size})[0]
    //          (-m).inverse_mod(2^{limb_size})
    {montm0inv}
);"""


def main():
    parser = argparse.ArgumentParser(
        description='Compute Montgomery constants for gridiron finite fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 31-bit limb constants for Fp256
  %(prog)s --limb-size 31 --field fp256

  # Generate 62-bit limb constants for Fp480
  %(prog)s --limb-size 62 --field fp480

  # Use custom prime
  %(prog)s --limb-size 31 --prime 12345... --module my_fp --classname MyFp --bits 256

  # Quiet mode (only output Rust macro)
  %(prog)s --limb-size 62 --field fp256 --quiet
        """
    )

    parser.add_argument('--limb-size', type=int, choices=[31, 62], required=True,
                        help='Limb size in bits (31 or 62)')
    parser.add_argument('--field', type=str, choices=['fp256', 'fp480'],
                        help='Use known field (fp256 or fp480)')
    parser.add_argument('--prime', type=str,
                        help='Prime number (decimal or hex with 0x prefix)')
    parser.add_argument('--module', type=str,
                        help='Module name (e.g., fp_256)')
    parser.add_argument('--classname', type=str,
                        help='Class name (e.g., Fp256)')
    parser.add_argument('--bits', type=int,
                        help='Number of bits in prime')
    parser.add_argument('--quiet', action='store_true',
                        help='Only output Rust macro (suppress explanations)')

    args = parser.parse_args()

    # Determine prime and metadata
    if args.field:
        field_info = KNOWN_PRIMES[args.field]
        prime = field_info['prime']
        bits = field_info['bits']
        module = field_info['module']
        classname = field_info['classname']
    elif args.prime:
        # Parse prime (support both decimal and hex)
        if args.prime.startswith('0x') or args.prime.startswith('0X'):
            prime = int(args.prime, 16)
        else:
            prime = int(args.prime)

        # Require other parameters for custom primes
        if not all([args.module, args.classname, args.bits]):
            parser.error("--prime requires --module, --classname, and --bits")

        module = args.module
        classname = args.classname
        bits = args.bits
    else:
        parser.error("Must specify either --field or --prime")

    # Compute constants
    prime_limbs, reduction_limbs, montgomery_one_limbs, montgomery_r2_limbs, montm0inv, num_limbs = \
        compute_constants(prime, args.limb_size, bits)

    # Output
    if not args.quiet:
        print("=" * 80)
        print(f"{classname} ({bits}-bit prime, {num_limbs} limbs @ {args.limb_size}-bit)")
        print("=" * 80)
        print(f"\nPrime (decimal): {prime}")
        print(f"\nPrime in {args.limb_size}-bit limbs:")
        print(f"  {prime_limbs}")
        print(f"\nMontgomery m0_inv ((-p[0])^-1 mod 2^{args.limb_size}):")
        print(f"  {montm0inv}")
        print(f"\nR = 2^({args.limb_size} * {num_limbs}) = 2^{args.limb_size * num_limbs}")
        print(f"\nMontgomery One (R mod p):")
        print(f"  {montgomery_one_limbs}")
        print(f"\nMontgomery R^2 (R^2 mod p):")
        print(f"  {montgomery_r2_limbs}")
        print(f"\nReduction constant (2^{args.limb_size * (2 * num_limbs - 1)} mod p):")
        print(f"  {reduction_limbs}")
        print("\n" + "=" * 80)
        print("Rust Macro Invocation:")
        print("=" * 80)
        print()

    # Always output the macro
    macro_code = generate_macro_invocation(
        module, classname, bits, args.limb_size,
        prime_limbs, reduction_limbs, montgomery_one_limbs, montgomery_r2_limbs,
        montm0inv, num_limbs
    )
    print(macro_code)

    if not args.quiet:
        print("\n" + "=" * 80)
        print("Copy the above macro invocation into your Rust code.")
        print("=" * 80)


if __name__ == '__main__':
    main()
