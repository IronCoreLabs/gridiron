# compute_constants.py

Computes Montgomery arithmetic constants for custom finite fields and outputs the complete Rust macro invocation ready to paste into your code.

## Usage

The script supports both 31-bit and 62-bit limb representations and can work with known fields (Fp256, Fp480) or custom primes.

### Using Known Fields

```bash
# Generate 31-bit constants for Fp256
nix-shell -p python3 --run "./scripts/compute_constants.py --limb-size 31 --field fp256"

# Generate 62-bit constants for Fp480
nix-shell -p python3 --run "./scripts/compute_constants.py --limb-size 62 --field fp480"
```

### Using Custom Primes

```bash
# Custom prime with all parameters
nix-shell -p python3 --run "./scripts/compute_constants.py \
  --limb-size 31 \
  --prime 123456789012345678901234567890 \
  --module my_field \
  --classname MyField \
  --bits 128"
```

The script will compute:

- Prime in limb representation
- Barrett reduction constant
- Montgomery One (R mod p)
- Montgomery R² (R² mod p)
- Montgomery m0_inv constant

And output a complete `fp31!` or `fp62!` macro invocation ready to paste into your Rust code.

## Requirements

- Python 3.x (available via `nix-shell -p python3`)
- No external dependencies (uses only Python standard library)

## Output Format

The script outputs:

1. **Verbose mode** (default): Detailed explanation of all computed values
2. **Quiet mode** (`--quiet`): Only the Rust macro invocation
