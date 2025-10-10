# Changelog

## 0.11.0

- [[#57]](https://github.com/IronCoreLabs/gridiron/pull/57) - Update rand to 0.9

- [[#58]](https://github.com/IronCoreLabs/gridiron/pull/58) - Update MSRV to 1.85.0 & rust edition 2024

- [[#59]](https://github.com/IronCoreLabs/gridiron/pull/59) - Update to criterion 0.6 (Test only change)


## 0.10.0

- [[#52]](https://github.com/IronCoreLabs/gridiron/pull/52)
  - Update to criterion 0.5 (Test only change)
  - Change MSRV to Rust 1.70.0

- [[#51]](https://github.com/IronCoreLabs/gridiron/pull/51) - Change MSRV to Rust 1.69.0

- [[#44]](https://github.com/IronCoreLabs/gridiron/pull/44) - Change MSRV to Rust 1.60.0

- [[#42]](https://github.com/IronCoreLabs/gridiron/pull/42) - Update to criterion 0.4 (Test only change)

- [[#40]](https://github.com/IronCoreLabs/gridiron/pull/40)
  - Remove arrayref dependency
  - Switch to Rust 2021 edition

## 0.9.0

- [[#37]](https://github.com/IronCoreLabs/gridiron/pull/37) - Update rand requirement from ~0.7.3 to ~0.8.0 (Test only change)

- [[#38]](https://github.com/IronCoreLabs/gridiron/pull/38) - Update proptest requirement from ~0.10 to ~1.0 (Test only change)

- [[#39]](https://github.com/IronCoreLabs/gridiron/pull/39) - Fix proptest issue where passing test wrote a regression file.

## 0.8.0

- [[#36]](https://github.com/IronCoreLabs/gridiron/pull/36)
  - Update to proptest to 0.10 (Test only change)
  - Change MSRV to Rust 1.40.0

## 0.7.0

- [[#30]](https://github.com/IronCoreLabs/gridiron/pull/30) - Update to Rand 0.7 (Test only change)

## 0.6.0

- [[#21]](https://github.com/IronCoreLabs/gridiron/pull/21) - Many fixes for edge cases, remove Barrett. Breaking changes on macro invocation.
- [[#20]](https://github.com/IronCoreLabs/gridiron/pull/20) - Fix defect in const_eq0 and related div_mod issue.
- [[#19]](https://github.com/IronCoreLabs/gridiron/pull/19) - Make abs constant time.

## 0.5.2

- [[#18]](https://github.com/IronCoreLabs/gridiron/pull/18) - Add From<[u8; 64]> to Fp480

## 0.5.1

- [[#17]](https://github.com/IronCoreLabs/gridiron/pull/17) - Clippy cleanup
- [[#16]](https://github.com/IronCoreLabs/gridiron/pull/16) - Add inverse and div to Monty

## 0.5.0

- [[#13]](https://github.com/IronCoreLabs/gridiron/pull/13) - Add const_swap_if to support swapping fp values in constant time.
- [[#14]](https://github.com/IronCoreLabs/gridiron/pull/14) - Standardize on non secret u32 values for mul and pow.

## 0.4.0

- [[#10]](https://github.com/IronCoreLabs/gridiron/pull/10) - Visibility and Documentation Changes
- [[#11]](https://github.com/IronCoreLabs/gridiron/pull/11) - Rust 2018 support
