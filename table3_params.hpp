#pragma once
//==============================================================================
// CKKS parameters for the Table 3 experiments.
//
// This is the "Table 3" row of Table 4 in the paper:
//     log N = 17,  log PQ = 2070,  log Delta = 51,  dnum = 3,  h = 64
// HEaaN2 ships no preset with this combination, so the modulus chain is
// spelled out here.
//
//   modulus chain: q0 (60 bit) + q1..q29 (51 bit)  =  chain length 30,
//                  so max_level = encryption_level = 29.
//
// Why q0 is 60 bit rather than 61. The integer PCMM accumulates in u128 and
// requires
//     available_bit = 128 - 2*modulus_bit - 1  >=  log2(TILE_K),
// and TILE_K is 64 for AVX512DQ with row-major operands, needing 6 bits:
//     61 bit : 128 - 122 - 1 = 5  <  6   -> rejected
//     60 bit : 128 - 120 - 1 = 7  >= 6   -> accepted
// The scale of a level is determined by the quantization primes above it, so
// the width of q0 does not change the scale at any level. Programs that do not
// use PCMM share these parameters anyway, for comparability.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include <stdexcept>
#include <vector>

namespace pel {

using heaan::u32;
using heaan::u64;

inline constexpr u32 TABLE3_LOG_DEGREE = 17;
inline constexpr u32 TABLE3_LOG_SLOTS = TABLE3_LOG_DEGREE - 1; // 16 -> 65536
inline constexpr u32 TABLE3_HAMMING_WEIGHT = 64;               // h

/// @brief Chain length: 1 base prime + 29 quantization primes, so the highest
/// usable level is 29.
inline constexpr u32 TABLE3_NUM_PRIMES = 30;

/// @brief Total QP budget in bits: the log PQ = 2070 of Table 4.
/// @details Q takes 60 + 29*51 = 1539 bits; the rest is P, the key-switching
/// temporary modulus. Given this budget SwKeyGenParamsBuilder settles on a
/// gadget decomposition of dnum = 3 in blocks of [10, 10, 10] with
/// log2(P) = 531, matching the dnum of Table 4.
inline constexpr u32 TABLE3_QP_MAX_BITS = 2070;

/// @brief The 60-bit base prime, 2^60 - 2^18 + 1.
/// @details 2*degree = 2^18 divides p-1, so the ring admits an NTT.
inline constexpr u64 TABLE3_Q0 = 1152921504606584833ULL;

/// @brief The modulus chain: one 60-bit base prime and 29 51-bit
/// quantization primes.
inline const std::vector<u64> &table3Primes() {
    static const std::vector<u64> primes = {
        // base prime (60 bit)
        TABLE3_Q0,
        // quantization primes (51 bit) x 29
        2251799756013569ULL, 2251799787995137ULL, 2251800352915457ULL,
        2251799780917249ULL, 2251799666884609ULL, 2251799678943233ULL,
        2251799696244737ULL, 2251800082382849ULL, 2251799776198657ULL,
        2251799929028609ULL, 2251799774887937ULL, 2251799849336833ULL,
        2251799883153409ULL, 2251799777771521ULL, 2251799879483393ULL,
        2251799772266497ULL, 2251799763091457ULL, 2251799844093953ULL,
        2251799823384577ULL, 2251799851958273ULL, 2251799789568001ULL,
        2251799797432321ULL, 2251799799267329ULL, 2251799836753921ULL,
        2251799806345217ULL, 2251799807131649ULL, 2251799818928129ULL,
        2251799816568833ULL, 2251799815520257ULL};
    return primes;
}

/// @brief Builds the Levels for the first num_primes primes of the chain.
/// @param num_primes How many primes to use, base prime included; the highest
/// level is then num_primes-1.
/// @details Always pass TABLE3_NUM_PRIMES. Scales are derived downwards from
/// the top of the chain, so truncating it shifts the scale of every level that
/// the circuit actually uses, and a deep circuit can no longer run under the
/// same parameters.
inline heaan::Levels makeTable3Levels(u32 num_primes) {
    const auto &primes = table3Primes();
    if (num_primes == 0 || num_primes > primes.size())
        throw std::invalid_argument("[makeTable3Levels] invalid num_primes");

    std::vector<heaan::PolyMod> chain;
    chain.reserve(num_primes);
    for (u32 n = 1; n <= num_primes; ++n) {
        heaan::PolyMod mod(n);
        for (u32 i = 0; i < n; ++i)
            mod[i] = primes[i];
        chain.push_back(mod);
    }

    // The top level takes its scale from the last prime in the chain.
    const heaan::Real128 top_scale =
        static_cast<heaan::Real128>(primes[num_primes - 1]);
    return heaan::Levels(chain, top_scale);
}

} // namespace pel
