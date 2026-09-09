#pragma once
//==============================================================================
// CKKS parameters for the Table 5 experiments (end-to-end encrypted FastText
// inference).
//
// This is the "Table 5 / Ours" row of Table 4 in the paper:
//     log N = 16,  log PQ = 1555,  log Delta = 42,  dnum = 5,  h = 192
// All three derived quantities were checked against the configuration below:
// log2(QP) = 1555, log2(scale) = 42.0 across levels 0..12, gadget_rank = 5.
//
//   modulus chain (25 primes, so max_level = 24):
//     base             58 bit x 1
//     slotToCoeff      42 bit x 3
//     multiplication   42 bit x 9    <- levels 0..12, the encryption region
//     evalSine         58 bit x 9
//     coeffToSlot      58 bit x 3
//   temporary primes   59 bit x 3 + 60 bit x 2   (the P part of key switching)
//
// The chain and the QP budget are kept bootstrappable even though this circuit
// never bootstraps, so that the parameter set is a realistic one.
//
// Two things differ from table3_params.hpp:
//
// 1. The scale is not uniform. In table3 every level converges to 2^51, but
//    here the primes fall into a 42-bit and a 58-bit region, so the scales do
//    too: 2^42 for levels 0..12 and 2^58 for levels 13..24. HEaaN2's
//    Levels(mods, top_scale) derives one scale from the next and cannot
//    produce this step in a single pass, so the Levels is built twice and
//    spliced -- see makeTable5Levels below.
//
// 2. q0 needs no adjustment. table3 has to widen q0 for the integer PCMM
//    bound, but 58 bits already leaves 128 - 2*58 - 1 = 11 bits of headroom
//    against the 6 that PCMM requires.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace pel {

using heaan::u32;
using heaan::u64;

inline constexpr u32 TABLE5_LOG_DEGREE = 16;
inline constexpr u32 TABLE5_LOG_SLOTS = TABLE5_LOG_DEGREE - 1; // 15 -> 32768
inline constexpr u32 TABLE5_HAMMING_WEIGHT = 192;              // h

/// @brief Chain length: 1 base prime + 24 others, so the highest level is 24.
/// @details Kept at full length so the parameter set stays bootstrappable,
/// even though this circuit does not bootstrap.
inline constexpr u32 TABLE5_NUM_PRIMES = 25;

/// @brief Top of the 42-bit region, and the level ciphertexts are encrypted
/// at.
/// @details Levels at or below this carry scale 2^42. It is the boundary the
/// splice below is built around, so changing it changes every scale.
inline constexpr u32 TABLE5_ENCRYPTION_LEVEL = 12;

/// @brief Total QP budget in bits: the log PQ = 1555 of Table 4.
/// @details Q = 58 + 12*42 + 12*58 = 1258 and P = 3*59 + 2*60 = 297, which sum
/// to 1555. HEaaN2 derives gadget_rank (dnum) = 5 from this budget.
inline constexpr u32 TABLE5_QP_MAX_BITS = 1555;

/// @brief The 25 primes of the modulus chain.
inline const std::vector<u64> &table5Primes() {
    static const std::vector<u64> primes = {
        // base prime (58 bit x 1)
        288230376147386369ULL,
        // slotToCoeff primes (42 bit x 3)
        4398044938241ULL, 4398043496449ULL, 4398042972161ULL,
        // multiplication primes (42 bit x 9)
        4398042185729ULL, 4398034845697ULL, 4398029340673ULL,
        4398028029953ULL, 4398021869569ULL, 4398021476353ULL,
        4398021345281ULL, 4398018723841ULL, 4398018592769ULL,
        // evalSine primes (58 bit x 9)
        288230376115535873ULL, 288230376138735617ULL, 288230376135196673ULL,
        288230376132182017ULL, 288230376131788801ULL, 288230376129691649ULL,
        288230376128643073ULL, 288230376126545921ULL, 288230376122613761ULL,
        // coeffToSlot primes (58 bit x 3)
        288230376121434113ULL, 288230376121040897ULL, 288230376118026241ULL};
    return primes;
}

/// @brief Builds the Levels for a prefix of the chain. Used by the splice.
inline heaan::Levels makeLevelsUpTo(u32 num_primes) {
    const auto &primes = table5Primes();
    if (num_primes == 0 || num_primes > (u32)primes.size())
        throw std::invalid_argument("[makeLevelsUpTo] invalid num_primes");

    std::vector<heaan::PolyMod> chain;
    chain.reserve(num_primes);
    for (u32 n = 1; n <= num_primes; ++n) {
        heaan::PolyMod mod(n);
        for (u32 i = 0; i < n; ++i)
            mod[i] = primes[i];
        chain.push_back(mod);
    }
    // The top level takes its scale from the last prime in the prefix.
    const heaan::Real128 top_scale =
        static_cast<heaan::Real128>(primes[num_primes - 1]);
    return heaan::Levels(chain, top_scale);
}

/// @brief Levels for the full chain, splicing the 42-bit and 58-bit regions.
/// @details Every operation checks, bit for bit, that a level's scale agrees
/// with the level above it and the prime between them, and throws
/// "[HomEval::mulRescale] Scale does not match" otherwise. Computing the
/// scales independently and overwriting them does not survive that check, so
/// the library's own values are used: mods and the scales of levels 13..24
/// come from the full chain, the scales of levels 0..12 from a chain
/// truncated at TABLE5_ENCRYPTION_LEVEL. Each region is then internally
/// consistent, and the only seam is at the 12/13 boundary, which this circuit
/// never crosses -- it uses at most level log2(p) = 8.
inline heaan::Levels makeTable5Levels() {
    // 58-bit region: source of mods and of the scales for levels 13..24.
    heaan::Levels full = makeLevelsUpTo(TABLE5_NUM_PRIMES);
    // 42-bit region: source of the scales for levels 0..12.
    const heaan::Levels low = makeLevelsUpTo(TABLE5_ENCRYPTION_LEVEL + 1);

    if (low.scales.size() != TABLE5_ENCRYPTION_LEVEL + 1 ||
        full.scales.size() != TABLE5_NUM_PRIMES)
        throw std::logic_error("[makeTable5Levels] unexpected Levels size");

    for (u32 i = 0; i <= TABLE5_ENCRYPTION_LEVEL; ++i)
        full.scales[i] = low.scales[i];

    return full;
}

} // namespace pel
