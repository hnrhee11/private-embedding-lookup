#pragma once
//==============================================================================
// Vector generation for the two lookup methods compared in
//
//   "Private Embedding Lookup with Encrypted Compact Queries under Fully
//    Homomorphic Encryption"
//
//   pel::ive   -- Algorithm 1 (IVE algorithm), Section 4.2 of that paper.
//                 Builds the DCT basis vector v_j from Enc(alpha_j).
//   pel::ICML24  -- the baseline it is compared against:
//                 Kim, Park, Lee, Cheon, "Privacy-Preserving Embedding via
//                 Look-up Table Evaluation with Fully Homomorphic Encryption".
//                 Builds a one-hot vector e_j by an indicator polynomial.
//
// Both write their p output ciphertexts back into the caller's flat array of
// K = ell * p ciphertexts, one block of p per sub-table.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include "pel_helpers.hpp" // compat::Mult, compat::Add

#include <algorithm>
#include <vector>

namespace pel {

//------------------------------------------------------------------------------
// IVE -- Algorithm 1 of the paper, applied to one sub-table block.
//
// The caller lays out K = ell * p ciphertexts flat; [base, base+p) is one
// block.
//
//   in  : ctxt[base]     = Enc(alpha_j)   at level log2(p)   (Algorithm 1 In)
//         ctxt[base+p-1] = Enc(sqrt(2))   at level 1         (Algorithm 1 L16)
//   out : ctxt[base+k]          = 2*cos((k+1)*theta)   k = 0..p_half-1
//         ctxt[base+p_half+k]   = 2*sin((k+1)*theta)   k = 0..p_half-2
//         ctxt[base+p-1]        untouched
//
// Everything ends at level 1, matching the constant ciphertext, so the block
// can go straight into PCMM. Depth = log2(p) as in Table 2 of the paper.
//------------------------------------------------------------------------------
inline void ive(const heaan::HomEval &eval, const heaan::ISwKey &relin_key,
                const heaan::ISwKey &conj_key,
                std::vector<heaan::Ptr<heaan::ICiphertext>> &ctxt, int base,
                int p) {
    const int p_half = p / 2; // n in Algorithm 1 L1

    auto ctxt_tmp = heaan::ICiphertext::make();
    auto ctxt_real = heaan::ICiphertext::make();
    auto ctxt_conj = heaan::ICiphertext::make();

    // Algorithm 1 L3-L9: power tree. ctxt[base+k-1] = alpha^k, k = 1..p_half
    for (int i = 1; i < p_half; i <<= 1) {
        const int t = std::min(i, p_half - i);
        for (int j = 1; j <= t; ++j) {
            compat::Mult(eval, relin_key, *ctxt[base + (i - 1)],
                         *ctxt[base + (j - 1)], *ctxt[base + (i + j - 1)]);
        }
    }

    // Algorithm 1 L10-L17: split each power into its real and imaginary parts
    for (int i = 0; i < p_half - 1; ++i) {
        eval.levelDownTo(*ctxt[base + i], *ctxt[base + i], 1); // unify levels

        eval.conj(*ctxt[base + i], *ctxt_tmp, conj_key);  // L11
        eval.add(*ctxt[base + i], *ctxt_tmp, *ctxt_real); // L12  2*Re
        eval.sub(*ctxt[base + i], *ctxt_tmp, *ctxt_conj); //      2i*Im
        ctxt_real->copyTo(*ctxt[base + i]);

        // L14: divide by sqrt(-1). GaussianInt(0, 1) is i, and 1/i == -i, so
        // multiplying by i and negating is the same. Gaussian-integer
        // multiplication is free of level consumption.
        eval.mul(*ctxt_conj, heaan::GaussianInt(0, 1), *ctxt_conj); // -2*Im
        eval.neg(*ctxt_conj, *ctxt[base + p_half + i]);             //  2*Im
    }

    // Algorithm 1 L10-L12 for k = p_half-1: real part only (L16 already holds
    // Enc(sqrt(2)) in ctxt[base+p-1])
    eval.levelDownTo(*ctxt[base + (p_half - 1)], *ctxt[base + (p_half - 1)], 1);
    eval.conj(*ctxt[base + (p_half - 1)], *ctxt_tmp, conj_key);
    eval.add(*ctxt[base + (p_half - 1)], *ctxt_tmp, *ctxt[base + (p_half - 1)]);
}

//------------------------------------------------------------------------------
// ICML24 -- baseline of Kim, Park, Lee, Cheon, "Privacy-Preserving Embedding via
// Look-up Table Evaluation with Fully Homomorphic Encryption", applied to one
// sub-table block. Referred to as the method of Kim et al. in Section 1.1 and
// Table 1 of our paper.
//
//   in  : ctxt[base]     = Enc(j), the raw index value, at level `level`+1
//   out : ctxt[base + i] = Enc(1) if j == i else Enc(0),  i = 0..p-1
//
// The indicator is
//     f(x) = 1 - 2*((x - i)/p)^2
// squared iter_r times, so it tends to 1 only at x == i, then smoothed by
// 3x^2 - 2x^3 applied iter_s times.
//
//   depth = 1 (constant mult) + 1 (square) + iter_r + 2*iter_s
//         = 2 + iter_r + 2*iter_s
//
// This is O(p) homomorphic operations per block against O(p) for IVE but with
// a far deeper circuit -- the cost gap our paper reports in Table 3.
//------------------------------------------------------------------------------
inline void ICML24(const heaan::HomEval &eval, const heaan::ISwKey &relin_key,
                 std::vector<heaan::Ptr<heaan::ICiphertext>> &ctxt, int base,
                 int p, int iter_r, int iter_s) {
    using heaan::Complex128;
    using heaan::ICiphertext;

    auto ctxt_origin = ICiphertext::make();
    auto ctxt_tmp = ICiphertext::make();
    auto ctxt_rst = ICiphertext::make();
    auto ctxt_tmp2 = ICiphertext::make();

    // ctxt[base] is overwritten at i == 0, so keep a copy of the query
    ctxt[(size_t)base]->copyTo(*ctxt_origin);

    for (int i = 0; i < p; ++i) {
        // f(x) = 1 - 2*((x - i)/p)^2
        eval.sub(*ctxt_origin, Complex128((double)i, 0.0), *ctxt_tmp); // x - i

        // mul by a scalar squares the scale, so rescale brings it back
        eval.mul(*ctxt_tmp, Complex128(1.0 / (double)p, 0.0), *ctxt_tmp2);
        eval.rescale(*ctxt_tmp2, *ctxt_tmp);                     // (x-i)/p
        compat::Mult(eval, relin_key, *ctxt_tmp, *ctxt_tmp, *ctxt_tmp);  // ^2
        eval.add(*ctxt_tmp, *ctxt_tmp, *ctxt_tmp);               // 2*
        eval.neg(*ctxt_tmp, *ctxt_tmp);                          // -2*
        eval.add(*ctxt_tmp, Complex128(1.0, 0.0), *ctxt_tmp);    // 1 - 2*

        for (int r = 0; r < iter_r; ++r)
            compat::Mult(eval, relin_key, *ctxt_tmp, *ctxt_tmp, *ctxt_tmp);

        ctxt_tmp->copyTo(*ctxt_rst);

        // smooth-step 3x^2 - 2x^3
        for (int s = 0; s < iter_s; ++s) {
            // x^2
            compat::Mult(eval, relin_key, *ctxt_tmp, *ctxt_tmp, *ctxt_tmp2);
            eval.add(*ctxt_tmp2, *ctxt_tmp2, *ctxt_rst);             // 2x^2
            eval.add(*ctxt_tmp2, *ctxt_rst, *ctxt_rst);              // 3x^2

            // x^3
            compat::Mult(eval, relin_key, *ctxt_tmp, *ctxt_tmp2, *ctxt_tmp);
            eval.neg(*ctxt_tmp, *ctxt_tmp);                          // -x^3
            eval.add(*ctxt_tmp, *ctxt_tmp, *ctxt_tmp);               // -2x^3
            // 3x^2 - 2x^3
            compat::Add(eval, *ctxt_tmp, *ctxt_rst, *ctxt_rst);

            ctxt_rst->copyTo(*ctxt_tmp);
        }

        ctxt_rst->copyTo(*ctxt[(size_t)(base + i)]);
    }
}

} // namespace pel
