//==============================================================================
// End-to-end encrypted FastText inference with the baseline lookup, the
// "Kim et al." rows of Table 5 in
//
//   "Private Embedding Lookup with Encrypted Compact Queries under Fully
//    Homomorphic Encryption"
//
// Same pipeline as ive_fasttext_spam.cpp -- the classifier weights are folded into the
// score table offline, so the server looks up one score per token and adds the
// scores together -- but the lookup vectors are built by the method of Kim,
// Park, Lee and Cheon instead of by IVE.
//
//   (1)(2)  encrypt the token codes           Section 5.1
//   (3)     one-hot vectors                    -> pel::ICML24
//   (4)     padding fix                        see below
//   (0)     score table U, offline             footnote 2
//   (5)     U . one-hot                        PCMM
//   (6)     sum over the token axis            7 rotations
//   (7)     decrypt and compare with the plaintext model
//
// Three things follow from using one-hot vectors rather than a DCT basis:
//
//  - The circuit is much deeper, level = 2 + iter_r + 2*iter_s against log2(p),
//    so ciphertexts are encrypted far higher up the modulus chain. Table 4
//    accordingly gives this method its own parameter set, the one that Table 3
//    also uses (log N = 17), so twice as many examples fit in a batch.
//  - The one-hot entries are already 0/1, so the raw score table is used with
//    no transform and no rescaling.
//  - Padding slots are filled with code 0, which switches the "code == 0"
//    indicator on and would add a phantom token's score to the total. Step (4)
//    subtracts a padding mask from the 0-th indicator of each sub-table.
//
// Notation: p = sub-table size, ell = 4 sub-tables per token, K = ell * p,
// N = degree, n = N/2 slots.
//==============================================================================
#include "HEaaN2/HEaaN2.hpp"

#include "basis_generation.hpp"
#include "filtering_data.hpp"
#include "pel_helpers.hpp"
#include "table3_params.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <omp.h>




using namespace heaan;

static int runExample() try {
    omp_set_num_threads(1);


    //==========================================================================
    // Parameters -- the "Table 5 / Kim et al." row of Table 4, which is the
    // same set Table 3 uses; see table3_params.hpp
    //==========================================================================
    const u32 log_degree = pel::TABLE3_LOG_DEGREE; // 17
    const u32 log_slots = pel::TABLE3_LOG_SLOTS;   // 16
    const u32 degree = 1u << log_degree;           // 131072
    const int n = 1 << log_slots;                  // 65536 slots

    const int p = pel::FILTER_P;                   // 256
    const int num_blocks = pel::FILTER_NUM_DIGITS; // ell in the paper
    const int K = p * num_blocks;                  // 1024
    const int M_out = pel::FILTER_NUM_CLASSES;     // 2
    const int max_tok = pel::FILTER_MAX_TOKENS;    // 128
    const int stride = n / max_tok;                // examples per batch

    // Indicator iterations for p = 256, and the depth they consume: 1 for the
    // constant multiply, 1 for the square, iter_r for the repeated squarings,
    // 2 per smooth-step. The query is encrypted one level above that.
    const int iter_r = 19;
    const int iter_s = 1;
    const int level = 2 + iter_r + 2 * iter_s; // 23
    const u32 enc_level = (u32)(level + 1);    // 24

    auto levels = pel::makeTable3Levels(pel::TABLE3_NUM_PRIMES);

    std::cout << "n=" << n << " p=" << p
              << " degree=" << degree << " stride=" << stride
              << " M_out=" << M_out << " r=" << iter_r << " s=" << iter_s
              << "\n";
    std::cout << "level=" << level << " enc_level=" << enc_level
              << " (levels.top()=" << levels.top() << ")\n";


    if (enc_level > levels.top()) {
        std::cerr << "[fatal] enc_level " << enc_level << " > levels.top() "
                  << levels.top() << "\n";
        return 1;
    }

    //==========================================================================
    // Key generation. The indicator uses only multiplications, so no
    // conjugation key is needed; the rotation keys drive the rotate-sum.
    //
    // PolyType::SIMPLE is the ordinary RNS system, rescaling one prime at a
    // time; NTTAlgorithm::NORMAL (the default) is the NTT for the usual
    // cyclotomic ring.
    //==========================================================================
    SKGenParams skgen_params(log_degree, pel::TABLE3_HAMMING_WEIGHT);
    SKGenerator skgen(skgen_params);

    std::cout << "Generating keys ... " << std::flush;
    auto sk = skgen.genKey();

    paramsUtils::SwKeyGenParamsBuilder swkgen_builder;
    swkgen_builder.setRing(log_degree, PolyType::SIMPLE);
    swkgen_builder.setModUpPrimes(pel::TABLE3_QP_MAX_BITS, 0.0);
    auto swkgen_params = swkgen_builder.build(levels.mods, false);
    SwKeyGenerator swkgen(swkgen_params);

    auto relin_key = swkgen.genRelinKey(*sk);

    // A positive step is a left rotation. Steps double so that the sum tree
    // below covers all max_tok token positions in log2(max_tok) steps.
    std::set<i32> rot_steps;
    for (int k = 1; k < max_tok; k <<= 1)
        rot_steps.insert((i32)(k * stride));
    auto rot_keys = swkgen.genRotKeys(*sk, rot_steps);
    std::cout << "done (" << rot_steps.size() << " rotation keys)\n";

    // Encoder for messages and ciphertexts: slot encoding.
    EncodeParams ecd_params(PolyType::SIMPLE, log_degree, levels);
    EnDecoder encoder(ecd_params);

    // Encoder for the score table U. pcmm accepts coefficient-encoded
    // operands only, so this one sets coeff_encoding. NTTAlgorithm::NORMAL is
    // just the default, repeated to reach the fifth argument.
    EncodeParams mat_ecd_params(PolyType::SIMPLE, log_degree, levels,
                                /*ntt_alg=*/NTTAlgorithm::NORMAL,
                                /*coeff_encoding=*/true);
    EnDecoder mat_encoder(mat_ecd_params);

    DiscreteGaussian noise_dist(/*sigma=*/3.2);
    EncryptParams enc_params(noise_dist);
    EnDecryptor encryptor(enc_params);

    // Public-key encryption: the client holds sk, the server sees only
    // ciphertexts. sk is used below only for key generation and for the final
    // decryption. Public-key noise sits about sqrt(N) above secret-key noise.
    EncKeyGenParams enckeygen_params(noise_dist, PolyType::SIMPLE,
                                     levels.mods[levels.top()]);
    EncKeyGenerator enckeygen(enckeygen_params);
    auto enc_key = enckeygen.genKey(*sk);
    HomEval eval{HomEvalParams(levels)}; // arithmetic; needs the prime chain
    HomEvalFlexible eval_flex;           // batching and encoding-flag surgery

    // Level the one-hot vectors come out at: the indicator consumes exactly
    // `level`, so this is 1 and PCMM can follow directly.
    const u32 basis_out_level = enc_level - (u32)level; // 1

    //==========================================================================
    // (0) Score table U [C x K] and dataset  [offline]
    //==========================================================================
    // Footnote 2 of the paper counts preparing the table as a one-time offline
    // cost, so it is loaded and encoded once, outside the batch loop and
    // outside every timer. Multiplying a one-hot vector by U selects a column,
    // so U is the raw score table: no transform and no rescaling.
    auto V_colmajor = pel::load_colmajor_f64_bin(
        pel::filterPath(pel::SPAM_SCORE_COL_MAJ), (std::uint32_t)M_out,
        (std::uint32_t)K);

    Matrix<Real> u_mat(log_degree, (u32)M_out, (u32)K);
    for (int r = 0; r < M_out; ++r)
        for (int c = 0; c < K; ++c)
            u_mat.at((u32)r, (u32)c) =
                V_colmajor[(size_t)r + (size_t)M_out * (size_t)c];

    auto pu = IPtMatrix::make();
    mat_encoder.encode(u_mat, *pu, levels.mods[basis_out_level],
                       levels.scales[basis_out_level]);

    HomEvalMatrix eval_mat{HomEvalParams(levels)};

    auto emails = pel::load_email_codes(pel::filterPath(pel::SPAM_CODES));
    auto ref_preds = pel::load_csv_pred_ids(pel::filterPath(pel::SPAM_CSV));

    const int num_samples = (int)emails.size();
    const int num_batches = (num_samples + stride - 1) / stride;
    std::cout << "Samples: " << num_samples << ", batches: " << num_batches
              << "\n\n";

    //==========================================================================
    // Batch loop
    //==========================================================================
    const auto to_sec = [](const auto b, const auto e) {
        return std::chrono::duration<double>(e - b).count();
    };
    double enc_sec = 0, vecgen_sec = 0, padfix_sec = 0, linear_sec = 0,
           rot_sec = 0;
    int correct = 0, total = 0;
    bool checked = false;

    for (int batch = 0; batch < num_batches; ++batch) {
        const int s_start = batch * stride;
        const int s_end = std::min(s_start + stride, num_samples);
        std::cout << "[Batch " << batch << "] samples " << s_start << " .. "
                  << (s_end - 1) << "\n";

        //----------------------------------------------------------------------
        // (1) Build the queries. This is client-side work, so it is not
        //     timed. Slot (i + stride*k) takes the code of token k of example
        //     s_start+i as a plain real number. Padding slots get code 0 and
        //     are corrected in step (4).
        //----------------------------------------------------------------------
        std::vector<Message> msg;
        msg.reserve(num_blocks);
        for (int b = 0; b < num_blocks; ++b)
            msg.emplace_back(log_slots);
        Message pad_mask(log_slots);

        for (int b = 0; b < num_blocks; ++b) {
            for (int i = 0; i < stride; ++i) {
                const int eidx = s_start + i;
                for (int k = 0; k < max_tok; ++k) {
                    const size_t slot = (size_t)(i + stride * k);
                    const bool valid =
                        (eidx < s_end && k < emails[(size_t)eidx].num_tokens);
                    msg[(size_t)b][slot].real(
                        valid ? (double)emails[(size_t)eidx].codes[k][b] : 0.0);
                    msg[(size_t)b][slot].imag(0.0);
                    if (b == 0) {
                        pad_mask[slot].real(valid ? 0.0 : 1.0);
                        pad_mask[slot].imag(0.0);
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // (2) Encrypt. Only the block base is encrypted; pel::ICML24 fills the
        //     other p-1 ciphertexts of the block.
        //----------------------------------------------------------------------
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<Ptr<ICiphertext>> ctxt((size_t)K);
        for (int i = 0; i < K; ++i)
            ctxt[(size_t)i] = ICiphertext::make();

        for (int b = 0; b < num_blocks; ++b)
            compat::Encrypt(encoder, encryptor, *enc_key, msg[(size_t)b],
                         enc_level, *ctxt[(size_t)(b * p)]);
        const auto t1 = std::chrono::steady_clock::now();

        //----------------------------------------------------------------------
        // (3) Vector generation: one-hot vectors, one block per sub-table
        //     ctxt[b*p + j][s] ~= indicator(codes[token_at_s][b] == j)
        //----------------------------------------------------------------------
        for (int b = 0; b < num_blocks; ++b)
            pel::ICML24(eval, *relin_key, ctxt, b * p, p, iter_r,
                              iter_s);
        const auto t2 = std::chrono::steady_clock::now();

        //----------------------------------------------------------------------
        // (4) Padding fix: switch off the code-0 indicator wherever there is
        //     no real token. Encoding the mask is charged to this step, since
        //     it is part of the correction rather than offline preparation.
        //----------------------------------------------------------------------
        {
            auto pad_ptxt = IPlaintext::make();
            encoder.encode(pad_mask, *pad_ptxt, basis_out_level);
            for (int b = 0; b < num_blocks; ++b) {
                auto fixed = ICiphertext::make();
                eval.sub(*ctxt[(size_t)(b * p)], *pad_ptxt, *fixed);
                ctxt[(size_t)(b * p)] = std::move(fixed);
            }
        }
        const auto t3 = std::chrono::steady_clock::now();

        // pcmm needs all inputs at one level; also check that the level U was
        // encoded at is the level the indicators actually came out at.
        {
            const u32 lv0 = eval.getLevel(*ctxt[0]);
            for (int i = 1; i < K; ++i)
                if (eval.getLevel(*ctxt[(size_t)i]) != lv0) {
                    std::cerr << "[fatal] level mismatch at ctxt[" << i
                              << "]\n";
                    return 1;
                }
            if (!checked) {
                if (lv0 != basis_out_level) {
                    std::cerr << "[fatal] basis level " << lv0 << " != "
                              << basis_out_level
                              << " (the level U was encoded at)\n";
                    return 1;
                }
                checked = true;
            }
        }

        //----------------------------------------------------------------------
        // (5) Linear (PCMM) :  W[M_out x degree] = U[M_out x K] . V[K x degree]
        //----------------------------------------------------------------------
        const auto t4 = std::chrono::steady_clock::now();

        std::vector<const ICiphertext *> ct_ptrs;
        ct_ptrs.reserve((size_t)K);
        for (int i = 0; i < K; ++i)
            ct_ptrs.push_back(&*ctxt[(size_t)i]);

        auto batch_in = ICiphertext::make(EncType::BatchRLWE);
        eval_flex.batch(ct_ptrs, *batch_in);
        eval_flex.setDFT(*batch_in, false);

        auto cv = ICtMatrix::make();
        cv->moveFrom(*batch_in, (u32)K, degree);

        auto cw = ICtMatrix::make();
        eval_mat.pcmm(*pu, *cv, *cw);
        eval_mat.rescale(*cw, *cw);

        auto batch_out = ICiphertext::make(EncType::BatchRLWE);
        cw->moveTo(*batch_out);
        eval_flex.setDFT(*batch_out, true);

        std::vector<Ptr<ICiphertext>> ctxt_rst((size_t)M_out);
        std::vector<ICiphertext *> rst_ptrs;
        rst_ptrs.reserve((size_t)M_out);
        for (int m = 0; m < M_out; ++m) {
            ctxt_rst[(size_t)m] = ICiphertext::make();
            rst_ptrs.push_back(&*ctxt_rst[(size_t)m]);
        }
        eval_flex.unbatch(*batch_out, rst_ptrs);

        const auto t5 = std::chrono::steady_clock::now();

        if (batch == 0)
            std::cout << "  linear done, result level = "
                      << eval.getLevel(*ctxt_rst[0]) << ", rows = " << M_out
                      << "\n";

        //----------------------------------------------------------------------
        // (6) Sum over the token axis, as a tree of log2(max_tok) rotations.
        //     Afterwards slot i holds the total score of example s_start+i.
        //----------------------------------------------------------------------
        std::vector<Ptr<ICiphertext>> ctxt_sum((size_t)M_out);
        for (int m = 0; m < M_out; ++m) {
            ctxt_sum[(size_t)m] = ICiphertext::make();
            ctxt_rst[(size_t)m]->copyTo(*ctxt_sum[(size_t)m]);
            for (int k = 1; k < max_tok; k <<= 1) {
                const i32 step = (i32)(k * stride);
                auto tmp = ICiphertext::make();
                eval.rot(*ctxt_sum[(size_t)m], step, *tmp,
                         *rot_keys.at(step));
                auto acc = ICiphertext::make();
                eval.add(*ctxt_sum[(size_t)m], *tmp, *acc);
                ctxt_sum[(size_t)m] = std::move(acc);
            }
        }
        const auto t6 = std::chrono::steady_clock::now();

        if (batch == 0)
            std::cout << "  rotate-sum done, level = "
                      << eval.getLevel(*ctxt_sum[0])
                      << " (rotation consumes no level)\n";

        enc_sec += to_sec(t0, t1);
        vecgen_sec += to_sec(t1, t2);
        padfix_sec += to_sec(t2, t3);
        linear_sec += to_sec(t4, t5);
        rot_sec += to_sec(t5, t6);

        //----------------------------------------------------------------------
        // (7) Decrypt and compare with the plaintext model's predictions
        //----------------------------------------------------------------------
        std::vector<Message> dmsg((size_t)M_out);
        for (int m = 0; m < M_out; ++m) {
            auto dptxt = IPlaintext::make();
            encryptor.decrypt(*ctxt_sum[(size_t)m], *sk, *dptxt);
            encoder.decode(*dptxt, dmsg[(size_t)m]);
            dmsg[(size_t)m].to(Device::CPU);
        }

        for (int i = 0; i < stride; ++i) {
            const int eidx = s_start + i;
            if (eidx >= num_samples)
                break;
            const double s0 = dmsg[0][(size_t)i].real();
            const double s1 = dmsg[1][(size_t)i].real();
            const int pred = (s1 > s0) ? 1 : 0;
            if (pred == ref_preds[(size_t)eidx])
                ++correct;
            ++total;
        }

        std::cout << "  Running: " << correct << "/" << total << " ("
                  << std::fixed << std::setprecision(2)
                  << (100.0 * correct / total) << "%)\n\n";

        break; // one batch is enough to report per-example timings
    }

    //==========================================================================
    // Results
    //==========================================================================
    std::cout << "=== Accuracy vs plaintext pred_id: " << correct << "/"
              << total << " (" << std::fixed << std::setprecision(2)
              << (100.0 * correct / total) << "%) ===\n";

    std::cout << "\n=== Latency (batch 0, " << stride << " examples) ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Encrypt            (s): " << enc_sec << "\n";
    std::cout << "Vector generation  (s): " << vecgen_sec
              << "\n";
    std::cout << "Pad-fix            (s): " << padfix_sec
              << "\n";
    std::cout << "Linear             (s): " << linear_sec
              << "\n";
    std::cout << "Rotate-sum         (s): " << rot_sec << "\n";

    return 0;
} catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
}

int main() { return runExample(); }
