#pragma once
//==============================================================================
// Dataset loaders for the end-to-end encrypted inference experiments of
// Section 6.2.
//
// A document is represented by up to FILTER_MAX_TOKENS tokens, and each token
// by FILTER_NUM_DIGITS sub-table indices in [0, FILTER_P). Everything the
// server needs is prepared offline by the preprocessing script:
//
//   codes (.bin)         one record per example
//                          uint32 num_tokens
//                          FILTER_MAX_TOKENS * FILTER_NUM_DIGITS uint8 codes,
//                            zero-padded past num_tokens
//                          uint8 label
//                        preceded by a uint32 example count.
//
//   score table (.bin)   (C x ell*p) column-major float64 with a uint32 rows,
//                        uint32 cols header. The classifier weights are folded
//                        into the table, so entry [c][l*p + code] is the score
//                        that token code contributes to class c.
//                          _col_maj                 raw table, for the
//                                                   one-hot baseline
//                          _transformed_colmaj_f64  M_L = table . D^T, for IVE
//
//   predictions (.csv)   the plaintext model's own output; only the pred_id
//                        column (0/1) is read, to check that encrypted
//                        inference agrees with it.
//==============================================================================
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pel {

//------------------------------------------------------------------------------
// Constants fixed by the data format; they must match the preprocessing script
//------------------------------------------------------------------------------
inline constexpr int FILTER_MAX_TOKENS = 128; ///< tokens kept per example
inline constexpr int FILTER_NUM_DIGITS = 4;   ///< ell, sub-tables per token
inline constexpr int FILTER_NUM_CLASSES = 2;  ///< C, number of classes
inline constexpr int FILTER_P = 256;          ///< p, sub-table size

/// @brief One example: its token codes and its ground-truth label.
/// @details codes[k][l] is sub-table index l of token k, zero-padded beyond
/// num_tokens.
struct EmailCodes {
    int num_tokens = 0;
    int label = 0;
    std::uint8_t codes[FILTER_MAX_TOKENS][FILTER_NUM_DIGITS] = {};
};

/// @brief Reads a codes file.
inline std::vector<EmailCodes> load_email_codes(const std::string &path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
        throw std::runtime_error("Cannot open: " + path);

    std::uint32_t n = 0;
    fin.read(reinterpret_cast<char *>(&n), 4);
    if (!fin)
        throw std::runtime_error("Failed to read sample count: " + path);

    std::vector<EmailCodes> emails(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t nt = 0;
        fin.read(reinterpret_cast<char *>(&nt), 4);
        emails[i].num_tokens = (int)nt;
        fin.read(reinterpret_cast<char *>(emails[i].codes),
                 FILTER_MAX_TOKENS * FILTER_NUM_DIGITS);
        std::uint8_t lab = 0;
        fin.read(reinterpret_cast<char *>(&lab), 1);
        emails[i].label = (int)lab;
        if (!fin)
            throw std::runtime_error("Truncated codes file: " + path);
    }
    return emails;
}

/// @brief Reads the pred_id column of a predictions CSV, in row order.
inline std::vector<int> load_csv_pred_ids(const std::string &path) {
    std::ifstream fin(path);
    if (!fin)
        throw std::runtime_error("Cannot open: " + path);

    std::string header;
    std::getline(fin, header);

    int pred_col = -1;
    {
        std::istringstream ss(header);
        std::string col;
        for (int c = 0; std::getline(ss, col, ','); ++c)
            if (col == "pred_id") {
                pred_col = c;
                break;
            }
    }
    if (pred_col < 0)
        throw std::runtime_error("pred_id column not found in CSV: " + path);

    std::vector<int> preds;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::string tok;
        for (int c = 0; c <= pred_col; ++c)
            std::getline(ss, tok, ',');
        preds.push_back(std::stoi(tok));
    }
    return preds;
}

//------------------------------------------------------------------------------
// File locations
//
// The data ships alongside this example, in my_example/data/. CMake passes the
// directory in as PEL_DATA_DIR at compile time (see examples/CMakeLists.txt),
// so no absolute path is baked into the source and the tree can be moved.
//------------------------------------------------------------------------------
#ifndef PEL_DATA_DIR
#error "PEL_DATA_DIR is not defined; see CMakeLists.txt"
#endif

/// @brief Directory holding the Enron-Spam data.
inline const char *filterDatasetBase() { return PEL_DATA_DIR "/"; }

/// @brief Absolute path of one data file.
inline std::string filterPath(const char *file_name) {
    return std::string(filterDatasetBase()) + file_name;
}

/// @brief Raw score table, for the one-hot baseline.
inline const char *SPAM_SCORE_COL_MAJ = "spam_C2_L4_p256_col_maj.bin";

/// @brief DCT-transformed score table M_L, for IVE.
inline const char *SPAM_SCORE_TRANSFORMED =
    "spam_C2_L4_p256_transformed_colmaj_f64.bin";

/// @brief Token codes and labels, one record per example.
inline const char *SPAM_CODES = "spam_codes_l4_p256.bin";

/// @brief The plaintext model's predictions, used as the accuracy reference.
inline const char *SPAM_CSV = "predictions_maxtokens128_d50_l4_p256.csv";

} // namespace pel
