cmake_minimum_required(VERSION 3.23)
project(
  PrivateEmbeddingLookup
  VERSION 1.0.0
  DESCRIPTION "Private Embedding Lookup with Encrypted Compact Queries"
  LANGUAGES CXX)

# When built on its own, locate the HEaaN2 devkit here. When this directory is
# pulled in with add_subdirectory(), the parent has already done it.
if(PROJECT_IS_TOP_LEVEL)
  list(PREPEND CMAKE_PREFIX_PATH "/home/user/devkit")
  find_package(HEaaN2 REQUIRED)
endif()

find_package(OpenMP REQUIRED)

# Data locations, passed in at compile time so that no absolute path is baked
# into the source:
#   data/fasttext_spam/      FastText classification experiment (Section 6.2)
#   data/{50d,300d,gpt2}/    embedding tables and ground truth (Table 3)
set(PEL_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/data")

set(PEL_EXAMPLES
    ive_pcmm
    ive_fasttext_spam
    ICML24_pcmm
    ICML24_fasttext_spam)

foreach(tgt IN LISTS PEL_EXAMPLES)
  add_executable(${tgt} ${tgt}.cpp)
  target_link_libraries(${tgt} PRIVATE HEaaN2::HEaaN2 OpenMP::OpenMP_CXX)
  target_compile_definitions(${tgt} PRIVATE
    PEL_DATA_DIR="${PEL_DATA_ROOT}/fasttext_spam"
    PEL_TABLE_DIR="${PEL_DATA_ROOT}")
endforeach()