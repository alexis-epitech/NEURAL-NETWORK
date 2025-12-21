#!/usr/bin/env bash
set -euo pipefail

mkdir -p training_data

cat dataset/checkmate/10_pieces.txt dataset/check/10_pieces.txt dataset/nothing/10_pieces.txt | shuf > training_data/train_10p_mixed.txt
cat dataset/checkmate/20_pieces.txt dataset/check/20_pieces.txt dataset/nothing/20_pieces.txt | shuf > training_data/train_20p_mixed.txt
cat dataset/checkmate/many_pieces.txt dataset/check/many_pieces.txt dataset/nothing/many_pieces.txt | shuf > training_data/train_many_mixed.txt

checkmate_tmp="$(mktemp)"
check_tmp="$(mktemp)"
nothing_tmp="$(mktemp)"

cat dataset/checkmate/*.txt > "${checkmate_tmp}"
cat dataset/check/*.txt > "${check_tmp}"
cat dataset/nothing/*.txt > "${nothing_tmp}"

n_checkmate=$(wc -l < "${checkmate_tmp}")
n_check=$(wc -l < "${check_tmp}")
n_nothing=$(wc -l < "${nothing_tmp}")

min_count="${n_checkmate}"
if [ "${n_check}" -lt "${min_count}" ]; then min_count="${n_check}"; fi
if [ "${n_nothing}" -lt "${min_count}" ]; then min_count="${n_nothing}"; fi

cat \
  <(shuf -n "${min_count}" "${checkmate_tmp}") \
  <(shuf -n "${min_count}" "${check_tmp}") \
  <(shuf -n "${min_count}" "${nothing_tmp}") \
  | shuf > training_data/train_all_balanced.txt

rm -f "${checkmate_tmp}" "${check_tmp}" "${nothing_tmp}"

echo "Done."
