#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy


def total_ad_order(adxy_order_seq):
  ad_order = 0
  for adxy_order in adxy_order_seq:
    adx_order, ady_order = adxy_order
    ad_order += adx_order + ady_order
  return ad_order


def split_order(ad_order, n):

  def helper(remaining, parts, current):
    if parts == 1:
      if remaining > 0:
        splits.append(tuple(current + [remaining]))
      return

    for i in range(1, remaining - parts + 2):
      helper(remaining - i, parts - 1, current + [i])

  splits = []
  if n > 0 and ad_order >= n:
    helper(ad_order, n, [])
  return splits


def get_all_possible_adxy_order_seqs(ad_order, merge_equivalence=True):
  adxy_order_seqs = []
  for k in range(1, ad_order + 1):
    for split in split_order(ad_order, k):
      if (k % 2) == 0:
        adxy_order_seq = [(split[2 * i], split[2 * i + 1])
                          for i in range(k // 2)]
      else:
        tmp = tuple([0] + list(split))
        adxy_order_seq = [(tmp[2 * i], tmp[2 * i + 1])
                          for i in range(len(tmp) // 2)]

      if merge_equivalence:
        # Check the Jacobi equivalence.
        #
        # The Jacobi Identitiy:
        #   [X,[Y,Z]] - [Y,[X,Z]] - [[X,Y], Z] = 0
        # Let Z = [Y,X], then [[X,Y], Z] = 0 and:
        #   [X,[Y,  Z  ]] = [Y,[X,  Z  ]]
        #   [X,[Y,[Y,X]]] = [Y,[X,[Y,X]]]
        # (adX * adY^2).X = (adY * adX * adY).X
        #
        # So the pattern
        #     (... * adY^{k} * adX * adY)     --- patern 1
        # is equivalent to
        #     (... * adY^{k-1} * adX * adY^2) --- patern 2
        # when they action on X. We'll regard them as the same
        # and only keep patern 1.

        # if adxy_order_seq[-1][0] == 1 and adxy_order_seq[-1][1] == 2:
        #   continue
        if adxy_order_seq[-1][0] >= 1 and adxy_order_seq[-1][1] == 2:
          continue

      adxy_order_seqs.append(adxy_order_seq)
  return adxy_order_seqs


def check_split(split, adxy_order_seq):
  assert (sum(split) == total_ad_order(adxy_order_seq))
  segment = 0
  kl_tuple = []
  for adxy_order in adxy_order_seq:
    adx_order, ady_order = adxy_order
    local_order = adx_order + ady_order
    local_offset = 0

    while local_offset < local_order:
      if split[segment] + local_offset > local_order:
        return None
      else:
        if local_offset >= adx_order:
          kl_tuple.append((0, split[segment]))
        elif split[segment] + local_offset < adx_order:
          kl_tuple.append((split[segment], 0))
        else:
          kl_tuple.append((adx_order - local_offset,
                           split[segment] + local_offset - adx_order))
        local_offset += split[segment]
        segment += 1
  return kl_tuple


def get_all_valid_n_splits(adxy_order_seq, n):
  ad_order = total_ad_order(adxy_order_seq)
  valid_splits = []
  for split in split_order(ad_order, n):
    kl_tuple = check_split(split, adxy_order_seq)
    if kl_tuple:
      valid_splits.append(kl_tuple)
  return valid_splits


def get_bch_coeff_for_adxy_order_seq(adxy_order_seq, merge_equivalence=True):
  coeff_from_equivalence = 0.0

  # Check Jacobi equivalence. We'll ignore the pattern 2
  # and add the coeff for pattern 2 to pattern 1.
  # See the comments in get_all_possible_adxy_order_seqs()
  # for information about pattern 1 and pattern 2.
  if merge_equivalence:
    if adxy_order_seq[-1][0] == 1 and adxy_order_seq[-1][1] == 2:
      return 0.0
    elif adxy_order_seq[-1][0] == 1 and adxy_order_seq[-1][1] == 1:
      if len(adxy_order_seq) > 1:
        equiv_seq = copy.copy(adxy_order_seq)
        if equiv_seq[-2][1] > 1:
          equiv_seq[-2] = (equiv_seq[-2][0], equiv_seq[-2][1] - 1)
          equiv_seq[-1] = (1, 2)
        else:
          # Merge the last two elements
          equiv_seq[-2] = (equiv_seq[-2][0] + 1, 2)
          equiv_seq = equiv_seq[:-1]  # Remove the last element
        coeff_from_equivalence = get_bch_coeff_for_adxy_order_seq(
            equiv_seq, merge_equivalence=False)

  total_adx_order = 0
  for adx_order, _ in adxy_order_seq:
    total_adx_order += adx_order

  # print("get_bch_coeff_for_adxy_order_seq: {},    total_adx_order: {}".format(adxy_order_seq, total_adx_order))

  coeff = 0.0
  ad_order = total_ad_order(adxy_order_seq)
  for n in range(1, ad_order + 1):
    valid_n_splits = get_all_valid_n_splits(adxy_order_seq, n)
    coeff_for_split_n = 0.0
    for kl_tuple in valid_n_splits:
      sumk = 0
      item_coeff = 1.0
      for k, l in kl_tuple:
        sumk += k
        item_coeff *= (1. / math.factorial(k)) * (1. / math.factorial(l))
      assert (sumk == total_adx_order)
      # item_coeff *= (1. / (total_adx_order + 1))  # we can do this only once at last
      # print("    coeff for kl_tuple {}: {}".format(kl_tuple, item_coeff))
      coeff_for_split_n += item_coeff

    coeff_for_split_n *= ((-1)**n) / (n + 1)
    # print("  total coeff for splits {}: {}".format(valid_n_splits, coeff_for_split_n))
    coeff += coeff_for_split_n
  coeff *= (1. / (total_adx_order + 1))
  coeff += coeff_from_equivalence
  # print("total coeff for {}: {}".format(adxy_order_seq, coeff))
  return coeff


def bch_high_order_ad_terms(max_ad_order, ignore_term_with_zero_coeff=True):
  output = {}
  for ad_order in range(1, max_ad_order + 1):
    adxy_order_seq_to_coeff = {}
    for adxy_order_seq in get_all_possible_adxy_order_seqs(ad_order):
      coeff = get_bch_coeff_for_adxy_order_seq(adxy_order_seq)
      if ignore_term_with_zero_coeff:
        if math.fabs(coeff) < 1e-15:
          continue
      adxy_order_seq_to_coeff[tuple(adxy_order_seq)] = coeff
    output[ad_order] = adxy_order_seq_to_coeff
  return output


def gen_bch_cpp_helper(max_order, jacobian_max_order=20):
  max_ad_order = max_order - 1
  high_order_terms = bch_high_order_ad_terms(max_ad_order,
                                             ignore_term_with_zero_coeff=True)
  code = "#pragma once\n\n"
  code += "#include <vector>\n"
  code += "#include <map>\n"
  code += "\n"
  # code += "using BchAdxyOrderSeqAndWeight = std::pair<std::vector<std::pair<int, int>>, double>;\n"
  # code += "using BchAdxyOrderSeqsAndWeights = std::vector<BchAdxyOrderSeqAndWeight>;\n\n"
  code += "using BchAdxyOrderSeqsAndWeights = std::map<std::vector<std::pair<int, int>>, double>;\n\n"

  def to_cpp(adxy_order_seq):
    cpp_str = ""
    for adxy_order in adxy_order_seq:
      adx_order, ady_order = adxy_order
      if cpp_str:
        cpp_str += ", "
      cpp_str += "{{{}, {}}}".format(adx_order, ady_order)
    cpp_str = "{{ {} }}".format(cpp_str)
    return cpp_str

  for ad_order in range(1, max_ad_order + 1):
    code += "// adxy order sequences with total ad_order {}\n".format(ad_order)
    code += "inline const BchAdxyOrderSeqsAndWeights _adxy_order_seqs_and_coeffs_{} = {{\n".format(
        ad_order)
    local_str = ""
    for adxy_order_seq, coeff in high_order_terms[ad_order].items():
      if local_str:
        local_str += ",\n"
      local_str += "  {{ {}, 1./{} }}".format(to_cpp(adxy_order_seq),
                                              1. / coeff)
    code += local_str
    code += "\n};\n\n"

  code += "inline const std::map<int, BchAdxyOrderSeqsAndWeights> _adxy_order_seqs_and_coeffs = {\n"
  code += "  {0, {}} /*placeholder for ad_order 0*/"
  for ad_order in range(1, max_ad_order + 1):
    code += ",\n  {{{}, _adxy_order_seqs_and_coeffs_{}}}".format(
        ad_order, ad_order)
  code += "\n};\n\n"

  code += "inline const std::vector<double> _inv_left_jacobian_coeffs = {\n"
  code += "  1  /*placeholder for adX_order 0*/"
  for jacobian_order in range(1, jacobian_max_order + 1):
    adxy_order_seq = [(0, jacobian_order)]
    coeff = get_bch_coeff_for_adxy_order_seq(adxy_order_seq)
    if math.fabs(coeff) < 1e-15:
      code += ",\n  0  /*order {}*/".format(jacobian_order)
    else:
      code += ",\n  1./{}  /*order {}*/".format(1. / coeff, jacobian_order)
  code += "\n};\n\n"

  return code


############## main ##############
if __name__ == "__main__":
  import sys
  max_order = 11
  jacobian_max_order = 20
  if len(sys.argv) > 1:
    max_order = int(sys.argv[1])
  print(gen_bch_cpp_helper(max_order, jacobian_max_order))
