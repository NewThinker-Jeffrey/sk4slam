#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy
from bch_helper import *


# For test
class so3(object):

  def __init__(self, x, y, z):
    self.d = (x, y, z)

  def __mul__(self, scalar):
    return so3(*[i * scalar for i in self.d])

  def __add__(self, other):
    return so3(*[i + j for i, j in zip(self.d, other.d)])

  def __sub__(self, other):
    return so3(*[i - j for i, j in zip(self.d, other.d)])

  def cross(X, Y):
    return so3(X.d[1] * Y.d[2] - X.d[2] * Y.d[1],
               X.d[2] * Y.d[0] - X.d[0] * Y.d[2],
               X.d[0] * Y.d[1] - X.d[1] * Y.d[0])


def BCH(X, Y, bracket, max_order=6):
  max_ad_order = max_order - 1
  high_order_terms = bch_high_order_ad_terms(max_ad_order,
                                             ignore_term_with_zero_coeff=True)
  Z = X + Y
  b = bracket
  for ad_order in range(1, max_ad_order + 1):
    for adxy_order_seq, coeff in high_order_terms[ad_order].items():
      item = X
      for adxy_order in reversed(adxy_order_seq):
        adx_order, ady_order = adxy_order
        # apply Y first, since Y is on the right
        if ady_order > 0:
          for i in range(ady_order):
            item = b(Y, item)
        # then X
        if adx_order > 0:
          for i in range(adx_order):
            item = b(X, item)
      Z += item * coeff
  return Z


def BCH_wiki(X, Y, bracket, max_order=6):
  Z = X + Y
  # https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
  # https://wikimedia.org/api/rest_v1/media/math/render/svg/936fe5484a33c662e0f0d7db978f00178cc5a2e9
  b = bracket
  if max_order >= 2:
    Z += b(X, Y) * (1. / 2)
  if max_order >= 3:
    Z += b(X, b(X, Y)) * (1. / 12)
    Z += b(Y, b(Y, X)) * (1. / 12)
  if max_order >= 4:
    Z -= b(Y, b(X, b(X, Y))) * (1. / 24)

  if max_order >= 5:
    Z -= b(Y, b(Y, b(Y, b(Y, X)))) * (1. / 720)
    Z -= b(X, b(X, b(X, b(X, Y)))) * (1. / 720)
    Z += b(X, b(Y, b(Y, b(Y, X)))) * (1. / 360)
    Z += b(Y, b(X, b(X, b(X, Y)))) * (1. / 360)
    Z += b(X, b(Y, b(X, b(Y, X)))) * (1. / 120)
    Z += b(Y, b(X, b(Y, b(X, Y)))) * (1. / 120)
  if max_order >= 6:
    Z += b(X, b(Y, b(X, b(Y, b(X, Y))))) * (1. / 240)
    Z += b(X, b(Y, b(X, b(X, b(X, Y))))) * (1. / 720)
    Z -= b(X, b(X, b(Y, b(Y, b(X, Y))))) * (1. / 720)
    Z += b(X, b(Y, b(Y, b(Y, b(X, Y))))) * (1. / 1440)
    Z -= b(X, b(X, b(Y, b(X, b(X, Y))))) * (1. / 1440)
  return Z


def print_bch(max_order, ignore_term_with_zero_coeff=True, bracket_form=True):
  max_ad_order = max_order - 1
  high_order_terms = bch_high_order_ad_terms(max_ad_order,
                                             ignore_term_with_zero_coeff)
  print("BCH(X,Y) = \n         X + Y")

  def to_formula(adxy_order_seq, bracket_form):
    formula_str = ""

    def left_append_factor(formula_str, factor_id, pow, bracket_form):
      if bracket_form:
        factor_id_to_str = ["X", "Y"]
        if not formula_str:
          formula_str = "X"
        for i in range(pow):
          formula_str = "[{}, {}]".format(factor_id_to_str[factor_id],
                                          formula_str)
      else:
        factor_id_to_str = ["adX", "adY"]
        if pow == 1:
          cur_factor_str = "{}".format(factor_id_to_str[factor_id])
        else:
          cur_factor_str = "{}^{}".format(factor_id_to_str[factor_id], pow)
        if formula_str:
          formula_str = " * " + formula_str
        formula_str = cur_factor_str + formula_str

      return formula_str

    for adxy_order in reversed(adxy_order_seq):
      adx_order, ady_order = adxy_order
      # apply Y first, since Y is on the right
      if ady_order > 0:
        formula_str = left_append_factor(formula_str, 1, ady_order,
                                         bracket_form)
      # then X
      if adx_order > 0:
        formula_str = left_append_factor(formula_str, 0, adx_order,
                                         bracket_form)

    return formula_str

  for ad_order in range(1, max_ad_order + 1):
    order_str = ""
    for adxy_order_seq, coeff in high_order_terms[ad_order].items():
      if coeff == 2**(-55):
        continue
      if not order_str:
        order_str += " {{order {}}} + ".format(ad_order + 1)
      else:
        order_str += "  +\n             "
      if coeff == 0:
        displayed_coeff = "0"
      else:
        displayed_coeff = "1./({:.2f})".format(1. / coeff)

      if bracket_form:
        order_str += " {} * {}".format(
            displayed_coeff, to_formula(adxy_order_seq, bracket_form))
      else:
        order_str += " {} * ({}).X".format(
            displayed_coeff, to_formula(adxy_order_seq, bracket_form))

    print(order_str)

  print("")


############## Unit tests ##############

import unittest


class TestSplitOrder(unittest.TestCase):

  def test_split_order(self):
    print("******* TestSplitOrder *******")
    self.assertEqual(sorted(split_order(4, 2)), sorted([(1, 3), (2, 2),
                                                        (3, 1)]))
    self.assertEqual(sorted(split_order(5, 2)),
                     sorted([(1, 4), (2, 3), (3, 2), (4, 1)]))
    self.assertEqual(
        sorted(split_order(6, 3)),
        sorted([(1, 1, 4), (1, 2, 3), (1, 3, 2), (1, 4, 1), (2, 1, 3),
                (2, 2, 2), (2, 3, 1), (3, 1, 2), (3, 2, 1), (4, 1, 1)]))
    self.assertEqual(split_order(3, 3), [(1, 1, 1)])
    self.assertEqual(split_order(3, 4), [])
    self.assertEqual(
        split_order(10, 4),
        sorted([(1, 1, 1, 7), (1, 1, 2, 6), (1, 1, 3, 5), (1, 1, 4, 4),
                (1, 1, 5, 3), (1, 1, 6, 2), (1, 1, 7, 1), (1, 2, 1, 6),
                (1, 2, 2, 5), (1, 2, 3, 4), (1, 2, 4, 3), (1, 2, 5, 2),
                (1, 2, 6, 1), (1, 3, 1, 5), (1, 3, 2, 4), (1, 3, 3, 3),
                (1, 3, 4, 2), (1, 3, 5, 1), (1, 4, 1, 4), (1, 4, 2, 3),
                (1, 4, 3, 2), (1, 4, 4, 1), (1, 5, 1, 3), (1, 5, 2, 2),
                (1, 5, 3, 1), (1, 6, 1, 2), (1, 6, 2, 1), (1, 7, 1, 1),
                (2, 1, 1, 6), (2, 1, 2, 5), (2, 1, 3, 4), (2, 1, 4, 3),
                (2, 1, 5, 2), (2, 1, 6, 1), (2, 2, 1, 5), (2, 2, 2, 4),
                (2, 2, 3, 3), (2, 2, 4, 2), (2, 2, 5, 1), (2, 3, 1, 4),
                (2, 3, 2, 3), (2, 3, 3, 2), (2, 3, 4, 1), (2, 4, 1, 3),
                (2, 4, 2, 2), (2, 4, 3, 1), (2, 5, 1, 2), (2, 5, 2, 1),
                (2, 6, 1, 1), (3, 1, 1, 5), (3, 1, 2, 4), (3, 1, 3, 3),
                (3, 1, 4, 2), (3, 1, 5, 1), (3, 2, 1, 4), (3, 2, 2, 3),
                (3, 2, 3, 2), (3, 2, 4, 1), (3, 3, 1, 3), (3, 3, 2, 2),
                (3, 3, 3, 1), (3, 4, 1, 2), (3, 4, 2, 1), (3, 5, 1, 1),
                (4, 1, 1, 4), (4, 1, 2, 3), (4, 1, 3, 2), (4, 1, 4, 1),
                (4, 2, 1, 3), (4, 2, 2, 2), (4, 2, 3, 1), (4, 3, 1, 2),
                (4, 3, 2, 1), (4, 4, 1, 1), (5, 1, 1, 3), (5, 1, 2, 2),
                (5, 1, 3, 1), (5, 2, 1, 2), (5, 2, 2, 1), (5, 3, 1, 1),
                (6, 1, 1, 2), (6, 1, 2, 1), (6, 2, 1, 1), (7, 1, 1, 1)]))


class TestGetSplitsFromXYOrderSeq(unittest.TestCase):
  # only for test
  def get_all_valid_splits(self, adxy_order_seq):
    ad_order = total_ad_order(adxy_order_seq)
    valid_splits = []
    for n in range(1, ad_order + 1):
      valid_splits += get_all_valid_n_splits(adxy_order_seq, n)
    return valid_splits

  def test_get_all_valid_splits(self):
    print("******* TestGetSplitsFromXYOrderSeq *******")
    adxy_order_seq = [
        (0, 2),
    ]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))
    # self.assertEqual(
    #     sorted(self.get_all_valid_splits(adxy_order_seq)),
    #     sorted([(1, 1), (2,)])
    # )

    adxy_order_seq = [
        (1, 1),
    ]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))
    # self.assertEqual(
    #     sorted(self.get_all_valid_splits(adxy_order_seq)),
    #     sorted([(1, 1), (2,)])
    # )

    adxy_order_seq = [
        (0, 3),
    ]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [
        (2, 1),
    ]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(1, 1), (1, 1)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(0, 1), (1, 1)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(0, 1), (1, 1)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(0, 1), (1, 1)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(1, 3)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(2, 2)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))

    adxy_order_seq = [(0, 1), (2, 1)]
    print("adxy_order_seq={}, kl_tuples={}".format(
        adxy_order_seq, sorted(self.get_all_valid_splits(adxy_order_seq))))


class TestGetBCHCoeffForXYOrderSeq(unittest.TestCase):

  def test_get_bch_coeff_for_adxy_order_seq(self):
    print("******* TestGetBCHCoeffForXYOrderSeq *******")
    adxy_order_seq = [
        (0, 2),
    ]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))
    self.assertAlmostEqual(get_bch_coeff_for_adxy_order_seq(adxy_order_seq),
                           1. / 12)

    adxy_order_seq = [
        (1, 1),
    ]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))
    self.assertAlmostEqual(get_bch_coeff_for_adxy_order_seq(adxy_order_seq),
                           -1. / 12)

    # adxy_order_seq = [(2, 1),]
    # print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(adxy_order_seq, 1./get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))
    # self.assertAlmostEqual(get_bch_coeff_for_adxy_order_seq(adxy_order_seq), 1./12)

    # adxy_order_seq = [(0, 3),]
    # print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(adxy_order_seq, 1./get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))
    # self.assertAlmostEqual(get_bch_coeff_for_adxy_order_seq(adxy_order_seq), 1./12)

    adxy_order_seq = [(0, 1), (1, 1)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))

    adxy_order_seq = [(0, 1), (1, 1)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))

    adxy_order_seq = [(1, 1), (1, 1)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))

    adxy_order_seq = [(0, 2), (1, 1)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))

    adxy_order_seq = [(1, 3)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))

    adxy_order_seq = [(2, 2)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))

    adxy_order_seq = [(0, 1), (2, 1)]
    print("adxy_order_seq={}, bch_coeff=1./{:.2f}".format(
        adxy_order_seq, 1. / get_bch_coeff_for_adxy_order_seq(adxy_order_seq)))


class TestGenerateXYOrderSeq(unittest.TestCase):

  def test_get_all_possible_adxy_order_seqs(self):
    print("******* TestGenerateXYOrderSeq *******")
    ad_order = 1
    adxy_order_seqs = get_all_possible_adxy_order_seqs(ad_order)
    print("ad_order {}: {}".format(ad_order, adxy_order_seqs))

    ad_order = 2
    adxy_order_seqs = get_all_possible_adxy_order_seqs(ad_order)
    print("ad_order {}: {}".format(ad_order, adxy_order_seqs))

    ad_order = 3
    adxy_order_seqs = get_all_possible_adxy_order_seqs(ad_order)
    print("ad_order {}: {}".format(ad_order, adxy_order_seqs))

    ad_order = 4
    adxy_order_seqs = get_all_possible_adxy_order_seqs(ad_order)
    print("ad_order {}: {}".format(ad_order, adxy_order_seqs))

    ad_order = 5
    adxy_order_seqs = get_all_possible_adxy_order_seqs(ad_order)
    print("ad_order {}: {}".format(ad_order, adxy_order_seqs))


class TestBCHFor_so3(unittest.TestCase):

  def test_so3(self):
    print("******* TestBCHFor_so3 *******")

    X = so3(0.42, 0.12, 0.34)
    Y = so3(0.12, 0.44, 0.29)
    X *= 5
    Y *= 5

    b = so3.cross
    b(X, b(X, b(Y, b(Y, X)))).d
    print("test [X, [X, [Y, [Y, X]]]]: {}\n".format(
        b(X, b(X, b(Y, b(Y, X)))).d))
    print("test [Y, [X, [X, [Y, X]]]]: {}\n".format(
        b(Y, b(X, b(X, b(Y, X)))).d))

    print("test [Y, [X, [Y, X]]]]: {}\n".format(b(Y, b(X, b(Y, X))).d))
    print("test [X, [Y, [Y, X]]]]: {}\n".format(b(X, b(Y, b(Y, X))).d))

    XY_order = 4
    Zwiki = BCH_wiki(X, Y, so3.cross, XY_order)
    Z = BCH(X, Y, so3.cross, XY_order)
    error = Z - Zwiki
    print("XY_order={}\nZwiki = {}\nZ     = {}\nerror = {}\n".format(
        XY_order, Zwiki.d, Z.d, error.d))

    XY_order = 5
    Zwiki = BCH_wiki(X, Y, so3.cross, XY_order)
    Z = BCH(X, Y, so3.cross, XY_order)
    error = Z - Zwiki
    print("XY_order={}\nZwiki = {}\nZ     = {}\nerror = {}\n".format(
        XY_order, Zwiki.d, Z.d, error.d))

    XY_order = 6
    Zwiki = BCH_wiki(X, Y, so3.cross, XY_order)
    Z = BCH(X, Y, so3.cross, XY_order)
    error = Z - Zwiki
    print("XY_order={}\nZwiki = {}\nZ     = {}\nerror = {}\n".format(
        XY_order, Zwiki.d, Z.d, error.d))


############## main ##############

if __name__ == "__main__":
  print_bch(7, ignore_term_with_zero_coeff=True, bracket_form=True)
  # print_bch(7, ignore_term_with_zero_coeff=True, bracket_form=False)
  # unittest.main()
