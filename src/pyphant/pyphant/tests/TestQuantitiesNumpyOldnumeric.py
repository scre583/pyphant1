# -*- coding: utf-8 -*-

# Copyright (c) 2015, Servicegroup Scientific Information Processing, FMF
# (servicegruppe.wissinfo@fmf.uni-freiburg.de)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the Freiburg Materials Research Center,
#   University of Freiburg nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import unittest
from pyphant.quantities import Quantity, _unit_table, PhysicalUnit,\
                               NumberDict, _base_names
import numpy as np
import numpy.oldnumeric


def int_sum_new(a, axis=0):
    return np.add.reduce(a, axis)


def int_sum_old(a, axis=0):
    return numpy.oldnumeric.add.reduce(a, axis)


def zeros_st_new(shape, other):
    return np.zeros(shape, dtype=other.dtype)


def zeros_st_old(shape, other):
    return numpy.oldnumeric.zeros(shape, dtype=other.dtype)


def sin_new(quantity):
    if quantity.unit.isAngle():
        return np.sin(quantity.value * \
                      quantity.unit.conversionFactorTo(_unit_table['rad'])
                      )
    else:
        raise TypeError('Argument of sin must be an angle')


def sin_old(quantity):
    if quantity.unit.isAngle():
        return numpy.oldnumeric.sin(quantity.value * \
                                    quantity.unit.conversionFactorTo(
                                        _unit_table['rad']
                                        )
                                    )
    else:
        raise TypeError('Argument of sin must be an angle')


def cos_new(quantity):
    if quantity.unit.isAngle():
        return np.cos(quantity.value * \
                      quantity.unit.conversionFactorTo(_unit_table['rad'])
                      )
    else:
        raise TypeError('Argument of sin must be an angle')


def cos_old(quantity):
    if quantity.unit.isAngle():
        return numpy.oldnumeric.cos(quantity.value * \
                                    quantity.unit.conversionFactorTo(
                                        _unit_table['rad']
                                        )
                                    )
    else:
        raise TypeError('Argument of sin must be an angle')


def tan_new(quantity):
    if quantity.unit.isAngle():
        return np.tan(quantity.value * \
                      quantity.unit.conversionFactorTo(_unit_table['rad'])
                      )
    else:
        raise TypeError('Argument of sin must be an angle')


def tan_old(quantity):
    if quantity.unit.isAngle():
        return numpy.oldnumeric.tan(quantity.value * \
                                    quantity.unit.conversionFactorTo(
                                        _unit_table['rad']
                                        )
                                    )
    else:
        raise TypeError('Argument of sin must be an angle')


def pow_new(physunit, other):
    if physunit.offset != 0:
        raise TypeError('cannot exponentiate units with non-zero offset')
    if isinstance(other, int):
        return PhysicalUnit(other * physunit.names,
                            pow(physunit.factor, other),
                            map(lambda x, p=other: x * p, physunit.powers))
    if (isinstance(other, float)) and (other != 0.):
        inv_exp = 1. / other
        rounded = round(inv_exp)
        if abs(inv_exp - rounded) < 1.e-10:
            if reduce(lambda a, b: a and b,
                      map(lambda x, e=rounded: x % e == 0, physunit.powers)):
                f = pow(physunit.factor, other)
                p = map(lambda x, p=rounded: x / p, physunit.powers)
                if reduce(lambda a, b: a and b,
                          map(lambda x, e=rounded: x % e == 0,
                              physunit.names.values())):
                    names = physunit.names / rounded
                else:
                    names = NumberDict()
                    if f != 1.:
                        names[str(f)] = 1
                    for i in range(len(p)):
                        names[_base_names[i]] = p[i]
                return PhysicalUnit(names, f, p)
            else:
                raise TypeError('Illegal exponent')
    raise TypeError('Only integer and inverse integer exponents allowed')


def pow_old(physunit, other):
    if physunit.offset != 0:
        raise TypeError('cannot exponentiate units with non-zero offset')
    if isinstance(other, int):
        return PhysicalUnit(other * physunit.names,
                            pow(physunit.factor, other),
                            map(lambda x, p=other: x * p, physunit.powers))
    if isinstance(other, float):
        inv_exp = 1. / other
        rounded = int(numpy.oldnumeric.floor(inv_exp + 0.5))
        if abs(inv_exp - rounded) < 1.e-10:
            if reduce(lambda a, b: a and b,
                      map(lambda x, e=rounded: x % e == 0, physunit.powers)):
                f = pow(physunit.factor, other)
                p = map(lambda x, p=rounded: x / p, physunit.powers)
                if reduce(lambda a, b: a and b,
                          map(lambda x, e=rounded: x % e == 0,
                              physunit.names.values())):
                    names = physunit.names / rounded
                else:
                    names = NumberDict()
                    if f != 1.:
                        names[str(f)] = 1
                    for i in range(len(p)):
                        names[_base_names[i]] = p[i]
                return PhysicalUnit(names, f, p)
            else:
                raise TypeError('Illegal exponent')
    raise TypeError('Only integer and inverse integer exponents allowed')


def round_new(x):
    if np.greater(x, 0.):
        return np.floor(x)
    else:
        return np.ceil(x)


def round_old(x):
    if numpy.oldnumeric.greater(x, 0.):
        return numpy.oldnumeric.floor(x)
    else:
        return numpy.oldnumeric.ceil(x)


class TestQuantityNumpyOldnumeric(unittest.TestCase):
    def setUp(self):
        self.arrays = []
        self.arrays.append([1, 2, 3])
        self.arrays.append([1.9, -2.2, 3.4])
        self.matrix = [[3.8, 4.5],
                       [-3.4, 546.6]]
        self.shapes = []
        self.shapes.append((5, np.array([6])))
        self.shapes.append((5, np.array([6.])))
        self.shapes.append(((3, 2), np.array([6])))
        self.shapes.append(((3, 2), np.array([6.])))
        self.angles = []
        self.angles.append(Quantity('1.4 rad'))
        self.angles.append(Quantity('2 rad'))
        self.angles.append(Quantity('30 deg'))
        self.angles.append(Quantity('-40 deg'))
        self.length = Quantity('5.3 m')
        self.phyunit = PhysicalUnit('m', 1., [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.phyunit **= 5
        self.exps = [0, 1, 2, 1. / 5, 1. / (5. + 1e-12), 1. / (5. - 1e-12),
                     -1. / (5. + 1e-12), -1. / (5. - 1e-12)]
        self.badexps = [0., 2., 1. / 6, 1. / (5. + 1e-8), 1. / (5. - 1e-8),
                        -1. / (5. + 1e-8), -1. / (5. - 1e-8)]
        self.values = [0, 1, 0., 1e-10, 4.049, 10.05, 3 - 1e-10]
        self.values += [-1, -1e-10, -4.049, -10.05, -(3 - 1e-10)]

    def testint_sum(self):
        for array in self.arrays:
            self.assertAlmostEqual(int_sum_new(array), int_sum_old(array))
        self.assertAlmostEqual(int_sum_new(self.matrix).tolist(),
                               int_sum_old(self.matrix).tolist())

    def testzero_st(self):
        for (shape, other) in self.shapes:
            self.assertEqual(zeros_st_new(shape, other).tolist(),
                             zeros_st_old(shape, other).tolist())

    def testsin(self):
        for angle in self.angles:
            self.assertAlmostEqual(sin_new(angle), sin_old(angle))

    def testsin_error(self):
        with self.assertRaises(TypeError):
            sin_new(self.length)
            sin_old(self.length)

    def testcos(self):
        for angle in self.angles:
            self.assertAlmostEqual(cos_new(angle), cos_old(angle))

    def testcos_error(self):
        with self.assertRaises(TypeError):
            cos_new(self.length)
            cos_old(self.length)

    def testtan(self):
        for angle in self.angles:
            self.assertAlmostEqual(tan_new(angle), tan_old(angle))

    def testtan_error(self):
        with self.assertRaises(TypeError):
            tan_new(self.length)
            tan_old(self.length)

    def testpow(self):
        for exp in self.exps:
            self.assertAlmostEqual(pow_new(self.phyunit, exp),
                                   pow_old(self.phyunit, exp)
                                   )

    def testpow_error(self):
        for exp in self.badexps:
            with self.assertRaises(TypeError):
                pow_new(self.phyunit, exp)
                pow_old(self.phyunit, exp)

    def testround(self):
        for value in self.values:
            self.assertEqual(round_new(value), round_old(value))

    def testpi(self):
        self.assertAlmostEqual(np.pi, numpy.oldnumeric.pi)

if __name__ == '__main__':
    unittest.main()
