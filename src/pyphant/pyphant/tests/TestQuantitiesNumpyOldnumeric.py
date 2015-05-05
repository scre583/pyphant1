import unittest
from pyphant.quantities import Quantity, _unit_table, PhysicalUnit,\
                               NumberDict, _base_names
import numpy as np
import numpy.oldnumeric


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


class TestQuantityNumpyOldnumeric(unittest.TestCase):
    def setUp(self):
        self.angles = []
        self.angles.append(Quantity('1.4 rad'))
        self.angles.append(Quantity('2 rad'))
        self.angles.append(Quantity('30 deg'))
        self.angles.append(Quantity('-40 deg'))
        self.length = Quantity('5.3 m')
        self.phyunit = PhysicalUnit('m', 1., [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.phyunit **= 5
        self.exps = [0, 1, 2, 1. / 5, 1. / (5. + 1e-12), 1. / (5. - 1e-12)]
        self.badexps = [0., 2., 1. / 6, 1. / (5. + 1e-8)]

    def testsin(self):
        for angle in self.angles:
            self.assertAlmostEqual(sin_new(angle), sin_old(angle))
        with self.assertRaises(TypeError):
            sin_new(self.length)
            sin_old(self.length)

    def testcos(self):
        for angle in self.angles:
            self.assertAlmostEqual(cos_new(angle), cos_old(angle))
        with self.assertRaises(TypeError):
            cos_new(self.length)
            cos_old(self.length)

    def testtan(self):
        for angle in self.angles:
            self.assertAlmostEqual(tan_new(angle), tan_old(angle))
        with self.assertRaises(TypeError):
            tan_new(self.length)
            tan_old(self.length)

    def testpow(self):
        for exp in self.exps:
            self.assertAlmostEqual(pow_new(self.phyunit, exp),
                                   pow_old(self.phyunit, exp)
                                   )
        for exp in self.badexps:
            with self.assertRaises(TypeError):
                pow_new(self.phyunit, exp)
                pow_old(self.phyunit, exp)

if __name__ == '__main__':
    unittest.main()
