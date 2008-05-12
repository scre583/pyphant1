# -*- coding: utf-8 -*-

# Copyright (c) 2006-2008, Rectorate of the University of Freiburg
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

u"""
"""

__id__ = "$Id$"
__author__ = "$Author$"
__version__ = "$Revision$"
# $Source$

import Connectors


class ParamFactory(object):
    @classmethod
    def createParam(cls, worker, paramName, displayName, values, subtype=None):
        if isinstance(values, list):
            return SelectionParam(worker, paramName, displayName, values, subtype)
        else:
            return Param(worker, paramName, displayName, values, subtype)


class Param(Connectors.Socket):
    def getValue(self):
        return self._value

    def __getValue(self):
        if self.isFull():
            return self.getResult()
        else:
            return self.getValue()

    def setValue(self, value):
        self._value=value

    def __setValue(self, value):
        oldValue=self.value
        if oldValue==value: return
#        if self._validator: ##reintroduce with appropriate subtype validator lookup
#            self._validator(oldValue, value)
        self.setValue(value)
        self.invalidate()

    value=property(__getValue, __setValue)

    def __init__(self, worker, name, displayName, value, subtype=None):
        Connectors.Socket.__init__(self, worker, name, type(value))
        self.isExternal=False
        self.valueType=type(value)
        self.displayName=displayName
        self._value=value
        self.subtype=subtype



class SelectionParam(Param):
    def __init__(self, worker, name, displayName, values, subtype=None):
        Param.__init__(self, worker, name, displayName, values[0], subtype)
        self.valueType=type(values)
        self._possibleValues=values

    def getPossibleValues(self):
        return self._possibleValues

    def __getPossibleValues(self):
        return self.getPossibleValues()

    def setPossibleValues(self, values):
        self._possibleValues=values

    def __setPossibleValues(self, values):
        oldValues=self.possibleValues
        if oldValues==values:
            return
#        if self._validator: ##see Param
#            self._validator(oldValues, values)
        self.setPossibleValues(values)
        self.invalidate()

    possibleValues=property(__getPossibleValues,__setPossibleValues)
