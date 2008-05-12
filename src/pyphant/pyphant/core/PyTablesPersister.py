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
This module provides the facilities to persist a Recipe including any
generated data into a PyTables structure or a HDF5 file.
A rudimentary description of the used file-format follows:
.:Attribut
+:group
#:table
*:array

/results/resId
.longname
.shortname
[field|sample]

sample:
.desc = sample.desc [pickled scipy.dtype]
#data(descDict["col_"+name.encode("utf-8")]=Col(dtype=str(desc.fields[name][0]).capitalize()))
#units(unit,name)

field:
*data
*error
.unit = repr(field.unit)
UNLESS field.dimensions==INDEX:
#dimensions(hash, id)
"""

__id__ = "$Id$"
__author__ = "$Author$"
__version__ = "$Revision$"
# $Source: /home/obi/Projekte/pyphant/sourceforge/pyphant/pyphant/devel/trunk/src/pyphant/core/PyTablesPersister.py,v $

import tables, datetime
import sys
from pyphant.core import (CompositeWorker, DataContainer)
from tables import StringCol, Col
from Scientific.Physics.PhysicalQuantities import PhysicalQuantity
import scipy
import logging
_logger = logging.getLogger("pyphant")

_reservedAttributes = ('longname','shortname','columns')

class Connection(tables.IsDescription):
    destinationWorker = tables.StringCol(len("worker_"+str(sys.maxint))+1)
    destinationSocket = tables.StringCol(64)

##########################################################################
# Saving Part
##########################################################################
def saveRecipeToHDF5File( recipe, filename ):
    _logger.info( "Saving to %s" % filename )
    h5 = tables.openFile(filename, 'w')
    recipeGroup = h5.createGroup("/", "recipe")
    resultsGroup = h5.createGroup("/", "results")
    workers=recipe.getWorkers()
    for worker in workers:
        saveWorker(h5, recipeGroup, worker)
    h5.close()

def saveWorker(h5, recipeGroup, worker):
    workerGroup = h5.createGroup(recipeGroup, "worker_"+str(hash(worker)))
    saveBaseAttributes(h5, workerGroup, worker)
    savePlugs(h5, workerGroup, worker)
    saveParameters(h5, workerGroup, worker)

def saveParameters(h5, workerGroup, worker):
    paramGroup = h5.createGroup(workerGroup, "parameters")
    for (paramName, param) in worker._params.iteritems():
        h5.setNodeAttr(paramGroup, paramName, param.value)

def savePlugs(h5, workerGroup, worker):
    plugs = h5.createGroup(workerGroup, "plugs")
    for (plugName, plug) in worker._plugs.iteritems():
        plugGroup = h5.createGroup(plugs, plugName)
        if plug.resultIsAvailable():
            resId = saveResult(plug._result, h5)
            h5.setNodeAttr(plugGroup, "result", resId)
        connectionTable = h5.createTable(plugGroup, 'connections', Connection, expectedrows=len(plug._sockets))
        connection=connectionTable.row
        for socket in plug._sockets:
            connection['destinationWorker'] = "worker_"+str(hash(socket.worker))
            connection['destinationSocket'] = socket.name
            connection.append()
        connectionTable.flush()

def saveBaseAttributes(h5, workerGroup, worker):
    h5.setNodeAttr(workerGroup, "module", worker.__class__.__module__)
    h5.setNodeAttr(workerGroup, "clazz", worker.__class__.__name__)
    h5.setNodeAttr(workerGroup, "WorkerAPIVersion", worker.API)
    h5.setNodeAttr(workerGroup, "WorkerVersion", worker.VERSION)
    h5.setNodeAttr(workerGroup, "WorkerRevision", worker.REVISION)
    h5.setNodeAttr(workerGroup, "Annotations", worker._annotations)

def saveResult(result, h5):
    hash, uriType = DataContainer.parseId(result.id)
    resId = u"result_"+hash
    try:
        resultGroup = h5.getNode("/results/"+resId)
    except tables.NoSuchNodeError, e:
        resultGroup = h5.createGroup("/results", resId, result.id.encode("utf-8"))
        if uriType=='field':
            saveField(h5, resultGroup, result)
        elif uriType=='sample':
            saveSample(h5, resultGroup, result)
        else:
            raise KeyError, "Unknown UriType %s in saving result %s." % (uriType, result.id)
    return resId


def saveSample(h5, resultGroup, result):
    h5.setNodeAttr(resultGroup, "longname", result.longname.encode("utf-8"))
    h5.setNodeAttr(resultGroup, "shortname", result.shortname.encode("utf-8"))
    for key,value in result.attributes.iteritems():
        if key in _reservedAttributes:
            raise ValueError, "Attribute should not be named %s!" % _reservedAttributes
        h5.setNodeAttr(resultGroup,key,value)
    #Store fields of sample Container and gather list of field IDs
    columns = []
    for column in result.columns:
        columns.append(saveResult(column,h5))
    h5.setNodeAttr(resultGroup, "columns", columns)

def loadSample(h5, resNode):
    result = DataContainer.SampleContainer.__new__(DataContainer.SampleContainer)
    result.longname = unicode(h5.getNodeAttr(resNode, "longname"), 'utf-8')
    result.shortname = unicode(h5.getNodeAttr(resNode, "shortname"), 'utf-8')
    result.attributes = {}
    for key in resNode._v_attrs._v_attrnamesuser:
        if key not in _reservedAttributes:
            result.attributes[key]=h5.getNodeAttr(resNode,key)
    columns = []
    for resId in h5.getNodeAttr(resNode,"columns"):
        columns.append(loadField(h5,h5.getNode("/results/"+resId)))
    result.columns=columns
    result.seal(resNode._v_title)
    return result

def saveField(h5, resultGroup, result):
    def dump(inputList):
        if type(inputList)==type([]):
            if type(inputList[0])==type(u' '):
                conversion = lambda s: s.encode('utf-8')
            else:
                conversion = lambda s: s.__repr__()
            return map(conversion,inputList)
        else:
            return map(dump,inputList)
    if result.data.dtype.char in ['U','O']:
        unicodeData = scipy.array(dump(result.data.tolist()))
        h5.createArray(resultGroup, "data", unicodeData, result.longname.encode("utf-8"))
    else:
        h5.createArray(resultGroup, "data", result.data, result.longname.encode("utf-8"))
    for key,value in result.attributes.iteritems():
        h5.setNodeAttr(resultGroup.data,key,value)
    h5.setNodeAttr(resultGroup, "longname", result.longname.encode("utf-8"))
    h5.setNodeAttr(resultGroup, "shortname", result.shortname.encode("utf-8"))

    if result.error != None:
        h5.createArray(resultGroup, "error", result.error,
                       (u"Error of "+result.longname).encode("utf-8"))
    h5.setNodeAttr(resultGroup, "unit", repr(result.unit).encode("utf-8"))
    if result.dimensions!=DataContainer.INDEX:
        idLen=max([len(dim.id.encode("utf-8")) for dim in result.dimensions])
        dimTable = h5.createTable(resultGroup, "dimensions",
                                  {"hash":StringCol(32), "id":StringCol(idLen)},
                                  (u"Dimensions of "+result.longname).encode("utf-8"),
                                  expectedrows=len(result.dimensions))
        for dim in result.dimensions:
            d = dimTable.row
            d["hash"]=dim.hash.encode("utf-8")
            d["id"]=dim.id.encode("utf-8")
            d.append()
            saveResult(dim, h5)
        dimTable.flush()

##########################################################################
# Loading Part
##########################################################################
def instantiateWorker( parent, workerGroup ):
    annotations = workerGroup._v_attrs.Annotations
    module = workerGroup._v_attrs.module
    exec "import "+module
    worker = eval(module+"."+workerGroup._v_attrs.clazz+"(parent, annotations)")
    for paramName in workerGroup.parameters._v_attrs._v_attrnamesuser:
        param = getattr(workerGroup.parameters._v_attrs, paramName)
        worker.getParam(paramName).value=param
    return worker

def loadRecipeFromHDF5File( filename ):
    h5 = tables.openFile(filename, 'r')
    recipeGroup = h5.root.recipe
    recipe = CompositeWorker.CompositeWorker()
    workers = {}
    createWorkerGraph(recipeGroup, workers, recipe)
    restoreResultsToWorkers(recipeGroup, workers, h5)
    h5.close()
    return recipe

def createWorkerGraph(recipeGroup, workers, recipe):
    for workerGroup in recipeGroup:
        workers[workerGroup._v_name]=instantiateWorker(recipe, workerGroup)
    for workerGroup in recipeGroup:
        for plugGroup in workerGroup.plugs:
            plug=workers[workerGroup._v_name].getPlug(plugGroup._v_name)
            for connection in plugGroup.connections.iterrows():
                workers[connection['destinationWorker']].getSocket(connection['destinationSocket']).insert(plug)

def restoreResultsToWorkers(recipeGroup, workers, h5):
    for workerGroup in recipeGroup:
        for plugGroup in workerGroup.plugs:
            plug=workers[workerGroup._v_name].getPlug(plugGroup._v_name)
            try:
                resId = plugGroup._v_attrs.result
                resNode = h5.getNode("/results/"+resId)
                hash, uriType = DataContainer.parseId(resNode._v_title)
                if uriType==u'field':
                    result=loadField(h5, resNode)
                elif uriType==u'sample':
                    _logger.info("Trying to load sample data...")
                    result=loadSample(h5, resNode)
                    _logger.info("...successfully loaded.")
                else:
                    raise TypeError, "Unknown result uriType in <%s>"%resNode._v_title
                plug._result = result
            except (AttributeError, tables.NoSuchNodeError), e:
                _logger.info( "Exception: "+str(e) )

def loadField(h5, resNode):
    longname = unicode(h5.getNodeAttr(resNode, "longname"), 'utf-8')
    shortname = unicode(h5.getNodeAttr(resNode, "shortname"), 'utf-8')
    data = scipy.array(resNode.data.read())
    def loads(inputList):
        if type(inputList)==type([]):
            try:
                return map(lambda s: eval(s),inputList)
            except:
                return map(lambda s: unicode(s, 'utf-8'),inputList)
        else:
            return map(loads,inputList)
    if data.dtype.char == 'S':
        data = scipy.array(loads(data.tolist()))
    attributes = {}
    for key in resNode.data._v_attrs._v_attrnamesuser:
        attributes[key]=h5.getNodeAttr(resNode.data,key)
    try:
        error = scipy.array(resNode.error.read())
    except tables.NoSuchNodeError, e:
        error = None
    unit = eval(unicode(h5.getNodeAttr(resNode, "unit"), 'utf-8'))
    try:
        dimTable = resNode.dimensions
        dimensions = [loadField(h5, h5.getNode("/results/result_"+DataContainer.parseId(row['id'])[0]))
                      for row in dimTable.iterrows()]
    except tables.NoSuchNodeError, e:
        dimensions = DataContainer.INDEX
    result = DataContainer.FieldContainer(data, unit, error, None,
                                          dimensions, longname, shortname,
                                          attributes)
    result.seal(resNode._v_title)
    return result


