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
Knowledge Manager for Pyphant
=============================

The ID of a DataContainer object is given by a emd5 string.

Responsibilities:
-----------------

 - register HDF5 files by their URLs
 - register remote knowledge managers by urls
 - share data containers via HTTP, they are requested by id
 - get references for these data containers (local or remote)

If an operation fails, a KnowledgeManagerException
will be raised. These exceptions have a method

  .getParentException()

in order to get additional information about the reason.

Usage:
------

 Get a reference to the KnowledgeManager instance, which is a
 singleton:

  import pyphant.core.KnowledgeManager as KM
  km = KM.KnowledgeManager.getInstance()

 Optionally: Start HTTP server for sharing data with others by

  km.startServer(<host>,<port>)

 Register a local HDF5 file:

  km.registerURL("file://tmp/data.h5")

 Register a remote HDF5 file:

  km.registerURL("http://example.com/repository/data.h5")

 Register another KnowledgeManager in order to benefit
 from their knowledge (see arguments of .startServer):

  km.registerKnowledgeManager("http://example.com:8000")

 Request data container by its id:

  dc = km.getDataContainer(id)

 Use the data container!

"""

__id__ = "$Id$"
__author__ = "$Author$"
__version__ = "$Revision$"
# $Source: $

from pyphant.core.singletonmixin import Singleton
from pyphant.core.DataContainer import parseId
import pyphant.core.PyTablesPersister as ptp

from types import TupleType
import urllib
import cgi

import tempfile

import tables

import sys
import os, os.path
import logging
import traceback

from SocketServer import ThreadingMixIn
import threading
from BaseHTTPServer import HTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler
from uuid import uuid1
import HTMLParser

WAITING_SECONDS_HTTP_SERVER_STOP = 5
HTTP_REQUEST_DC_URL_PATH="/request_dc_url"
HTTP_REQUEST_KM_ID_PATH="/request_km_id"


class KnowledgeManagerException(Exception):
    def __init__(self, message, parent_excep=None, *args, **kwds):
        super(KnowledgeManagerException, self).__init__(message, *args, **kwds)
        self._message = message
        self._parent_excep = parent_excep

    def __str__(self):
        return self._message+" (reason: %s)" % (str(self._parent_excep),)

    def getParentException(self):
        return self._parent_excep

class KnowledgeManager(Singleton):

    def __init__(self):
        super(KnowledgeManager, self).__init__()
        self._logger = logging.getLogger("pyphant")
        self._refs = {}
        self._remoteKMs = {} # key:id, value:url
        self._server = None
        self._server_id = uuid1()

    def __del__(self):
        if self.isServerRunning():
            self.stopServer()

    def _getServerURL(self):
        if self._server is None:
            return None
        return "http://%s:%d" % (self._http_host, self._http_port)

    def getServerId(self):
        """Return uniqe id of the KnowledgeManager.
        """
        return self._server_id

    def startServer(self, host, port=8000):
        """Start the HTTP server. When the server was running already, it is restartet with the new parameters.

          host -- full qualified domain name or IP address under which
                  server can be contacted via HTTP
          port -- port of HTTP server (integer), default: 8000

          A temporary directory is generated in order to
          save temporary HDF5 files.
          The data may be announced to other KnowledgeManagers.
        """
        logger = self._logger
        if self.isServerRunning():
            logger.warn("Server is running at host %s, port %d already. Stopping server...", self._http_host, self._http_port)
            self.stopServer()
        self._http_host = host
        self._http_port = port
        self._http_dir = tempfile.mkdtemp(prefix='pyphant-knowledgemanager')
        self._server = _HTTPServer((host,port),_HTTPRequestHandler)

        class _HTTPServerThread(threading.Thread):
            def run(other):
                self._server.start()
        self._http_server_thread = _HTTPServerThread()
        self._http_server_thread.start()
        self._logger.debug("Started HTTP server."\
                         +" host: %s, port: %d,"\
                         +" temp dir: %s", host, port, self._http_dir)


    def stopServer(self):
        """Stop the HTTP server.

        The temporary directory is removed.
        """

        logger = self._logger
        if self.isServerRunning():
            self._server.stop_server = True
            # do fake request
            try:
                urllib.urlopen(self._getServerURL())
            except:
                logger.warn("Fake HTTP request failed when "+\
                                "stopping HTTP server.")
            logger.info("Waiting for HTTP server thread to die...")
            self._http_server_thread.join(WAITING_SECONDS_HTTP_SERVER_STOP)
            if self._http_server_thread.isAlive():
                logger.warn("HTTP server thread could not be stopped.")
            else:
                logger.info("HTTP server has been stopped.")
            self._server = None
            self._server_id = None
            self._http_host = None
            self._http_port = None
            try:
                logger.debug("Deleting temporary directory '%s'..", self._http_dir)
                os.removedirs(self._http_dir)
            except Exception, e:
                logger.warn("Failed to delete temporary directory '%s'.", self._http_dir)
            self._http_dir = None
        else:
            self._logger.warn("HTTP server should be stopped but isn't running.")

    def isServerRunning(self):
        """Return whether HTTP server is running."""
        return self._server is not None

    def registerKnowledgeManager(self, host, port=8000, share_knowledge=False):
        """Register a knowledge manager.

        host -- full qualified domain name or IP address under which
                server can be contacted via HTTP

        port -- port of HTTP server (integer), default: 8000

        share_knowledge -- local knowledge is made available to the remote KM when set to True and the HTTP server is running at the local KM, default: False

        The remote KnowledgeManager is contacted immediately in order
        to save its unique ID.
        """
        logger = self._logger
        try:
            km_url = "http://%s:%d"%(host, port)
            # get unique id from KM via HTTP
            logger.debug("Requesting ID from Knowledgemanager with URL '%s'...", km_url)
            # request url for given id over http
            local_km_host = ''
            local_km_port = ''
            if self.isServerRunning() and share_knowledge:
                local_km_host = self._http_host
                local_km_port = str(self._http_port)
            post_data = urllib.urlencode({'kmhost':local_km_host, 'kmport':local_km_port})
            answer = urllib.urlopen(km_url+HTTP_REQUEST_KM_ID_PATH, post_data)
            logger.debug("Info from HTTP answer: %s", answer.info())
            km_id = answer.readline().strip()
            answer.close()
            logger.debug("KM ID read from HTTP answer: %s", km_id)
        except Exception, e:
            raise KnowledgeManagerException(
                "Couldn't get ID for knowledge manager under URL %s." % (km_url,),e)

        self._remoteKMs[km_id] = km_url

    def registerURL(self, url):
        """Register an HDF5 file downloadable from given URL.

        url -- URL of the HDF5 file

        The HDF5 file is downloaded and all DataContainers
        in the file are registered with their identifiers.
        """
        self._retrieveURL(url)

    def registerDataContainer(self, datacontainer):
        """Register a DataContainer located in memory using a given reference.

        datacontainer -- reference to the DataContainer object

        The DataContainer must have an .id attribute,
        which could be generated by the datacontainer.seal() method.
        """
        try:
            assert datacontainer.id is not None
            self._refs[datacontainer.id] = datacontainer
        except Exception, e:
            raise KnowledgeManagerException("Invalid id for DataContainer '" +\
                                                datacontainer.longname+"'", e)


    def _retrieveURL(self, url):
        """Retrieve HDF5 file from a given URL.

        url -- URL of the HDF5 file

        The HDF5 file is downloaded and all DataContainers
        in the file are registered with their identifiers.
        """

        self._logger.info("Retrieving url '%s'..." % (url,))
        localfilename, headers = urllib.urlretrieve(url)
        self._logger.info("Using local file '%s'." % (localfilename,))
        self._logger.info("Header information: %s", (str(headers),))

        #
        # Save index entries
        #
        h5 = tables.openFile(localfilename)
        # title of 'result_' groups has id in TITLE attribute
        dc = None
        for group in h5.walkGroups(where="/results"):
            dc_id = group._v_attrs.TITLE
            if len(dc_id)>0:
                self._logger.debug("Registering DC ID '%s'.." % (dc_id,))
                self._refs[dc_id] = (url, localfilename, group._v_pathname)

        h5.close()

    def _retrieveRemoteKMs(self, dc_id, omit_km_ids):
        """Retrieve datacontainer by its id from remote KnowledgeManagers.

        dc_id -- unique id of the requested DataContainer
        """
        dc_url = self._getURLFromRemoteKMs(dc_id, omit_km_ids)
        if dc_url is None:
            raise KnowledgeManagerException(
                "Couldn't retrieve DC ID '%s' from remote knowledgemanagers" % (dc_id,))
        else:
            self._retrieveURL(dc_url)

    def _getURLFromRemoteKMs(self, dc_id, omit_km_ids):
        """Return URL for a DataContainer by requesting remote KnowledgeManagers.

        dc_id -- ID of the requested DataContainer
        omit_km_ids -- list of KnowledgeManager IDs which shouldn't be
                       asked
        """
        logger = self._logger
        #
        # build query for http request with
        # id of data container and
        # list of URLs which should not be requested by
        # the remote side
        #
        query = { 'dcid': dc_id}
        idx = -1 # needed if omit_km_ids is empty
        for idx,km_id in enumerate(omit_km_ids):
            query['kmid%d' % (idx,)] = km_id

        serverID = self._server_id
        if serverID is not None:
            idx += 1
            query['kmid%d' % (idx,)] = serverID

        #
        # ask every remote KnowledgeManager for id
        #
        logger.debug("Requesting knowledge managers for DC id '%s'..." % (dc_id,))
        dc_url = None
        for km_id, km_url in self._remoteKMs.iteritems():
            if not (km_id in omit_km_ids):
                logger.debug("Requesting Knowledgemanager with ID '%s' and URL '%s'...", km_id, km_url)
                # request url for given id over http
                try:
                    data = urllib.urlencode(query)
                    logger.debug("URL encoded query: %s", data)
                    answer = urllib.urlopen(km_url+HTTP_REQUEST_DC_URL_PATH, data)
                    code = answer.headers.dict['code']
                    if code < 400:
                        dc_url = answer.headers.dict['location']
                        logger.debug("URL for id read from HTTP answer: %s", dc_url)
                        break
                    else:
                        # message for everyone: do not ask this KM again
                        idx += 1
                        query['kmid%d' % (idx),] = km_id
                except:
                    logger.debug("Could not contact KM with ID '%s'", km_id)
                    # message for everyone: do not ask this KM again
                    idx += 1
                    query['kmid%d' % (idx),] = km_id
                finally:
                    answer.close()
        return dc_url


    def _getDataContainerURL(self, dc_id, omit_km_ids=[]):
        """Return a URL from which a DataContainer can be downloaded.

        dc_id -- ID of requested DataContainer
        omit_km_ids -- list of KnowledgeManager IDs which shouldn't be
                       asked (Default: [])

        The DataContainer can be downloaded as HDF5 file.
        The server must be running before calling this method.
        """
        assert self.isServerRunning(), "Server is not running."

        if dc_id in self._refs.keys():
            dc = self.getDataContainer(dc_id, omit_km_ids=omit_km_ids)

            #
            # Wrap data container in temporary HDF5 file
            #
            h5fileid, h5name = tempfile.mkstemp(suffix='.h5',
                                                prefix='dcrequest-',
                                                dir=self._http_dir)
            os.close(h5fileid)

            h5 = tables.openFile(h5name,'w')
            resultsGroup = h5.createGroup("/", "results")
            ptp.saveResult(dc, h5)
            h5.close()
            dc_url = self._getServerURL()+"/"+os.path.basename(h5name)
        else:
            try:
                dc_url = self._getURLFromRemoteKMs(dc_id, omit_km_ids)
            except Exception, e:
                raise KnowledgeManagerException(
                    "URL for DC ID '%s' not found." % (dc_id,), e)
        return dc_url

    def getDataContainer(self, dc_id, try_cache=True, omit_km_ids=[]):
        """Request reference on DataContainer having the given id.

        dc_id       -- Unique ID of the DataContainer
        try_cache   -- Try local cache first (default: True)
        omit_km_ids -- list of KnowledgeManager IDs which shouldn't be
                       asked (Default: [])
        """
        if dc_id not in self._refs.keys():
            # raise KnowledgeManagerException("DC ID '%s'unknown."%(dc_id,))
            try:
                self._retrieveRemoteKMs(dc_id, omit_km_ids)
            except Exception, e:
                raise KnowledgeManagerException(
                    "DC ID '%s' unknown." % (dc_id,), e)

        ref = self._refs[dc_id]
        if isinstance(ref, TupleType):
            dc = self._getDCfromURLRef(dc_id, try_cache = try_cache)
        else:
            dc = ref

        return dc

    def _getDCfromURLRef(self, dc_id, try_cache=True, omit_km_ids=[]):
        """Return DataContainer.

        dc_id       -- Unique ID of the DataContainer
        try_cache   -- Try local cache first (default: True)
        omit_km_ids -- list of KnowledgeManager IDs which shouldn't be
                       asked (Default: [])

        The following request order is used:

         1. Use local cache file, if available (only for try_cache=True)
         2. Try to download HDF5 file (again).
         3. Ask remote KnowledgeManagers for the given ID.
            Download the DataContainer as HDF5 file (if available).

        Afterwards open the file and extract the DataContainer.
        The given dc_id must be known to the KnowledgeManager.
        """
        dc_url, localfilename, h5path = self._refs[dc_id]
        if not try_cache:
            os.remove(localfilename)

        if not os.path.exists(localfilename):
            try:
                # download URL and save ids as references
                self._retrieveURL(dc_url)
            except Exception, e_url:
                try:
                    self._retrieveRemoteKMs(dc_id, omit_km_ids)
                except Exception, e_rem:
                    raise KnowledgeManagerException(
                        "DC ID '%s' not found on remote sites."% (dc_id,),
                        KnowledgeManagerException(
                            "DC ID '%s' could not be resolved using URL '%s'" \
                                % (dc_id, dc_url)), e_url)

            dc_url, localfilename, h5path = self._refs[dc_id]

        h5 = tables.openFile(localfilename)

        hash, type = parseId(dc_id)
        assert type in ['sample','field']
        if type=='sample':
            loader = ptp.loadSample
        elif type=='field':
            loader = ptp.loadField
        else:
            raise KnowledgeManagerException("Unknown result type '%s'" \
                                                % (type,))
        try:
            self._logger.debug("Loading data from '%s' in file '%s'.." % (localfilename, h5path))
            dc = loader(h5, h5.getNode(h5path))
        except Exception, e:
            raise KnowledgeManagerException(
                "DC ID '%s' known, but cannot be read from file '%s'." \
                    % (dc_id,localfilename), e)
        finally:
            h5.close()
        return dc

class _HTTPRequestHandler(SimpleHTTPRequestHandler):

    _knowledge_manager = KnowledgeManager.getInstance()
    _logger = logging.getLogger("pyphant")
    
    def send_response(self, code, message=None):
        self.log_request(code)
        if message is None:
            if code in self.responses:
                message = self.responses[code][0]
            else:
                message = ''
        if self.request_version != 'HTTP/0.9':
            self.wfile.write("%s %d %s\r\n" % (self.protocol_version, code, message))
        self.send_header('Server', self.version_string())
        self.send_header('Date', self.date_time_string())
        #for older versions of urllib.urlopen which do not support .getcode() method
        self.send_header('code', code)
    
    def do_POST(self):
        self._logger.debug("POST request from client (host,port): %s",
                           self.client_address)
        self._logger.debug("POST request path: %s", self.path)
        # self.log_request()
        if self.path==HTTP_REQUEST_DC_URL_PATH:
            httpanswer = self._do_POST_request_dc_url()
        elif self.path==HTTP_REQUEST_KM_ID_PATH:
            httpanswer = self._do_POST_request_km_id()
        else:
            code = 400
            message = "Unknown request path '%s'." % (self.path,)
            httpanswer = _HTTPAnswer(code, message)
        
        httpanswer.sendTo(self)

        
    def _do_POST_request_km_id(self):
        """Return the KnowledgeManager ID."""
        km = _HTTPRequestHandler._knowledge_manager
        if self.headers.has_key('content-length'):
            length= int( self.headers['content-length'] )
            query = self.rfile.read(length)
            query_dict = cgi.parse_qs(query)
            remote_host = ''
            remote_port = ''
            try:
                remote_host = query_dict['kmhost'][0]
                remote_port = query_dict['kmport'][0]
            except:
                self._logger.warn("Remote knowledge is not being shared.")
            if remote_host != '' and remote_port != '':
                km.registerKnowledgeManager(remote_host, int(remote_port), False)

        code = 200
        answer = km._server_id
        self._logger.debug("Returning ID '%s'...", answer)
        
        #TODO
        htmlheaders = {}
        httpheaders = {}
        htmlbody = answer
        message = answer
        contenttype = 'text/html'
        return _HTTPAnswer(code, message, httpheaders, contenttype, htmlheaders, htmlbody)


    def _do_POST_request_dc_url(self):
        """Return a URL for a given DataContainer ID."""
        if self.headers.has_key('content-length'):
            length= int( self.headers['content-length'] )
            query = self.rfile.read(length)
            query_dict = cgi.parse_qs(query)

            dc_id = query_dict['dcid'][0]
            omit_km_ids = [ value[0] for (key,value) in query_dict.iteritems()
                             if key!='dcid']
            self._logger.debug("Query data: dc_id: %s, omit_km_ids: %s",
                               dc_id, omit_km_ids)

            try:
                km = _HTTPRequestHandler._knowledge_manager
                redirect_url = km._getDataContainerURL(dc_id, omit_km_ids)
                if redirect_url != None:
                    self._logger.debug("Returning URL '%s'...", redirect_url)
                    httpanswer = _HTTPAnswer(201, None, {'location':redirect_url}, 'text/html', {}, '<a href="%s"></a>'%(redirect_url,))
                else:
                    self._logger.debug("Returning Error Code 404: DataContainer ID '%s' not found.", dc_id)
                    httpanswer = _HTTPAnswer(404, "DataContainer ID '%s' not found." % (dc_id,))
            except Exception, e:
                self._logger.warn("Catched exception: %s", traceback.format_exc())
                code = 404 #file not found
                answer = "Failed: DC ID '%s' not found." % (dc_id,) # 'Failed' significant!
                httpanswer = _HTTPAnswer(500, "Internal server error occured durin lookup of DataContainer with ID '%s'"%(dc_id,))   
        else:
            httpanswer = _HTTPAnswer(400, "Cannot interpret query.")

        return httpanswer

    def do_GET(self):
        """Return a requested HDF5 from temporary directory.
        """
        log = self._logger
        f = self.send_head()
        if f:
            self.copyfile(f, self.wfile)
            f.close()
            try:
                log.debug("Trying to remove temporary file '%s'..", f.name)
                os.remove(f.name)
            except Exception, e:
                log.warn("Cannot delete temporary file '%s'.", f.name)



    def send_head(self): # see SimpleHTTPServer.SimpleHTTPRequestHandler
        """Send header for HDF5 file request.
        """
        log = self._logger

        km = _HTTPRequestHandler._knowledge_manager
        source_dir = km._http_dir # this is intended
        log.debug("HTTP GET request: "\
                      +"Reading files from directory '%s'..", source_dir)

        try:
            # build filename, remove preceding '/' in path
            filename = os.path.join(source_dir, self.path[1:])
            log.debug("Returning file '%s' as answer for HTTP request..",
                      filename)
            f = open(filename, 'rb')
        except IOError:
            self.send_error(404, "File not found")
            return None
        self.send_response(200)
        self.send_header("Content-type", "application/x-hdf")
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f


class _HTTPServer(ThreadingMixIn,HTTPServer):
    """Threaded HTTP Server for the KnowledgeManager.
    """
    stop_server = False
    _logger = logging.getLogger("pyphant")

    def start(self):
        while not self.stop_server:
            self.handle_request()
        self._logger.info("Stopped HTTP server.")

class _HTMLParser(HTMLParser.HTMLParser):
    def __init__(self):
        HTMLParser.HTMLParser.__init__(self)
        self._isinhead = False
        self._isinbody = False
        self._headitems = {} #tag : [content, attributes]
        self._currentheadtag = None
        self._bodytext = ''

    def handle_starttag(self, tag, attrs):
        if tag == 'head':
            self._isinhead = True
        elif tag == 'body':
            self._isinbody = True
        elif self._isinhead:
            self._currentheadtag = tag
            self._headitems[tag] = ''
    
    def handle_endtag(self, tag):
        if tag == 'head':
            self._isinhead = False
        elif self._isinhead:
            self._currentheadtag = None
        elif tag == 'body':
            self._isinbody = False
    
    def handle_data(self, data):
        if self._isinhead and self._currentheadtag != None:
            self._headitems[self._currentheadtag] += data+", "
        elif self._isinbody:
            self._bodytext += data
        
class _HTTPAnswer():
    def __init__(self, code, message=None, httpheaders = {}, contenttype='text/html', htmlheaders={}, htmlbody=''):
        self._code = code
        self._message = message
        self._httpheaders = httpheaders
        self._htmlheaders = htmlheaders
        self._htmlbody = htmlbody
        self._httpheaders['Content-type'] = contenttype

    def sendTo(self, handler):
        _logger = logging.getLogger("pyphant")
        if self._code >= 400:
            #send error response
            handler.send_error(self._code, self._message)
        else:
            #send HTTP headers...
            handler.send_response(self._code, self._message)
            for key, value in self._httpheaders.items():
                handler.send_header(key, value)
                _logger.debug("key: %s, value: %s\n", key, value)
            handler.end_headers()
        
            #send HTML headers...
            handler.wfile.write('<head>\n')
            for key, value in self._htmlheaders.items():
                handler.wfile.write('<%s>\n'%(key,))
                handler.wfile.write(value+'\n')
                handler.wfile.write('</%s>\n'%(key,))
            handler.wfile.write('</head>\n')
            
            #send HTML body...
            handler.wfile.write('<body>\n')
            handler.wfile.write(self._htmlbody+'\n')
            handler.wfile.write('</body>\n')
            handler.wfile.write('\n')        

def _enableLogging():
    """Enable logging for debug purposes."""
    l = logging.getLogger("pyphant")
    l.setLevel(logging.DEBUG)
    f = logging.Formatter('%(asctime)s [%(name)s|%(levelname)s] %(message)s')
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(f)
    l.addHandler(h)
    l.info("Logger 'pyphant' has been configured for debug purposes.")

