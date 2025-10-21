from __future__ import print_function, absolute_import, division
import _mockdt
import f90wrap.runtime
import logging
import numpy
import warnings

class Constant_Parameters(f90wrap.runtime.FortranModule):
    """
    Module constant_parameters
    Defined at ./Source/BasicDefs/aa1_modules.fpp lines 14-30
    """
    pass
    _dt_array_initialisers = []
    

constant_parameters = Constant_Parameters()

class Gaussian(f90wrap.runtime.FortranModule):
    """
    Module gaussian
    Defined at ./Source/BasicDefs/aa1_modules.fpp lines 35-44
    """
    @property
    def ng(self):
        """
        Element ng ftype=integer         pytype=int
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 41
        """
        return _mockdt.f90wrap_gaussian__get__ng()
    
    @ng.setter
    def ng(self, ng):
        _mockdt.f90wrap_gaussian__set__ng(ng)
    
    def get_ng(self):
        return self.ng
    
    def set_ng(self, value):
        self.ng = value
    
    @property
    def ngpsi(self):
        """
        Element ngpsi ftype=integer         pytype=int
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 41
        """
        return _mockdt.f90wrap_gaussian__get__ngpsi()
    
    @ngpsi.setter
    def ngpsi(self, ngpsi):
        _mockdt.f90wrap_gaussian__set__ngpsi(ngpsi)
    
    def get_ngpsi(self):
        return self.ngpsi
    
    def set_ngpsi(self, value):
        self.ngpsi = value
    
    @property
    def ecinv(self):
        """
        Element ecinv ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdt.f90wrap_gaussian__array__ecinv(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            ecinv = self._arrays[array_hash]
        else:
            try:
                ecinv = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _mockdt.f90wrap_gaussian__array__ecinv)
            except TypeError:
                ecinv = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = ecinv
        return ecinv
    
    @ecinv.setter
    def ecinv(self, ecinv):
        self.ecinv[...] = ecinv
    
    def set_array_ecinv(self, value):
        self.ecinv[...] = value
    
    def get_array_ecinv(self):
        return self.ecinv
    
    @property
    def xg(self):
        """
        Element xg ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdt.f90wrap_gaussian__array__xg(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            xg = self._arrays[array_hash]
        else:
            try:
                xg = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _mockdt.f90wrap_gaussian__array__xg)
            except TypeError:
                xg = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = xg
        return xg
    
    @xg.setter
    def xg(self, xg):
        self.xg[...] = xg
    
    def set_array_xg(self, value):
        self.xg[...] = value
    
    def get_array_xg(self):
        return self.xg
    
    @property
    def fcinv(self):
        """
        Element fcinv ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdt.f90wrap_gaussian__array__fcinv(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            fcinv = self._arrays[array_hash]
        else:
            try:
                fcinv = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _mockdt.f90wrap_gaussian__array__fcinv)
            except TypeError:
                fcinv = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = fcinv
        return fcinv
    
    @fcinv.setter
    def fcinv(self, fcinv):
        self.fcinv[...] = fcinv
    
    def set_array_fcinv(self, value):
        self.fcinv[...] = value
    
    def get_array_fcinv(self):
        return self.fcinv
    
    @property
    def wg(self):
        """
        Element wg ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdt.f90wrap_gaussian__array__wg(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            wg = self._arrays[array_hash]
        else:
            try:
                wg = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _mockdt.f90wrap_gaussian__array__wg)
            except TypeError:
                wg = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = wg
        return wg
    
    @wg.setter
    def wg(self, wg):
        self.wg[...] = wg
    
    def set_array_wg(self, value):
        self.wg[...] = value
    
    def get_array_wg(self):
        return self.wg
    
    @property
    def xgpsi(self):
        """
        Element xgpsi ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 45
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdt.f90wrap_gaussian__array__xgpsi(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            xgpsi = self._arrays[array_hash]
        else:
            try:
                xgpsi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _mockdt.f90wrap_gaussian__array__xgpsi)
            except TypeError:
                xgpsi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = xgpsi
        return xgpsi
    
    @xgpsi.setter
    def xgpsi(self, xgpsi):
        self.xgpsi[...] = xgpsi
    
    def set_array_xgpsi(self, value):
        self.xgpsi[...] = value
    
    def get_array_xgpsi(self):
        return self.xgpsi
    
    @property
    def wgpsi(self):
        """
        Element wgpsi ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa1_modules.fpp line 45
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdt.f90wrap_gaussian__array__wgpsi(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            wgpsi = self._arrays[array_hash]
        else:
            try:
                wgpsi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _mockdt.f90wrap_gaussian__array__wgpsi)
            except TypeError:
                wgpsi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = wgpsi
        return wgpsi
    
    @wgpsi.setter
    def wgpsi(self, wgpsi):
        self.wgpsi[...] = wgpsi
    
    def set_array_wgpsi(self, value):
        self.wgpsi[...] = value
    
    def get_array_wgpsi(self):
        return self.wgpsi
    
    def __str__(self):
        ret = ['<gaussian>{\n']
        ret.append('    ng : ')
        ret.append(repr(self.ng))
        ret.append(',\n    ngpsi : ')
        ret.append(repr(self.ngpsi))
        ret.append(',\n    ecinv : ')
        ret.append(repr(self.ecinv))
        ret.append(',\n    xg : ')
        ret.append(repr(self.xg))
        ret.append(',\n    fcinv : ')
        ret.append(repr(self.fcinv))
        ret.append(',\n    wg : ')
        ret.append(repr(self.wg))
        ret.append(',\n    xgpsi : ')
        ret.append(repr(self.xgpsi))
        ret.append(',\n    wgpsi : ')
        ret.append(repr(self.wgpsi))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

gaussian = Gaussian()

class Defineallproperties(f90wrap.runtime.FortranModule):
    """
    Module defineallproperties
    Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp lines 13-47
    """
    @f90wrap.runtime.register_class("mockdt.SolverOptionsDef")
    class SolverOptionsDef(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=solveroptionsdef)
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp lines 19-46
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for solveroptionsdef
            
            self = Solveroptionsdef()
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp lines 19-46
            
            Returns
            -------
            this : Solveroptionsdef
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _mockdt.f90wrap_defineallproperties__solveroptionsdef_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for solveroptionsdef
            
            Destructor for class Solveroptionsdef
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp lines 19-46
            
            Parameters
            ----------
            this : Solveroptionsdef
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _mockdt.f90wrap_defineallproperties__solveroptionsdef_finalise(this=self._handle)
        
        @property
        def trimswitch(self):
            """
            Element trimswitch ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 24
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__trimswitch(self._handle)
        
        @trimswitch.setter
        def trimswitch(self, trimswitch):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__trimswitch(self._handle, \
                trimswitch)
        
        @property
        def updateguess(self):
            """
            Element updateguess ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 24
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__updateguess(self._handle)
        
        @updateguess.setter
        def updateguess(self, updateguess):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__updateguess(self._handle, \
                updateguess)
        
        @property
        def deltaairloads(self):
            """
            Element deltaairloads ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 24
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__deltaaief83(self._handle)
        
        @deltaairloads.setter
        def deltaairloads(self, deltaairloads):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__deltaai9421(self._handle, \
                deltaairloads)
        
        @property
        def linrzswitch(self):
            """
            Element linrzswitch ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 25
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__linrzswitch(self._handle)
        
        @linrzswitch.setter
        def linrzswitch(self, linrzswitch):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__linrzswitch(self._handle, \
                linrzswitch)
        
        @property
        def timemarchswitch(self):
            """
            Element timemarchswitch ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 26
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__timemare3b3(self._handle)
        
        @timemarchswitch.setter
        def timemarchswitch(self, timemarchswitch):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__timemar4f99(self._handle, \
                timemarchswitch)
        
        @property
        def freewakeswitch(self):
            """
            Element freewakeswitch ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 27
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__freewak3c80(self._handle)
        
        @freewakeswitch.setter
        def freewakeswitch(self, freewakeswitch):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__freewak8069(self._handle, \
                freewakeswitch)
        
        @property
        def windtunnelswitch(self):
            """
            Element windtunnelswitch ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 28
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__windtund117(self._handle)
        
        @windtunnelswitch.setter
        def windtunnelswitch(self, windtunnelswitch):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__windtun0496(self._handle, \
                windtunnelswitch)
        
        @property
        def rigidbladeswitch(self):
            """
            Element rigidbladeswitch ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 29
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__rigidbldcbe(self._handle)
        
        @rigidbladeswitch.setter
        def rigidbladeswitch(self, rigidbladeswitch):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__rigidbl1493(self._handle, \
                rigidbladeswitch)
        
        @property
        def fet_qddot(self):
            """
            Element fet_qddot ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 30
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__fet_qddot(self._handle)
        
        @fet_qddot.setter
        def fet_qddot(self, fet_qddot):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__fet_qddot(self._handle, \
                fet_qddot)
        
        @property
        def fet_response(self):
            """
            Element fet_response ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 30
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__fet_resc250(self._handle)
        
        @fet_response.setter
        def fet_response(self, fet_response):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__fet_res15d8(self._handle, \
                fet_response)
        
        @property
        def store_fet_responsejac(self):
            """
            Element store_fet_responsejac ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 31
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__store_f538c(self._handle)
        
        @store_fet_responsejac.setter
        def store_fet_responsejac(self, store_fet_responsejac):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__store_fb24a(self._handle, \
                store_fet_responsejac)
        
        @property
        def fet_responsejacavail(self):
            """
            Element fet_responsejacavail ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 31
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__fet_res230e(self._handle)
        
        @fet_responsejacavail.setter
        def fet_responsejacavail(self, fet_responsejacavail):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__fet_resf178(self._handle, \
                fet_responsejacavail)
        
        @property
        def airframevib(self):
            """
            Element airframevib ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 32
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__airframevib(self._handle)
        
        @airframevib.setter
        def airframevib(self, airframevib):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__airframevib(self._handle, \
                airframevib)
        
        @property
        def fusharm(self):
            """
            Element fusharm ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 32
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__fusharm(self._handle)
        
        @fusharm.setter
        def fusharm(self, fusharm):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__fusharm(self._handle, \
                fusharm)
        
        @property
        def axialdof(self):
            """
            Element axialdof ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 33
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__axialdof(self._handle)
        
        @axialdof.setter
        def axialdof(self, axialdof):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__axialdof(self._handle, \
                axialdof)
        
        @property
        def composite_coupling(self):
            """
            Element composite_coupling ftype=logical pytype=bool
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 33
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__composiee25(self._handle)
        
        @composite_coupling.setter
        def composite_coupling(self, composite_coupling):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__composic943(self._handle, \
                composite_coupling)
        
        @property
        def trimtechnique(self):
            """
            Element trimtechnique ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 34
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__trimtecb616(self._handle)
        
        @trimtechnique.setter
        def trimtechnique(self, trimtechnique):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__trimtec7319(self._handle, \
                trimtechnique)
        
        @property
        def trimsweepoption(self):
            """
            Element trimsweepoption ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 35
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__trimswe87ad(self._handle)
        
        @trimsweepoption.setter
        def trimsweepoption(self, trimsweepoption):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__trimswe913a(self._handle, \
                trimsweepoption)
        
        @property
        def ntimeelements(self):
            """
            Element ntimeelements ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 35
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__ntimeel2005(self._handle)
        
        @ntimeelements.setter
        def ntimeelements(self, ntimeelements):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__ntimeelb6b3(self._handle, \
                ntimeelements)
        
        @property
        def nbladeharm(self):
            """
            Element nbladeharm ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 36
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nbladeharm(self._handle)
        
        @nbladeharm.setter
        def nbladeharm(self, nbladeharm):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nbladeharm(self._handle, \
                nbladeharm)
        
        @property
        def nblademodes(self):
            """
            Element nblademodes ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 36
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nblademodes(self._handle)
        
        @nblademodes.setter
        def nblademodes(self, nblademodes):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nblademodes(self._handle, \
                nblademodes)
        
        @property
        def modeorder(self):
            """
            Element modeorder ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 37
            """
            array_ndim, array_type, array_shape, array_handle = \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__array__modeorder(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                modeorder = self._arrays[array_hash]
            else:
                try:
                    modeorder = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _mockdt.f90wrap_defineallproperties__solveroptionsdef__array__modeorder)
                except TypeError:
                    modeorder = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = modeorder
            return modeorder
        
        @modeorder.setter
        def modeorder(self, modeorder):
            self.modeorder[...] = modeorder
        
        @property
        def ncosinflowharm(self):
            """
            Element ncosinflowharm ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 38
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__ncosinf3b81(self._handle)
        
        @ncosinflowharm.setter
        def ncosinflowharm(self, ncosinflowharm):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__ncosinf375e(self._handle, \
                ncosinflowharm)
        
        @property
        def nmaxinflowpoly(self):
            """
            Element nmaxinflowpoly ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 38
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nmaxinf0dda(self._handle)
        
        @nmaxinflowpoly.setter
        def nmaxinflowpoly(self, nmaxinflowpoly):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nmaxinff41c(self._handle, \
                nmaxinflowpoly)
        
        @property
        def linflm(self):
            """
            Element linflm ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 38
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__linflm(self._handle)
        
        @linflm.setter
        def linflm(self, linflm):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__linflm(self._handle, \
                linflm)
        
        @property
        def linrzpts(self):
            """
            Element linrzpts ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 39
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__linrzpts(self._handle)
        
        @linrzpts.setter
        def linrzpts(self, linrzpts):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__linrzpts(self._handle, \
                linrzpts)
        
        @property
        def controlhistoption(self):
            """
            Element controlhistoption ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 40
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__control2fe2(self._handle)
        
        @controlhistoption.setter
        def controlhistoption(self, controlhistoption):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__control0e2b(self._handle, \
                controlhistoption)
        
        @property
        def nrevolutions(self):
            """
            Element nrevolutions ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 40
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nrevolu91d0(self._handle)
        
        @nrevolutions.setter
        def nrevolutions(self, nrevolutions):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nrevolu57c7(self._handle, \
                nrevolutions)
        
        @property
        def nazim(self):
            """
            Element nazim ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 40
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nazim(self._handle)
        
        @nazim.setter
        def nazim(self, nazim):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nazim(self._handle, \
                nazim)
        
        @property
        def ntimesteps(self):
            """
            Element ntimesteps ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 41
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__ntimesteps(self._handle)
        
        @ntimesteps.setter
        def ntimesteps(self, ntimesteps):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__ntimesteps(self._handle, \
                ntimesteps)
        
        @property
        def nred(self):
            """
            Element nred ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 42
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nred(self._handle)
        
        @nred.setter
        def nred(self, nred):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nred(self._handle, \
                nred)
        
        @property
        def nred2(self):
            """
            Element nred2 ftype=integer            pytype=int
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 42
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__nred2(self._handle)
        
        @nred2.setter
        def nred2(self, nred2):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__nred2(self._handle, \
                nred2)
        
        @property
        def trimconvergence(self):
            """
            Element trimconvergence ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 43
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__trimcona726(self._handle)
        
        @trimconvergence.setter
        def trimconvergence(self, trimconvergence):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__trimcon7c3a(self._handle, \
                trimconvergence)
        
        @property
        def integerror(self):
            """
            Element integerror ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 43
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__integerror(self._handle)
        
        @integerror.setter
        def integerror(self, integerror):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__integerror(self._handle, \
                integerror)
        
        @property
        def linrzpert(self):
            """
            Element linrzpert ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 44
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__linrzpert(self._handle)
        
        @linrzpert.setter
        def linrzpert(self, linrzpert):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__linrzpert(self._handle, \
                linrzpert)
        
        @property
        def controlamplitude(self):
            """
            Element controlamplitude ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 45
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__control513b(self._handle)
        
        @controlamplitude.setter
        def controlamplitude(self, controlamplitude):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__controlfbaf(self._handle, \
                controlamplitude)
        
        @property
        def controlfrequency(self):
            """
            Element controlfrequency ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 45
            """
            return \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__get__control3a84(self._handle)
        
        @controlfrequency.setter
        def controlfrequency(self, controlfrequency):
            _mockdt.f90wrap_defineallproperties__solveroptionsdef__set__control732d(self._handle, \
                controlfrequency)
        
        @property
        def jac(self):
            """
            Element jac ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 46
            """
            array_ndim, array_type, array_shape, array_handle = \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__array__jac(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                jac = self._arrays[array_hash]
            else:
                try:
                    jac = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _mockdt.f90wrap_defineallproperties__solveroptionsdef__array__jac)
                except TypeError:
                    jac = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = jac
            return jac
        
        @jac.setter
        def jac(self, jac):
            self.jac[...] = jac
        
        @property
        def jac2(self):
            """
            Element jac2 ftype=real(kind=rdp) pytype=float
            Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 46
            """
            array_ndim, array_type, array_shape, array_handle = \
                _mockdt.f90wrap_defineallproperties__solveroptionsdef__array__jac2(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                jac2 = self._arrays[array_hash]
            else:
                try:
                    jac2 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _mockdt.f90wrap_defineallproperties__solveroptionsdef__array__jac2)
                except TypeError:
                    jac2 = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = jac2
            return jac2
        
        @jac2.setter
        def jac2(self, jac2):
            self.jac2[...] = jac2
        
        def __str__(self):
            ret = ['<solveroptionsdef>{\n']
            ret.append('    trimswitch : ')
            ret.append(repr(self.trimswitch))
            ret.append(',\n    updateguess : ')
            ret.append(repr(self.updateguess))
            ret.append(',\n    deltaairloads : ')
            ret.append(repr(self.deltaairloads))
            ret.append(',\n    linrzswitch : ')
            ret.append(repr(self.linrzswitch))
            ret.append(',\n    timemarchswitch : ')
            ret.append(repr(self.timemarchswitch))
            ret.append(',\n    freewakeswitch : ')
            ret.append(repr(self.freewakeswitch))
            ret.append(',\n    windtunnelswitch : ')
            ret.append(repr(self.windtunnelswitch))
            ret.append(',\n    rigidbladeswitch : ')
            ret.append(repr(self.rigidbladeswitch))
            ret.append(',\n    fet_qddot : ')
            ret.append(repr(self.fet_qddot))
            ret.append(',\n    fet_response : ')
            ret.append(repr(self.fet_response))
            ret.append(',\n    store_fet_responsejac : ')
            ret.append(repr(self.store_fet_responsejac))
            ret.append(',\n    fet_responsejacavail : ')
            ret.append(repr(self.fet_responsejacavail))
            ret.append(',\n    airframevib : ')
            ret.append(repr(self.airframevib))
            ret.append(',\n    fusharm : ')
            ret.append(repr(self.fusharm))
            ret.append(',\n    axialdof : ')
            ret.append(repr(self.axialdof))
            ret.append(',\n    composite_coupling : ')
            ret.append(repr(self.composite_coupling))
            ret.append(',\n    trimtechnique : ')
            ret.append(repr(self.trimtechnique))
            ret.append(',\n    trimsweepoption : ')
            ret.append(repr(self.trimsweepoption))
            ret.append(',\n    ntimeelements : ')
            ret.append(repr(self.ntimeelements))
            ret.append(',\n    nbladeharm : ')
            ret.append(repr(self.nbladeharm))
            ret.append(',\n    nblademodes : ')
            ret.append(repr(self.nblademodes))
            ret.append(',\n    modeorder : ')
            ret.append(repr(self.modeorder))
            ret.append(',\n    ncosinflowharm : ')
            ret.append(repr(self.ncosinflowharm))
            ret.append(',\n    nmaxinflowpoly : ')
            ret.append(repr(self.nmaxinflowpoly))
            ret.append(',\n    linflm : ')
            ret.append(repr(self.linflm))
            ret.append(',\n    linrzpts : ')
            ret.append(repr(self.linrzpts))
            ret.append(',\n    controlhistoption : ')
            ret.append(repr(self.controlhistoption))
            ret.append(',\n    nrevolutions : ')
            ret.append(repr(self.nrevolutions))
            ret.append(',\n    nazim : ')
            ret.append(repr(self.nazim))
            ret.append(',\n    ntimesteps : ')
            ret.append(repr(self.ntimesteps))
            ret.append(',\n    nred : ')
            ret.append(repr(self.nred))
            ret.append(',\n    nred2 : ')
            ret.append(repr(self.nred2))
            ret.append(',\n    trimconvergence : ')
            ret.append(repr(self.trimconvergence))
            ret.append(',\n    integerror : ')
            ret.append(repr(self.integerror))
            ret.append(',\n    linrzpert : ')
            ret.append(repr(self.linrzpert))
            ret.append(',\n    controlamplitude : ')
            ret.append(repr(self.controlamplitude))
            ret.append(',\n    controlfrequency : ')
            ret.append(repr(self.controlfrequency))
            ret.append(',\n    jac : ')
            ret.append(repr(self.jac))
            ret.append(',\n    jac2 : ')
            ret.append(repr(self.jac2))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    

defineallproperties = Defineallproperties()

class Precision(f90wrap.runtime.FortranModule):
    """
    Module precision
    Defined at ./Source/BasicDefs/aa0_typelist.fpp lines 9-23
    """
    @property
    def rdp(self):
        """
        Element rdp ftype=integer pytype=int
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 19
        """
        return _mockdt.f90wrap_precision__get__rdp()
    
    def get_rdp(self):
        return self.rdp
    
    @property
    def zero(self):
        """
        Element zero ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__zero()
    
    @zero.setter
    def zero(self, zero):
        _mockdt.f90wrap_precision__set__zero(zero)
    
    def get_zero(self):
        return self.zero
    
    def set_zero(self, value):
        self.zero = value
    
    @property
    def one(self):
        """
        Element one ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__one()
    
    @one.setter
    def one(self, one):
        _mockdt.f90wrap_precision__set__one(one)
    
    def get_one(self):
        return self.one
    
    def set_one(self, value):
        self.one = value
    
    @property
    def half(self):
        """
        Element half ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__half()
    
    @half.setter
    def half(self, half):
        _mockdt.f90wrap_precision__set__half(half)
    
    def get_half(self):
        return self.half
    
    def set_half(self, value):
        self.half = value
    
    @property
    def two(self):
        """
        Element two ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__two()
    
    @two.setter
    def two(self, two):
        _mockdt.f90wrap_precision__set__two(two)
    
    def get_two(self):
        return self.two
    
    def set_two(self, value):
        self.two = value
    
    @property
    def three(self):
        """
        Element three ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__three()
    
    @three.setter
    def three(self, three):
        _mockdt.f90wrap_precision__set__three(three)
    
    def get_three(self):
        return self.three
    
    def set_three(self, value):
        self.three = value
    
    @property
    def four(self):
        """
        Element four ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__four()
    
    @four.setter
    def four(self, four):
        _mockdt.f90wrap_precision__set__four(four)
    
    def get_four(self):
        return self.four
    
    def set_four(self, value):
        self.four = value
    
    @property
    def six(self):
        """
        Element six ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__six()
    
    @six.setter
    def six(self, six):
        _mockdt.f90wrap_precision__set__six(six)
    
    def get_six(self):
        return self.six
    
    def set_six(self, value):
        self.six = value
    
    @property
    def eight(self):
        """
        Element eight ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__eight()
    
    @eight.setter
    def eight(self, eight):
        _mockdt.f90wrap_precision__set__eight(eight)
    
    def get_eight(self):
        return self.eight
    
    def set_eight(self, value):
        self.eight = value
    
    @property
    def pi(self):
        """
        Element pi ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__pi()
    
    @pi.setter
    def pi(self, pi):
        _mockdt.f90wrap_precision__set__pi(pi)
    
    def get_pi(self):
        return self.pi
    
    def set_pi(self, value):
        self.pi = value
    
    @property
    def twopi(self):
        """
        Element twopi ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__twopi()
    
    @twopi.setter
    def twopi(self, twopi):
        _mockdt.f90wrap_precision__set__twopi(twopi)
    
    def get_twopi(self):
        return self.twopi
    
    def set_twopi(self, value):
        self.twopi = value
    
    @property
    def d2r(self):
        """
        Element d2r ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__d2r()
    
    @d2r.setter
    def d2r(self, d2r):
        _mockdt.f90wrap_precision__set__d2r(d2r)
    
    def get_d2r(self):
        return self.d2r
    
    def set_d2r(self, value):
        self.d2r = value
    
    @property
    def r2d(self):
        """
        Element r2d ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__r2d()
    
    @r2d.setter
    def r2d(self, r2d):
        _mockdt.f90wrap_precision__set__r2d(r2d)
    
    def get_r2d(self):
        return self.r2d
    
    def set_r2d(self, value):
        self.r2d = value
    
    @property
    def xk2fps(self):
        """
        Element xk2fps ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__xk2fps()
    
    @xk2fps.setter
    def xk2fps(self, xk2fps):
        _mockdt.f90wrap_precision__set__xk2fps(xk2fps)
    
    def get_xk2fps(self):
        return self.xk2fps
    
    def set_xk2fps(self, value):
        self.xk2fps = value
    
    @property
    def lb2n(self):
        """
        Element lb2n ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__lb2n()
    
    @lb2n.setter
    def lb2n(self, lb2n):
        _mockdt.f90wrap_precision__set__lb2n(lb2n)
    
    def get_lb2n(self):
        return self.lb2n
    
    def set_lb2n(self, value):
        self.lb2n = value
    
    @property
    def ftlb2nm(self):
        """
        Element ftlb2nm ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__ftlb2nm()
    
    @ftlb2nm.setter
    def ftlb2nm(self, ftlb2nm):
        _mockdt.f90wrap_precision__set__ftlb2nm(ftlb2nm)
    
    def get_ftlb2nm(self):
        return self.ftlb2nm
    
    def set_ftlb2nm(self, value):
        self.ftlb2nm = value
    
    @property
    def one80(self):
        """
        Element one80 ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__one80()
    
    @one80.setter
    def one80(self, one80):
        _mockdt.f90wrap_precision__set__one80(one80)
    
    def get_one80(self):
        return self.one80
    
    def set_one80(self, value):
        self.one80 = value
    
    @property
    def ft2m(self):
        """
        Element ft2m ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__ft2m()
    
    @ft2m.setter
    def ft2m(self, ft2m):
        _mockdt.f90wrap_precision__set__ft2m(ft2m)
    
    def get_ft2m(self):
        return self.ft2m
    
    def set_ft2m(self, value):
        self.ft2m = value
    
    @property
    def gsi(self):
        """
        Element gsi ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__gsi()
    
    @gsi.setter
    def gsi(self, gsi):
        _mockdt.f90wrap_precision__set__gsi(gsi)
    
    def get_gsi(self):
        return self.gsi
    
    def set_gsi(self, value):
        self.gsi = value
    
    @property
    def gfps(self):
        """
        Element gfps ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__gfps()
    
    @gfps.setter
    def gfps(self, gfps):
        _mockdt.f90wrap_precision__set__gfps(gfps)
    
    def get_gfps(self):
        return self.gfps
    
    def set_gfps(self, value):
        self.gfps = value
    
    @property
    def three60(self):
        """
        Element three60 ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__three60()
    
    @three60.setter
    def three60(self, three60):
        _mockdt.f90wrap_precision__set__three60(three60)
    
    def get_three60(self):
        return self.three60
    
    def set_three60(self, value):
        self.three60 = value
    
    @property
    def in2ft(self):
        """
        Element in2ft ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__in2ft()
    
    @in2ft.setter
    def in2ft(self, in2ft):
        _mockdt.f90wrap_precision__set__in2ft(in2ft)
    
    def get_in2ft(self):
        return self.in2ft
    
    def set_in2ft(self, value):
        self.in2ft = value
    
    @property
    def five(self):
        """
        Element five ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
        """
        return _mockdt.f90wrap_precision__get__five()
    
    @five.setter
    def five(self, five):
        _mockdt.f90wrap_precision__set__five(five)
    
    def get_five(self):
        return self.five
    
    def set_five(self, value):
        self.five = value
    
    def __str__(self):
        ret = ['<precision>{\n']
        ret.append('    rdp : ')
        ret.append(repr(self.rdp))
        ret.append(',\n    zero : ')
        ret.append(repr(self.zero))
        ret.append(',\n    one : ')
        ret.append(repr(self.one))
        ret.append(',\n    half : ')
        ret.append(repr(self.half))
        ret.append(',\n    two : ')
        ret.append(repr(self.two))
        ret.append(',\n    three : ')
        ret.append(repr(self.three))
        ret.append(',\n    four : ')
        ret.append(repr(self.four))
        ret.append(',\n    six : ')
        ret.append(repr(self.six))
        ret.append(',\n    eight : ')
        ret.append(repr(self.eight))
        ret.append(',\n    pi : ')
        ret.append(repr(self.pi))
        ret.append(',\n    twopi : ')
        ret.append(repr(self.twopi))
        ret.append(',\n    d2r : ')
        ret.append(repr(self.d2r))
        ret.append(',\n    r2d : ')
        ret.append(repr(self.r2d))
        ret.append(',\n    xk2fps : ')
        ret.append(repr(self.xk2fps))
        ret.append(',\n    lb2n : ')
        ret.append(repr(self.lb2n))
        ret.append(',\n    ftlb2nm : ')
        ret.append(repr(self.ftlb2nm))
        ret.append(',\n    one80 : ')
        ret.append(repr(self.one80))
        ret.append(',\n    ft2m : ')
        ret.append(repr(self.ft2m))
        ret.append(',\n    gsi : ')
        ret.append(repr(self.gsi))
        ret.append(',\n    gfps : ')
        ret.append(repr(self.gfps))
        ret.append(',\n    three60 : ')
        ret.append(repr(self.three60))
        ret.append(',\n    in2ft : ')
        ret.append(repr(self.in2ft))
        ret.append(',\n    five : ')
        ret.append(repr(self.five))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

precision = Precision()

def set_defaults(self, interface_call=False):
    """
    =======================================================================
                             EXECUTABLE CODE
    =======================================================================
    
    set_defaults(self)
    Defined at ./Source/HeliSrc/set_defaults.fpp lines 9-26
    
    Parameters
    ----------
    solver : Solveroptionsdef
    """
    _mockdt.f90wrap_set_defaults(solver=self._handle)

def assign_constants(interface_call=False):
    """
    =======================================================================
    local constants (one-time use)
    =======================================================================
    
    assign_constants()
    Defined at ./Source/BasicDefs/assign_constants.fpp lines 8-74
    
    """
    _mockdt.f90wrap_assign_constants()

