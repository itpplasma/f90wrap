"""
Module defineallproperties
Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp lines 13-47
"""
from __future__ import print_function, absolute_import, division
import _mockdtpkg
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_mockdtpkg = _SafeDirectCExecutor(_mockdtpkg, module_import_name='_mockdtpkg')

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("mockdtpkg.SolverOptionsDef")
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
            result = _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef_initialise()
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
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef_finalise(this=self._handle)
    
    @property
    def trimswitch(self):
        """
        Element trimswitch ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 24
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__trimswitch(self._handle)
    
    @trimswitch.setter
    def trimswitch(self, trimswitch):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__trimswitch(self._handle, \
            trimswitch)
    
    @property
    def updateguess(self):
        """
        Element updateguess ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 24
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__updateguess(self._handle)
    
    @updateguess.setter
    def updateguess(self, updateguess):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__updateguess(self._handle, \
            updateguess)
    
    @property
    def deltaairloads(self):
        """
        Element deltaairloads ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 24
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__deltaaief83(self._handle)
    
    @deltaairloads.setter
    def deltaairloads(self, deltaairloads):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__deltaai9421(self._handle, \
            deltaairloads)
    
    @property
    def linrzswitch(self):
        """
        Element linrzswitch ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 25
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__linrzswitch(self._handle)
    
    @linrzswitch.setter
    def linrzswitch(self, linrzswitch):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__linrzswitch(self._handle, \
            linrzswitch)
    
    @property
    def timemarchswitch(self):
        """
        Element timemarchswitch ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 26
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__timemare3b3(self._handle)
    
    @timemarchswitch.setter
    def timemarchswitch(self, timemarchswitch):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__timemar4f99(self._handle, \
            timemarchswitch)
    
    @property
    def freewakeswitch(self):
        """
        Element freewakeswitch ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 27
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__freewak3c80(self._handle)
    
    @freewakeswitch.setter
    def freewakeswitch(self, freewakeswitch):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__freewak8069(self._handle, \
            freewakeswitch)
    
    @property
    def windtunnelswitch(self):
        """
        Element windtunnelswitch ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 28
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__windtund117(self._handle)
    
    @windtunnelswitch.setter
    def windtunnelswitch(self, windtunnelswitch):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__windtun0496(self._handle, \
            windtunnelswitch)
    
    @property
    def rigidbladeswitch(self):
        """
        Element rigidbladeswitch ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 29
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__rigidbldcbe(self._handle)
    
    @rigidbladeswitch.setter
    def rigidbladeswitch(self, rigidbladeswitch):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__rigidbl1493(self._handle, \
            rigidbladeswitch)
    
    @property
    def fet_qddot(self):
        """
        Element fet_qddot ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 30
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__fet_qddot(self._handle)
    
    @fet_qddot.setter
    def fet_qddot(self, fet_qddot):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__fet_qddot(self._handle, \
            fet_qddot)
    
    @property
    def fet_response(self):
        """
        Element fet_response ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 30
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__fet_resc250(self._handle)
    
    @fet_response.setter
    def fet_response(self, fet_response):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__fet_res15d8(self._handle, \
            fet_response)
    
    @property
    def store_fet_responsejac(self):
        """
        Element store_fet_responsejac ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 31
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__store_f538c(self._handle)
    
    @store_fet_responsejac.setter
    def store_fet_responsejac(self, store_fet_responsejac):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__store_fb24a(self._handle, \
            store_fet_responsejac)
    
    @property
    def fet_responsejacavail(self):
        """
        Element fet_responsejacavail ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 31
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__fet_res230e(self._handle)
    
    @fet_responsejacavail.setter
    def fet_responsejacavail(self, fet_responsejacavail):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__fet_resf178(self._handle, \
            fet_responsejacavail)
    
    @property
    def airframevib(self):
        """
        Element airframevib ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 32
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__airframevib(self._handle)
    
    @airframevib.setter
    def airframevib(self, airframevib):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__airframevib(self._handle, \
            airframevib)
    
    @property
    def fusharm(self):
        """
        Element fusharm ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 32
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__fusharm(self._handle)
    
    @fusharm.setter
    def fusharm(self, fusharm):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__fusharm(self._handle, \
            fusharm)
    
    @property
    def axialdof(self):
        """
        Element axialdof ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 33
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__axialdof(self._handle)
    
    @axialdof.setter
    def axialdof(self, axialdof):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__axialdof(self._handle, \
            axialdof)
    
    @property
    def composite_coupling(self):
        """
        Element composite_coupling ftype=logical pytype=bool
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 33
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__composiee25(self._handle)
    
    @composite_coupling.setter
    def composite_coupling(self, composite_coupling):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__composic943(self._handle, \
            composite_coupling)
    
    @property
    def trimtechnique(self):
        """
        Element trimtechnique ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 34
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__trimtecb616(self._handle)
    
    @trimtechnique.setter
    def trimtechnique(self, trimtechnique):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__trimtec7319(self._handle, \
            trimtechnique)
    
    @property
    def trimsweepoption(self):
        """
        Element trimsweepoption ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 35
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__trimswe87ad(self._handle)
    
    @trimsweepoption.setter
    def trimsweepoption(self, trimsweepoption):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__trimswe913a(self._handle, \
            trimsweepoption)
    
    @property
    def ntimeelements(self):
        """
        Element ntimeelements ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 35
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__ntimeel2005(self._handle)
    
    @ntimeelements.setter
    def ntimeelements(self, ntimeelements):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__ntimeelb6b3(self._handle, \
            ntimeelements)
    
    @property
    def nbladeharm(self):
        """
        Element nbladeharm ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 36
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nbladeharm(self._handle)
    
    @nbladeharm.setter
    def nbladeharm(self, nbladeharm):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nbladeharm(self._handle, \
            nbladeharm)
    
    @property
    def nblademodes(self):
        """
        Element nblademodes ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 36
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nblademodes(self._handle)
    
    @nblademodes.setter
    def nblademodes(self, nblademodes):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nblademodes(self._handle, \
            nblademodes)
    
    @property
    def modeorder(self):
        """
        Element modeorder ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 37
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__array__modeorder(self._handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            modeorder = self._arrays[array_hash]
        else:
            try:
                modeorder = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        self._handle,
                                        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__array__modeorder)
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
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__ncosinf3b81(self._handle)
    
    @ncosinflowharm.setter
    def ncosinflowharm(self, ncosinflowharm):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__ncosinf375e(self._handle, \
            ncosinflowharm)
    
    @property
    def nmaxinflowpoly(self):
        """
        Element nmaxinflowpoly ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 38
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nmaxinf0dda(self._handle)
    
    @nmaxinflowpoly.setter
    def nmaxinflowpoly(self, nmaxinflowpoly):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nmaxinff41c(self._handle, \
            nmaxinflowpoly)
    
    @property
    def linflm(self):
        """
        Element linflm ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 38
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__linflm(self._handle)
    
    @linflm.setter
    def linflm(self, linflm):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__linflm(self._handle, \
            linflm)
    
    @property
    def linrzpts(self):
        """
        Element linrzpts ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 39
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__linrzpts(self._handle)
    
    @linrzpts.setter
    def linrzpts(self, linrzpts):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__linrzpts(self._handle, \
            linrzpts)
    
    @property
    def controlhistoption(self):
        """
        Element controlhistoption ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 40
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__control2fe2(self._handle)
    
    @controlhistoption.setter
    def controlhistoption(self, controlhistoption):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__control0e2b(self._handle, \
            controlhistoption)
    
    @property
    def nrevolutions(self):
        """
        Element nrevolutions ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 40
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nrevolu91d0(self._handle)
    
    @nrevolutions.setter
    def nrevolutions(self, nrevolutions):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nrevolu57c7(self._handle, \
            nrevolutions)
    
    @property
    def nazim(self):
        """
        Element nazim ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 40
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nazim(self._handle)
    
    @nazim.setter
    def nazim(self, nazim):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nazim(self._handle, \
            nazim)
    
    @property
    def ntimesteps(self):
        """
        Element ntimesteps ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 41
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__ntimesteps(self._handle)
    
    @ntimesteps.setter
    def ntimesteps(self, ntimesteps):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__ntimesteps(self._handle, \
            ntimesteps)
    
    @property
    def nred(self):
        """
        Element nred ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 42
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nred(self._handle)
    
    @nred.setter
    def nred(self, nred):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nred(self._handle, \
            nred)
    
    @property
    def nred2(self):
        """
        Element nred2 ftype=integer            pytype=int
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 42
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__nred2(self._handle)
    
    @nred2.setter
    def nred2(self, nred2):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__nred2(self._handle, \
            nred2)
    
    @property
    def trimconvergence(self):
        """
        Element trimconvergence ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 43
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__trimcona726(self._handle)
    
    @trimconvergence.setter
    def trimconvergence(self, trimconvergence):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__trimcon7c3a(self._handle, \
            trimconvergence)
    
    @property
    def integerror(self):
        """
        Element integerror ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 43
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__integerror(self._handle)
    
    @integerror.setter
    def integerror(self, integerror):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__integerror(self._handle, \
            integerror)
    
    @property
    def linrzpert(self):
        """
        Element linrzpert ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 44
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__linrzpert(self._handle)
    
    @linrzpert.setter
    def linrzpert(self, linrzpert):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__linrzpert(self._handle, \
            linrzpert)
    
    @property
    def controlamplitude(self):
        """
        Element controlamplitude ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 45
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__control513b(self._handle)
    
    @controlamplitude.setter
    def controlamplitude(self, controlamplitude):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__controlfbaf(self._handle, \
            controlamplitude)
    
    @property
    def controlfrequency(self):
        """
        Element controlfrequency ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 45
        """
        return \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__get__control3a84(self._handle)
    
    @controlfrequency.setter
    def controlfrequency(self, controlfrequency):
        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__set__control732d(self._handle, \
            controlfrequency)
    
    @property
    def jac(self):
        """
        Element jac ftype=real(kind=rdp) pytype=float
        Defined at ./Source/BasicDefs/aa2_defineAllProperties.fpp line 46
        """
        array_ndim, array_type, array_shape, array_handle = \
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__array__jac(self._handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            jac = self._arrays[array_hash]
        else:
            try:
                jac = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        self._handle,
                                        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__array__jac)
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
            _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__array__jac2(self._handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            jac2 = self._arrays[array_hash]
        else:
            try:
                jac2 = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        self._handle,
                                        _mockdtpkg.f90wrap_defineallproperties__solveroptionsdef__array__jac2)
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
    


_array_initialisers = []
_dt_array_initialisers = []


try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module \
        "defineallproperties".')

for func in _dt_array_initialisers:
    func()
