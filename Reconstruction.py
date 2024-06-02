from aspire.nufft import nufft, anufft
from aspire.numeric import fft
from functools import partial
import numpy as np
from pynufft import NUFFT
from skimage.data import camera, shepp_logan_phantom
from scipy.sparse.linalg import cg, LinearOperator
from scipy.spatial.transform import Rotation as sp_rot
from skimage.transform import resize
from time import time


class TestData:
    """Create simple normalized 2D and 3D sample images"""
        
    @staticmethod
    def sphere(res, ndim=2, r=None, seed=None):
        """
        Construct normalized N-dimensional hypersphere.

        Parameters:
        -----------
        res : int, resolution of object.
        ndim : int, optional (default 2), dimension of hypersphere.
        r : list, length ndim, optional (default uniform between 0.1 and 0.4), 
            radii in each dimension.
        seed : int, optional (default None), random seed.
        
        Returns:
        --------
        ndim np.ndarray representing hypersphere.
        """
        if seed is not None: np.random.seed(seed)
        if r is None: r = np.random.uniform(0.1,0.4,ndim)
        grid = slice(0,1,res*1j)
        Mgrid = np.mgrid[(grid,)*ndim]
        return 1.0*(np.sum([(Mgrid[i]-0.5)**2/r[i]**2 for i in range(ndim)],axis=0)<1.)

    @staticmethod
    def rect(res, ndim=2, w=None, seed=None):
        """
        Construct normalized N-dimensional rectangular hypercuboid.

        Parameters:
        -----------
        res : int, resolution of object.
        ndim : int, optional (default 2), dimension of hypercuboid.
        w : list, length ndim, optional (default uniform between 0.1 and 0.4), 
            widths in each dimension.
        seed : int, optional (default None), random seed.
        
        Returns:
        --------
        ndim np.ndarray representing hypercuboid.
        """
        if seed is not None: np.random.seed(seed)
        if w is None: w = np.random.uniform(0.1,0.4,ndim)
        grid = slice(0,1,res*1j)
        Mgrid = np.mgrid[(grid,)*ndim]
        return 1.0*np.prod([np.abs(Mgrid[i]-0.5)<w[i] for i in range(ndim)],axis=0)

    @staticmethod
    def gauss(res, ndim=2, s=None, seed=None):
        """
        Construct normalized N-dimensional Gaussian.

        Parameters:
        -----------
        res : int, resolution of object.
        ndim : int, optional (default 2), dimension of hypersphere.
        s : list, length ndim, optional (default uniform between 0.05 and 0.2), 
            standard width in each dimension.
        seed : int, optional (default None), random seed.

        Returns:
        --------
        ndim np.ndarray representing Gaussian.
        """
        if seed is not None: np.random.seed(seed)
        if s is None: s = np.random.uniform(0.05,0.2,ndim)
        grid = slice(0,1,res*1j)
        Mgrid = np.mgrid[(grid,)*ndim]
        g = np.prod([np.exp(-(Mgrid[i]-0.5)**2/(2*s[i]**2)) for i in range(ndim)],axis=0)
        return g/g.max()

    @staticmethod
    def sl_phantom(res, ndim=2, _=None, seed=None):
        """
        Returns normalized Shepp-Logan phantom image rescaled to resolution `res`.
        Dummy arguments are added to match the form of the other data functions.
        """
        if ndim != 2: 
            print('\033[91mOnly ndim=2 available. Returned `None`.\033[0m')
            return None
        slp = resize(shepp_logan_phantom(),(res,res))
        return slp/slp.max()

    @staticmethod
    def camera_man(res, ndim=2, _=None, seed=None): 
        """
        Returns normalized camera man image rescaled to resolution `res`.
        Dummy arguments are added to match the form of the other data functions.
        """
        if ndim != 2: 
            print('\033[91mOnly ndim=2 available. Returned `None`.\033[0m')
            return None
        cm = resize(camera(),(res,res))
        return cm/cm.max()

    @staticmethod
    def molecule(res, ndim=3, _=None, seed=None):
        """
        Returns EMDB-2660 molecule structure rescaled to resolution `res`.
        Dummy arguments are added to match the form of the other data functions.
        """
        if ndim != 3: 
            print('\033[91mOnly ndim=3 available. Returned `None`.\033[0m')
            return None
        from aspire.downloader import emdb_2660
        molecule = emdb_2660().downsample(res)
        return molecule._data[0].astype(np.float64)

    @staticmethod
    def channelspin(res, ndim=3, _=None, seed=0):
        """
        Returns simulated ChannelSpin data set of potassium ion channel,
        rescaled to resolution `res`. `Seed` selects one of 54 rotations of the 
        top part of the channel. 
        Dummy arguments are added to match the form of the other data functions.
        """
        if ndim != 3: 
            print('\033[91mOnly ndim=3 available. Returned `None`.\033[0m')
            return None
        from aspire.downloader import simulated_channelspin
        molecule = simulated_channelspin()['vols'][seed%54].downsample(res)
        return molecule._data[0].astype(np.float64)

class Projector:
    
    def __init__(self, data, orientations):
        """
        Projector object initiated with 2D or 3D data and orientations in 
        which to project this data
    
        Parameters
        ----------
        data : np.ndarray, image data in a square array
        orientations : float, list or array-like, rotation angles in radians.
        """
        self.data = data
        self.orientations = orientations
        self.res = self.data.shape[-1]
        self.n_rots = orientations.shape[0]
        self.ndim = len(self.data.shape)

    def project(self):
        """
        Uses the Fourier projection slice theorem to generate projections
        of an image or volume under different orientations: A slice of the Fourier
        transformed image/volume under given orientation equals the Fourier transform 
        of a projection onto a line/plane under the same orientation.
        
        Returns
        -------
        Projections onto lines/planes of the provided orientations, numpy.ndarray of 
        shape K-by-L(-by-L) with K the number of orientations and L the image resolution.
        """
        if self.ndim == 2:
            return self._project_2d()
        elif self.ndim == 3:
            return self._project_3d()
        
    def _project_2d(self):
        """
        Uses the Fourier projection slice theorem to generate 1D projections
        of a 2D image under different orientations.
        
        Returns
        -------
        Projections onto planes of the provided orientations, numpy.ndarray of 
        shape K-by-L with K the number of orientations and L the image resolution.
        """
        # Generate rotation matrices
        rot_matrices = self.rotation_matrices_2d(self.orientations)
        # Generate rotated grids in Fourier space (points of Fourier slices)
        fslice_pts = self.rotated_grids(self.res, rot_matrices)
        fslice_pts = fslice_pts.reshape((2, self.n_rots * self.res))
        
        # Sample image in Fourier space on rotated grid
        fslices = nufft(self.data, fslice_pts) / self.res
        fslices = fslices.reshape(-1, self.res)
        if self.res % 2 == 0:
            fslices[:, 0] = 0
        
        # Revert back to real space
        # The centered ifft first applies ifft_shift, then ifft, finished by fft_shift
        self.projections = fft.centered_ifft(fslices).real
        return self.projections

    def _project_3d(self):
        """
        Uses the Fourier projection slice theorem to generate 2D projections
        of a 3D volumes under different orientations.
        
        Returns
        -------
        Projections onto planes of the provided orientations, numpy.ndarray of 
        shape K-by-L-by-L with K the number of orientations and L the volume resolution.
        """
        # Generate rotation matrices
        rot_matrices = self.rotation_matrices_3d(self.orientations)
        # Generate rotated grids in Fourier space (points of Fourier slices)
        fslice_pts = self.rotated_grids(self.res, rot_matrices, ndim=3)
        fslice_pts = fslice_pts.reshape(3, self.n_rots * self.res**2)
        
        # Sample image in Fourier space on rotated grid
        fslices = nufft(self.data, fslice_pts) / self.res
        fslices = fslices.reshape(-1, self.res, self.res)
        if self.res % 2 == 0:
            fslices[:, 0, :] = 0
            fslices[:, :, 0] = 0
        
        # Revert back to real space
        # The centered ifft first applies ifft_shift, then ifft, finished by fft_shift
        self.projections = fft.centered_ifft2(fslices).real
        return self.projections

    @staticmethod
    def rotation_matrices_2d(angles):
        """
        Generate 2D rotation matrices with provided rotation angles
        
        Parameters
        ----------
        angles : float, list or array-like, rotation angles in radians.
    
        Returns
        -------
        2D rotation matrices of shape K-by-2-by-2 with K the number of angles provided.
        """
        angles = np.reshape(angles,(np.size(angles),))
        rots = np.array([[np.cos(angles),-np.sin(angles)],
                         [np.sin(angles),np.cos(angles)]]).transpose((2,0,1))
        return rots

    @staticmethod
    def rotation_matrices_3d(angles,dtype=None):
        """
        Generate 3D rotation matrices with provided ZYZ Euler angles
        
        Parameters
        ----------
        angles : float, K-by-3 np.ndarray, ZYZ Eulerian rotation angles in radians.
    
        Returns
        -------
        3D rotation matrices of shape K-by-3-by-3 with K the number of angles provided.
        """
        dtype = dtype or getattr(angles, "dtype", np.float64)
        rotations = sp_rot.from_euler("ZYZ", angles, degrees=False)
        matrices = rotations.as_matrix().astype(dtype, copy=False)
        return matrices

    @staticmethod
    def rotated_grids(L, rot_matrices, ndim=2):
        """
        Generate ndim-1 dimensional Fourier grids rotated in ndim dimensional space
    
        Parameters
        ----------
        L : int, the resolution of the desired grids.
        rot_matrices : An array of size K-by-3-by-3 containing K rotation matrices.
        ndim : dimension of space in which to place the grids (i.e. image 
            reconstruction --> 2, volume --> 3).
        
        Returns
        -------
        A set of rotated Fourier grids ndim dimensions as specified by the 
        rotation matrices. Frequencies are in the range [-pi, pi].
        """
        # 0-centered grid spanning [-1,1) for even and (-1,1) for odd resolution
        start = (-L // 2 + 1 - 1*(L%2 == 0)) / (L/2)
        end = (L // 2 - 1*(L%2 == 0)) / (L/2)
        grid = slice(start,end,L*1j)
    
        Mgrid = np.mgrid[(grid,)*(ndim-1)].astype(rot_matrices.dtype)
    
        # Number of rotation matrices
        num_rots = rot_matrices.shape[0]
        num_pts = L**(ndim-1)
    
        # Frequency points placed in xy.. order to apply rotations.
        pts = np.pi * np.vstack([coord.flatten() for coord in Mgrid[::-1]] + [np.zeros(num_pts, dtype=rot_matrices.dtype),])
        pts_rot = np.zeros((ndim, num_rots, num_pts), dtype=rot_matrices.dtype)
        
        for i in range(num_rots):
            pts_rot[:, i, :] = rot_matrices[i, :, :] @ pts
    
        # Reshape rotated frequency points and convert back into ..yx convention.
        return pts_rot.reshape((ndim, num_rots,) + (L,)*(ndim-1))[::-1]


class Reconstructor(Projector):
    
    def __init__(self,projections,orientations):
        """
        Reconstructor object initiated by projections and corresponding 
        orientations.
        
        Parameters:
        -----------
        projections : np.ndarray, K-by-L or K-by-L-by-L, 1D or 2D projections
        orientations : np.ndarray, K or K-by-3, (ZYZ Eulerian) rotation angles 
            corresponding to projections
        """
        self.projections = projections
        self.orientations = orientations
        self.res = self.projections.shape[-1] # image resolution
        self.n_rots = orientations.shape[0]
        self.ndim = len(self.projections.shape[1:])+1

        if self.ndim == 2:
            # Generate rotation matrices
            self.rot_matrices = self.rotation_matrices_2d(self.orientations)
            # Fourier transformed projections are equivalent to slices of the 2D Fourier transform
            fslices = fft.centered_fft(self.projections)    
            if self.res % 2 == 0: fslices[:, 0] = 0 
        elif self.ndim == 3:
            # Generate rotation matrices
            self.rot_matrices = self.rotation_matrices_3d(self.orientations)
            # Fourier transformed projections are equivalent to slices of the 3D Fourier transform
            fslices = fft.centered_fft2(self.projections)
            if self.res % 2 == 0: fslices[:, 0, :] = 0; fslices[:, :, 0] = 0
        
        self.fslices = fslices.reshape(self.n_rots * self.res**(self.ndim-1))   
        # Generate rotated grids in Fourier space bast on known projection orientations
        fslice_pts = self.rotated_grids(self.res, self.rot_matrices, ndim=self.ndim)
        self.fslice_pts = fslice_pts.reshape(self.ndim, self.n_rots * self.res**(self.ndim-1))

        self.dtype = self.fslice_pts.dtype
    
    def reconstruct(self, method='lscg', maxiter=50, store_iters=False, **kwargs):
        """
        Uses the Fourier projection slice theorem to reconstruct a image/volume
        from projections under known orientations: A slice of the Fourier
        transformed image/volume under given orientation equals the Fourier 
        transform of a projection onto a line/plane under the same orientation.
        
        Parameters
        ----------
        method : str, optional (default 'lscg'), inverse solving method (for 
            inverse non-uniform FFT).
        maxiter : int, optional (default 50), maximum number of iterations in cg.
        store_iters : bool, optional (default False), store reconstruction at 
            intermediat CG iterations in the `info` dictionary.
        
        Returns
        -------
        reconstructed : L-by-L reconstruction of image of which projections where provided.
        info : dictionary with info on iterations, time and intermediate iteration results.
        """
        # 2D inverse Fourier to reconstruct 2D images, obtained with non-uniform FFT
        self.info = {'niter':0, 'time':-time(), 'intermediate':[]}
        if not store_iters:
            def cb(xb):
                self.info['niter'] += 1
        
        method = method.lower()
        if method not in ('dfi', 'lscg', 'adj', 'adjoint'): 
            print(f'\033[93mMethod {method} is not implemented, returned None.\033[0m')
            return None
        
        if method in ('adjoint', 'adj'):
            NufftObj = self._plan_PyNUFFT(**kwargs)
            self.reconstructed = NufftObj.adjoint(self.fslices).real.astype(self.dtype)
        
        elif method == 'dfi': # Direct method: direct Fourier inversion
            NufftObj = self._plan_PyNUFFT(**kwargs)
            if store_iters:
                def cb(xb):
                    inter = NufftObj.k2xx(xb.reshape(NufftObj.Kd))/NufftObj.sn*self.res
                    self.info['intermediate'].append(inter.reshape((self.res,)*self.ndim).T.real.astype(self.dtype))
                    self.info['niter'] += 1
            self.reconstructed = NufftObj.solve(
                self.fslices, solver="cg", maxiter=maxiter, atol=0, tol=1e-5, callback=cb
            ).real.astype(self.dtype) * self.res
        
        elif method == 'lscg': # Variational method: least squares optimal solution found using conjugate gradient on normal equations
            if store_iters:
                def cb(xb):
                    self.info['intermediate'].append(xb.reshape((self.res,)*self.ndim).T.real.astype(self.dtype))
                    self.info['niter'] += 1
            kernel, ATb = self._precompute_normaleq()
            self.reconstructed = self.conj_grad(ATb, kernel, maxiter=maxiter, callback=cb, **kwargs)

        self.info['time'] += time()
        return self.reconstructed, self.info

    def _precompute_normaleq(self):
        """
        Precompute A* b and A* A (kernel) from normal equations.
        
        Returns
        -------
        Ker : np.ndarray, convolution kernel representing A* A operation.
        ATb : np.ndarray, backwards projected data A* b.
        """
        fslices = self.fslices/self.res
        b = fslices.flatten()
    
        ## Pre-compute the back-projection A* b using NUFFT
        ATb = anufft(b, self.fslice_pts[::-1], (self.res,)*self.ndim, real=True) / (self.n_rots*self.res**(self.ndim-1))
        
        ## Pre-compute the convolution kernel Ker(n-l) = A* A(n,l) using NUFFT
        W = np.repeat(np.ones((self.n_rots,1))/self.n_rots,self.res)
        if self.res % 2 == 0: W[::self.res] = 0
        if self.ndim == 3: 
            W = np.repeat(W,self.res)
            if self.res % 2 == 0: W[::self.res] = 0
        Ker = anufft(W, self.fslice_pts[::-1], (2*self.res,)*self.ndim, real=True) / self.res**(self.ndim+1)
        Ker[0, :] = 0; Ker[:, 0] = 0
        if self.ndim == 3: Ker[:, :, 0] = 0
        
        return Ker, ATb

    def _plan_PyNUFFT(self, KSOR=1.2, IS=6):
        """
        Create a plannend PyNUFFT object.
        
        Parameters
        ----------
        KSOR : float, k-space oversampling ratio.
        IS : int, interpolation size.

        Returns
        -------
        NUFFT object
        """
        NufftObj = NUFFT()
        Nd = (self.res,)*self.ndim  # image size
        Kd = (int(np.ceil(self.res*KSOR)),)*self.ndim  # k-space size with slight oversampling to remove artifacts
        Jd = (IS,)*self.ndim  # interpolation size
        NufftObj.plan(self.fslice_pts.T, Nd, Kd, Jd)
        return NufftObj
            
    def conj_grad(self, ATb, kernel, tol=1e-5, maxiter=None, regularizer=0, callback=None):
        """
        Solves Ax=b where Ax is a convolution of kernel with x, using conjugant 
        gradient technique on the normal equations.
        
        Parameters
        ----------
        ATb : np.ndarray, back-projected data
        kernel : np.ndarray, kernel representing the A* A operator in the 
            normal equations.
        tol : float, optional (default 1e-5), relative tollerance.
        maxiter : int, optional (default None), maximum number of iterations.
        regularizer : float, optional (default 0), constant added to kernel as 
            optional means of regularization.
        callback : callable, optional (default None), callback function called 
            at each CG iterations. Must have 1 argument for the reconstruction 
            at the current iteration.

        Returns
        -------
        Reconstructed image/volume, as output from the CG.
        """
        N = ATb.shape[0]
        count = ATb.size
    
        # CG kernel providing linear operator A* A on x such that A* A x = A* b
        if regularizer > 0: kernel += regularizer
        operator = LinearOperator((count, count), matvec = partial(self.apply_kernel, kernel=kernel))
        
        x, info = cg(operator, ATb.flatten(), M=None, tol=tol, atol=0, maxiter=maxiter, callback=callback)
    	
        if info != 0: 
            print(f"\033[91mCG did not converge within maximum number of iterations {maxiter}\033[0m")
        
        return x.reshape((N,)*self.ndim).T

    @staticmethod
    def apply_kernel(x, kernel=None):
        """Convolves kernel with x in Fourier space"""
        ndim = len(kernel.shape)
        N = round((x.shape[0])**(1/ndim))  # Assuming x is a flattened square matrix
        N_ker = kernel.shape[0]  # size of convolution kernel
        x = x.reshape((N,)*ndim)  # Reshape x to 2D
        # Fourier transform kernel
        kernel = fft.mdim_ifftshift(kernel, range(0,ndim))
        kermat_f = np.real(fft.fftn(kernel, axes=tuple(range(0,ndim))))
        # Fourier transform x
        x_f = fft.fftn(np.pad( x, [(0,N_ker-N)]*ndim ))
        # Inverse Fourier of product is convolution
        if ndim == 2:
            return np.real(fft.ifftn( x_f * kermat_f ))[:N,:N]
        elif ndim == 3:
            return np.real(fft.ifftn( x_f * kermat_f ))[:N,:N,:N]


class Inspect():
    
    import matplotlib.pyplot as plt
    
    @staticmethod
    def nLSE(d,r,axis=None):
        r"""Returns normalized least squares error $\|d-r\|/\|d\|$."""
        return np.sqrt(np.sum((d-r)**2,axis=axis)/np.sum(d**2))
        
    @classmethod
    def plot_3d_recon(cls, data, reconstruction, projection_angles=None, seed=None, title=None, show=True):
        """Create plot comparing projections from ground truth data and reconstruction."""
        # Generate random projections of ground truth data and reconstruction
        if projection_angles is None:
            if isinstance(seed, int): np.random.seed(seed)
            projection_angles = np.random.uniform([-np.pi,-np.pi/2,-np.pi],[np.pi,np.pi/2,np.pi],size=(1000,3))
        data_projections = Projector(data,projection_angles).project()
        reconstruction_projections = Projector(reconstruction,projection_angles).project()
    
        fig, ax = cls.plt.subplots(3,4+1,figsize=(13.5,6))
        cls.plt.suptitle(title)
        for i in range(4+1):
            if i<4:
                Id = ax[0,i].imshow(data_projections[i], cmap="gray"); cls.plt.colorbar(Id,ax=ax[0,i])
                Ir = ax[1,i].imshow(reconstruction_projections[i], cmap="gray"); cls.plt.colorbar(Ir,ax=ax[1,i])
                Ie = ax[2,i].imshow(np.abs(data_projections[i]-reconstruction_projections[i]),cmap='plasma'); cls.plt.colorbar(Ie,ax=ax[2,i])
                ax[0,i].set_title(fr"Projection {i+1}",fontsize=10)
                ax[2,i].set_title(fr"NLSE = {cls.nLSE(data_projections[i],reconstruction_projections[i]):1.2f}",fontsize=8)
            else:
                Id = ax[0,i].imshow(data_projections.mean(axis=0), cmap="gray"); cls.plt.colorbar(Id,ax=ax[0,i])
                Ir = ax[1,i].imshow(reconstruction_projections.mean(axis=0), cmap="gray"); cls.plt.colorbar(Ir,ax=ax[1,i])
                Ie = ax[2,i].imshow(np.abs(data_projections-reconstruction_projections).mean(axis=0),cmap='plasma'); cls.plt.colorbar(Ie,ax=ax[2,i])
                ax[0,i].set_title(fr"Average projection",fontsize=10)
                ax[2,i].set_title(fr"Mean NLSE = {cls.nLSE(data_projections,reconstruction_projections,axis=(-2,-1)).mean():1.2f}",fontsize=8)
            for j in range(3):
                ax[j,i].set_xticks([]); ax[j,i].set_yticks([])
        ax[0,0].set_ylabel("Projections ground truth")
        ax[1,0].set_ylabel("Projections reconstructed")
        ax[2,0].set_ylabel("Absolute error")
        cls.plt.tight_layout()
        if show: cls.plt.show()
        return fig, ax

    @classmethod
    def plot_3d_FOV_compare(cls, dat1, dat2, XYZ, title=None, show=True, 
                            dpi=None, cmap='plasma', tight_layout=False):
        """Create plot comparing FOV sensitivity stored in dat1 and dat2."""
        plt = cls.plt
        fig = plt.figure(figsize=(7, 8),dpi=dpi)
        ax = [fig.add_subplot(221, projection='3d')]
        ax.append(fig.add_subplot(222, projection='3d'))
        ax.append(fig.add_subplot(223, projection='3d'))
        ax.append(fig.add_subplot(224, projection='3d'))
            
        plt.suptitle(title,x=0.62,y=0.92)
        X,Y,Z = XYZ
    
        datmin = np.min([dat1.min(),dat2.min()])
        datmax = np.max([dat1.max(),dat2.max()])
        kw = {
            'vmin': datmin,
            'vmax': datmax,
            'levels': np.linspace(datmin, datmax, 10),
            'cmap': cmap
        }
        
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
    
        for a in ax: 
            a.set_box_aspect((1,0.6,1))
            a.view_init(40,-30, 0)
            a.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
            a.set_xlabel(r"$FOV_\alpha$ ($\pi$ rad)",labelpad=1)
            a.set_ylabel(r"$FOV_\beta$ ($\pi$ rad)",labelpad=-3)
            a.set_zlabel(r"$FOV_\gamma$ ($\pi$ rad)",labelpad=-1)
            a.tick_params(axis='y', pad=-2,labelsize=8) 
            a.tick_params(axis='x', pad=-2,labelsize=8)
            a.tick_params(axis='z', pad=-1,labelsize=8)
        ax[0].set_title("LSCG",rotation='vertical',x=0.05,y=0.65,fontsize=11,
                        verticalalignment='center',horizontalalignment='center')
        ax[2].set_title("DFI",rotation='vertical',x=0.05,y=0.6,fontsize=11,
                        verticalalignment='center',horizontalalignment='center')

        for i, dat in enumerate([dat1, dat2]):
            ax[2*i].contourf(dat[-1,:,:],Y[-1,:,:],Z[-1,:,:],zdir='x',offset=xmax,**kw) # 4, 0-4, 0-4
            ax[2*i].contourf(X[:,:,-1],Y[:,:,-1],dat[:,:,-1],zdir='z',offset=zmax,**kw) # 0-4, 0-4, 4
            ax[2*i].contourf(X[:,0,:],dat[:,0,:],Z[:,0,:],zdir='y',offset=0,**kw) # 0-4,0,0-4
            ax[2*i].plot([xmin,xmax],[ymin,ymin],zmax,c="grey",zorder=1e3)
            ax[2*i].plot([xmax,xmax],[ymin,ymax],zmax,c="grey",zorder=1e3)
            ax[2*i].plot([xmax,xmax],[ymin,ymin],[zmin,zmax],c="grey",zorder=1e3)
            ax[2*i].text(0.5*(xmax-xmin)+xmin,0*(xmax-xmin)+xmin,0.5*(xmax-xmin)+xmin,
                         f"{dat.max():1.4f}",zdir='x',color='black',
                         verticalalignment='center',horizontalalignment='center',zorder=1e4)
        
            ax[2*i+1].contourf(X[:,-1,:],dat[:,-1,:],Z[:,-1,:],zdir='y',offset=ymax,**kw) # 0-4, 4, 0-4
            ax[2*i+1].contourf(dat[0,:,:],Y[0,:,:],Z[0,:,:],zdir='x',offset=0,**kw) # 0, 0-4, 0-4
            c = ax[2*i+1].contourf(X[:,:,0],Y[:,:,0],dat[:,:,0],zdir='z',offset=0,**kw) # 0-4, 0-4, 0
            ax[2*i+1].plot([xmin,xmax],[ymax,ymax],zmin,c="grey",zorder=1e3)
            ax[2*i+1].plot([xmin,xmin],[ymin,ymax],zmin,c="grey",zorder=1e3)
            ax[2*i+1].plot([xmin,xmin],[ymax,ymax],[zmin,zmax],c="grey",zorder=1e3)
            ax[2*i+1].text(0.5*(xmax-xmin)+xmin,0.5*(xmax-xmin)+xmin,0.5*(xmax-xmin)+xmin,
                           f"{dat.min():1.4f}",zdir='x',color='white',
                           verticalalignment='center',horizontalalignment='center',zorder=1e4)
    
        cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.75])
        fig.colorbar(c, cax=cb_ax, label='Normalized least-squares error')
        if tight_layout: plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if show: plt.show()
        return fig, ax


