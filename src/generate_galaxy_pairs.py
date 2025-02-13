import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
from astropy.cosmology import Planck18 as cosmo
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
from skimage.measure import block_reduce
import scipy
from scipy.spatial import ConvexHull
import scipy.ndimage as ndi
import random
import copy
import joblib
from sklearn.neighbors import KernelDensity
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.robust.scale import mad as MAD
from skimage import measure as skmeasure
from matplotlib.path import Path
import argparse
from config import get_config
import h5py
import pdb
from skimage.transform import resize
import time
import ast
import tensorflow as tf



tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

t0 = time.time()


current_time = int(t0)
random.seed(current_time)


warnings.filterwarnings('ignore', category=np.RankWarning)


class CALIFAAugmentor:
    def __init__(self, config, file_cube,extended=True,recenter=False):
        
        
        self.config = config
        self.extended = extended
        self.recenter = recenter


        
        self.desired_dim_cube = np.array((0,0))
        self.range_shift = np.array((0,0))
        if self.extended == True :
            self.desired_dim_cube[-1] = 192
            self.desired_dim_cube[-2] = 184
            self.range_shift[-1] = 16
            self.range_shift[-2] = 16
            self.min_num_pix = 743
            self.min_flux = 5e-6

            
        else :
            self.desired_dim_cube[-1] = 96
            self.desired_dim_cube[-2] = 92
            self.range_shift[-1] = 8
            self.range_shift[-2] = 8
            self.min_num_pix = 167
            self.min_flux = 1e-5


        self.folder_data = self.config.get('FOLDER_INFO', 'folder_data')
        folder_aux = self.config.get('FOLDER_INFO', 'folder_aux')
        self.file_pca = config.get('FILES_INFO', 'file_pca')
        
        if self.extended == True :
            folder_cubes = self.folder_data + '/ecubes/'
        else :
            folder_cubes = self.folder_data + '/cubes/'



        '''
        Load data cube
        '''
        
        x = fits.open(folder_cubes + file_cube)
        if self.extended == False :
            self.name_galaxy = file_cube.split('.V500.rscube.fits')[0]
        else :
            self.name_galaxy = file_cube.split('.V500.drscube.fits')[0]


        len_gal = len(self.name_galaxy.split('_'))
        if len_gal > 1 :
            self.interaction = True
        else :
            self.interaction = False

            

        self.data_cube = x[0].data
        self.error_cube = x[1].data
        self.badpix_cube = x[3].data


        '''
        Flag nan or inf values
        '''
        
        where_nan = (np.isnan(self.data_cube) == True) | (np.isinf(self.data_cube) == True)
        where_zero = (self.error_cube <= 0)
        where_flag = where_zero | where_nan
        self.data_cube[where_flag] = 0.0
        self.badpix_cube[where_flag] = 1
        

        self.SN_min = 1.
        self.SN_max = 200.
        
                    

        
        '''
        Wavelenght range
        '''
        wavelength_sampling = x[0].header['CD3_3']
        wavelength_start = x[0].header['CRVAL3']
        self.old_wavelengths = wavelength_start + wavelength_sampling * np.arange(x[0].header['NAXIS3'])
        self.wavelengths = wavelength_start + wavelength_sampling * np.arange(x[0].header['NAXIS3'])
              
        '''
        Cube dimensions
        '''
        
        self.len_wavelength = len(self.wavelengths)
        self.len_x = self.data_cube.shape[1]
        self.len_y = self.data_cube.shape[2]

        
        
        '''
        Load Mask stars
        '''
        
        if self.extended == True :
            self.mask_stars = x[5].data
            self.mask_stars = self.mask_stars == 1.
            
        else :
        
            try :
                file_mask_star = file_cube.split('rscube.fits')[0] + 'mask.fits'

                y = fits.open(self.folder_data + 'mask_stars/' + file_mask_star)

            
                self.mask_stars = y[0].data
            
                self.mask_stars = (self.mask_stars == 2.) | (self.mask_stars == 1.)
            except :
                self.mask_stars = np.zeros((self.len_x,self.len_y)).astype(bool)
                

        '''
        Merge mask and antimask
        '''

        try :
            self.upload_my_custome_antimask()

            self.mask_stars = self.mask_stars.astype(int)
            self.mask_stars = self.mask_stars + self.my_antimask

            where_g1 = self.mask_stars >= 1
            self.mask_stars[where_g1] = True
            self.mask_stars[~where_g1] = False
            self.mask_stars = self.mask_stars.astype(bool)

        except :
            print('NO ANTIMASK')



        '''
        Load galaxy mask

        '''

        try :
            if self.extended == False :
                with h5py.File(self.folder_data + '/galaxy_mask_cube/' + self.name_galaxy + '.h5', 'r') as f:
                    self.galaxy_mask = f['gal_mask'][:,:]
                f.close()
            else :
                with h5py.File(self.folder_data + '/galaxy_mask_ecube/' + self.name_galaxy + '.h5', 'r') as f:
                    self.galaxy_mask = f['gal_mask'][:,:]
                f.close()
            print('Charching galaxy mask')
        except :
            self.galaxy_mask = np.zeros((self.len_x,self.len_y)).astype(bool)
            print('No galaxy mask')





        
        '''
        Homegenized cube dimension for spatial coordinates
        '''
        
        

        self.data_cube = self.update_dim(self.data_cube,0)
        self.error_cube = self.update_dim(self.error_cube,1e10)
        self.badpix_cube = self.update_dim(self.badpix_cube,1)
        self.mask_stars = self.update_dim(self.mask_stars,False)
        self.galaxy_mask = self.update_dim(self.galaxy_mask,True)

        self.len_x = self.data_cube.shape[1]
        self.len_y = self.data_cube.shape[2]




        '''
        Define border mask, 1 (0) is a bad (good) pixel
        '''
        self.border_mask = np.zeros((self.len_x,self.len_y))
        borders = self.badpix_cube.sum(axis=0) == self.len_wavelength
        borders = borders & (self.data_cube.sum(axis=0) == 0.)
        self.border_mask[borders] = 1

        '''
        Move galaxy to rest frame (z = 0)
        '''
        
        file_catalogue_EDR = folder_aux + 'galaxies_properties.fits'
        ed = fits.open(file_catalogue_EDR)
        mother_sample = {'redshift' : ed[1].data['z'], 'ID' : ed[1].data['cubename']}
        
        file_catalogue_DR = folder_aux  + 'CALIFA_DR3_Catalogs.fits'
        ec = fits.open(file_catalogue_DR)
        mother_sample_dr = {'redshift' : ec[1].data['redshift_CALIFA'],
              'ID' :  ec[1].data['REALNAME']}
        
                    
         
        try:
            self.redshift = self._get_redshift(self.name_galaxy, mother_sample)
            if self.redshift is None:
                self.redshift = self._get_redshift(self.name_galaxy, mother_sample_dr)
                if self.redshift is None:
                    raise ValueError
        except:
            self.redshift = 0
            print('REDSHIFT NOT FOUND')



        self.wavelengths = self.wavelengths/(1 + self.redshift)


        D_L = cosmo.angular_diameter_distance(self.redshift).value

        self.data_cube *= 4*np.pi * D_L**2 * 1e-5 * (1 + self.redshift)
        self.error_cube *= 4*np.pi * D_L**2 * 1e-5 * (1 + self.redshift)



        '''
        Remove the edges of the spectrum
        '''


        self.lam_range = np.array([3739,6803])

        cond_wave = ( self.wavelengths >= self.lam_range[0]) & (self.wavelengths <= self.lam_range[1])


        self.wavelengths = self.wavelengths[cond_wave]
        self.data_cube = self.data_cube[cond_wave,:,:]
        self.error_cube = self.error_cube[cond_wave,:,:]
        self.badpix_cube = self.badpix_cube[cond_wave,:,:]



        self.badpix_cube = self.expand_bad_values(self.badpix_cube,20)


        self.SN = (self.data_cube/self.error_cube)


        '''
        Update cube dimensions
        '''


        self.len_wavelength = len(self.wavelengths)
        self.len_x = self.data_cube.shape[1]
        self.len_y = self.data_cube.shape[2]
        self.center_pixel = (self.len_x // 2, self.len_y // 2)


        '''
        Charge PCA, file generated with PCA_fit.py
        '''
        
        self.pca = joblib.load(folder_aux + self.file_pca)
 



        '''
        Charge the distribution of stars, i.e. positions in polar
        coordinate, radii, and number of stars per galaxy
        '''

        if self.extended == True :
            self.star_distribution = joblib.load(folder_aux + 'star_distribution_edr.pkl')
        else :
            self.star_distribution = joblib.load(folder_aux + 'star_distribution_dr.pkl')


        r_polar = self.star_distribution['coordinate'][:,0]
        theta = self.star_distribution['coordinate'][:,1]
        data = np.vstack([r_polar, theta]).T
        self.star_position_dist = KernelDensity(kernel='gaussian', bandwidth=1).fit(data)
      
      
      
            
    def _get_redshift(self, name, sample):
        where_gal = name == sample['ID']
        if where_gal.sum() == 0:
            where_gal = name.split('_')[0] == sample['ID']
        if where_gal.sum() > 0:
            return sample['redshift'][where_gal][0]
        return None

            
            
    def update_dim(self,cube,value) :
        current_dim = cube.shape
        desired_dim = np.array(copy.deepcopy(current_dim))
        desired_dim[-1] = self.desired_dim_cube[-1]
        desired_dim[-2] = self.desired_dim_cube[-2]
        dif = desired_dim - np.array(current_dim)
        
        add_dif = []
        for i in range(len(dif)) :
            add_dif.append((0,0))
        
        
        add_dif[-2] = (dif[-2]//2,dif[-2] - dif[-2]//2)
        add_dif[-1] = (dif[-1]//2,dif[-1] - dif[-1]//2)
        
        
        cube = np.pad(cube, add_dif,mode='constant', constant_values=value)
        return cube
        
    def polar_to_cartesian(self,coordinates):
        r = coordinates[:,0]
        theta = coordinates[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    def cartesian_to_pixel_coordinates(self,cartesian_coordinates):
        # Find the image center
        cx, cy = ( self.len_x  - 1) / 2, (self.len_y  - 1) / 2
        # Add the image center to the Cartesian coordinates
        pixel_coordinates = cartesian_coordinates + np.array([cx, cy])
        # Round the resulting values to integers
        pixel_coordinates = np.round(pixel_coordinates).astype(int)

        return pixel_coordinates
    
    def upload_my_custome_antimask(self) :
        self.my_antimask = np.zeros((self.len_x,self.len_y))
        if self.extended == False :
            name_file = self.folder_data + 'my_custome_antimask_cube/' + self.name_galaxy + '.my.antimask.txt'
        else :
            name_file = self.folder_data + 'my_custome_antimask_ecube/' + self.name_galaxy + '.my.antimask.txt'
        
        if os.path.isfile(name_file):
            print('Charching antimask')
            pixels = []
            flags = []
            with open(name_file, "r") as f:
                next(f)
                for line in f:
                    line = line.strip().split("\t")
                    x, y, flag = int(line[0]), int(line[1]), int(line[2])
                    pixels.append((x, y))
                    flags.append(flag)
            f.close()
            pixels = np.array(pixels)
            where_inside = pixels < self.my_antimask.shape
            where_inside = where_inside.sum(axis=1) == 2
            pixels = pixels[where_inside,:]
            indices = np.ravel_multi_index(np.transpose(pixels), self.my_antimask.shape)

            np.put(self.my_antimask, indices, flags)



    
    def expand_bad_values(self,flags, window_size=20):

        bool_flags = (flags > 0) & (self.border_mask[np.newaxis,:,:] == 0)

        new_flags = copy.deepcopy(bool_flags)
        bad_indices = np.where(bool_flags)


        for i, j, k in zip(*bad_indices):

            lower_limit = self.wavelengths[i] - window_size / 2
            upper_limit = self.wavelengths[i] + window_size / 2

            s = (self.wavelengths >= lower_limit) & (self.wavelengths <= upper_limit)
            new_flags[s,j,k] = 1

        
        return new_flags

    def create_el_mask(self) :
        
        '''
        Emission line vector
        '''
        
        file_SDSS = '../auxiliary_files/Standard.SDSS.mask'
        with open(file_SDSS) as f:
            lines = f.readlines()

        lines = lines[1:38]
        w1, w2 = ([],[])
        for line in lines:
            w1.append(line[0:4])
            w2.append(line[8:12])
        w1 = np.array(w1) ; w2 = np.array(w2)


        self.el_mask = np.zeros(self.len_wavelength).astype(bool)
        self.el_mask[:] = False
        for i in range(len(w1)) :
            cond_true = (self.wavelengths >= float(w1[i])) & (self.wavelengths <= float(w2[i]))
            self.el_mask[cond_true] = True
        
                     
        file_skyline = '../auxiliary_files/sky_lines.mask'
        with open(file_skyline) as f:
            lines = f.readlines()

        lines = lines[0:7]
        w1, w2 = ([],[])
        for line in lines:
            w1.append(line[0:6])
            w2.append(line[7:13])
        w1 = np.array(w1) ; w2 = np.array(w2)
        
        w1 = w1.astype(float)
        w2 = w2.astype(float)
        
        w1 = w1.astype(float)/(1 + self.redshift)
        w2 = w2.astype(float)/(1 + self.redshift)
        
        self.el_sky = np.zeros(self.len_wavelength).astype(bool)
        self.el_sky[:] = False
        for i in range(len(w1)) :
            cond_true = (self.wavelengths >= w1[i] - 10) & (self.wavelengths <= w2[i] + 10)
            self.el_sky[cond_true] = True

        file_skyline = '../extra_files/sky_lines.mask'
        with open(file_skyline) as f:
            lines = f.readlines()
            
        lines = lines[7:]
        w1, w2 = ([],[])
        for line in lines:
            w1.append(line[0:6])
            w2.append(line[7:13])
        w1 = np.array(w1) ; w2 = np.array(w2)
        
        w1 = w1.astype(float)
        w2 = w2.astype(float)
        
        w1 = w1.astype(float)/(1 + self.redshift)
        w2 = w2.astype(float)/(1 + self.redshift)

        self.el_sky_astro = np.zeros(self.len_wavelength).astype(bool)
        self.el_sky_astro[:] = False
        for i in range(len(w1)) :
            cond_true = (self.wavelengths >= w1[i] - 10) & (self.wavelengths <= w2[i] + 10)
            self.el_sky_astro[cond_true] = True


            
    def update_equidistant_values(self,SN,equidistant_bool) :
    
        where_inf = equidistant_bool & ((np.isinf(SN) == True) | (np.isnan(SN) == True))
    

        if where_inf.sum() > 0 :
            vec_inf = np.where(where_inf == True)[0]
            where_no_inf = (np.isinf(SN) == False) & (np.isnan(SN) == False)
            vec_no_inf = np.where(where_no_inf == True)[0]
            for i in range(where_inf.sum()) :
                dif  = np.abs(vec_no_inf - vec_inf[i])
                where_min = np.where(dif == np.min(dif))[0]
                equidistant_bool[where_min] = True
        
                                    
                                        
        return equidistant_bool
        
        


     
    def get_galaxy_contour(self,data_cube,SN_map,SN_threshold,replace=False) :
        contours = self.image_contours(SN_map, SN_threshold)
        contours = self.convex_hull_polygon(contours)
        self.mask_contour = self.contours_to_mask(contours, SN_map)
        if replace == True :
            data_cube[:,~self.mask_contour] = 0.0
        
        mask_outside = ~self.mask_contour
        
        return data_cube, mask_outside

                
    def image_contours(self,image, level, smooth=False, gauss=True, single=True, upper=False,
                       sigma=2.0, kernel=6.0, bval=-100., origin=1.0, chunks=None):
        # x = contours[:, 1] | y = contours[:, 0]
        ny, nx = image.shape
        fimg = np.zeros((ny + 2, nx + 2)) + bval
        y, x = np.indices(fimg.shape)
        if smooth:
            if gauss:
                img = scipy.ndimage.gaussian_filter(image, sigma=sigma, order=0)
            else:
                img = scipy.ndimage.filters.median_filter(image, kernel)
        else:
            img = image.copy()
        fimg[1:-1, 1:-1] = img
        if upper:
            fimg[fimg >= level] = 100. * level
            fimg[fimg < level]  = bval
        if chunks is not None:
            mask = image_quadrant(fimg, chunks=chunks)
            fimg[~mask] = bval
        # list of (n,2) - ndarrays
        # Each contour is an ndarray of shape (n, 2), consisting of n (x, y) coordinates
        contours = skmeasure.find_contours(fimg, level)
        contours = [contour - origin for contour in contours]
        if single:
            len_contours = np.array([len(contour) for contour in contours])
            clevel = contours[len_contours.argmax()] if len_contours.size > 0 else None
            return clevel
        else:
            return contours

    def contours_to_mask(self,contours, shape):
        if isinstance(contours, (list, tuple)):
            contours = np.array(contours)
        if isinstance(shape, np.ndarray):
            shape = shape.shape
        ny, nx = shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((y, x)).T
        path = Path(contours)
        mask = path.contains_points(points)
        mask = mask.reshape(shape)
        return mask

    def convex_hull_polygon(self,points):
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        return vertices

    def noise_realisation(self,SN_fit_cond):

        cov = np.zeros((self.len_wavelength,self.len_wavelength))
        np.fill_diagonal(cov, 1)
        random_field = np.random.multivariate_normal(np.zeros(self.len_wavelength), cov, size=(self.len_x, self.len_y))
        random_field = random_field.reshape(self.len_wavelength, self.len_x, self.len_y)
        self.SN = (self.data_cube/self.error_cube)
        cond = self.SN <= self.SN_min
        
        self.SN[cond] = self.SN_min
        
        equidistant_values = np.linspace(0, self.len_wavelength-1, 10, dtype=int)
        equidistant_bool = np.zeros(self.len_wavelength).astype(bool)
        equidistant_bool[:] = False
        equidistant_bool[equidistant_values] = True


        
        if SN_fit_cond == True :
            data_vector = self.data_cube.reshape(self.len_wavelength,self.len_x*self.len_y)
            edata_vector = self.error_cube.reshape(self.len_wavelength,self.len_x*self.len_y)
            badpix_vector = self.badpix_cube.reshape(self.len_wavelength,self.len_x*self.len_y)
            self.SN_vector = (data_vector/edata_vector)
            cond = (self.SN_vector <= self.SN_min) | (np.isnan(self.SN_vector) == True)
            self.SN_vector[cond] = self.SN_min
            self.SN_fit = copy.deepcopy(self.SN_vector)
            self.good_val = np.zeros((self.len_wavelength,self.len_x*self.len_y)).astype(bool)
            

            
            for i in range(self.len_x*self.len_y) :
                
                self.good_val[:,i] = (self.SN_vector[:,i] < self.SN_max) & (self.SN_vector[:,i] > self.SN_min) & (np.isnan(self.SN_vector[:,i]) == False) & (np.isinf(self.SN_vector[:,i]) == False) & (badpix_vector[:,i] == False) & (self.el_sky == False) & (self.el_mask == False)

                if (self.good_val[:,i].sum() > 0) & (data_vector[:,i].sum() != 0.0) :
                    equidistant_bool = self.update_equidistant_values(self.SN_vector[:,i],equidistant_bool)
                    self.good_val[equidistant_bool,i] = True
                    degree = 8
                    coefficients = np.polyfit(self.wavelengths[self.good_val[:,i]], self.SN_vector[self.good_val[:,i],i], degree)
                    polynomial = np.poly1d(coefficients)
                    self.SN_fit[:,i] = polynomial(self.wavelengths)
                    

                    
                    where_good_lines = (self.SN_vector[:,i] > self.SN_fit[:,i]) & (self.el_mask == True) & (self.SN_vector[:,i] >= self.SN_min) & (np.isnan(self.SN_vector[:,i]) == False) & (np.isinf(self.SN_vector[:,i]) == False) & (badpix_vector[:,i] == False)
                    


                    
                    self.SN_fit[where_good_lines,i] = self.SN_vector[where_good_lines,i]
                    where_very_low = (self.SN_fit[:,i] < self.SN_min)
                    self.SN_fit[where_very_low,i] = self.SN_min
                    
           
            self.SN_fit = self.SN_fit.reshape(self.len_wavelength, self.len_x, self.len_y)
            self.SN_vector = self.SN_vector.reshape(self.len_wavelength, self.len_x, self.len_y)
            self.good_val = self.good_val.reshape(self.len_wavelength, self.len_x,self.len_y)
            self.SN_map = np.nanmedian(self.SN_fit,axis=0)
            self.SN_map[self.center_pixel] = 1000

#            random_field *= (1/self.SN_fit)

            SN_random_value = random.randint(1,10)
            self.SN[:,:] = SN_random_value
            print('random S/N realization :', SN_random_value)
            random_field *= (1/self.SN)
        
        return self.data_cube * (1 + random_field)
     
    def degrade_spectral_resolution(self,cube1,new_step,update_wavelenght=False) :
        
        '''
        Reduce the spectral resolution in order to have lower dimensions
        '''
        
        wavelengths_new = np.arange(self.lam_range[0],self.lam_range[-1]+1,new_step)
        len_wavelength_new = len(wavelengths_new)
        
        new_cube1 = np.zeros((len_wavelength_new,self.len_x,self.len_y))
        new_badpix_cube = np.zeros((len_wavelength_new,self.len_x,self.len_y))
         

        for i in range(self.len_x) :
            for j in range(self.len_y) :
                new_cube1[:,i,j] = np.interp(wavelengths_new,self.wavelengths,cube1[:,i,j])
                new_badpix_cube[:,i,j] = np.interp(wavelengths_new,self.wavelengths,self.badpix_cube[:,i,j])

        
        if update_wavelenght == True :
            self.badpix_cube = new_badpix_cube
            self.wavelengths_old = copy.deepcopy(self.wavelengths)
            self.wavelengths = copy.deepcopy(wavelengths_new)
            self.len_wavelength = len(self.wavelengths)
        
        return new_cube1
        
        
    def gaussian_filter(self,cube,sigma) :
        
        '''
        Pass a Multidimensional Gaussian filter with a sigma
        '''
        sigma_spec, sigma_spatial = sigma
        new_cube = gaussian_filter(cube, sigma = (sigma_spec,sigma_spatial,sigma_spatial))
                                   
        return new_cube
    
   
    def get_SN_my_cube(self,cube) :
        data_vector = cube.reshape((self.len_wavelength,self.len_x*self.len_y))
        
        data_vector = cube.reshape((self.len_wavelength,self.len_x*self.len_y))
        cond_5600 = (self.wavelengths > 5600) & (self.wavelengths < 6000)
        mean_at_5600 = np.nanmedian(data_vector[cond_5600,:],axis=0)
        where_mean_zero = mean_at_5600 <= 0
        data_vector_norm = data_vector/mean_at_5600[np.newaxis,:]
        data_vector_norm[:,where_mean_zero] = 0

        
        data_vector_t = self.pca.inverse_transform(self.pca.transform(np.transpose(data_vector_norm)))
        
        dif_spec = (data_vector_t - np.transpose(data_vector_norm))/np.transpose(data_vector_norm)
 
        sigma_spec = MAD(dif_spec[:,~self.el_mask],axis=1)
        where_very_low = np.median(np.transpose(data_vector),axis=1) < self.min_flux
        sigma_spec[where_very_low] = 1000
        SN_map = 1/sigma_spec
        SN_map = SN_map.reshape((self.len_x,self.len_y))
        where_nan = (np.isnan(SN_map) == True)
        SN_map[where_nan] = 1

        SN_map = gaussian_filter(SN_map,(2,2))
        

        

        return SN_map
        
    def remove_teluric_lines(self,cube,sigma) :
        
        border_mask_vector = self.border_mask.reshape(self.len_x*self.len_y)
        cond_inside = border_mask_vector == 0
        
                        

                
        data_vector = cube.reshape((self.len_wavelength,self.len_x*self.len_y))

        cond_5600 = (self.wavelengths > 5600) & (self.wavelengths < 6000)
        mean_at_5600 = np.nanmedian(data_vector[cond_5600,:],axis=0)
        
        where_mean_zero = mean_at_5600 <= 0
        data_vector_norm = data_vector/mean_at_5600[np.newaxis,:]
        data_vector_norm[:,where_mean_zero] = 0



        data_vector_t = self.pca.inverse_transform(self.pca.transform(np.transpose(data_vector_norm)))
        data_vector_t *= mean_at_5600[:,np.newaxis]

        
        self.data_vector = data_vector
        self.data_vector_t = data_vector_t
        
        dif_spec = data_vector - np.transpose(data_vector_t)
        mean_value_spec = np.nanmedian(dif_spec,axis=0)
        sigma_value_spec = np.nanstd(dif_spec,axis=0)

        fluctuation = (dif_spec > mean_value_spec[np.newaxis,:] + sigma*sigma_value_spec[np.newaxis,:])
        fluctuation = fluctuation | (dif_spec < mean_value_spec[np.newaxis,:] - sigma*sigma_value_spec[np.newaxis,:])


        
        fluctuation_down = (dif_spec < mean_value_spec[np.newaxis,:] - sigma*sigma_value_spec[np.newaxis,:])
        
        
        sigma_huge = 4
        
        huge_fluctuation = (dif_spec > mean_value_spec[np.newaxis,:] + sigma*sigma_value_spec[np.newaxis,:])
        huge_fluctuation = huge_fluctuation | (dif_spec < mean_value_spec[np.newaxis,:] - sigma*sigma_value_spec[np.newaxis,:])





        new_shape = (self.el_sky.shape[0],self.len_x*self.len_y)
        el_sky_mask = np.repeat(self.el_sky,new_shape[1]).reshape(new_shape)
        el_sky_astro_mask = np.repeat(self.el_sky_astro,new_shape[1]).reshape(new_shape)
        el_astro_mask = np.repeat(self.el_mask,new_shape[1]).reshape(new_shape)
        bad_pix_vector = self.badpix_cube.reshape((self.len_wavelength,self.len_x*self.len_y))
     
        
        teluric_lines = fluctuation & (el_sky_mask == True) & (el_astro_mask == False)
        
        teluric_in_astroline = fluctuation_down & (el_sky_astro_mask == True)

        bad_pix = fluctuation  & (bad_pix_vector > 0)

        fake_line = huge_fluctuation & (el_sky_mask == False) & (el_astro_mask == False)
        

        amb_lines = fluctuation & (el_sky_mask == True) & (el_astro_mask == True)


        replace_spec = (teluric_lines | bad_pix | fake_line | amb_lines | teluric_in_astroline) & (cond_inside == True)
        data_vector[replace_spec] = np.transpose(data_vector_t)[replace_spec]
        self.teluric_lines = teluric_lines.reshape((self.len_wavelength,self.len_x,self.len_y))
        cube = data_vector.reshape((self.len_wavelength,self.len_x,self.len_y))
        self.cube_pca = np.transpose(data_vector_t).reshape((self.len_wavelength,self.len_x,self.len_y))
        

        return cube
   
    
    def generate_mask_stars(self) :
        
        my_mask_stars = np.zeros((self.len_x,self.len_y)).astype(bool)
        x, y = np.indices(my_mask_stars.shape)
        if self.extended == True :
            r_max = 16
        else :
            r_max = 12

        

        self.n_stars = np.random.choice(self.star_distribution['n_stars'], 1)[0]
        print('N stars: ', self.n_stars)
        if self.n_stars > 0 :
            r_star = np.random.choice(self.star_distribution['r_stars'], self.n_stars)
            coor = self.star_position_dist.sample(self.n_stars)
            cartesian = self.polar_to_cartesian(coor)
            pix_pos = self.cartesian_to_pixel_coordinates(cartesian)
            
            for i in range(self.n_stars) :
                d = np.sqrt((x - pix_pos[i,0]) ** 2 + (y - pix_pos[i,1]) ** 2)
                if r_star[i] > r_max : r_star[i] = r_max
                where_star = d <= r_star[i]
                my_mask_stars[where_star] = True
                print('Radio: ', r_star[i])
                

        return my_mask_stars
    
    
               
    def rotate_galaxy(self,data_cube, mask_stars, gal_mask, SN_map, rot=True) :
        
        '''
        Random rotation + mirror flip in x or y
        '''
        
        if rot == True :
            angle = random.uniform(1, 359)
            flip_direction = random.randint(1, 2)
            
            data_cube = rotate(data_cube, angle, axes=(1,2), order=3, reshape=False)
            data_cube = np.flip(data_cube, axis=flip_direction)
     
            mask_stars = rotate(mask_stars, angle, axes=(0,1), order=0, reshape=False)
            mask_stars = np.flip(mask_stars, axis=flip_direction-1)
            
            gal_mask = rotate(gal_mask, angle, axes=(0,1), order=0, reshape=False)
            gal_mask = np.flip(gal_mask, axis=flip_direction-1)
            
                
            SN_map = rotate(SN_map, angle, axes=(0,1), order=3, reshape=False)
            SN_map = np.flip(SN_map, axis=flip_direction-1)
           
        else :
            angle = False
            flip_direction = False
            
        
        SN_map[mask_stars] = 0
        SN_map[gal_mask] = 0
        SN_map[self.center_pixel] = 1000

        
        return data_cube, mask_stars, gal_mask, SN_map, angle, flip_direction
    
       
       
    def compute_shift(self,cube,gal_mask,star_mask, N=20) :
        
        x = np.arange(0, self.len_x)
        y = np.arange(0, self.len_y)
        X, Y = np.meshgrid(x, y, indexing='ij')  # create a grid of coordinates
        
        lam_window = (self.wavelengths > 5400) & (self.wavelengths < 5600)

        image_2d = np.nansum(cube[lam_window,:,:], axis=0)
        
        mask = gal_mask | star_mask
        
        image_2d[mask] = 0.0
        
        flat_indices = np.argsort(-image_2d.ravel())
        top_N_indices = flat_indices[:N]

        indices = np.unravel_index(top_N_indices, image_2d.shape)
        

        mean_x = np.mean(indices[0])
        mean_y = np.mean(indices[1])

        shift_x = self.len_x // 2 - mean_x
        shift_y = self.len_y // 2 - mean_y
        
        shift = shift_x, shift_y
        mean = mean_x, mean_y
        
        
        return shift, mean


    
    def center_cube(self,cube,order,shift,gal_mask=False) :
        shift_x, shift_y = shift

        if len(cube.shape) == 3 :
            cube = ndi.shift(cube, (0, shift_x, shift_y),order=order)
        elif len(cube.shape) == 2 :
            cube = ndi.shift(cube, (shift_x, shift_y),order=order)
            if gal_mask == True :
                dim_mask = cube.shape
                white_box = np.ones((dim_mask)).astype(bool)
                white_box = ndi.shift(white_box, (shift_x, shift_y), order=0)
                where_move = white_box == False
                cube[where_move] = True

        else :
            print('CUBE DOES NOT HAVE EXPECTED DIMENSIONS')
        
        return cube


    
       
    def generate_pair(self,SN_threshold,triming,original=False) :

        # Set up
        self.SN_threshold = SN_threshold

        sigma_teluric = 2
        lambda_step = 2
        sigma_spec = 0
        sigma_spatial = 1
        ratio_field = 0.
        self.create_el_mask()

        galaxies_not_to_center = ['MCG-02-08-014', 'SN2002dl', 'MCG-01-10-015', 'MCG-02-08-014',
        'NGC5630', 'NGC5616', 'NGC0523', 'UGC12688', 'NGC3991', 'IC2098','2MASXJ01331766+1319567', 'MGC+11-08-25']
        
        
        galaxies_with_other_SN_cut = ['MGC+08-01-03', 'SN2008bz','UGC05990','UGC09708','UGC11922']
        gal2SN = { 'MGC+08-01-03' : 2.5, 'SN2008bz' : 2.5,'UGC05990' : 1.5,'UGC09708' : 2.5,'UGC11922' : 2}
        
        if self.name_galaxy in galaxies_not_to_center:
            forbiden_list = True
        else :
            forbiden_list = False
        

        if (self.name_galaxy in galaxies_with_other_SN_cut) & (self.extended == True):
            self.SN_threshold = gal2SN[self.name_galaxy]
            
        '''
        Adding gausian noise
        '''
        if (original == False) & (self.extended == False) :
            print('Two new cubes have been generated')
            self.cube1 = self.data_cube
            self.cube2 = self.noise_realisation(True)

        else :
            print('Original cube is taken')
            self.cube1 = self.data_cube
            self.cube2 = self.noise_realisation(True)


        
        '''
        Degrading spectral resolution to lambda_step
        '''

        self.cube1 = self.degrade_spectral_resolution(self.cube1,lambda_step)
        self.cube2 = self.degrade_spectral_resolution(self.cube2,lambda_step,True)
        
        

#
#
        self.create_el_mask()
#
        self.cube1 = self.remove_teluric_lines(self.cube1,sigma_teluric)
        self.cube2 = self.remove_teluric_lines(self.cube2,sigma_teluric)






        '''
        Taking the mask stars or generating new
        '''

#
        self.cond_star = self.mask_stars.sum()

        if self.cond_star == 0 :
            print('Generating new stars')
            self.cube_mask_star = self.generate_mask_stars()
        else :
            print('Taking the original mask stars')
            self.cube_mask_star = self.mask_stars
            self.n_stars = 0





        '''
        Centering image if is an interacting galaxy

        '''


        if (self.interaction == True) | (self.name_galaxy == 'UGC01162') :
            print('Centering an interacting galaxy')
            shift, center = self.compute_shift(self.cube1,self.galaxy_mask,self.cube_mask_star)

            self.cube1 = self.center_cube(self.cube1,3,shift)
            self.cube2 = self.center_cube(self.cube2,3,shift)

            self.SN_map = self.center_cube(self.SN_map,3,shift)
            self.cube_mask_star = self.center_cube(self.cube_mask_star,0,shift)
            self.galaxy_mask = self.center_cube(self.galaxy_mask,0,shift,True)

            self.cube1[:,self.galaxy_mask] = 0.0
            self.cube1[:,self.cube_mask_star] = 0.0

        self.SN_map = self.get_SN_my_cube(self.cube1)


        '''
        Rotating and/or fliping the galaxy
        '''

        if original == False :
            self.cube1, self.cube1_mask_star, self.galaxy_mask1, self.SN_cube1, self.angle1, self.flip1 = self.rotate_galaxy(self.cube1,self.cube_mask_star,self.galaxy_mask,self.SN_map)
            self.cube2, self.cube2_mask_star, self.galaxy_mask2, self.SN_cube2,self.angle2, self.flip2 = self.rotate_galaxy(self.cube2,self.cube_mask_star,self.galaxy_mask,self.SN_map)

        else :
            self.cube1, self.cube1_mask_star, self.galaxy_mask1, self.SN_cube1, self.angle1, self.flip1 = self.rotate_galaxy(self.cube1,self.cube_mask_star,self.galaxy_mask,self.SN_map,False)
            self.cube2, self.cube2_mask_star, self.galaxy_mask2, self.SN_cube2, self.angle2, self.flip2 = self.rotate_galaxy(self.cube2,self.cube_mask_star,self.galaxy_mask,self.SN_map)



        '''
        Applying a gaussina blur
        '''

        sigma = sigma_spec, sigma_spatial

        my_random_number = random.randint(0,1)
        if ((original == False) & (my_random_number == 0)) | (original == True) :
            self.cube2 = self.gaussian_filter(self.cube2,sigma)
            print('Gaussian blur applied to cube 2')
        elif (original == False) & ( my_random_number  == 1) :
            self.cube1 = self.gaussian_filter(self.cube1,sigma)
            print('Gaussian blur applied to cube 1')




        '''
        Masking stars
        '''


        replacement_stars = 0.0

        if (self.cond_star > 0) | (self.n_stars > 0) :
            self.cube1[:,self.cube1_mask_star] = replacement_stars
            self.cube2[:,self.cube2_mask_star] = replacement_stars
        


        '''
        Masking the galaxy
        '''

        self.cube1[:,self.galaxy_mask1] = replacement_stars
        self.cube2[:,self.galaxy_mask2] = replacement_stars



#
        '''
        Triming the galaxy
        '''

        if triming == True :
            self.cube1, self.mask1_outside = self.get_galaxy_contour(self.cube1,self.SN_cube1,self.SN_threshold,True)
            self.cube2, self.mask2_outside = self.get_galaxy_contour(self.cube2,self.SN_cube2,self.SN_threshold,True)
        else :

            self.cube1, self.mask1_outside = self.get_galaxy_contour(self.cube1,self.SN_cube1,1)
            self.cube2, self.mask2_outside = self.get_galaxy_contour(self.cube2,self.SN_cube2,1)
            
    
        if (self.extended == False) | (triming == True) :
            
            self.cube_none, self.mask1_outsidev2 = self.get_galaxy_contour(self.cube1,self.SN_cube1,self.SN_threshold,False)

            n_tot_pix = self.cube1.shape[1] * self.cube1.shape[2]

            ratio_field_of_view = self.mask_contour.sum()/n_tot_pix
 
            
            if (ratio_field_of_view < ratio_field) & (self.interaction == False) & (forbiden_list == False):
                self.recenter = True
            else :
                self.recenter = False


         
        if self.recenter == True :
        
                print('RECENTERING GALAXY')
                mask_none = np.zeros((self.cube1.shape[1],self.cube1.shape[2])).astype(bool)
                
                self.shift1, self.center1 = self.compute_shift(self.cube1,mask_none,mask_none)
            
                self.cube1 = self.center_cube(self.cube1,3,self.shift1)
                self.SN_cube1 = self.center_cube(self.SN_cube1,3,self.shift1)

                self.my_total_mask1 = self.galaxy_mask1 | self.cube1_mask_star
                self.my_total_mask1 = self.center_cube(self.my_total_mask1,0,self.shift1,True)
                self.cube1[:,self.my_total_mask1] = replacement_stars
                self.SN_cube1[self.my_total_mask1] = replacement_stars
                self.SN_cube1[self.center_pixel] = 1000
            
                
                self.shift2, self.center2 = self.compute_shift(self.cube2,mask_none,mask_none)
                
                self.cube2 = self.center_cube(self.cube2,3,self.shift2)
                self.SN_cube2 = self.center_cube(self.SN_cube2,3,self.shift2)
                self.SN_cube2[self.center_pixel] = 1000


                self.my_total_mask2 = self.galaxy_mask2 | self.cube2_mask_star
                self.my_total_mask2 = self.center_cube(self.my_total_mask2,0,self.shift2,True)
                self.cube2[:,self.my_total_mask2] = replacement_stars
                self.SN_cube2[self.my_total_mask2] = replacement_stars

                        
        else :
            self.shift1 = (0,0)
            self.shift2 = (0,0)
            self.center1 = self.center_pixel
            self.center2 = self.center_pixel


    


class last_tranformation :
    def __init__(self,cube1,mask_star1,galaxy_mask1,cube2,mask_star2,galaxy_mask2,range_shift,min_num_pix,original):
        self.cube1 = cube1
        self.mask_star1 = mask_star1
        self.mask_outside1 = galaxy_mask1

        
        self.cube2 = cube2
        self.mask_star2 = mask_star2
        self.mask_outside2 = galaxy_mask2

        self.range_shift = range_shift
        self.original = original
        self.min_num_pix = min_num_pix
   
        self.desired_dim_cube = cube1.shape

   
   
    def update_dim(self,cube,value) :
        current_dim = cube.shape
        desired_dim = np.array(copy.deepcopy(current_dim))
        desired_dim[-1] = self.desired_dim_cube[-1]
        desired_dim[-2] = self.desired_dim_cube[-2]
        dif = desired_dim - np.array(current_dim)
        
        add_dif = []
        for i in range(len(dif)) :
            add_dif.append((0,0))
        
        
        add_dif[-2] = (dif[-2]//2,dif[-2] - dif[-2]//2)
        add_dif[-1] = (dif[-1]//2,dif[-1] - dif[-1]//2)
        
        
        cube = np.pad(cube, add_dif,mode='constant', constant_values=value)
        return cube
    
    def resize_galaxy(self,data_cube, kernel_size,bool=False):
        if kernel_size <= 0:
            raise ValueError("Kernel size must be a positive integer")
        # Creating a kernel with all elements equal to 1/kernel_size**2
        
        elif kernel_size == 1 :
            downsampled_cube = data_cube
        else :
            kernel = np.ones((kernel_size, kernel_size), dtype=float)

            # Applying convolution+
            if bool == False :
                channels = data_cube.shape[0]

                for dd in range(channels):
                    if dd == 0 :
                        a = block_reduce(data_cube[dd], (kernel_size, kernel_size), np.sum)
                        downsampled_cube = np.zeros((channels, a.shape[0], a.shape[1]))
                        downsampled_cube[dd] = block_reduce(data_cube[dd], (kernel_size, kernel_size), np.sum)

                    else :
                        downsampled_cube[dd] = block_reduce(data_cube[dd], (kernel_size, kernel_size), np.sum)
                        
            else :
                downsampled_cube = block_reduce(data_cube, (kernel_size, kernel_size), np.any)

        
        return downsampled_cube

        
    def redshift_galaxy(self,cube,cube_mask_star,mask_outside) :
        num_pix = (mask_outside == False).sum()
        
        possible_scale = np.array([1,2,3,4,5])
        num_pix_afeter_scaling = num_pix//possible_scale**2
        
        where_poissible = num_pix_afeter_scaling > self.min_num_pix
        
        if where_poissible.sum() > 1 :
            possible_scale = possible_scale[where_poissible]
            
            scale_factor = np.random.randint(possible_scale[0], possible_scale[-1]+1)
            print('POSSIBLE SCALE FACTOR: ', possible_scale)
            print('SCALE FACTOR: ', scale_factor)
            #Resize cube
            cube = self.resize_galaxy(cube, scale_factor,False)

            cube_mask_star = self.resize_galaxy(cube_mask_star, scale_factor,True)
            mask_outside = self.resize_galaxy(mask_outside, scale_factor,True)
            
        
            #Update dimension
            cube = self.update_dim(cube,0)
            cube_mask_star = self.update_dim(cube_mask_star,False)
            mask_outside = self.update_dim(mask_outside,True)
        else :
            print('NO POSSIBLE SCALING')
            
        return cube,cube_mask_star,mask_outside



    def cube_translation(self,cube,gal_mask,mask_stars) :
        shift_x = np.random.randint(-self.range_shift[0], self.range_shift[0]+1)
        shift_y = np.random.randint(-self.range_shift[1], self.range_shift[1]+1)
        print(shift_x,shift_y)
        dim_mask = gal_mask.shape
        
        white_box = np.ones((dim_mask)).astype(bool)
        white_box = ndi.shift(white_box, (shift_x, shift_y), order=0)
        where_move = white_box == False
        
        cube = ndi.shift(cube, (0, shift_x, shift_y), order=3)
        mask_stars = ndi.shift(mask_stars, (shift_x, shift_y), order=0)
        gal_mask = ndi.shift(gal_mask, (shift_x, shift_y), order=0)
        gal_mask[where_move] = True
        
        cube[:,gal_mask] = 0.0
        cube[:,mask_stars] = 0.0
        

        return cube, gal_mask, mask_stars
    

        
    def make_last_transformation(self) :
        
        self.cube2 ,self.mask_star2, self.mask_outside2 = self.redshift_galaxy(self.cube2,self.mask_star2,self.mask_outside2)
        self.cube2, self.mask_outside2, self.mask_stars2 = self.cube_translation(self.cube2,self.mask_outside2,self.mask_star2)

        if self.original == False :
            self.cube1 ,self.mask_star1, self.mask_outside1 = self.redshift_galaxy(self.cube1,self.mask_star1,self.mask_outside1)
            
            self.cube1, self.mask_outside1, self.mask_stars1 = self.cube_translation(self.cube1,self.mask_outside1,self.mask_star1)





         
def generate_sky_cube(cube) :

    data_vector = cube.reshape((cube.shape[0],cube.shape[1]*cube.shape[2]))
        
    try :
        cond_5600 = (galaxy_dr.wavelengths > 5600) & (galaxy_dr.wavelengths < 6000)
    except :
        cond_5600 = (galaxy_edr.wavelengths > 5600) & (galaxy_edr.wavelengths < 6000)
    mean_at_5600 = np.nanmedian(data_vector[cond_5600,:],axis=0)
    where_mean_zero = mean_at_5600 <= 0
    
    flux_zero_min = 0.15 * np.mean(mean_at_5600[~where_mean_zero])
    flux_zero_max = 0.25 * np.mean(mean_at_5600[~where_mean_zero])
    
    flux_zero_map = flux_zero_min + (flux_zero_max - flux_zero_min)*np.random.rand(cube.shape[1],cube.shape[2])
    
    cov = np.zeros((cube.shape[0],cube.shape[0]))
    np.fill_diagonal(cov, 1)
    random_field = np.random.multivariate_normal(np.zeros(cube.shape[0]), cov, size=(cube.shape[1],cube.shape[2]))
    random_field = random_field.reshape(cube.shape[0], cube.shape[1],cube.shape[2])
    
    image = 1 + np.random.rand(cube.shape[1],cube.shape[2])
    image = flux_zero_map[np.newaxis,:,:]*(1 + random_field/image[np.newaxis,:,:])
    
    return image

def get_cube_pca(cube,cube_sky) :
    len_wavelength = cube.shape[0]
    len_x = cube.shape[1]
    len_y = cube.shape[2]

    data_vector = cube.reshape((len_wavelength,len_x*len_y))
    cube_sky_vector = cube_sky.reshape((len_wavelength,len_x*len_y))

    try :
        cond_5600 = (galaxy_dr.wavelengths > 5600) & (galaxy_dr.wavelengths < 6000)
        el_mask = galaxy_dr.el_mask
        min_flux = galaxy_dr.min_flux
    except :
        cond_5600 = (galaxy_edr.wavelengths > 5600) & (galaxy_edr.wavelengths < 6000)
        el_mask = galaxy_edr.el_mask
        min_flux = galaxy_edr.min_flux

    mean_at_5600 = np.nanmedian(data_vector[cond_5600,:],axis=0)
    mean_at_5600_sky = np.nanmedian(cube_sky_vector[:,:],axis=0)

    
    where_mean_zero = mean_at_5600 <= 0
    data_vector_norm = data_vector/mean_at_5600[np.newaxis,:]
    
    data_vector_norm[:,where_mean_zero] = cube_sky_vector[:,where_mean_zero]/mean_at_5600_sky[where_mean_zero][np.newaxis,:]
        


    pca = joblib.load(folder_aux + file_pca)


    data_vector_t = pca.inverse_transform(pca.transform(np.transpose(data_vector_norm)))

    dif_spec = (data_vector_t - np.transpose(data_vector_norm))/np.transpose(data_vector_norm)
    
    
    sigma_spec = MAD(dif_spec[:,~el_mask],axis=1)
    where_very_low = np.median(np.transpose(data_vector),axis=1) < min_flux
    sigma_spec[where_very_low] = 1000

    SN_spec = 1/sigma_spec
    where_nan = (np.isnan(SN_spec) == True) | (SN_spec < 1)
    SN_spec[where_nan] = 0
        
    pca_component = pca.transform(np.transpose(data_vector_norm))


    where_pca_extreme1 = (np.abs(pca_component[:,15:]) >= 3).sum(axis=1) >= 1
#    where_pca_extreme1 = where_pca_extreme1 & (SN_spec < 8)
    where_pca_extreme1 = where_pca_extreme1 & (SN_spec < 5)

        
    where_pca_extreme2 = (np.abs(pca_component[:,0:]) >= 40).sum(axis=1) >= 1
    where_pca_extreme2 = where_pca_extreme2 & (SN_spec < 5)

    where_pca_extreme = where_pca_extreme1 | where_pca_extreme2


    data_vector_norm[:,where_pca_extreme] = cube_sky_vector[:,where_pca_extreme]/mean_at_5600_sky[where_pca_extreme][np.newaxis,:]
    
    
    pca_component = pca.transform(np.transpose(data_vector_norm))


    where_flag = where_pca_extreme | where_mean_zero
    mean_at_5600[where_flag] =  mean_at_5600_sky[where_flag]



    reduce_cube = np.zeros((pca_component.shape[0],pca_component.shape[1] + 1))
    reduce_cube[:,0] =  np.log10(mean_at_5600 + 1e-16)
    reduce_cube[:,1:] = pca_component

    reduce_cube = np.transpose(reduce_cube.reshape((len_x,len_y,reduce_cube.shape[-1])))
    where_flag = np.transpose(where_flag.reshape((len_x,len_y)))

    return reduce_cube, where_flag

            

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

class Generate_gakaxy_mask :
    def __init__(self, cube1_l, SN_cube1_l, rot1_l, cube2_l, SN_cube2_l, rot2_l, cube1_h, SN_cube1_h, rot1_h, cube2_h, SN_cube2_h, rot2_h, SN_threshold,shift1_l,shift2_l,shift1_h,shift2_h,recenter):
        
        '''
        CUBE
        '''
        self.cube1_l = cube1_l
        self.cube2_l = cube2_l
        
        self.SN_cube1_l = SN_cube1_l
        self.SN_cube2_l = SN_cube2_l
        
    
        self.rot1_l = rot1_l
        self.rot2_l = rot2_l
        
        self.shift1_l = shift1_l
        self.shift2_l = shift2_l

        '''
        ECUBE
        '''

        self.cube1_h = cube1_h
        self.cube2_h = cube2_h
        
        self.SN_cube1_h = SN_cube1_h
        self.SN_cube2_h = SN_cube2_h
        
        self.rot1_h = rot1_h
        self.rot2_h = rot2_h
        
        self.shift1_h = shift1_h
        self.shift2_h = shift2_h

        
        
        
        self.SN_threshold = SN_threshold
      
        self.recenter = recenter
   
    def resize_mask(self,mask_l,mask_h,high2low=True) :
        if high2low == True :
            bool_mask = mask_h
            new_shape = (mask_l.shape[0],mask_l.shape[1])

        else :
            bool_mask = mask_l
            new_shape = (mask_h.shape[0],mask_h.shape[1])


        bool_mask = bool_mask.astype(float)

        resized_bool_mask = resize(bool_mask, (new_shape[0], new_shape[1]), order=0)
        resized_bool_mask = (resized_bool_mask * np.iinfo(np.uint8).max).astype(bool)

        return resized_bool_mask

    
    def center_cube(self,cube,order,shift,gal_mask=False) :
        shift_x, shift_y = shift

        if len(cube.shape) == 3 :
            cube = ndi.shift(cube, (0, shift_x, shift_y),order=order)
        elif len(cube.shape) == 2 :
            cube = ndi.shift(cube, (shift_x, shift_y),order=order)
            if gal_mask == True :
                dim_mask = cube.shape
                white_box = np.ones((dim_mask)).astype(bool)
                white_box = ndi.shift(white_box, (shift_x, shift_y), order=0)
                where_move = white_box == False
                cube[where_move] = True

        else :
            print('CUBE DOES NOT HAVE EXPECTED DIMENSIONS')
        
        return cube
                   
    def rotate_galaxy(self, gal_mask, rot, derot=True) :
        
    
        if (rot[0] != False) & (rot[1] != False) :


            if derot == True :
            
                gal_mask = np.flip(gal_mask, axis=rot[1]-1)
                gal_mask = rotate(gal_mask, 360-rot[0], axes=(0,1), order=0, reshape=False)
            else :
                gal_mask = rotate(gal_mask, rot[0], axes=(0,1), order=0, reshape=False)
                gal_mask = np.flip(gal_mask, axis=rot[1]-1)

    
            
        
        return gal_mask
    


    def get_galaxy_contour(self,data_cube,SN_map,SN_threshold) :
        contours = self.image_contours(SN_map, SN_threshold)
        contours = self.convex_hull_polygon(contours)
        self.mask_contour = self.contours_to_mask(contours, SN_map)
        
        return self.mask_contour
        


                
    def image_contours(self,image, level, smooth=False, gauss=True, single=True, upper=False,
                       sigma=2.0, kernel=6.0, bval=-100., origin=1.0, chunks=None):
        # x = contours[:, 1] | y = contours[:, 0]
        ny, nx = image.shape
        fimg = np.zeros((ny + 2, nx + 2)) + bval
        y, x = np.indices(fimg.shape)
        if smooth:
            if gauss:
                img = scipy.ndimage.gaussian_filter(image, sigma=sigma, order=0)
            else:
                img = scipy.ndimage.filters.median_filter(image, kernel)
        else:
            img = image.copy()
        fimg[1:-1, 1:-1] = img
        if upper:
            fimg[fimg >= level] = 100. * level
            fimg[fimg < level]  = bval
        if chunks is not None:
            mask = image_quadrant(fimg, chunks=chunks)
            fimg[~mask] = bval
        # list of (n,2) - ndarrays
        # Each contour is an ndarray of shape (n, 2), consisting of n (x, y) coordinates
        contours = skmeasure.find_contours(fimg, level)
        contours = [contour - origin for contour in contours]
        if single:
            len_contours = np.array([len(contour) for contour in contours])
            clevel = contours[len_contours.argmax()] if len_contours.size > 0 else None
            return clevel
        else:
            return contours

    def contours_to_mask(self,contours, shape):
        if isinstance(contours, (list, tuple)):
            contours = np.array(contours)
        if isinstance(shape, np.ndarray):
            shape = shape.shape
        ny, nx = shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((y, x)).T
        path = Path(contours)
        mask = path.contains_points(points)
        mask = mask.reshape(shape)
        return mask

    def convex_hull_polygon(self,points):
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        return vertices

        


    def apply_commun_mask(self) :
        
        
        '''
        LOW RESOLUTION
        '''
        self.mask1_l = self.get_galaxy_contour(self.cube1_l,self.SN_cube1_l,self.SN_threshold)
        self.mask1_h = self.get_galaxy_contour(self.cube1_h,self.SN_cube1_h,self.SN_threshold)
  

        mm = copy.deepcopy(self.mask1_l)
        ii = copy.deepcopy(self.cube1_l)
        mmh = copy.deepcopy(self.mask1_h)


        if self.recenter == True :
            array_shift1_l = np.array(self.shift1_l)
            antiarray_shift1_l = -array_shift1_l
            antituple_shift1_l = tuple(antiarray_shift1_l)
            self.mask1_l = self.center_cube(self.mask1_l,0,antituple_shift1_l,True)

            array_shift1_h = np.array(self.shift1_h)
            antiarray_shift1_h = -array_shift1_h
            antituple_shift1_h = tuple(antiarray_shift1_h)
            self.mask1_h = self.center_cube(self.mask1_h,0,antituple_shift1_h,True)
        
        
        self.mask1_l = self.rotate_galaxy(self.mask1_l,self.rot1_l,True)
        self.mask1_h = self.rotate_galaxy(self.mask1_h,self.rot1_h,True)
        
        self.mask1_h2l = self.resize_mask(self.mask1_l,self.mask1_h,True)
        self.mask1_l = self.mask1_l | self.mask1_h2l

        self.mask1_l2h = self.resize_mask(self.mask1_l,self.mask1_h,False)

        self.mask1_l = self.rotate_galaxy(self.mask1_l,self.rot1_l,False)
        self.mask1_h = self.rotate_galaxy(self.mask1_l2h,self.rot1_h,False)

        if self.recenter == True :
            self.mask1_l = self.center_cube(self.mask1_l,0,self.shift1_l,True)
            self.mask1_h = self.center_cube(self.mask1_h,0,self.shift1_h,True)

        self.cube1_l[:,~self.mask1_l] = 0.0
        self.cube1_h[:,~self.mask1_h] = 0.0
        self.mask1_l = ~self.mask1_l
        self.mask1_h = ~self.mask1_h


        '''
        HIGH RESOLUTION
        '''

        self.mask2_l = self.get_galaxy_contour(self.cube2_l,self.SN_cube2_l,self.SN_threshold)
        self.mask2_h = self.get_galaxy_contour(self.cube2_h,self.SN_cube2_h,self.SN_threshold)
        

        if self.recenter == True :
            array_shift2_l = np.array(self.shift2_l)
            antiarray_shift2_l = -array_shift2_l
            antituple_shift2_l = tuple(antiarray_shift2_l)
            self.mask2_l = self.center_cube(self.mask2_l,0,antituple_shift2_l,True)

            array_shift2_h = np.array(self.shift2_h)
            antiarray_shift2_h = -array_shift2_h
            antituple_shift2_h = tuple(antiarray_shift2_h)
            self.mask2_h = self.center_cube(self.mask2_h,0,antituple_shift2_h,True)
    
        self.mask2_l = self.rotate_galaxy(self.mask2_l,self.rot2_l,True)
        self.mask2_h = self.rotate_galaxy(self.mask2_h,self.rot2_h,True)
        
        
        self.mask2_h2l = self.resize_mask(self.mask2_l,self.mask2_h,True)
        self.mask2_l = self.mask2_l | self.mask2_h2l
        self.mask2_l2h = self.resize_mask(self.mask2_l,self.mask2_h,False)

        self.mask2_l = self.rotate_galaxy(self.mask2_l,self.rot2_l,False)
        self.mask2_h = self.rotate_galaxy(self.mask2_l2h,self.rot2_h,False)
                

        if self.recenter == True :
            self.mask2_l = self.center_cube(self.mask2_l,0,self.shift2_l,True)
            self.mask2_h = self.center_cube(self.mask2_h,0,self.shift2_h,True)
        
        self.cube2_l[:,~self.mask2_l] = 0.0
        self.cube2_h[:,~self.mask2_h] = 0.0
        self.mask2_l = ~self.mask2_l
        self.mask2_h = ~self.mask2_h




def create_rgb_image(cube, red_range, green_range, blue_range, gamma=2.2):
    red_channel = np.nanmean(cube[red_range, :, :], axis=0)
    green_channel = np.nanmean(cube[green_range, :, :], axis=0)
    blue_channel = np.nanmean(cube[blue_range, :, :], axis=0)

    max_value = np.max([red_channel, green_channel, blue_channel])

    red_channel /= max_value
    green_channel /= max_value
    blue_channel /= max_value

    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    rgb_image = np.power(rgb_image, 1 / gamma)
    
    return rgb_image
    

def plot_rgb_images(cube1_l,cube2_l,cube1_h,cube2_h,name_my_gal,wavelenghts,el_mask):

    red_range = (wavelenghts > 6400) & (wavelenghts < 6700) & (el_mask == False)
    green_range = (wavelenghts > 5000) & (wavelenghts < 5600) & (el_mask == False)
    blue_range = (wavelenghts > 4200) & (wavelenghts< 4700) & (el_mask == False)

    rgb_image1 = create_rgb_image(cube1_l, red_range, green_range, blue_range)
    rgb_image2 = create_rgb_image(cube2_l, red_range, green_range, blue_range)
    
    
    rgb_image3 = create_rgb_image(cube1_h, red_range, green_range, blue_range)
    rgb_image4 = create_rgb_image(cube2_h, red_range, green_range, blue_range)

    




    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    
    
    
    ax1.imshow(rgb_image1, origin='lower', extent=[0, cube1_l.shape[2], 0, cube1_l.shape[1]])
#    contours = ax1.contour(mask1_l, levels=[0.5], colors='r')
    ax1.set_title(name_my_gal)
    
    ax2.imshow(rgb_image2, origin='lower', extent=[0, cube2_l.shape[2], 0, cube2_l.shape[1]])
#    contours = ax2.contour(mask2_l, levels=[0.5], colors='r')

    ax2.set_title(name_my_gal)

    ax3.imshow(rgb_image3, origin='lower', extent=[0, cube1_h.shape[2], 0, cube1_h.shape[1]])
#    contours = ax3.contour(mask1_h, levels=[0.5], colors='r')

    ax3.set_title(name_my_gal)
    
    ax4.imshow(rgb_image4, origin='lower', extent=[0, cube2_h.shape[2], 0, cube2_h.shape[1]])
#    contours = ax4.contour(mask2_h, levels=[0.5], colors='r')

    ax4.set_title(name_my_gal)

        
    return fig


#
#
def plot_rgb_images_pca(cube1_l,cube2_l,cube1_h,cube2_h,where_zero1_l,where_zero2_l,where_zero1_h,where_zero2_h,name_my_gal):


    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

#    cube1_l[:,where_zero1_l] = np.nan
#    cube2_l[:,where_zero2_l] = np.nan
#    cube1_h[:,where_zero1_h] = np.nan
#    cube2_h[:,where_zero2_h] = np.nan

    ax1.imshow(np.transpose(cube1_l[0,:,:]), origin='lower',  vmin=np.nanmedian(cube1_l[0,~where_zero1_l]) - 2*np.nanstd(cube1_l[0,~where_zero1_l]),vmax=np.nanmedian(cube1_l[0,~where_zero1_l]) + 2*np.nanstd(cube1_l[0,~where_zero1_l]), cmap='Reds', extent=[0, cube1_l.shape[1], 0, cube1_l.shape[2]])
    
    ax1.set_title(name_my_gal)

    ax2.imshow(np.transpose(cube2_l[0,:,:]), origin='lower', vmin=np.nanmedian(cube2_l[0,~where_zero2_l]) - 2*np.nanstd(cube2_l[0,~where_zero2_l]),vmax=np.nanmedian(cube2_l[0,~where_zero2_l]) + 2*np.nanstd(cube2_l[0,~where_zero2_l]), cmap='Reds', extent=[0, cube2_l.shape[1], 0, cube2_l.shape[2]])

    ax2.set_title(name_my_gal)

    ax3.imshow(np.transpose(cube1_h[1,:,:]), origin='lower',  vmin=np.nanmedian(cube1_h[1,~where_zero1_h]) - 2*np.nanstd(cube1_h[1,~where_zero1_h]),vmax=np.nanmedian(cube1_h[1,~where_zero1_h]) + 2*np.nanstd(cube1_h[1,~where_zero1_h]), cmap='Reds', extent=[0, cube1_h.shape[1], 0, cube1_h.shape[2]])

    ax3.set_title(name_my_gal)

    ax4.imshow(np.transpose(cube2_h[1,:,:]), origin='lower',  vmin=np.nanmedian(cube2_h[1,~where_zero2_h]) - 2*np.nanstd(cube2_h[1,~where_zero2_h]),vmax=np.nanmedian(cube2_h[1,~where_zero2_h]) + 2*np.nanstd(cube2_h[1,~where_zero2_h]), cmap='Reds', extent=[0, cube2_h.shape[1], 0, cube2_h.shape[2]])

#    cube1_l[:,where_zero1_l] = 0.
#    cube2_l[:,where_zero2_l] = 0.
#    cube1_h[:,where_zero1_h] = 0.
#    cube2_h[:,where_zero2_h] = 0.
    
    return fig





parser = argparse.ArgumentParser(description='Genarate galaxies pair')
parser.add_argument('--config_file', type=str, help='Path to the config file')
parser.add_argument('--n_times', type=int, help='Augmentation factor')
parser.add_argument('--range', nargs='+', type=int, help='low and high limit')


args = parser.parse_args()
name_config = args.config_file
n_times = args.n_times
range_it = args.range

n_gal = range_it[1] - range_it[0]



config = get_config(name_config)
folder_data = config.get('FOLDER_INFO', 'folder_data')
folder_pdf = config.get('FOLDER_INFO', 'folder_pdf')
folder_aux = config.get('FOLDER_INFO', 'folder_aux')
folder_cube_pair = folder_data + 'cube_pairs/'
folder_ecube_pair = folder_data + 'ecube_pairs/'
file_pca = config.get('FILES_INFO', 'file_pca')

n_pca_component = ast.literal_eval(config.get('FILES_INFO', 'n_pca_component'))


SN_threshold = 3.


pdf_name = 'Galaxy_augmentation_' + str(range_it[0]) + '_' + str(range_it[1]) + '.pdf'
file_name = 'parameters_rot_' + str(range_it[0]) + '_' + str(range_it[1]) + '.txt'



file_cubes = [f for f in os.listdir(folder_data + 'cubes/') if not f.startswith('.') and f.endswith('.fits')]
file_cubes = list(map(lambda x: x.replace('.V500.rscube.fits', ''), file_cubes))

file_ecubes = [f for f in os.listdir(folder_data + 'ecubes/') if not f.startswith('.') and f.endswith('.fits')]
file_ecubes = list(map(lambda x: x.replace('.V500.drscube.fits', ''), file_ecubes))

combined_galaxies = set(file_cubes) | set(file_ecubes)
combined_galaxies = list(combined_galaxies)
combined_galaxies = sorted(list(combined_galaxies))

combined_galaxies = np.array(combined_galaxies)
print(len(combined_galaxies))
combined_galaxies = combined_galaxies[range_it[0]:range_it[1]]



data_info_cube = np.load(folder_aux + "train_test_dr.npz")
data_info_ecube = np.load(folder_aux + "train_test_edr.npz")


index_cubes = [index for index, element in enumerate(data_info_cube['ID']) if element in combined_galaxies]
n_gal_train_cube = data_info_cube['train_sample'][index_cubes].sum()

index_ecubes = [index for index, element in enumerate(data_info_ecube['ID']) if element in combined_galaxies]
n_gal_train_ecube = data_info_ecube['train_sample'][index_ecubes].sum()

n_gal_test = data_info_ecube['test_sample'][index_ecubes].sum()


with PdfPages(folder_pdf + pdf_name) as pdf:

    for j in range(n_times) :
        itr_c = 0
        its_c = 0
        itr_ec = 0
        its_ec = 0
        if j == 0 :
            original_cube = True
            original_ecube = True
        elif j == n_times - 1 :
            original_cube = False
            original_ecube = False
        else :
            original_cube = False
            original_ecube = False
            

            

        list_gal = []
        for i in range(n_gal) :
            try :
                name_my_gal = combined_galaxies[i]
                file_c = name_my_gal + '.V500.rscube.fits'
                file_e = name_my_gal + '.V500.drscube.fits'
                print(j,i,name_my_gal)

                cube_exist = os.path.exists(folder_data + 'cubes/' + file_c)
                ecube_exist = os.path.exists(folder_data + 'ecubes/' + file_e)

                both_cubes = cube_exist & ecube_exist
                
                both_cubes = False
                cube_exist = False

                if both_cubes == True :
                    triming = False
                else :
                    triming = True
                
                if cube_exist :
                    galaxy_dr = CALIFAAugmentor(config,file_c,False)
                    galaxy_dr.generate_pair(SN_threshold,triming,original_cube)
                    cube1_l = galaxy_dr.cube1
                    cube2_l = galaxy_dr.cube2
    
                    SN_cube1_l = galaxy_dr.SN_cube1
                    SN_cube2_l = galaxy_dr.SN_cube2

                    mask_star1_l = galaxy_dr.cube1_mask_star
                    mask_star2_l = galaxy_dr.cube2_mask_star

                    galaxy_mask1_l = galaxy_dr.galaxy_mask1 | galaxy_dr.mask1_outside
                    galaxy_mask2_l = galaxy_dr.galaxy_mask2 | galaxy_dr.mask2_outside

                    rot1_l = galaxy_dr.angle1, galaxy_dr.flip1
                    rot2_l = galaxy_dr.angle2, galaxy_dr.flip2
    
                    recenter = galaxy_dr.recenter


                else :
                    print('Cube does not exist')
                    recenter = False
                  
                if ecube_exist :
                    galaxy_edr = CALIFAAugmentor(config,file_e,True,recenter)
                    galaxy_edr.generate_pair(SN_threshold,triming,original_ecube)
                    
                    cube1_h = galaxy_edr.cube1
                    cube2_h = galaxy_edr.cube2
    
                    SN_cube1_h = galaxy_edr.SN_cube1
                    SN_cube2_h = galaxy_edr.SN_cube2
                    
                    mask_star1_h = galaxy_edr.cube1_mask_star
                    mask_star2_h = galaxy_edr.cube2_mask_star

                    galaxy_mask1_h = galaxy_edr.galaxy_mask1 | galaxy_edr.mask1_outside
                    galaxy_mask2_h = galaxy_edr.galaxy_mask2 | galaxy_edr.mask2_outside
                    
                    rot1_h = galaxy_edr.angle1, galaxy_edr.flip1
                    rot2_h = galaxy_edr.angle2, galaxy_edr.flip2

                else :
                    print('Ecube does not exist')
        
        
                if both_cubes == True :
                    galmask = Generate_gakaxy_mask(cube1_l,SN_cube1_l,rot1_l,cube2_l,SN_cube2_l,rot2_l,cube1_h,SN_cube1_h,rot1_h,cube2_h,SN_cube2_h,rot2_h,SN_threshold,galaxy_dr.shift1,galaxy_dr.shift2,galaxy_edr.shift1,galaxy_edr.shift2,recenter)
                    galmask.apply_commun_mask()
                    
                    galaxy_mask1_l = galmask.mask1_l
                    galaxy_mask2_l = galmask.mask2_l
                    
                    galaxy_mask1_h = galmask.mask1_h
                    galaxy_mask2_h = galmask.mask2_h

                    cube1_l = galmask.cube1_l
                    cube2_l = galmask.cube2_l
                    cube1_h = galmask.cube1_h
                    cube2_h = galmask.cube2_h

                
#                plot_rgb_images_mask(galaxy_mask1_l,galaxy_mask2_l,cube1_l,cube2_l,name_my_gal,galaxy_dr.wavelengths,galaxy_dr.el_mask)
                
                if cube_exist :
                    transformation_l = last_tranformation(cube1_l,mask_star1_l,galaxy_mask1_l,cube2_l,mask_star2_l,galaxy_mask2_l,galaxy_dr.range_shift,galaxy_dr.min_num_pix,original_cube)
                    transformation_l.make_last_transformation()
                    
                    cube1_l = transformation_l.cube1
                    cube2_l = transformation_l.cube2
                    cube1_l_sky = generate_sky_cube(cube1_l)
                    cube2_l_sky = generate_sky_cube(cube2_l)


                if ecube_exist :
                    transformation_h = last_tranformation(cube1_h,mask_star1_h,galaxy_mask1_h,cube2_h,mask_star2_h,galaxy_mask2_h,galaxy_edr.range_shift,galaxy_edr.min_num_pix,original_ecube)
                    transformation_h.make_last_transformation()

                    cube1_h = transformation_h.cube1
                    cube2_h = transformation_h.cube2
                    cube1_h_sky = generate_sky_cube(cube1_h)
                    cube2_h_sky = generate_sky_cube(cube2_h)


                if (both_cubes == True) & (j == 0) :
                    fig = plot_rgb_images(cube1_l,cube2_l,cube1_h,cube2_h,name_my_gal,galaxy_dr.wavelengths,galaxy_dr.el_mask)
                    pdf.savefig(fig,dpi = 100)
                    
                elif (ecube_exist == True) & (j == 0) :
                    fig = plot_rgb_images(cube1_h,cube2_h,cube1_h,cube2_h,name_my_gal,galaxy_edr.wavelengths,galaxy_edr.el_mask)
                    pdf.savefig(fig,dpi = 100)

                elif (cube_exist == True) & (j == 0) :
                    fig = plot_rgb_images(cube1_l,cube2_l,cube1_l,cube2_l,name_my_gal,galaxy_dr.wavelengths,galaxy_dr.el_mask)
                    pdf.savefig(fig,dpi = 100)


#                plot_rgb_images_onclik(cube1_l,cube2_l,galaxy_dr.wavelengths,galaxy_dr.el_mask)
                if cube_exist :
                    
                    cube1_l, where_zero1_l = get_cube_pca(cube1_l,cube1_l_sky)
                    cube2_l,where_zero2_l = get_cube_pca(cube2_l,cube2_l_sky)

                if ecube_exist :
                    cube1_h,where_zero1_h = get_cube_pca(cube1_h,cube1_h_sky)
                    cube2_h, where_zero2_h = get_cube_pca(cube2_h,cube2_h_sky)
                
#                plot_rgb_images_onclik_pca(cube1_l,cube2_l)

                if (both_cubes == True) & (j == 0) :
                    fig = plot_rgb_images_pca(cube1_l,cube2_l,cube1_h,cube2_h,where_zero1_l,where_zero2_l,where_zero1_h,where_zero2_h,name_my_gal)
                    pdf.savefig(fig,dpi = 100)

                elif (ecube_exist == True) & (j == 0) :
                    fig = plot_rgb_images_pca(cube1_h,cube2_h,cube1_h,cube2_h,where_zero1_h,where_zero2_h,where_zero1_h,where_zero2_h,name_my_gal)
                    pdf.savefig(fig,dpi = 100)

                elif (cube_exist == True) & (j == 0) :
                    fig = plot_rgb_images_pca(cube1_l,cube2_l,cube1_l,cube2_l,where_zero1_l,where_zero2_l,where_zero1_l,where_zero2_l,name_my_gal)
                    pdf.savefig(fig,dpi = 100)


                if i == 0 :
                    len_wavelength_dr, len_x_dr, len_y_dr = n_pca_component + 1, 96, 92
                    len_wavelength_edr, len_x_edr, len_y_edr = n_pca_component + 1, 192, 184

                    if n_gal_train_cube > 0 :
                        gal_train_cube = np.zeros((2,n_gal_train_cube,len_wavelength_dr,len_x_dr,len_y_dr))

                    if n_gal_train_ecube > 0 :
                        gal_train_ecube = np.zeros((2,n_gal_train_ecube,len_wavelength_edr,len_x_edr,len_y_edr))

                    if n_gal_test > 0 :
                        gal_test_cube = np.zeros((2,n_gal_test,len_wavelength_dr,len_x_dr,len_y_dr))
                        gal_test_ecube = np.zeros((2,n_gal_test,len_wavelength_edr,len_x_edr,len_y_edr))


                if cube_exist :
                    index_my_cube = [index for index, element in enumerate(data_info_cube['ID']) if element == name_my_gal]
                    print('CUBE ' + name_my_gal)
                    if (data_info_cube['train_sample'][index_my_cube][0] == True) :
                        gal_train_cube[0,itr_c,:,:,:] = cube1_l
                        gal_train_cube[1,itr_c,:,:,:] = cube2_l
                        itr_c += 1
                    else :
                        gal_test_cube[0,its_c,:,:,:] = cube1_l
                        gal_test_cube[1,its_c,:,:,:] = cube2_l
                        its_c += 1

                if ecube_exist :
                    index_my_ecube = [index for index, element in enumerate(data_info_ecube['ID']) if element == name_my_gal]
                    print('eCUBE ' + name_my_gal)
                    if (data_info_ecube['train_sample'][index_my_ecube][0] == True) :
                        gal_train_ecube[0,itr_ec,:,:,:] = cube1_h
                        gal_train_ecube[1,itr_ec,:,:,:] = cube2_h
                        itr_ec += 1
                    else :
                        gal_test_ecube[0,its_ec,:,:,:] = cube1_h
                        gal_test_ecube[1,its_ec,:,:,:] = cube1_h
                        its_ec += 1
            except :
                list_gal.append(name_my_gal)
                
        if j == 0 : plt.close()

        with open('/new_list_' + str(j) + '.txt', 'a') as filehandle:
            for listitem in list_gal:
                filehandle.write('%s\n' % listitem)

        if n_gal_train_cube > 0 :
            with tf.io.TFRecordWriter(folder_cube_pair + 'train_sample/A/' + f'train_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_train_cube):
                    feature = gal_train_cube[0,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_train_cube[0,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

            with tf.io.TFRecordWriter(folder_cube_pair + 'train_sample/B/' + f'train_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_train_cube):
                    feature = gal_train_cube[1,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_train_cube[1,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())


        if n_gal_train_ecube > 0 :
            with tf.io.TFRecordWriter(folder_ecube_pair + 'train_sample/A/' + f'train_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_train_ecube):
                    feature = gal_train_ecube[0,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_train_ecube[0,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

            with tf.io.TFRecordWriter(folder_ecube_pair + 'train_sample/B/' + f'train_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_train_ecube):
                    feature = gal_train_ecube[1,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_train_ecube[1,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())


        if n_gal_test > 0 :
            with tf.io.TFRecordWriter(folder_ecube_pair + 'test_sample/A/' + f'test_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_test):
                    feature = gal_test_ecube[0,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_test_ecube[0,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

            with tf.io.TFRecordWriter(folder_ecube_pair + 'test_sample/B/' + f'test_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_test):
                    feature = gal_test_ecube[1,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_test_ecube[1,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())


            with tf.io.TFRecordWriter(folder_cube_pair + 'test_sample/A/' + f'test_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:
                for i in range(n_gal_test):
                    feature = gal_test_cube[0,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_test_cube[0,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

            with tf.io.TFRecordWriter(folder_cube_pair + 'test_sample/B/' + f'test_sample_{range_it[0]}_{range_it[1]}_{j+1}.tfrecord') as writer:

                for i in range(n_gal_test):
                    feature = gal_test_cube[1,i,:,:,:].astype(np.float32).tobytes()
                    shape = list(gal_test_cube[1,i,:,:,:].shape)
                    features = {
                        'image': _bytes_feature(feature),
                        'shape': _int64_feature(shape),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

x = (time.time()-t0)/60
print(x)


