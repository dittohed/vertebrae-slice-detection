"""
Utility functions for preprocessing CTs.
"""

import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
import pydicom

# --- constants ---
MIN_HU = -1000 # air HU
FAT_HU = -120

# --- processing functions ---
def read_bin(filename, size, h, nbytes, signed='Y', byte_order='BE'):
    """
    Reads a .raw format binary CT. 
    Returns a numpy array with depth x width x height axes order.
    """
    
    if nbytes == 2:
        if signed == 'N':
            img = np.zeros((size, size, h), np.uint16)
        else:
            img = np.zeros((size, size, h), np.int16)
    elif nbytes == 1:
        if signed == 'N':
            img = np.zeros((size, size, h), np.uint8)
        else:
            img = np.zeros((size, size, h), np.int8)
    else:
        return None
    
    f = open(filename, 'rb')
    for i in range(h):
        for j in range(size):
            for k in range(size):
                byte = f.read(nbytes)
                
                if nbytes == 2:
                    if byte_order == 'BE':
                        a = 256*byte[0] + byte[1]
                    else:
                        a = byte[0] + 256*byte[1]
                else:
                    a = byte[0]
                    
                img[j, k, i] = a
                
    f.close()
    return img

def read_dicom(dicom_dir, verbose=False):
    """
    Reads .dcm files in a given dicom_dir.
    Returns a numpy array with depth x width x height axes order.
    """

    dicom_files = [file for file in os.listdir(dicom_dir) if file.endswith('dcm')]
    dicoms = []
    
    for file in dicom_files:
        path = os.path.join(dicom_dir, file)
        ds = pydicom.dcmread(path)
        dicoms.append((path, int(ds[0x0020, 0x0013].value)))
            
    dicoms = sorted(dicoms, key=lambda x: x[1])

    img = []
    for d in dicoms:
        ds = pydicom.dcmread(d[0])
#         try:
        img.append(ds.pixel_array)
#         except ValueError as e:
#             print(f'An exception for {DICOM_DIR} occured: {str(e)}')
#             return None, None, None

    img = np.asarray(img, dtype=np.int16)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    if verbose:
        print(img.shape)

        for s in range(img.shape[0]):
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(img[s], cmap='gray')
            plt.show()

    return img, ds.PixelSpacing, int(ds.SliceThickness)

def get_mass_center(img, verbose=False):
    """
    Takes a 3D CT with depth x width x height axes order,
    calculates body mask and zeroes everything else.
    Returns processed img & body mask mass center indexes.
    """
    
    middle_index = img.shape[2] // 2 # for verbose & calculating body mask mass center
    
    # for each slice leave body only
    for i in range(img.shape[2]):
        mask = np.zeros(img[:, :, i].shape, dtype = np.uint8)
        mask[img[:, :, i] >= FAT_HU] = 255 # body binary mask

        if i == middle_index and verbose:
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            fig.tight_layout()

            axs[0].imshow(img[:, :, i], cmap='gray')
            axs[0].set_title('Original')
            axs[0].axis('off')

            axs[1].imshow(mask, cmap='gray')
            axs[1].set_title('Binary mask (body & other)')
            axs[1].axis('off')

            plt.show()
            plt.close(fig)
      
        _, labels = cv2.connectedComponents(mask, connectivity=4) 
        unique, counts = np.unique(labels, return_counts=True)
        
        if len(counts) == 1: # air found only
            continue
            
        if i == middle_index and verbose:
            show_img(labels, 'Indexed objects (body & other)', 'viridis')

        # finding body index (0 for air, assuming body is the biggest object)
        body_ind = np.argmax(counts[1 : ]) + 1

        # final binarization
        labels[labels != body_ind] = 0
        labels[labels == body_ind] = 255
        mask = labels.astype(np.uint8)
 
        if i == middle_index and verbose:
            show_img(mask, 'Binary mask (body only)', 'gray')

        # filling body holes steps
        # non-body objects indexing (body holes, background)
        ret, labels = cv2.connectedComponents(255 - mask, connectivity=4)
        unique = np.unique(labels)
    
        if i == middle_index and verbose:
            show_img(labels, 'Indexed objects (background & others)', 'viridis')
    
        # finding body holes vs. other objects
        for ind in unique:
            indices = np.argwhere(labels == ind)[:, 0]
            min_col = np.min(indices)
            max_col = np.max(indices)

            if min_col > 0 and max_col < mask.shape[0] - 1: # TODO: jeżeli nie rozciąga się od góry do dołu
                labels[labels == ind] = 0

        # filling body holes
        mask[labels != 0] = 0
        mask[labels == 0] = 255
        
        # consider everything outside body mask as air
        img[:, :, i][mask == 0] = MIN_HU

        if i == middle_index and verbose:
            show_img(mask, 'Binary mask (no holes)', 'gray')
            
        if i == middle_index:
        
            # calculating mass center
            moments = cv2.moments(mask)
            center_w = moments['m10'] / moments['m00']
            center_h = moments['m01'] / moments['m00']

            if verbose:
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(mask, cmap='gray')
                plt.scatter(center_w, center_h, s=500, c='red', marker='o')
                plt.axis('off')
                plt.title('Binary mask (no holes) with center of mass')
                plt.show()
                plt.close()
                
    return img, int(center_h), int(center_w)

def rescale_mip(img, pixel_spacing, slice_thickness):
    """
    Rescales 2D MIP image to 1mm spacing.
    """
    
    return cv2.resize(img, 
                      (round(img.shape[1] * pixel_spacing), round(img.shape[0] * slice_thickness)), 
                      interpolation=cv2.INTER_CUBIC)

def to_HU(img):
    """
    Converts 3D CT image values to Hounsfield scale for any intial values range.
    """
    
    min_val = np.amin(img)
    diff = abs(MIN_HU-min_val)
    
    if min_val > MIN_HU:
        img -= diff 
    else:
        img += diff
        
    return img

def crop_ct(img, center_h, center_w, for_frontal=True, sides_cut=30):
    """
    Takes a 3D CT (depth x width x height axes order) image and crops it:
    * as a step before calculating frontal MIP - leaves posterior half only;
    * as a step before calculating sagittal MIP - leaves [-sides_cut, sides_cut] width range from the center of image.
    """
    
    if for_frontal:
        return img[center_h:, :, :]
    else:
        return img[:, center_w - sides_cut : center_w + sides_cut, :]
        
def show_img(image, title, cmap):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

# --- VerSe repo functions (https://github.com/anjany/verse/tree/main/utils) below --- 
def reorient_to(img, axcodes_to=('P', 'I', 'R')):
    """Reorients the nifti from its original orientation to another specified orientation
    
    Parameters:
    ----------
    img: nibabel image
    axcodes_to: a tuple of 3 characters specifying the desired orientation
    
    Returns:
    ----------
    newimg: The reoriented nibabel image 
    
    """
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
 
    # print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    
    return newimg

def load_centroids(ctd_path):
    """loads the json centroid file
    
    Parameters:
    ----------
    ctd_path: the full path to the json file
    
    Returns:
    ----------
    ctd_list: a list containing the orientation and coordinates of the centroids
    
    """
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):            #skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']]) 
            
    return ctd_list

def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    
    # print("[*] Image resampled to voxel size:", voxel_spacing)
    
    return new_img

def reorient_centroids_to(ctd_list, img, decimals=1):
    """reorient centroids to image orientation
    
    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image 
    decimals: rounding decimal digits
    
    Returns:
    ----------
    out_list: reoriented list of centroids 
    
    """
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present") 
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
        
    # print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    
    return out_list

def rescale_centroids(ctd_list, img, voxel_spacing=(1, 1, 1)):
    """rescale centroid coordinates to new spacing in current x-y-z-orientation
    
    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image 
    voxel_spacing: desired spacing
    
    Returns:
    ----------
    out_list: rescaled list of centroids 
    
    """
    ornt_img = nio.io_orientation(img.affine)
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] * zms[0] / voxel_spacing[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] * zms[1] / voxel_spacing[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] * zms[2] / voxel_spacing[2], decimals=1)
    out_list = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    
    # print("[*] Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm")
    
    return out_list