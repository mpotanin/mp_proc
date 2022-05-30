import os
import argparse
import sys
import json

from common_utils import raster_proc as rproc
from common_utils import vector_operations as vop
import numpy as np
import s2_meta

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import scipy.ndimage as ndi

NDV_FLOAT_IMG = -10000
MAX_BAND_VAL = 10000
FID_COL = 'fid'




def smooth_with_mask(image, mask):
    """Smooth an image with a linear function, ignoring masked pixels.

    Parameters
    ----------
    image : array
        Image you want to smooth.
    function : callable
        A function that does image smoothing.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.

    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """

    bleed_over = ndi.gaussian_filter(mask.astype(float), sigma=1, mode='constant')
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = ndi.gaussian_filter(masked_image, sigma=1, mode='constant')
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image




def are_enough_valid_pixels(feat_ndvi_img, feat_geometry):
    valid_pixels = np.count_nonzero(feat_ndvi_img != NDV_FLOAT_IMG)
    if valid_pixels < 10: return False
    else:
        return False if 0.75*(feat_geometry.GetArea()/100) > valid_pixels else True

def get_scl_metrics(in_mem_scl_tif,feat_vec_in_mem):
    scl_img = rproc.open_clipped_raster_as_image(in_mem_scl_tif,dst_nodata=0,cutline=feat_vec_in_mem)
    cloudless_pixels = np.count_nonzero((scl_img==4) | (scl_img==5) | (scl_img==6) )
    return (np.count_nonzero(scl_img>0),cloudless_pixels)

# select cloudless images:
# 0501 < date < 0731
# 0.4<=mean_ndvi <= max_ndvi
# cloudless = 100%
# get ndvi
# find date with min max-min brigtness: sqrt(G^2 + R^2 + NIR^2 + SWIR^2)
# calc based on MSAVI2

#(2*NIR+1-SQRT((2*NIR+1)^2 - 8*(NIR-RED)))/2

def classify_by_quantiles (img, quantiles):

    quant_vals = list()
    img_only_valid = img[img!=NDV_FLOAT_IMG]

    img_classified = np.full(img.shape,fill_value=NDV_FLOAT_IMG, dtype=int)
    lower_val = NDV_FLOAT_IMG
    class_num = 1
    for q in quantiles:
        upper_val = np.quantile(img_only_valid,q)
        img_classified = np.where(img>lower_val & img <=upper_val, class_num, img_classified)
        lower_val = upper_val
        class_num+=1
    return img_classified

def calc_nodata_dilation(img_classified):
    img_out = np.full(img_classified.shape,fill_value=NDV_FLOAT_IMG,dtype=int)

    for i in range(1,img_classified.shape(0)-1):
        for j in range(1,img_classified.shape(1)-1):
            if np.where(img_classified[i-1:i+2,j-1:j+2] == NDV_FLOAT_IMG)[0].size == 0:
                img_out[i][j] = img_classified[i][j]
            else:
                img_out[i][j] = NDV_FLOAT_IMG

    return img_out

def calc_majority_vote_filter (img_classified):
    img_out = np.full(img_classified.shape,fill_value=NDV_FLOAT_IMG,dtype=int)
    img_3_3 = None

    for i in range(1,img_classified.shape(0)-1):
        for j in range(1,img_classified.shape(1)-1):
            if img_classified[i,j] == NDV_FLOAT_IMG:
                img_out[i][j] = NDV_FLOAT_IMG
            else:
                img_3_3 = img_classified[i-1:i+2,j-1:j+2]
                img_out[i][j] = np.argmax(np.bincount(img_3_3[img_3_3!=NDV_FLOAT_IMG]))
    return img_out

def calc_under_nodata_dialation (img_before_filtering, img_after_filtering):

    def cut_out_window (img, i, j, w):
        i1 = i-w if i-w>=0 else 0
        i2 = i+w+1 if i+w+1<=img.shape[0] else img.shape[0]
        j1 = j-w if j-w>=0 else 0
        j2 = j+w+1 if j+w+1<=img.shape[1] else img.shape[1]

        return img[i1:i2,j1:j2]

    img_out = np.full(img_classified.shape,fill_value=NDV_FLOAT_IMG,dtype=int)
    img_tmp = None
    for i in range(0,img_classified.shape(0)):
        for j in range(0,img_classified.shape(1)):
            if (img_before_filtering[i,j] != NDV_FLOAT_IMG) and (img_after_filtering[i,j]==NDV_FLOAT_IMG):
                img_tmp = cut_out_window(img_after_filtering,i,j,2)
                if np.where(img_tmp!=NDV_FLOAT_IMG)[0].size!=0:
                    img_out[i, j] = np.argmax(np.bincount(img_tmp[img_tmp!=NDV_FLOAT_IMG]))
                else:
                    img_out[i, j] = img_before_filtering[i,j]
            else:
                img_out[i,j] = img_after_filtering[i,j]


def colorize (img):
    palette = [
        {"color": "#800101"},
        {"color": "#CA0105"},
        {"color": "#E8DA43"}, #"#DF5F0B"
        {"color": "#FEF015"},
        {"color": "#BCF520"}, #E3F518
        {"color": "#75A327"},
        {"color": "#13713C"}
    ]
    img_valid_only = img[img!=NDV_FLOAT_IMG]
    img_valid_only = np.reshape(img_valid_only,img_valid_only.size)

    range_values = []
    for i in range(len(palette)):
        range_values.append(np.quantile(img_valid_only,(i+1)/len(palette)))
    range_values.insert(0,NDV_FLOAT_IMG)

    img_out = np.full(img.shape,fill_value=NDV_FLOAT_IMG,dtype=int)
    for i in range(1,len(palette)+1):
        color_val = int(palette[i-1]['color'][1:],16)
        img_out = np.where((img>range_values[i-1]) & (img<=range_values[i]),color_val,img_out)

    return img_out

def transform_to_rgb(img):
    img_rgb = np.full([3,img.shape[0],img.shape[1]],fill_value=0,dtype=np.uint8)
    img_rgb[0] = np.where(img!=NDV_FLOAT_IMG,np.floor_divide(img,65536),0)
    img_rgb[1] = np.where(img!=NDV_FLOAT_IMG,np.floor_divide(img - 65536*img_rgb[0], 256),0)
    img_rgb[2] = np.where(img!=NDV_FLOAT_IMG,np.remainder(img,256),0)

    return img_rgb

def calc_smoothed_img (img):
    img_smoothed = smooth_with_mask(img,img!=NDV_FLOAT_IMG)
    return np.where(img!=NDV_FLOAT_IMG,img_smoothed,NDV_FLOAT_IMG)

def calc_best_scene (candidates : list, base_path : str, feat_in_mem_vec : str, ext='.jp2'):

    min_diff = 10
    best_scene = ''

    for scene in candidates:
        if not os.path.exists(os.path.join(base_path,scene)):
            continue
        band_imgs = extract_band_refl_imgs(base_path,scene,feat_in_mem_vec,ext)
        min,max = calc_min_max_brightness(band_imgs[0],band_imgs[1],band_imgs[2],band_imgs[3])
        if max-min < min_diff:
            min_diff = max-min
            best_scene = scene

    return best_scene



def select_candidate_dates (feat_stat:dict):

    output = list()
    max_ndvi = -1
    max_ndvi_date = None
    for scene in feat_stat.keys():
        if s2_meta.SceneID.month(scene, type=int) < 5 and s2_meta.SceneID.month(scene, type=int) > 7:
            continue
        if feat_stat[scene]['ndvi'][2] > max_ndvi:
            max_ndvi = feat_stat[scene]['ndvi'][2]
            max_ndvi_date = int(s2_meta.SceneID.date(scene))

    if max_ndvi_date is not None:
        for scene in feat_stat.keys():
            if s2_meta.SceneID.month(scene,type=int) < 5 and s2_meta.SceneID.month(scene,type=int) > 7:
                continue

            if feat_stat[scene]['ndvi'][2] < 0.4:
                continue

            if int(s2_meta.SceneID.date(scene)) > max_ndvi_date:
                continue

            if 0.98*feat_stat[scene]['scl'][0] > feat_stat[scene]['scl'][1]:
                continue

            output.append(scene)

    return output

def calc_msavi (red, nir):
    return np.where((red!=NDV_FLOAT_IMG)&(nir!=NDV_FLOAT_IMG),
                    nir + 0.5 - 0.5*np.sqrt( (2*nir + 1) * (2*nir + 1) - 8*(nir - red)),
                    NDV_FLOAT_IMG)


def calc_brightness (green, red, nir, swir):
    return np.where((green!=NDV_FLOAT_IMG) & (red!=NDV_FLOAT_IMG) & (nir!=NDV_FLOAT_IMG) & (swir!=NDV_FLOAT_IMG),
                    np.sqrt(green*green + red*red + nir*nir + swir*swir),
                    NDV_FLOAT_IMG)

def calc_min_max_brightness (green, red, nir, swir):
    brightness = calc_brightness (green, red, nir, swir)
    return (np.where(brightness!=NDV_FLOAT_IMG,brightness,10).min(),
            np.where(brightness != NDV_FLOAT_IMG, brightness, 0).max())

def extract_band_refl_imgs (base_path, sceneid, feat_in_mem_vec, ext = 'jp2'):
    bands_10m=['B03','B04','B08'] # output + SWIR 'B11'
    output_refl_imgs = list()
    scene_full_path = os.path.join(base_path,sceneid)
    for b in bands_10m:
        band_refl_img = rproc.open_clipped_raster_as_image(
                                    s2_meta.L2AScene.get_band_file(scene_full_path,b,ext),
                                    cutline=feat_in_mem_vec,crop_to_cutline=True,dst_nodata=0)
        band_refl_img = np.where(band_refl_img!=0,1e-4*band_refl_img,NDV_FLOAT_IMG)

        output_refl_imgs.append(band_refl_img)

    band_refl_img = rproc.open_clipped_raster_as_image(
                        s2_meta.L2AScene.get_band_file(scene_full_path,'B11',ext),
                        cutline=feat_in_mem_vec,crop_to_cutline=True,dst_nodata=0,
                        pixel_width=output_refl_imgs[0].shape[1],pixel_height=output_refl_imgs[0].shape[0])
    band_refl_img = np.where(band_refl_img!=0,1e-4*band_refl_img,NDV_FLOAT_IMG)
    output_refl_imgs.append(band_refl_img)

    return output_refl_imgs


def collect_tile_stat (input_vector, base_path, tile, filter_start, filter_end):
    all_stat = dict()
    scenes = os.listdir(base_path)

    for sceneid in scenes:
        if not (sceneid.startswith('S2A_MSIL2A') or sceneid.startswith('S2B_MSIL2A')) : continue
        scene_full_path = os.path.join(base_path, sceneid)
        if not os.path.isdir(scene_full_path): continue

        if s2_meta.SceneID.tile_name(sceneid) != tile: continue
        if int(s2_meta.SceneID.date(sceneid))<int(filter_start): continue
        if int(s2_meta.SceneID.date(sceneid)) > int(filter_end): continue


        print(sceneid)

        red_band_file = s2_meta.L2AScene.get_band_file(scene_full_path, 'B04', 'tif')
        nir_band_file = s2_meta.L2AScene.get_band_file(scene_full_path, 'B08', 'tif')
        scl_band_file = s2_meta.L2AScene.get_band_file(scene_full_path, 'SCL', 'tif')

        srs, geotr = rproc.extract_georeference(red_band_file)

        ndvi_img = rproc.calc_ndvi_as_image(red_band_file, nir_band_file)
        in_mem_ndvi_tif = rproc.generate_virtual_random_tif_path()
        rproc.array2geotiff(in_mem_ndvi_tif, [geotr[0], geotr[3]], geotr[1], srs, ndvi_img, NDV_FLOAT_IMG)
        ndvi_img = None

        in_mem_scl_tif = rproc.get_clipped_inmem_raster(scl_band_file, dst_nodata=0)

        vec_srs = vop.VectorFile.get_srs_from_file(input_vector)
        features = vop.VectorFile.get_all_features(input_vector, t_srs=vec_srs)



        for feat in features:

            if feat.GetGeometryRef().IsValid():
                wkt = feat.GetGeometryRef().ExportToWkt()
            else:
                continue


            feat_vec_in_mem = vop.VectorFile.create_virtual_vector_file(feat.GetGeometryRef().ExportToWkt(),
                                                                        srs=vec_srs)
            feat_ndvi_in_mem = rproc.get_clipped_inmem_raster(in_mem_ndvi_tif, cutline=feat_vec_in_mem,
                                                              dst_nodata=NDV_FLOAT_IMG, crop_to_cutline=True)
            feat_ndvi_img = rproc.open_clipped_raster_as_image(feat_ndvi_in_mem)


            if are_enough_valid_pixels(feat_ndvi_img, feat.GetGeometryRef()):
                feat_total_pix, feat_cloudless_pix = get_scl_metrics(in_mem_scl_tif, feat_vec_in_mem)
                feat_id = feat.GetFieldAsInteger(FID_COL)
                if feat_id not in all_stat.keys():
                    all_stat[feat_id] = dict()
                all_stat[feat_id][sceneid] = dict()
                all_stat[feat_id][sceneid]['ndvi'] = rproc.get_statistics(feat_ndvi_in_mem)
                all_stat[feat_id][sceneid]['scl'] = [feat_total_pix, feat_cloudless_pix]

            #print(f'{rproc.get_statistics(feat_ndvi_in_mem)} | {feat_bbox_pix}, {feat_cloudless_pix}')

            feat_ndvi_img = None
            gdal.Unlink(feat_ndvi_in_mem)
            vop.VectorFile.remove_virtual_vector_file(feat_vec_in_mem)

        gdal.Unlink(in_mem_ndvi_tif)
        gdal.Unlink(in_mem_scl_tif)

    return all_stat