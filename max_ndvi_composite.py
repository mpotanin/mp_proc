import os
import argparse
import sys
import json

import s2_meta
from common_utils import vector_operations as vop
from common_utils import raster_proc as rproc
import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr



input_vector_file = 'D:\\work\\inno\\rt\\winter_crops_21\\winter_crops_21.gpkg'
tile_filter = '39UUA 39UUB 39UVB 39UWB 39UXA 39UXB 39UXV 39VUC 39VVC 39VWC 39VXC 39UVA 39UWA 38UPF 38UPG'
input_json_path = 'D:\\work\\inno\\rt\\winter_crops_21'

FID_COL = 'fid'
NDVI_COL = 'ndvi'
tile_filter = tile_filter.split(' ')

for t in tile_filter:
    print(t)

    stat_data = None
    with open(os.path.join(input_json_path,f'{t}.json')) as f:
        stat_data = json.load(f)
        f.close()

    ds = ogr.Open(input_vector_file,1)
    layer = ds.GetLayer()

    feat = layer.GetFeature(1)
    for feat in layer:
        fid = str(feat.GetFID())
        print(fid)
        update = False
        if fid in stat_data:
            for scene in stat_data[fid]:
                if (stat_data[fid][scene]['ndvi'][2] > feat.GetField(NDVI_COL) and
                        0.95 * stat_data[fid][scene]['scl'][0] < stat_data[fid][scene]['scl'][1]):
                    feat.SetField(NDVI_COL,stat_data[fid][scene]['ndvi'][2])
                    update = True
            if update: layer.SetFeature(feat)
    ds=layer=None

exit(0)
#################################################################################################
tile_filter = '39UUA 39UUB 39UVB 39UWB 39UXA 39UXB 39UXV 39VUC 39VVC 39VWC 39VXC 39UVA 39UWA 38UPF 38UPG'
input_folder = 'D:\\work\\39UVB_test'
tile_filter = tile_filter.split(' ')
start = 20210915
end = 20211031
output_folder = 'D:\\work\\39UVB_test\\composite_max'
NODATA_VAL = -10000




for t in tile_filter:
    ndvi_max_img = None
    srs,geotr = None,None
    for sceneid in os.listdir(input_folder):
        if not sceneid.startswith('S2'): continue

        if s2_meta.SceneID.tile_name(sceneid) != t: continue
        if int(s2_meta.SceneID.date(sceneid)) < start: continue
        if int(s2_meta.SceneID.date(sceneid)) > end: continue

        print(sceneid)

        output_file = os.path.join(output_folder,f'{t}_ndvi_max_{start}{end}.tif')
        scene_path = os.path.join(input_folder,sceneid)
        red_band_file = s2_meta.L2AScene.get_band_file(scene_path,'B04',extension='tif')
        nir_band_file = s2_meta.L2AScene.get_band_file(scene_path, 'B08', extension='tif')
        ndvi_img = rproc.calc_ndvi_as_image(red_band_file,nir_band_file,0,NODATA_VAL)
        if ndvi_max_img is None:
            ndvi_max_img = ndvi_img
            srs,geotr = rproc.extract_georeference(red_band_file)
        else:
            ndvi_max_img = np.maximum(ndvi_max_img,ndvi_img)


        #calc ndvi with NODATA -9999
        #recalculate max
    if ndvi_max_img is not None:
        output_file = os.path.join(output_folder, f'{t}_ndvi_max_{start}{end}.tif')
        rproc.array2geotiff(output_file,[geotr[0],geotr[3]],geotr[1],srs,ndvi_max_img,nodata_val=NODATA_VAL)
    #save output