import os
import argparse
import sys
import json

import numpy as np
import s2_meta

import zoning
from common_utils import vector_operations as vop
from common_utils import raster_proc as rproc


from osgeo import gdal
from osgeo import ogr
from osgeo import osr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     ('Calculate zones color RGB 24bit GeoTIFF image for each feature'))

    parser.add_argument('-i', required=True, metavar='L2A folder',
                        help='Folder with sentinel 2 L2 products')
    parser.add_argument('-iv', required=True, metavar='vector file',
                        help='Input vector file')
    parser.add_argument('-ib', required=True, metavar='best scene json',
                        help='Input best scene json')
    parser.add_argument('-ext', required=False, default='tif', metavar='tif|jp2',
                        help='tif|jp2')
    parser.add_argument('-o', required=True, metavar='output folder',
                        help='Output folder to save zones to')

    if (len(sys.argv) == 1):
        parser.print_usage()
        exit(0)
    args = parser.parse_args()


    vec_srs = vop.VectorFile.get_srs_from_file(args.iv)
    features = vop.VectorFile.get_all_features(args.iv,t_srs=vec_srs)

    best_scenes = None
    with open(args.ib) as json_file:
        best_scenes = json.load(json_file)

    for feat in features:
        feat_id = feat.GetFieldAsInteger(zoning.FID_COL)
        if str(feat_id) not in best_scenes.keys(): continue

        scene = best_scenes[str(feat_id)]
        if (scene is None) or (scene == '') : continue
        print(feat_id)

        scene_full_path = os.path.join(args.i,scene)
        feat_wkt = feat.GetGeometryRef().ExportToWkt()
        vec_in_mem = vop.VectorFile.create_virtual_vector_file(feat_wkt,srs=vec_srs)
        rast_srs,geotr = rproc.extract_georeference(s2_meta.L2AScene.get_band_file(scene_full_path,'B08',args.ext),
                                                cutline=vec_in_mem)
        band_imgs = zoning.extract_band_refl_imgs(args.i, scene, vec_in_mem, args.ext)
        msavi_img = zoning.calc_msavi(band_imgs[1], band_imgs[2])
        msavi_img_smoothed = zoning.calc_smoothed_img(msavi_img)
        zones_rgb = zoning.transform_to_rgb(zoning.colorize(msavi_img_smoothed))

        rproc.array2geotiff(os.path.join(args.o,f'{feat_id}.tif'),
                            [geotr[0], geotr[3]], geotr[1], rast_srs,
                            zones_rgb,
                            nodata_val=0)
        vop.VectorFile.remove_virtual_vector_file(vec_in_mem)





