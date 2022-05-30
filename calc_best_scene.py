import os
import argparse
import sys
import json

import zoning
from common_utils import vector_operations as vop

from osgeo import gdal
from osgeo import ogr
from osgeo import osr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     ('Calculate best scenes for each feature'))

    parser.add_argument('-i', required=True, metavar='L2A folder',
                        help='Folder with sentinel 2 L2 products')
    parser.add_argument('-iv', required=True, metavar='vector file',
                        help='Input vector file')
    parser.add_argument('-istat', required=True, metavar='stat file',
                        help='Input stat file')
    parser.add_argument('-ext', required=False, default='tif', metavar='tif|jp2',
                        help='tif|jp2')
    parser.add_argument('-o', required=True, metavar='output file',
                        help='Output json file')


    if (len(sys.argv) == 1):
        parser.print_usage()
        exit(0)
    args = parser.parse_args()




    with open(args.istat) as fp:
        all_stat = json.load(fp)


    best_scenes = dict()
    vec_srs = vop.VectorFile.get_srs_from_file(args.iv)
    features = vop.VectorFile.get_all_features(args.iv,t_srs=vec_srs)

    for feat in features:

        if feat.GetGeometryRef().IsValid():
            wkt = feat.GetGeometryRef().ExportToWkt()
        else:
            continue

        feat_vec_in_mem = vop.VectorFile.create_virtual_vector_file(feat.GetGeometryRef().ExportToWkt(),
                                                                    srs=vec_srs)
        feat_id = feat.GetFieldAsInteger(zoning.FID_COL)
        candidates = zoning.select_candidate_dates(all_stat[str(feat_id)])
        best_scenes[feat_id] = zoning.calc_best_scene(candidates,args.i,feat_vec_in_mem,args.ext)

        vop.VectorFile.remove_virtual_vector_file(feat_vec_in_mem)

    with open(args.o,'w') as fp:
        json.dump(best_scenes,fp)
    print('OK!')
