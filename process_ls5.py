import os
import sys
import argparse
import numpy as np
from common_utils import raster_proc as rproc
from common_utils import vector_operations as vop
from osgeo import gdal, osr

from eolearn.core import EOPatch
from eolearn.core import EOExecutor, OverwritePermission, EOTask, EOPatch, LinearWorkflow, FeatureType
from eolearn.io import ImportFromTiff, ExportToTiff

from fiona.env import NullContextManager
from sentinelhub import UtmZoneSplitter, BBox, CRS, DataCollection, UtmGridSplitter, Geometry
import geopandas as gpd


class SceneID:
    # LT05_L2SP_170022_19850609_20200918_02_T1_SR_B4.TIF
    @staticmethod
    def date(sceneid):
        return sceneid[17:25]

    @staticmethod
    def day(sceneid, type='string'):
        return sceneid[23:25] if type == 'string' else int(sceneid[23:25])

    @staticmethod
    def month(sceneid, type='string'):
        return sceneid[21:23] if type == 'string' else int(sceneid[21:23])

    @staticmethod
    def year(sceneid, type='string'):
        return sceneid[17:21] if type == 'string' else int(sceneid[17:21])

    @staticmethod
    def path_row(sceneid):
        return sceneid[10:16]

    @staticmethod
    def get_band(sceneid):
        return sceneid[44:46]

class L2AScene:
    BANDS = ['B1','B2','B3','B4','B5','B6','B7']

    @staticmethod
    def get_scene_id_from_path (scene_full_path):
        while scene_full_path[-1] == '\\' or scene_full_path[-1] == '/':
            scene_full_path = scene_full_path[:-1]
        return os.path.basename(scene_full_path)

    @staticmethod
    def get_band_file(scene_full_path, band_name):
        scene_full_path = L2AScene.get_scene_id_from_path(scene_full_path)

        if band_name not in L2AScene.BANDS:
            return None

        scene = os.path.basename(scene_full_path)
        if band_name != 'B6':
            return f'{scene}_SR_{band_name}.TIF'
        else:
            return f'{scene}_ST_{band_name}.TIF'

    @staticmethod
    def get_pixel_quality_file (scene_full_path):
        scene_full_path = L2AScene.get_scene_id_from_path(scene_full_path)
        return f'{os.path.basename(scene_full_path)}_QA_PIXEL.TIF'

    @staticmethod
    def get_cloud_mask_file (scene_full_path):
        scene_full_path = L2AScene.get_scene_id_from_path(scene_full_path)
        return f'{os.path.basename(scene_full_path)}_SR_CLOUD_QA.TIF'

    @staticmethod
    def transform_sr_values (raw_values):
        return np.minimum(np.maximum((0.275 * raw_values - 2000)*0.0001, 0),0.9)

    @staticmethod
    def calc_SR_values (band_file_full_path):
        return L2AScene.transform_sr_values(rproc.open_clipped_raster_as_image(band_file_full_path))

    @staticmethod
    def calc_valid_pixels_mask (cloud_qa_pixels, qa_pixels):
        return (
            np.where( (cloud_qa_pixels>>1 & 1 == 1) | (cloud_qa_pixels>>2 & 1 == 1) | (cloud_qa_pixels>>3 & 1 == 1), 0, 1)
            * np.where((qa_pixels == 1) | (qa_pixels>>1 & 1 == 1)  | (qa_pixels>>3 & 1 == 1) | (qa_pixels>>4 & 1 == 1),0,1)
        )



    """
    @staticmethod
    def calc_valid_pixels_mask(scene_full_path):

        img = rproc.open_clipped_raster_as_image(
            os.path.join(scene_full_path, L2AScene.get_cloud_mask_file(scene_full_path)))
        mask = np.where( (img>>1 & 1 == 1) | (img>>2 & 1 == 1) | (img>>3 & 1 == 1), 0, 1)
        img = rproc.open_clipped_raster_as_image(
            os.path.join(scene_full_path,L2AScene.get_pixel_quality_file(scene_full_path)))
        mask *= np.where((img == 1) | (img>>1 & 1 ==1) | (img>>3 & 1 == 1) | (img>>4 & 1 ==1),0 ,1 )
        for b in L2AScene.BANDS:
            img = rproc.open_clipped_raster_as_image(
                os.path.join(scene_full_path,L2AScene.get_band_file(scene_full_path,b)))
            mask *= np.where(img==0,0,1)

        return mask
    """

class IntegrateMaskTask(EOTask):

    def execute(self, eopatch, *, patch_composite = None):
        if 'VALID_COUNT' not in patch_composite.mask_timeless:
            patch_composite.mask_timeless['VALID_COUNT'] = np.zeros(
                    shape=eopatch.mask_timeless['IS_VALID'].shape,dtype=np.uint8)
        patch_composite.mask_timeless['VALID_COUNT'] += eopatch.mask_timeless['IS_VALID']

        return patch_composite

class IntegrateMaxNDVITask(EOTask):
    def execute(self,eopatch, *, patch_composite = None ):
        if 'BANDS' not in patch_composite.data_timeless:
            patch_composite.data_timeless['BANDS'] = eopatch.data_timeless['BANDS'].copy()
            patch_composite.data_timeless['NDVI'] = eopatch.data_timeless['NDVI'].copy()
            patch_composite.mask_timeless['VALID_COUNT'] = eopatch.mask_timeless['IS_VALID'].copy()
            patch_composite.mask_timeless['BRIGHT_CLOUD'] = eopatch.mask_timeless['BRIGHT_CLOUD'].copy()

        else:
            for b in range(0,patch_composite.data_timeless['BANDS'].shape[2]):
                patch_composite.data_timeless['BANDS'][:,:,b] = np.where(
                    ((eopatch.data_timeless['NDVI'][:,:,0] > patch_composite.data_timeless['NDVI'][:,:,0])
                    & (eopatch.mask_timeless['BRIGHT_CLOUD'][:,:,0] == 0))
                    | (eopatch.mask_timeless['BRIGHT_CLOUD'][:,:,0] < patch_composite.mask_timeless['BRIGHT_CLOUD'][:,:,0]),
                    eopatch.data_timeless['BANDS'][:,:,b],patch_composite.data_timeless['BANDS'][:,:,b]
                )
            patch_composite.data_timeless['NDVI'] = np.where(
                ((eopatch.data_timeless['NDVI'] > patch_composite.data_timeless['NDVI'])
                 & (eopatch.mask_timeless['BRIGHT_CLOUD'] == 0))
                | (eopatch.mask_timeless['BRIGHT_CLOUD'] < patch_composite.mask_timeless['BRIGHT_CLOUD']),
                eopatch.data_timeless['NDVI'], patch_composite.data_timeless['NDVI']
            )
            patch_composite.mask_timeless['BRIGHT_CLOUD'] = np.minimum(
                patch_composite.mask_timeless['BRIGHT_CLOUD'],eopatch.mask_timeless['BRIGHT_CLOUD']
            )

            patch_composite.mask_timeless['VALID_COUNT'] += eopatch.mask_timeless['IS_VALID']

        return patch_composite

class CalcNDVITask(EOTask):

    def execute(self,eopatch):
        eopatch.data_timeless['NDVI'] = np.zeros(
                        shape=(eopatch.data_timeless['BANDS'].shape[0],eopatch.data_timeless['BANDS'].shape[1],1),
                        dtype=np.float32
        )
        eopatch.data_timeless['NDVI'][:,:,0] = rproc.calc_ndvi_as_image_from_mem(
            eopatch.data_timeless['BANDS'][:,:,2],eopatch.data_timeless['BANDS'][:,:,3]
        )
        return eopatch

class LoadPathTask(EOTask):
    def execute(self, bbox, scenes, skip_bands_load = False):

        load_task = LoadSceneTask()
        patch_composite = load_task.execute(bbox,scenes[0],skip_bands_load)

        for i in range(1,len(scenes)):
            patch_scene = load_task.execute(bbox,scenes[i],skip_bands_load)
            if not skip_bands_load:
                for b in range(0,patch_composite.data_timeless['BANDS'].shape[2]):
                    patch_composite.data_timeless['BANDS'][:,:,b] = np.where(
                        patch_composite.mask_timeless['IS_DATA'][:,:,0] == 1,
                        patch_composite.data_timeless['BANDS'][:,:,b],patch_scene.data_timeless['BANDS'][:,:,b]
                    )
            patch_composite.mask_timeless['IS_VALID'] |= patch_scene.mask_timeless['IS_VALID']
            patch_composite.mask_timeless['IS_DATA'] |= patch_scene.mask_timeless['IS_DATA']
            patch_composite.mask_timeless['BRIGHT_CLOUD'] |= patch_scene.mask_timeless['BRIGHT_CLOUD']

        return patch_composite


class LoadSceneTask(EOTask):

    def execute(self, bbox, scene_full_path, skip_bands_load = False):
        patch = EOPatch()
        patch.bbox = bbox
        output_bounds = (bbox.lower_left[0],bbox.lower_left[1],bbox.upper_right[0],bbox.upper_right[1])
        patch_srs = osr.SpatialReference()
        patch_srs.ImportFromEPSG(bbox.crs.epsg)
        pixel_res = 30

        pixel_width = int(0.5 + ((bbox.upper_right[0] - bbox.lower_left[0])/pixel_res))
        pixel_height = int(0.5 + ((bbox.upper_right[1] - bbox.lower_left[1])/pixel_res))


        if not skip_bands_load:
            patch.data_timeless['BANDS'] = np.zeros(shape=(pixel_height,pixel_width,len(L2AScene.BANDS)-1),
                                                    dtype=np.float32
                                                    )
            i = 0
            for band in L2AScene.BANDS:
                if band =='B6': continue
                band_file = os.path.join(scene_full_path,L2AScene.get_band_file(scene_full_path,band))
                patch.data_timeless['BANDS'][:,:,i] = L2AScene.transform_sr_values(
                                rproc.open_clipped_raster_as_image(raster_file=band_file,dst_nodata=0,
                                                                  output_bounds=output_bounds,
                                                                  pixel_width=pixel_width,pixel_height=pixel_height,
                                                                  dst_srs=patch_srs,resample_alg=gdal.GRA_Cubic,
                                                                   output_type=gdal.GDT_UInt16)
                )
                i+=1

        mask_files = [os.path.join(scene_full_path,L2AScene.get_cloud_mask_file(scene_full_path)),
                      os.path.join(scene_full_path, L2AScene.get_pixel_quality_file(scene_full_path))]
        mask_imgs = ([rproc.open_clipped_raster_as_image( raster_file=mf,dst_nodata=1,
                                output_bounds=output_bounds,pixel_width=pixel_width, pixel_height=pixel_height,
                                dst_srs=patch_srs, resample_alg=gdal.GRA_NearestNeighbour, output_type=gdal.GDT_UInt16)
                     for mf in mask_files]
        )

        patch.mask_timeless['IS_DATA'] = np.empty( shape=[pixel_height,pixel_width,1], dtype=np.uint8 )
        patch.mask_timeless['IS_DATA'][:,:,0] = np.where(
            (patch.data_timeless['BANDS'][:,:,3] > 0) & (patch.data_timeless['BANDS'][:,:,2] > 0),1,0
        )

        patch.mask_timeless['BRIGHT_CLOUD'] = np.empty( shape=[pixel_height,pixel_width,1], dtype=np.uint8 )
        patch.mask_timeless['BRIGHT_CLOUD'][:,:,0] = np.where(
            (patch.data_timeless['BANDS'][:,:,3] > 0.6) & (patch.data_timeless['BANDS'][:,:,2] > 0.6),1,0
        )

        patch.mask_timeless['IS_VALID'] = np.empty( shape=[pixel_height,pixel_width,1], dtype=np.uint8 )
        patch.mask_timeless['IS_VALID'][:,:,0] = L2AScene.calc_valid_pixels_mask(mask_imgs[0],mask_imgs[1])

        return patch

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description=(''))

    parser.add_argument('-i', required=True, metavar='input folder', help='Folder with L5 L2A products')
    parser.add_argument('-o', required=True, metavar='output file', help='Output file')
    parser.add_argument('-ys', type=int, default=1000, required=False, metavar='year start', help='Year start')
    parser.add_argument('-ye', type=int, default=3000, required=False, metavar='year end', help='Year end')
    parser.add_argument('-ms', type=int, default=1, required=False, metavar='month start', help='Month start')
    parser.add_argument('-me', type=int, default=12, required=False, metavar='month end', help='Month end')
    parser.add_argument('-sf', required=False, metavar='scene filter', help='Scene filter')
    parser.add_argument('-m', required=True, metavar='method', help='composite method')


    if (len(sys.argv) == 1):
        parser.print_usage()
        #exit(0)
    #args = parser.parse_args()

    input_folder = 'D:\\work\\inno\\rt\\L5' #args.i
    output_file = 'D:\\work\\inno\\rt\\test\\max_ndvi.tif'#args.o
    year_start = 1000 #args.ys
    year_end = 3000#args.ye
    month_start = 1#args.ms
    month_end = 12#args.me
    max_workers = 1
    method = 'ndvi'#args.m
    scene_filter = None
    #if args.sf is not None:
    if False:
        scene_filter = list()
        with open(args.sf) as f:
            lines = f.read().splitlines()
        for l in lines:
            scene_filter.append(f'{l.split(",")[0]}_{l.split(",")[1]}')

    daily_paths = dict()
    for scene in os.listdir(input_folder):
        if not scene.startswith('LT05_L2SP'): continue
        if SceneID.year(scene, type=int) < year_start or SceneID.year(scene, type=int) > year_end: continue
        if SceneID.month(scene, type=int) < month_start or SceneID.month(scene, type=int) > month_end: continue
        if scene_filter is not None:
            if f'{SceneID.path_row(scene)}_{SceneID.date(scene)}' not in scene_filter: continue

        if f'{SceneID.date(scene)}' not in daily_paths:
            daily_paths[f'{SceneID.date(scene)}'] = list()
        daily_paths[f'{SceneID.date(scene)}'].append(os.path.join(input_folder, scene))

    execution_args = []

    #BBOX_RT_UTM39 = BBox(((513320,6130870), (514340, 6132880)), crs=CRS('32639'))
    #BBOX_RT_UTM39 = BBox(((408429, 6073254), (555189, 6191784)), crs=CRS('32639'))
    BBOX_RT_UTM39 = BBox(((258429, 5983254), (705189, 6281784)), crs=CRS('32639'))

    load_task = LoadPathTask()
    patch_composite = EOPatch()
    patch_composite.bbox = BBOX_RT_UTM39
    workflow = None


    if method.upper() == 'NDVI':
        composite_ndvi_task = IntegrateMaxNDVITask()
        calc_ndvi_task = CalcNDVITask()
        workflow = LinearWorkflow(load_task,calc_ndvi_task,composite_ndvi_task)
        for dp in daily_paths:
            execution_args.append({
                load_task: {'bbox': BBOX_RT_UTM39, 'scenes' : daily_paths[dp], 'skip_bands_load' : False},
                calc_ndvi_task: {},
                composite_ndvi_task : {'patch_composite' : patch_composite}
            })
    else:
        integrate_mask_task = IntegrateMaskTask()
        workflow = LinearWorkflow(load_task,integrate_mask_task)

        for dp in daily_paths:
            execution_args.append({
                load_task: {'bbox': BBOX_RT_UTM39, 'scenes' : daily_paths[dp], 'skip_bands_load' : True},
                integrate_mask_task : {'patch_composite' : patch_composite}
            })



    executor = EOExecutor(workflow, execution_args, save_logs=False)
    executor.run(workers=max_workers, multiprocess=False)


    if method.upper() == 'NDVI':

        export_task = ExportToTiff(feature = (FeatureType.DATA_TIMELESS, 'BANDS'),
                                       folder = os.path.dirname(output_file),
                                       band_indices=[0,1,2,3,4,5],
                                       no_data_value=0
                                   )
        export_task.execute(eopatch=patch_composite,filename=os.path.basename(output_file))

        export_task = ExportToTiff(feature=(FeatureType.DATA_TIMELESS, 'NDVI'),
                                   folder=os.path.dirname(output_file),
                                   band_indices=[0],
                                   no_data_value=-10000
                                   )
        export_task.execute(eopatch=patch_composite, filename=os.path.basename(output_file).replace('.tif','_ndvi.tif'))
    else:
        export_task = ExportToTiff(feature = (FeatureType.MASK_TIMELESS, 'VALID_COUNT'),
                                       folder = os.path.dirname(output_file),
                                       band_indices=[0],
                                       no_data_value=0
                                   )
        export_task.execute(eopatch=patch_composite,filename=os.path.basename(output_file))

    exit(0)