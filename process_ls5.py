import os

import numpy as np
from common_utils import raster_proc as rproc
from common_utils import vector_operations as vop

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
        return np.maximum((0.275 * raw_values - 2000)*0.0001, 0)

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


class LoadSceneTask(EOTask):

    def execute(self, bbox, scene_full_path, skip_bands_load = False):
        patch = EOPatch()
        patch.bbox = bbox

        if not skip_bands_load:
            import_task = ImportFromTiff((FeatureType.DATA_TIMELESS, 'BANDS'),
                            folder=scene_full_path,
                            image_dtype=np.float32,
                            no_data_value=0)
            band_files = [L2AScene.get_band_file(scene_full_path,b) for b in L2AScene.BANDS if b!='B6']
            import_task.execute(filename=band_files, eopatch=patch)
            patch.data_timeless['BANDS'] = L2AScene.transform_sr_values(patch.data_timeless['BANDS'])


        patch_mask_bands = EOPatch()
        patch_mask_bands.bbox = bbox
        import_task = ImportFromTiff((FeatureType.DATA_TIMELESS, 'BANDS'),
                        folder=scene_full_path,
                        image_dtype=np.ushort,
                        no_data_value=1)
        import_task.execute(
            filename= [L2AScene.get_cloud_mask_file(scene_full_path), L2AScene.get_pixel_quality_file(scene_full_path)],
            eopatch=patch_mask_bands )

        cloudless_mask = L2AScene.calc_valid_pixels_mask(patch_mask_bands.data_timeless['BANDS'][:,:,0],
                                                         patch_mask_bands.data_timeless['BANDS'][:,:,1])

        patch_mask_bands = None

        patch.mask_timeless['IS_VALID'] = np.empty(
            shape=[cloudless_mask.shape[0],cloudless_mask.shape[1],len(L2AScene.BANDS)-1 ],
            dtype=np.byte)

        for b in range(0,patch.mask_timeless['IS_VALID'].shape[2]):
            patch.mask_timeless['IS_VALID'][:,:,b] = cloudless_mask

        return patch

if __name__ == '__main__':

    BBOX_RT_UTM39 = BBox(((258429, 5983254), (705189, 6281784)), crs=CRS('32639'))

    load_task = LoadSceneTask()
    patch = load_task.execute(bbox=BBOX_RT_UTM39,
                      scene_full_path='D:\\work\\inno\\rt\\L5\\LT05_L2SP_170022_19860527_20200918_02_T1',
                              skip_bands_load=True)

    export_task = ExportToTiff(feature = (FeatureType.MASK_TIMELESS, 'IS_VALID'),
                                   folder = "D:\\work\\inno\\rt\\test",
                                   band_indices=[0],
                                   no_data_value=0)
    export_task.execute(eopatch=patch,filename='mask_3.tif')



    exit(0)