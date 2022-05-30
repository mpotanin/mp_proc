import os

BANDS = {
    'B01':60,'B02':10,'B03':10,'B04':10,'B05':20,'B06':20,
    'B07':20,'B08':10,'B8A':20,'B09':60,'B10':60,'B11':20,'B12':20,'SCL':20
    }

class SceneID:
    #S2A_MSIL1C_20150704T101337_N0202_R022_T33UUP_20160606T205155
    #S2B_MSIL2A_20191109T081049_N0213_R078_T37TGK_20191109T101247
    @staticmethod
    def date(sceneid):
        return sceneid[11:19]

    @staticmethod
    def datetime(sceneid):
        return sceneid[11:26]

    @staticmethod
    def day(sceneid, type='string'):
        return sceneid[17:19] if type=='string' else int(sceneid[17:19])

    @staticmethod
    def month(sceneid, type='string'):
        return sceneid[15:17] if type=='string' else int(sceneid[15:17])
    @staticmethod
    def year(sceneid, type='string'):
        return sceneid[11:15] if type=='string' else int(sceneid[11:15])

    @staticmethod
    def tile_name(sceneid):
        return sceneid[39:44]



"""
Class l2a_scene
    def get_band_file()

Class l1c_scene
    def get_band_file()

Class bands_bundle
"""


# 1) aws to scihub S2-L2 converter: subfolder structures and filenames
# 2) jp2 to tiff converter (using gdal_translate command line util)


# Converts S2-L2 scene folder (nested folders structure and filenames)
# from AWS format to Sci-HUB format

class L2AScene:

    @staticmethod
    def get_granule(scene_full_path):
        for el in os.listdir(os.path.join(scene_full_path,'GRANULE')):
            if el.startswith('L2A'):
                return el
        return None

    @staticmethod
    def aws2schihub(scene_full_path):
        scene = os.path.basename(scene_full_path)

        # metadata.xml -> MTD_MSIL2A.xml
        os.rename(os.path.join(scene_full_path, 'metadata.xml'), os.path.join(scene_full_path, 'MTD_MSIL2A.xml'))
        # productInfo.json isn't needed
        os.remove(os.path.join(scene_full_path, 'productInfo.json'))

        # it's rare but is possible that there are two version inside scene folder: "0" and "1"
        # if it is the case then one of them is removed and the rest is converted
        # by default there is only "0" version
        # if subfolder "1" exists we've to decide
        # which version of scene ("0" or "1") to delete.
        # For that purpose we extract 'name' from 0/productInfo.json
        # if extracted name equals scene then "0" is converted and "1" is deleted
        # other way vice versa
        sub_scene_path = os.path.join(scene_full_path, '0')
        sub_scene_to_delete = None
        if os.path.exists(os.path.join(scene_full_path, '1')):
            product_info_file = open(os.path.join(sub_scene_path, 'productInfo.json'))
            data = json.load(product_info_file)
            if (data['name'] == scene):
                sub_scene_to_delete = os.path.join(scene_full_path, '1')
            else:
                sub_scene_to_delete = sub_scene_path = os.path.join(scene_full_path, '0')
                sub_scene_path = os.path.join(scene_full_path, '1')
            product_info_file.close()
        # delete second version of scene
        if sub_scene_to_delete is not None:
            shutil.rmtree(os.path.join(scene_full_path, sub_scene_to_delete))
        # move all nested folders and files from
        # sub scene folder (0 or 1) one level up to scene folder
        for elem in os.listdir(sub_scene_path):
            shutil.move(os.path.join(sub_scene_path, elem), scene_full_path)
        shutil.rmtree(sub_scene_path)

        # extract from metadata file granule name,
        # creates sub folder GRANULE/{granule_name}
        # and moves all raster bands and other files inside it
        metadata_file = open(os.path.join(scene_full_path, 'MTD_MSIL2A.xml'), 'r')
        metadata = metadata_file.read()
        metadata_file.close()
        granule_name = ''
        m = re.search("GRANULE/([^/]*)/", metadata)
        if m:
            granule_name = m.group(1)

        granule_path = os.path.join(scene_full_path, 'GRANULE')
        os.mkdir(granule_path)
        granule_path = os.path.join(granule_path, granule_name)
        os.mkdir(granule_path)

        shutil.move(os.path.join(scene_full_path, 'metadata.xml'), os.path.join(granule_path, 'MTD_TL.xml'))

        granule_path = os.path.join(granule_path, 'IMG_DATA')
        os.mkdir(granule_path)

        shutil.move(os.path.join(scene_full_path, 'R10m'), granule_path)
        shutil.move(os.path.join(scene_full_path, 'R20m'), granule_path)
        shutil.move(os.path.join(scene_full_path, 'R60m'), granule_path)

        # renames band files for different resolutions
        prod_base = scene[38:44] + '_' + scene[11:26]
        for (dirpath, dirnames, filenames) in os.walk(granule_path):
            for f in filenames:
                if (f.endswith('.jp2')):
                    os.rename(os.path.join(dirpath, f),
                              os.path.join(dirpath, prod_base + '_' + f.replace('.jp2', '_' + dirpath[-3:] + '.jp2')))

        return True

    @staticmethod
    def get_band_file(scene_full_path, band_name, extension='jp2'):

        if band_name not in BANDS.keys():
            return None

        scene = os.path.basename(scene_full_path)
        granule = L2AScene.get_granule(scene_full_path)
        return os.path.join(scene_full_path,
                    f'GRANULE/{granule}/IMG_DATA/R{BANDS[band_name]}m/' \
                    f'T{SceneID.tile_name(scene)}_{SceneID.datetime(scene)}_{band_name}_{BANDS[band_name]}m.{extension}'
                    )
        return band_file

class L1CScene:
    @staticmethod
#gs://gcp-public-data-sentinel-2/tiles/33/U/UP/S2A_MSIL1C_20150704T101337_N0204_R022_T33UUP_20160809T050727.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_EPA__20160809T015434_A000162_T33UUP_N02.04/IMG_DATA/S2A_OPER_MSI_L1C_TL_EPA__20160809T015434_A000162_T33UUP_B02.jp2

    def get_band_file(scene_full_path, band_name):
        return True