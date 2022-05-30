import os
import sys
import argparse
import boto3
from mp_proc import s2_meta
import pathlib



class S3:
    @staticmethod
    def check_s3_path (s3_path) :
        if len(s3_path) == 0: return ''
        elif s3_path[-1] != '/': return s3_path + '/'
        else: return s3_path

    @staticmethod
    def get_file_data(s3_client, bucket_name, key, req_pays=False):

        response = None
        fun_kwargs = ({'Bucket':bucket_name, 'Key':key, 'RequestPayer':'requester'} if req_pays
                      else {'Bucket':bucket_name, 'Key':key})
        response = s3_client.get_object(**fun_kwargs)

        return response['Body'].read()


    @staticmethod
    def download_file(s3_client,bucket_name,key,output_path,req_pays=False):
        file_data = S3.get_file_data(s3_client,bucket_name,key,req_pays)
        f = open(output_path, 'wb')
        f.write(file_data)
        f.close()



    def listsubfolders(s3_client, bucket_name, remote_dir, req_pays=False):

        remote_dir = S3.check_s3_path(remote_dir)

        all_objects = None
        fun_kwargs = {'Bucket':bucket_name, 'Delimiter':'/'}
        if req_pays: fun_kwargs['RequestPayer'] = 'requester'
        if remote_dir != '' and remote_dir is not None: fun_kwargs['Prefix'] = remote_dir

        objects = s3_client.list_objects(**fun_kwargs)
        subfolders = list()
        for o in objects.get('CommonPrefixes'):
            subfolders.append(o.get('Prefix')[len(remote_dir):-1])

        return subfolders




    @staticmethod
    def download_dir_recursive(s3_client,bucket_name,remote_dir,local_dir,req_pays=False):
        if remote_dir is None or len(remote_dir) == 0: return  False
        else: remote_dir = S3.check_s3_path(remote_dir)


        fun_kwargs = ({'Bucket': bucket_name, 'Prefix' : remote_dir, 'RequestPayer' : 'requester'} if req_pays
                        else {'Bucket': bucket_name, 'Prefix' : remote_dir})

        all_objects = s3_client.list_objects(**fun_kwargs)
        if 'Contents' not in all_objects.keys(): return False
        remote_dir = os.path.dirname(remote_dir[:-1])
        for obj in all_objects['Contents']:
            filename = os.path.basename(obj['Key'])
            rel_path = os.path.dirname(obj['Key'])[len(remote_dir) + 1:]

            loc_path = os.path.join(local_dir,rel_path)
            if not os.path.exists(loc_path):
                pathlib.Path(loc_path).mkdir(parents=True, exist_ok=True)

            S3.download_file(s3_client,bucket_name,obj['Key'],os.path.join(loc_path,filename),req_pays)

        return True

class AWS:
    s3_client = None
    l2a_bucket = 'sentinel-s2-l2a'

    @staticmethod
    def init(aws_access_key_id, aws_secret_access_key) :
        AWS.s3_client = boto3.client('s3',
                                       aws_access_key_id=aws_access_key_id,
                                       aws_secret_access_key=aws_secret_access_key)

    @staticmethod
    def get_l2a_prod_dir (sceneid):
        return f'products/{s2_meta.SceneID.year(sceneid)}' \
               f'/{s2_meta.SceneID.month(sceneid,type=int)}' \
               f'/{s2_meta.SceneID.day(sceneid,type=int)}' \
               f'/{sceneid}'

    @staticmethod
    def get_l2a_tile_dir(sceneid):
        return f'tiles/{s2_meta.SceneID.tile_name(sceneid)[0:2]}' \
               f'/{s2_meta.SceneID.tile_name(sceneid)[2:3]}' \
               f'/{s2_meta.SceneID.tile_name(sceneid)[3:5]}' \
               f'/{s2_meta.SceneID.year(sceneid)}' \
               f'/{s2_meta.SceneID.month(sceneid,type=int)}' \
               f'/{s2_meta.SceneID.day(sceneid,type=int)}'

    @staticmethod
    def download_l2a_scene (sceneid, dest_folder):
        scene_path = os.path.join(dest_folder,sceneid)
        if not os.path.exists(scene_path):
            os.mkdir(scene_path)

    @staticmethod
    def download_bands(sceneid,bands,dest_folder):
        return True

    def download_metadata(sceneid,dest_folder):
        return True

    def aws2scihub (sceneid_loc_folder):
        return True

class GCS:
    s3_client = None
    l1c_bucket = 'gcp-public-data-sentinel-2'

    @staticmethod
    def init(google_access_key_id, google_access_key_secret) :
        GCS.s3_client = boto3.client('s3',
                                    region_name="auto",
                                    endpoint_url="https://storage.googleapis.com",
                                    aws_access_key_id=google_access_key_id,
                                    aws_secret_access_key=google_access_key_secret)

    def get_l1c_dir (sceneid):
        return  f'tiles/{s2_meta.SceneID.tile_name(sceneid)[0:2]}' \
                f'/{s2_meta.SceneID.tile_name(sceneid)[2:3]}' \
                f'/{s2_meta.SceneID.tile_name(sceneid)[3:5]}' \
                f'/{sceneid}.SAFE'

            # products/2021/5/15/S2A_MSIL2A_20210515T004641_N0300_R045_T56TNS_20210515T025256/
# Key: products/2021/5/15/S2A_MSIL2A_20210515T004641_N0300_R045_T56UNU_20210515T025256/metadata.xml

"""
    @staticmethod
    def download_dir_from_s3(bucket_name, remote_dir):
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(bucketName)
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key) # save to same path
"""




