import os
import pathlib
from datetime import datetime,timedelta
import subprocess, shlex
import time
import warnings
import piexif
from PIL import Image as pil_image
from pathlib import Path

# This script must be run from within the scripts folder, otherwise the relative paths do not work out anymore!

remove_corrupted_images       = True            # each image will checked if it loads correctly. Takes more time

# relative destination path
# Note that if the destination_path is changed also the symlink path must be adjusted below (*NOTE)
destination_path = 'JPG/'

os.makedirs(destination_path,exist_ok=True)


p = Path('timelapse_images')

start = time.time()
for dir in sorted(p.glob('*/')):
    dir_date = datetime.strptime(str(dir.name), '%Y-%m-%d')
    os.makedirs(destination_path + dir.name,exist_ok=True)

    last_timestamp = None # used to discard images that are too frequent (more than one image every 4 minutes)
    img_file_list = sorted(dir.glob('*'))        
    for i in range(len(img_file_list)):
        img_file = img_file_list[i]
        # print(file.stem)
        start_time_str = img_file.stem

        # The files' naming convention changed at some point
        # We are going to check which one the current file has and adjust it
        try:
            # try new convention first
            timestamp = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S')
        except ValueError:
            # try old naming convention
            try:
                timestamp = datetime.strptime(start_time_str, '%Y-%m-%d_%H%M%S')
            except ValueError:    
                warnings.warn('The following is not a valid image filename and will be ignored: %s'%img_file)
                continue

        # we want to discard some images from days when the camera setup was not properly and there are too many images
        # the desired frequency is 1/(4 min)
        if last_timestamp is not None:
            if timestamp - last_timestamp < timedelta(minutes=3.7):	# add some error margin 3.7 minutes instead of 4
               print('Discarded file',img_file)
               continue
        last_timestamp = timestamp

        dst = destination_path + dir.name + '/' + timestamp.strftime('%Y%m%d_%H%M%S') + '.JPG' # create filename with new convention

        # Incremental update: do nothing if the file already exists
        if os.path.isfile(str(dst)):
            continue

        # If the file is a .NEF file (Nikon's RAW format), we want to extract the JPG from the file
        # Otherwise we create a symlink to the original JPG
        if img_file.suffix in ['.JPG','.jpg']:
            # we need to adjust the relative path to where we the file will be placed
            # destination_path/dir.name/img_file.name
            # *NOTE: If destination_path is changed this must be adjusted here
            src = '../../' + str(img_file)
            # try:
            #     if remove_corrupted_images:
            #         img = pil_image.open(img_file)
            #         img.load()
            #     os.symlink(src,dst)
            # except OSError:
            #     warnings.warn('A file might be corrupted and will be ignored. An error occured opening the file %s'%img_file)

        # elif img_file.suffix in ['.NEF','.nef']:
            # Extract the JPG, by issuing the nconvert command
            src = str(img_file)
            resize_w, resize_h = 1424, 2144 
            cmd = shlex.split('./NConvert/nconvert -quiet -out jpeg -embedded_jpeg -resize %d %d -overwrite -o %s %s' % (resize_w,resize_h,dst,src))
            proc_util = subprocess.run(cmd, stdout=subprocess.PIPE)
            cmd = shlex.split('./NConvert/nconvert -quiet -out jpeg -jpegtrans rot90 -overwrite -o %s %s' % (dst,dst))
            proc_util = subprocess.run(cmd, stdout=subprocess.PIPE)
            exif_dict = piexif.load(dst)
            exif_dict["0th"][piexif.ImageIFD.Orientation] = 8
            piexif.insert(piexif.dump(exif_dict),dst)
        else:
            warnings.warn('The file has an unknown extension an will be ignored: %s'%img_file)

        # print(src,dst)

print('Finished creating the file structure. It took', time.time()-start, 'seconds')
