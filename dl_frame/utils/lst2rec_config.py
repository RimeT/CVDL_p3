import subprocess
import sys
import os

im2rec_path = "/home/tiansong/HDD/code/PycharmProjects/p3_mxml/dl_frame/utils/im2rec.py"
lst_path = "/home/tiansong/HDD/code/PycharmProjects/p3_mxml/mx_rec/dcm_lymmass_train.lst"
image_root = "/home/tiansong/HDD/infer_dataset/obj_det/breast/Xiehe2700/dcm"
resize = 512

if __name__ == "__main__":
    arguments = [sys.executable,
                 lst_path,
                 image_root,
                 '--dicom',
                 '--resize=%d' % resize,
                 '--pack-label',
                 '--num-thread=8']
    p = subprocess.Popen(arguments,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         close_fds=True,
                         env=os.environ.copy())
    try:
        while p.poll() is None:
            for line in p.stdout:
                if line is not None and len(line) > 1:
                    print(line)

    except Exception as e:
        print(e)
        if p.poll() is None:
            p.terminate()
