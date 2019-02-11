import argparse
import fnmatch
import os

import pydicom

test_image = "/home/tiansong/HDD/infer_dataset/obj_det/wuhan_50/dcm/1216052767/1216052767_004.dcm"

SENSITIVE_TAGS = [
    'InstitutionName',
    'Institution Address',
    'InstitutionalDepartmentName',
    'PatientName',
    'PatientAddress'
]


def masking(dcm_path, sense_tags):
    image = pydicom.dcmread(dcm_path)
    image.remove_private_tags()
    for i in sense_tags:
        if image.__contains__(i):
            image.__delattr__(i)
    return image


def dcm_copy(in_dir, out_dir, sense_tags):
    DCM_SUFFIX = '.dcm'
    for base_dir, _, file_iter in os.walk(in_dir):
        for targ_file in fnmatch.filter(file_iter, "*" + DCM_SUFFIX):
            dcm_path = os.path.join(base_dir, targ_file)
            image = masking(dcm_path, sense_tags)
            # copy
            temp_dir = os.path.join(out_dir, os.path.split(base_dir)[-1])
            if not os.path.isdir(temp_dir):
                os.mkdir(temp_dir)
            # write image
            image.save_as(os.path.join(temp_dir, targ_file))
            print(targ_file)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='dicom images masking')
parser.add_argument('in_dir', help='path to folder containing dicoms.')
parser.add_argument('out_dir', help='path to generate masked dicoms.')
opt = parser.parse_args()


def main():
    dcm_copy(opt.in_dir, opt.out_dir, SENSITIVE_TAGS)


if __name__ == "__main__":
    main()
