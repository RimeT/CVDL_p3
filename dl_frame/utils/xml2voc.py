import argparse
import fnmatch
import os
import xml.etree.ElementTree as ET


def xml2voc(in_dir, out_dir, dcm_width, dcm_height):
    XML_SUFFIX = '.xml'
    for base_dir, _, file_iter in os.walk(in_dir):
        for targ_file in fnmatch.filter(file_iter, "*" + XML_SUFFIX):
            label_file = os.path.join(base_dir, targ_file)
            tree = ET.parse(label_file)
            tree_root = tree.getroot()
            tree_root.find('folder').text = ''
            # xml_fn = tree_root.find('filename').text
            # series, snumber = xml_fn.split("_")
            # tree_root.find('filename').text = series
            # slice_num = ET.SubElement(tree_root, 'slice_num')
            # slice_num.text = snumber
            size = tree_root.find('size')
            size.find('width').text = str(dcm_width)
            size.find('height').text = str(dcm_height)
            size.find('depth').text = '1'
            # size = ET.SubElement(tree_root, 'size')
            # width = ET.SubElement(size, 'width')
            # width.text = str(dcm_width)
            # height = ET.SubElement(size, 'height')
            # height.text = str(dcm_height)
            # depth = ET.SubElement(size, 'depth')
            # depth.text = '1'
            # copy
            temp_dir = os.path.join(out_dir, os.path.split(base_dir)[-1])
            if not os.path.isdir(temp_dir):
                os.mkdir(temp_dir)
            tree.write(open(os.path.join(temp_dir, targ_file), 'w'))
            print(targ_file)


parser = argparse.ArgumentParser(description='XML to MXNET LST parser')
parser.add_argument('indir',
                    help='Annotation directory')
parser.add_argument('outdir',
                    help='Output directory path')
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
# parser.add_argument('--multi-slices', action='store_true',
#                     help='Is filename include slice number?')
opt = parser.parse_args()


def main():
    xml2voc(opt.indir, opt.outdir, opt.width, opt.height)


if __name__ == "__main__":
    main()
