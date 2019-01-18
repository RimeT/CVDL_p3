import fnmatch
import os
import argparse
import xml.etree.ElementTree


class XmlToLst(object):

    def __init__(self, out_dir, lst_name, custom_classes):
        self.out_dir = out_dir
        self.lst_name = lst_name
        self.custom_classes = custom_classes

    def start_parse(self, img_folder, label_folder):
        if os.path.isdir(label_folder):
            self.detection_to_lst(self.out_dir, img_folder, label_folder, self.lst_name)
        else:
            raise ValueError("please input a folder includes xml files")

    def format_lst_line(self, index, width, height, objs, fname):
        line = str(index) + "\t" + "4\t" + "6\t"
        line += width + "\t" + height + "\t"
        line += objs
        line += fname + "\n"
        return line

    def parse_obj(self, objs, nclasses, width, height):
        result = ''
        for obj in objs:
            name = obj.find('name').text
            if self.custom_classes:
                if name.lower() not in self.custom_classes:
                    continue
            if obj.find('difficult'):
                difficult = int(obj.find('difficult').text)
            else:
                difficult = 0
            if name not in nclasses:
                nclasses.append(name)
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / float(width)
            ymin = float(bndbox.find('ymin').text) / float(height)
            xmax = float(bndbox.find('xmax').text) / float(width)
            ymax = float(bndbox.find('ymax').text) / float(height)
            result += str.format(
                "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t" % (nclasses.index(name), xmin, ymin, xmax, ymax, difficult))

        return result

    def detection_to_lst(self, out_dir, img_folder, label_folder, lst_name):
        XML_SUFFIX = '.xml'
        LABEL_SUFFIX = '.lst'
        lst_file = os.path.join(out_dir, lst_name + LABEL_SUFFIX)
        fnum = 0
        for _, _, files in os.walk(label_folder):
            for _ in fnmatch.filter(files, "*" + XML_SUFFIX):
                fnum += 1
        index = 0
        nclasses = []
        with open(lst_file, 'w') as f:
            for base_dir, _, file_iter in os.walk(label_folder):
                for targ_file in fnmatch.filter(file_iter, "*" + XML_SUFFIX):
                    label_file = os.path.join(base_dir, targ_file)
                    tree = xml.etree.ElementTree.parse(label_file)
                    fname = os.path.join(tree.find('folder').text, tree.find('filename').text)
                    if not os.path.isfile(os.path.join(img_folder, fname)):
                        # process may not be continuous
                        continue
                    size = tree.find('size')
                    width = size.find('width').text
                    height = size.find('height').text
                    objs = tree.iter('object')
                    obj_dict = self.parse_obj(objs, nclasses, width, height)
                    if len(obj_dict) < 12:
                        continue
                    f.write(self.format_lst_line(index, width, height, obj_dict, fname))
                    print('Processed %d/%d' % (index, fnum))
                    index += 1
        # generate lymmass.txt
        label_file = lst_name + ".txt"
        with open(os.path.join(out_dir, label_file), 'w') as f:
            for num, item in enumerate(nclasses):
                line = item
                if num != len(nclasses) - 1:
                    line += '\n'
                f.write(line)


parser = argparse.ArgumentParser(description='XML to MXNET LST parser')
parser.add_argument('--out-dir', type=str,
                    help='output directory path')
parser.add_argument('--lst-name', type=str,
                    help='lst file basename')
parser.add_argument('--custom-classes', type=str, default=None,
                    help='classes to train. split with ,')
parser.add_argument('--image-folder', type=str,
                    help='images root directory path')
parser.add_argument('--xml-folder', type=str,
                    help='label xml root directory path')
opt = parser.parse_args()


def main():
    custom_classes = opt.custom_classes.split(',') if opt.custom_classes else opt.custom_classes
    xml2lst = XmlToLst(opt.out_dir, opt.lst_name, custom_classes)
    xml2lst.start_parse(opt.image_folder, opt.xml_folder)


if __name__ == "__main__":
    main()
