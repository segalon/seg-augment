from os import listdir
from xml.etree import ElementTree
from mrcnn.utils import Dataset
import numpy as np

class KangarooDataset(Dataset):
    def load_kangaroos(self, dataset_dir, is_train=True):
            self.add_class("dataset", 1, "kangaroo")

            self.images_dir = dataset_dir + '/images/'
            self.annotations_dir = dataset_dir + '/annots/'

            for f in listdir(self.images_dir):
                    # get original id
                    image_id = f[:-4]
                    # skip bad image
                    if image_id in ['00090']:
                            continue
                    if is_train and int(image_id) >= 150:
                            continue
                    if not is_train and int(image_id) < 150:
                            continue

                    img_path = self.images_dir + f
                    ann_path = self.annotations_dir + image_id + '.xml'

                    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)


    def get_ids(self):
            f2id = dict()
            for i, info in enumerate(self.image_info):
                    path = self.image_reference(i)
                    orig_id = int(path[:-4].split('/')[-1])
                    f2id[orig_id] = i
            return f2id

        
    def extract_boxes(self, path):
            root = ElementTree.parse(path).getroot()
            boxes = []
            bndboxes = root.findall('.//bndbox')
            for box in bndboxes:
                    xmin = int(box.find('xmin').text)
                    ymin = int(box.find('ymin').text)
                    xmax = int(box.find('xmax').text)
                    ymax = int(box.find('ymax').text)
                    boxes.append([xmin, ymin, xmax, ymax])
            
            width = int(root.find('.//size/width').text)
            height = int(root.find('.//size/height').text)
            return boxes, width, height


    def image_reference(self, image_id):
            return self.image_info[image_id]['path']



