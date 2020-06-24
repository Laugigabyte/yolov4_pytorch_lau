import os.path as osp
from lxml import etree
from collections import defaultdict
from tqdm import tqdm


name_box_id = defaultdict(list)

VOC_CLASSES = (  # always index 0
   'aeroplane', 'bicycle', 'bird', 'boat',
   'bottle', 'bus', 'car', 'cat', 'chair',
   'cow', 'diningtable', 'dog', 'horse',
   'motorbike', 'person', 'pottedplant',
   'sheep', 'sofa', 'train', 'tvmonitor')

class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
# print(class_to_ind)
# annopath = osp.join('/Users/quan/VOC2007/sample_train/Annotations/' + '000005' + '.xml')
# target = etree.parse(annopath).getroot()
# print(type(target))
# for obj in target.iter('object'):
#     name = obj.find('name')
#     print(name.text.lower().strip())

"""hyper parameters"""
file_path = '/Users/quan/VOC2007/sample_test/Annotations/' 
images_dir_path = '/Users/quan/VOC2007/sample_test/JPEGImages'
output_path = '../data/val.txt'

"""load xml file"""
name_box_id = defaultdict(list)
id_name = dict()

sign = 'val'
if sign == 'train':
    idx_list = ['05', '07', '09', '12', '16']
else:
    idx_list = ['01', '02', '03', '04', '06', '08', '10']
for i in idx_list:
    idx = '0000' + i
    annopath = osp.join(file_path + idx + '.xml')
    target = etree.parse(annopath).getroot()
    target_name = target.find('filename').text
    target_name = osp.join(images_dir_path, target_name)
    res = []
    for obj in target.iter('object'):
        bbox = obj.find('bndbox')
        name = obj.find('name').text
        # print(name)
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = bbox.find(pt).text
            bndbox.append(cur_pt)
        label_idx = class_to_ind[name]
        res += [bndbox, label_idx]  # [[xmin, ymin, xmax, ymax], label_ind]
        # [[[xmin, ymin, xmax, ymax], label_ind]], ... ]
        name_box_id[target_name].append(res)

"""write to txt"""
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')