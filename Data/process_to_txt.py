import os
import pyrootutils

# Root Path ..\VersionTorch
root = pyrootutils.setup_root(search_from=__file__,
                            indicator=["pyproject.toml"],
                            pythonpath=True,
                            dotenv=True)

'''
map NEU-CLS image directory to path file
NEU-CLS resource: http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.html
put NEU-CLS resource to file path Data/
'''
src_image_path = os.path.join(root, "Data", "NEU-CLS")
path_file_path = os.path.join(root, "Data")

for img_path in os.listdir(src_image_path):
    all_img_path = os.path.join(src_image_path, img_path)
    with open(os.path.join(path_file_path, 'neu_cls.txt'), 'a') as f:
        if '.db' not in img_path:
            f.write(' ' + img_path[0:2] + ' ' + all_img_path + '\n')
            pass
        pass

'''
map Mini-ImageNet image directory to path file
Mini-ImageNet resource: https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download
put archive resource to file path Data/ 
'''
label_pair = {}
src_image_path = os.path.join(root, "Data", "Mini-ImageNet", "archive")
path_file_path = os.path.join(root, "Data")
for idx, cls_path in enumerate(os.listdir(src_image_path)):
    cls_img_path = os.path.join(src_image_path, cls_path)
    for img_path in os.listdir(cls_img_path):
        path = os.path.join(cls_img_path, img_path)
        with open(os.path.join(path_file_path, 'mini-imagenet.txt'), 'a') as f:
            f.write(' ' + cls_path + ' ' + path + '\n')
            pass
        pass
    label_pair[cls_path] = idx

