import json
import os
import numpy as np
   
def gen_simple_labels(path):
    valjson_dir = os.path.join(path, 'bdd100k_labels_images_val.json')
    trainjson_dir = os.path.join(path, 'bdd100k_labels_images_train.json')
    simple_train_dir = os.path.join(path, 'simple_train_labels.json')
    simple_val_dir = os.path.join(path, 'simple_val_labels.json')
    
    if os.path.exists(simple_val_dir) or os.path.exists(simple_train_dir):
        print('Simplified labels files already exist!')
        return
    
    else:    
        with open(valjson_dir, 'r') as f_v:
            val_dict = json.load(f_v)
        with open(trainjson_dir, 'r') as f_t:
            train_dict = json.load(f_t)
        
        simple_train_dict = simplify_labels(train_dict)
        simple_val_dict = simplify_labels(val_dict)
        
        with open(simple_val_dir, 'w') as s_v:
            json.dump(simple_val_dict, s_v)
        with open(simple_train_dir, 'w') as s_t:
            json.dump(simple_train_dict, s_t)
            
        print('Simplified labels files created!')
        return

def simplify_labels(labels_dict):
    length = len(labels_dict)
    for i in range(length):
        if 'timestamp' in labels_dict[i].keys():
            labels_dict[i].pop('timestamp')
        
        labels = labels_dict[i]['labels']
        ptr = 0
        while True:
            if ptr == len(labels):
                break
            else:
                if labels[ptr]['category'] == 'drivable area' or labels[ptr]['category'] == 'lane':
                    labels.pop(ptr)
                else:
                    ptr+=1
    
    return labels_dict

def prepare_labels(label_path, classname_path):
    ''' 
    from original label files(json), generate a list of numpy array, 
    each array contains all labels corresponding to each img:
    
    [
     [[category, x1,y1,x2,y2],
      [category, x1,y1,x2,y2], ...](numpy array),  # 1st img
     [[category, x1,y1,x2,y2],
      [category, x1,y1,x2,y2], ...](numpy array),  # 2nd img
     ...
    ]
    
    '''
    try:
        with open(label_path, 'r') as f:
            ori_dict = json.load(f)
        with open(classname_path, 'r') as f_name:
            names = f_name.readlines()
    except:
        raise FileNotFoundError
        
    class_names = [n.strip() for n in names[:10]]
    
    names = []
    labels = []
    for i in range(len(ori_dict)):
        img_name = ori_dict[i]['name']
        img_labels = ori_dict[i]['labels']
        boxes = []
        for j in range(len(img_labels)):
            box = []
            x1 = img_labels[j]['box2d']['x1']
            y1 = img_labels[j]['box2d']['y1']
            x2 = img_labels[j]['box2d']['x2']
            y2 = img_labels[j]['box2d']['y2']
            xc = (x1+x2)/2 
            yc = (y1+y2)/2
            w = x2-x1
            h = y2-y1
            
            box.append(class_names.index(img_labels[j]['category']))
            box.append(xc)
            box.append(yc)
            box.append(w)
            box.append(h)
            boxes.append(box)
        
        names.append(img_name)
        labels.append(np.array(boxes))
        
    return names, labels
            
            
            
            
            
            
            
    
    