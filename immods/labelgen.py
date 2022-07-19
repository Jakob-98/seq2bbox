
import os

def generateLabel(label):
    return str(str(label) + " 0.5 0.5 1 1") 


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def createLabelsSingle(imageList, basedir, labeldirname, metadata_full):
    # For single objects only

    basedir = basedir.parent
    os.makedirs(str(basedir) + "/labels/" + labeldirname,exist_ok=True)

    ids = [i.get('image_id') for i in imageList]
    # generate lookup for bbox and category id based on image id


    # print("!WARNING: hardcoded fix for islands dataset")

    lookup = {}
    for meta in metadata_full:
        if meta["image_id"] not in ids: continue

        bb = [0, 0, 0, 1] 

        try:
            bb = meta['bbox']
        except KeyError:
            raise KeyError('Keyerror on boundingbox!')

        lookup[meta['image_id']] = {"bbox": bb, "category_id": meta["category_id"]}


    for im in imageList:

        ann = lookup.get(im['image_id'])

        dw = 1. / im['width']
        dh = 1. / im['height']
        
        
        filename = im['file_name'].replace(".jpg", ".txt").replace("/", "-")
        # print(Path(basedir).parent.__str__() + "/labels/" + labeldirname + filename, "a")
        with open(str(basedir) + "/labels/" + labeldirname + filename, "a") as myfile:
            xmin = ann["bbox"][0]
            ymin = ann["bbox"][1]
            xmax = ann["bbox"][2] + ann["bbox"][0]
            ymax = ann["bbox"][3] + ann["bbox"][1]
            
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2
            
            w = xmax - xmin
            h = ymax-ymin
            
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            
            mystring = str(str(ann['category_id']) + " " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
            myfile.write(mystring)
            myfile.write("\n")

        myfile.close()