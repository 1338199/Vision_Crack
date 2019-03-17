import numpy as np
import tensorflow as tf
import os as os
from dataset import cache
from Train_CD import Model
import cv2,sys
import argparse
from pathlib import Path


def break_image(test_image, size):
    
    h,w= np.shape(test_image)[0],np.shape(test_image)[1]
    broken_image = []
    h_no = h//size
    w_no = w//size
    h=h_no*size
    w=w_no*size
    for i in range(0,h_no):
        for j in range(0,w_no):
            split = test_image[size*i:size*(i+1),size*j:size*(j+1),:]
            broken_image.append(split); 
            
    return broken_image,h,w,h_no,w_no

class Dataset_test:
    def __init__(self, in_dir, exts='.jpg'):
        # Extend the input directory to the full path.
        in_dir = os.path.abspath(in_dir)

        # Input directory.
        self.in_dir = in_dir
        
        model=Model(in_dir)
        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Filenames for all the files in the test-set
        self.filenames = []

        # Class-number for each file in the test-set.
        self.class_numbers_test = []

        # Total number of classes in the data-set.
        self.num_classes = model.num_classes
        
        # If it is a directory.
        if os.path.isdir(in_dir):
          
            # Get all the valid filenames in the dir
            self.filenames = self._get_filenames_and_paths(in_dir)
         
        else:
            print("Invalid Directory")
        self.images = self.load_images(self.filenames)
        
    def _get_filenames_and_paths(self, dir):
        """
        Create and return a list of filenames with matching extensions in the given directory.
        :param dir:
            Directory to scan for files. Sub-dirs are not scanned.
        :return:
            List of filenames. Only filenames. Does not include the directory.
        """

        # Initialize empty list.
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    path = os.path.join(self.in_dir, filename)
                    filenames.append(os.path.abspath(path))

        return filenames


    def load_images(self,image_paths):
        # Load the images from disk.
        images = [cv2.imread(path) for path in image_paths]
    
        # Convert to a numpy array and returns it in the form of [num_images,size,size,channel]
        return np.asarray(images)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing Network')
    parser.add_argument('--in_dir',dest='in_dir',type=str,default='cracky_test')
    parser.add_argument('--meta_file',dest='meta_file',type=str,default='model.meta')
    parser.add_argument('--CP_dir',dest='chk_point_dir',type=str,default=os.getcwd())
    parser.add_argument('--save_dir',type=str,default='save')
    return parser.parse_args()

def main(args):



    #File names are saved into a cache file
    args=parse_arguments()
    dataset_test = cache(cache_path='my_dataset_cache_test.pkl',
                    fn=Dataset_test, 
                    in_dir=args.in_dir)
    test_images = dataset_test.images
    nums = len(list(enumerate(test_images)))
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            #import the model dir
            try:
                file_=Path(args.meta_file)
                abs_path=file_.resolve()
            except FileNotFoundError:
                sys.exit('Meta File Not found')
            else:
                imported_meta = tf.train.import_meta_graph(args.meta_file)
                       
            if os.path.isdir(args.chk_point_dir):
                imported_meta.restore(sess, tf.train.latest_checkpoint(args.chk_point_dir))
            else:
                sys.exit("Check Point Directory does not exist")
            
            x = graph.get_operation_by_name("x").outputs[0]
            predictions = graph.get_operation_by_name("predictions").outputs[0]
            
            leng = []
            leng.append(nums)
            #Take one image at a time, pass it through the network and save it
            for counter,image in enumerate(test_images):
                cnt = 0
                broken_image,h,w,h_no,w_no = break_image(image,128)
        
                output_image = np.zeros((h_no*128,w_no*128,3),dtype = np.uint8)
                                            
                feed_dict = {x: broken_image}
                batch_predictions = sess.run(predictions, feed_dict = feed_dict)
            
                matrix_pred = batch_predictions.reshape((h_no,w_no))
                #Concentrate after this for post processing
                for i in range(0,h_no):
                    for j in range(0,w_no):
                        a = matrix_pred[i,j]
                        output_image[128*i:128*(i+1),128*j:128*(j+1),:] = 1-a
                        if 1 - a == 1:
                            cnt += 1

                cropped_image = image[0:h_no*128,0:w_no*128,:]                    
                pred_image = np.multiply(output_image,cropped_image)

                print("Saved {} Image(s)".format(counter+1))
                leng.append(cnt * 128 * 1.414 )
                cv2.imwrite(os.path.join(args.save_dir,'outfile_{}.jpg'.format(counter+1)), pred_image)

            index = 0
            for img in os.listdir("./Save"):  # read all the images in dir "Save"
                # if not hidden folder
                if not img.startswith('.'):
                    index += 1
                    # read the image img1
                    img1 = cv2.imread("./Save" + "/" + img, cv2.IMREAD_GRAYSCALE)
                    # get the shape
                    size = img1.shape
                    lengthcnt = 0
                    widthcnt = 0
                    length = [0] * size[1]
                    width = [0] * size[0]
                    for i in range(size[0]):
                        for j in range(size[1]):
                            # if this pix is in the crack-zone
                            if img1[i][j] > 0 and length[j] == 0:
                                length[j] = 1
                                lengthcnt += 1
                            if img1[i][j] > 0 and width[i] == 0:
                                width[i] = 1
                                widthcnt += 1

                    # get the length of the crack
                    len_vec = leng
                    totLen = len_vec[len_vec[0] - index + 1]

                    print(img + "中裂缝的行投影为:%d像素，占图片长度的%.2f%%，裂缝的列投影为:%d像素，占图片高度的%.2f%%。裂缝的总长约为:%.3f像素"
                          % (lengthcnt, lengthcnt * 100.0 / size[1], widthcnt, widthcnt * 100.0 / size[0], totLen))


                                
if __name__ == '__main__':
    global leng
    main(sys.argv)
    
