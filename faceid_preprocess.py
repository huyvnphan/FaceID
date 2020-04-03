import os, sys, shutil, requests, zipfile, io
import numpy as np
from tqdm import tqdm as pbar
from PIL import Image 
sys.path.insert(0, '/home/huy/Projects/FaceID')

data_dir = '/raid/data/pytorch_dataset/faceid/'

#Dataset link http://www.vap.aau.dk/rgb-d-face-database/

train_list=['http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(151751).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(153054).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(154211).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(160440).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(160931).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(161342).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(163349).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-16)(164248).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(141550).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(142154).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(142457).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-17)(143016).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(132824).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(133201).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(133846).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(134239).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(134757).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(140516).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(143345).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(144316).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(145150).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(145623).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(150303).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(150650).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(151337).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(151650).zip']

val_list=['http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(152717).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(153532).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(154129).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(154728).zip',
 'http://vap.aau.dk/wp-content/uploads/VAPRBGD/(2012-05-18)(155357).zip']

def get_data(links, directory):
    """
    Download and unzip data
    """
    for link in pbar(links):
        r = requests.get(link, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(directory)
        
get_data(train_list, 'train_raw')
get_data(val_list, 'val_raw')

def crop_center(img, cropx, cropy):
    y,x = img.shape[0], img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def process_data(raw_dir, processed_dir):
    """
    Stack RGB and D images to a numpy array of shape 400 x 400 x 4
    Save images in format of personID_poseID
    Print the mean and std of whole dataset
    """
    mean = np.array([0.0, 0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0, 0.0])
    counter = 0
    
    all_people = [x for x in os.listdir(raw_dir) if '2012' in x]
    for person_id, a_person in enumerate(pbar(all_people)):
        
        all_poses = [x for x in os.listdir(os.path.join(raw_dir, a_person)) if '.bmp' in x]
        for pose_id, a_pose in enumerate(all_poses):
            photo_id = a_pose[:-5] # drop 'c.bmp'

            photo_rgb = photo_id + 'c.bmp'
            photo_rgb = Image.open(os.path.join(raw_dir, a_person, photo_rgb))
            photo_rgb = np.array(photo_rgb.convert("RGB").resize((640, 480)))

            photo_depth = photo_id + 'd.dat'
            photo_depth = np.loadtxt(os.path.join(raw_dir, a_person, photo_depth))
            photo_depth = np.where((photo_depth > 400) & (photo_depth < 3000), photo_depth, 0) # Valid range is from 400 to 3000
            photo_depth = (photo_depth - 400 / (3000 - 400)) * 255.0
            photo_depth = np.expand_dims(photo_depth, -1)
            
            # Save to disk
            rgbd = np.concatenate((photo_rgb, photo_depth), axis=2)
            rgbd = np.uint8(rgbd)
            rgbd = crop_center(rgbd, 400, 400)
            name = 'person' + str(person_id) + '_pose' + str(pose_id) + '.npy'
            np.save(os.path.join(processed_dir, name), rgbd)
            
            # Calculate mean and std
            counter += 1
            for i in range(4):
                mean[i] += (rgbd[:,:,i]/255.0).mean()
                std[i] += (rgbd[:,:,i]/255.0).std()
    
    mean = mean / counter
    mean = [round(x, 4) for x in mean]
    std = std / counter
    std = [round(x, 4) for x in std]
    print("Mean: ", mean)
    print("STD: ", std)
    
os.mkdir('train')
process_data('train_raw', 'train')

os.mkdir('val')
process_data('val_raw', 'val')

# mean = [0.5255, 0.5095, 0.4861, 0.7114]
# std = [0.2075, 0.1959, 0.1678, 0.2599]

# Move data to desire place
os.mkdir(data_dir)
shutil.move('train_raw', data_dir)
shutil.move('val_raw', data_dir)
shutil.move('train', data_dir)
shutil.move('val', data_dir)