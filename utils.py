import cv2
import numpy as np
from sklearn.utils import shuffle

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def generator(samples, batch_size, is_training, angle_correct, trans_range):

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = np.array(samples)
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:

                if is_training & (np.random.rand() < 0.6):
                    image, angle = choose_image(batch_sample, angle_correct)
                    image, angle = augment(image,angle, trans_range)
                    image = preprocess(image)

                else:
                    name = 'E:/CarND-Behavioral-Cloning/CarND-Behavioral-Cloning-P3/data/IMG/'+batch_sample[0].split('/')[-1]
                    image2 = cv2.imread(name,1)
                    #image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                    image = preprocess(image2)
                    angle = float(batch_sample[3])

                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def brighten(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand()-0.5)
    hsv[:,:,2] = hsv[:,:,2]*ratio

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def preprocess(image):

    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)

    return image

def choose_image(sample, angle_correct):
    choice = np.random.choice(3)
    correction = angle_correct
    name = 'E:/CarND-Behavioral-Cloning/CarND-Behavioral-Cloning-P3/data/IMG/'+sample[choice].split('/')[-1]
    image = cv2.imread(name,1)
    #image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    if choice == 0:                 #center
        angle = float(sample[3])

    elif choice == 1:               #left
        angle = float(sample[3])+correction
        
    elif choice == 2:               #right
        angle = float(sample[3])-correction
        
    return image, angle

def augment(image, angle, trans_range):

    #if np.random.rand() > 0.5:
    image = brighten(image)
    #if np.random.rand() > 0.5:
    image = shadow(image)
    #if np.random.rand() > 0.5:
    image, angle = translate(image,angle, trans_range)
    if np.random.rand() > 0.5:
        image, angle = flip(image,angle)

    return image, angle

def shadow(image):
    x1 = 320*np.random.rand()
    y1 = 0
    x2 = 320*np.random.rand()
    y2 = image.shape[0]
    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    ym, xm = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1
    if np.random.randint(2)==1:
        random_bright = .25+.7*np.random.uniform()
        side1 = shadow_mask==1
        side0 = shadow_mask==0 
        if np.random.randint(2)==1:
            image_hls[:,:,1][side1] = image_hls[:,:,1][side1]*random_bright
        else:
            image_hls[:,:,1][side0] = image_hls[:,:,1][side0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)
    return image

def translate(image, angle, trans_range):
    rows,cols = image.shape[:2]
    trans_x = trans_range * (np.random.rand() - 0.5)
    trans_y = trans_range * (np.random.rand() - 0.5)
    angle += trans_x * 0.002
    M = np.float32([[1,0, trans_x],[0,1, trans_y]])
    image = cv2.warpAffine(image,M,(cols,rows))

    return image, angle

def flip (image, angle):
    image = cv2.flip(image, flipCode = 1)
    angle = -angle

    return image, angle


def crop (image):

    return image[60:135, :, :]

def resize(image):
    
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)

    return image

def rgb2yuv(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    return image

 
    
    
