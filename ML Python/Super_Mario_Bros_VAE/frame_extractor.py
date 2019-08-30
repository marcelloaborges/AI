import cv2
import os
import glob

def extract_from_video():
    cam = cv2.VideoCapture("supermariobros.mp4") 
    
    try: 
        
        # creating a folder named data 
        if not os.path.exists('data/frame'): 
            os.makedirs('data/frame') 

        if not os.path.exists('data/frame_resized'): 
            os.makedirs('data/frame_resized') 
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
        
        # reading from frame 
        ret, frame = cam.read() 
    
        if ret:                 
            if currentframe % 10 != 0:
                currentframe += 1
                continue

            # if video is still left continue creating images 
            name = './data/frame/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created        
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

data_files = './data/frame'
resize_dim = ( 256, 240 )

def resize_imgs():

    for img in glob.glob(data_files+'/*.*'):        
        var_img = cv2.imread(img)

        resized_img = cv2.resize( var_img, resize_dim, interpolation = cv2.INTER_AREA )

        name = './data/frame_resized/' + str(img).split('\\')[-1]
        print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, resized_img)

    cv2.destroyAllWindows() 


# extract_from_video()
# resize_imgs()