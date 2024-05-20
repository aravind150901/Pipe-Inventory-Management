import cv2
#pip install Augmentor
import Augmentor as aug
import os

def normalize_and_resize_image(image_path, target_size):
    # Read the image using OpenCV
    image = cv2.imread(r"D:\360DigiTMG\Project\Annotated dataset\train\images\IMG_3785_MOV-8_jpg.rf.4799a85cd1424c5521570d8f2368aab6.jpg")
    
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    
    return resized_image


# Example usage:
image_path = cv2.imread(r'D:\360DigiTMG\Project\Annotated dataset\train\images\IMG_3785_MOV-8_jpg.rf.4799a85cd1424c5521570d8f2368aab6.jpg')
target_size = (640, 640)  # Specify the desired target size

resized_normalized_image = normalize_and_resize_image(image_path, target_size)

if resized_normalized_image is not None:
    # Save the normalized image
    normalized_image = cv2.imwrite(r'D:\360DigiTMG\Project\Annotated dataset\preprocessed.jpg', resized_normalized_image) 


p = aug.Pipeline(r"D:\360DigiTMG\Project\Annotated dataset\train")

# Add the normalized image to the pipeline
p.ground_truth(r"D:\360DigiTMG\Project\Annotated dataset\preprocessed.jpg")

p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5) # Rotates the images left and right 
p.zoom_random(probability=0.5, percentage_area=0.8) # zooms randomly with given probability
p.flip_top_bottom(probability=0.5) # flips images upside down
p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2) # adds Contrast
p.shear_range=0.2  # Shear transformation causes a kind of 'slanting' effect by shifting one part of the image along an axis.

p.sample(10)
