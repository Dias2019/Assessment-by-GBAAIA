import os
import errno
from PIL import Image
import torchvision.transforms as transforms



# Data Augmentation in the First Model:
# These class is run only once to augment and save rotated images into dataset folder

class Data_Aug_first_model:

    def __init__(self, input_file):

        # let's assume that input_file is something like this: "C:/Users/Admin/Facial_emotions_dataset/JAFFE/"
        self.file_location = input_file


    def image_rotation(self):

        # for different dataset we apply different types of rotational augmentation

        # JAFFE dataset
        # for example with JAFFE, we rotate within 5 degrees at a step of 0.1 degrees
        # therefore, we ended up acquiring 50 times more images
        if str(self.file_location).split("/")[-1] == "JAFFE":

            for filename in os.listdir(self.file_location):
                if filename.endswith(".jpg"): 
                    path = os.path.join(self.file_location, filename)
                    original_image = Image.open(path)
                    for angle in range(0, 5, 0.1):

                        # this is a PIL function for rotation which accepts angle (in degrees) as input
                        # it uses inverse mapping or reverse transformation via transformation matrix for affine transformation
                        rotated_image = original_image.rotate(angle)
                        rotated_image_name = str(filename).split(".")[0] + "_" + str(angle) + ".jpg"
                        rotated_image.save(rotated_image_name)


        # CK+ dataset:
        # with CK+ dataset, we rotate within 10 degrees at a step of 1 degrees
        # because we need to increase our dataset 10 times
        elif str(self.file_location).split("/")[-1] == "CK+":
            for filename in os.listdir(self.file_location):
                if filename.endswith(".jpg"): 
                    path = os.path.join(self.file_location, filename)
                    original_image = Image.open(path)
                    for angle in range(0, 10, 1):
                        rotated_image = original_image.rotate(angle)
                        rotated_image_name = str(path).split(".")[0] + "_" + str(angle) + ".jpg"
                        rotated_image.save(rotated_image_name)

        
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.file_location)




# Apart from rotational data augmentation, we should apply several data preprocessings on every input image
# This can be achieved via PyTorch's Compose function which automatically applies all indicated data preprocessing 
# on the input image before putting into a model


# our personally defined function for cropping:
def face_crop(image, face_coordinates):
    x, y = face_coordinates
    x1, x2 = x; y1, y2 = y
    return image[x1:x2, y1:y2]

# Data processing of the first model consist from Image Cropping, Image Normalization, Down-Sampling
Data_Processing_first_model = transforms.Compose([
    
    # built-in Lambda function helps to utilize and compose functions defined by ourselves
    transforms.Lambda(face_crop(img=img, face_coordinates=coordinates)),

    # normalizes a picture via given mean and std
    # these mean and std were defined from ImageNet dataset and considered as approximate mean and std of all images
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

    # the first cnn model accepts only 48x48 image size
    # so, all input images must be resized
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])