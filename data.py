import torchvision.transforms as transforms



# Data Preprocessing in the First Model:

class data_first_model:

    def __init__(self) -> None:
        pass

# Image Cropping:

    def image_cropping(face_coordinates):
        return 0

# Image Normalization:

    def image_normalization():
        return 0

# Down-sampling:

    def down_sampling():
        return 0

# Data Augmentation:

    def image_rotation():
        return 0



def face_crop(image, face_coordinates):
    x, y = face_coordinates
    x1, x2 = x; y1, y2 = y
    return image[x1:x2, y1:y2]

torchvision_transform = transforms.Compose([
    
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

    transforms.functional.rotate(),
    transforms.ToTensor(),
])