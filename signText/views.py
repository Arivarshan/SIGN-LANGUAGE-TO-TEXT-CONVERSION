from django.shortcuts import render

from rest_framework.decorators import api_view,permission_classes
from rest_framework.response import Response
from rest_framework import status

from collections import Counter
from time import sleep
import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image
import base64

from django.core.files.base import ContentFile

model = load_model("C:/Users/ELCOT/Django/FinelYearProject/signText/model.h5")  # load pre-trained model
#model = load_model("/api/signText/model.h5")
model.make_predict_function()


def prediction_to_char(pred):
    """
    Convert the prediction to ASCII character
    Parameters
    ----------
    pred: integer
        an integer valued between 0 to 25 that indicates the prediction
    Returns
    -------
    char
        an ASCII character
    """

    return chr(pred + 65)


def predict(model, image):
    """
    predicts the character from input image
    Parameters
    ----------
    model: a keras model instance
        pre-trained CNN model saved in HDF5 format
    image: OpenCV Image
        image of hand captured from camera
    Returns
    -------
    float
        probability of the predicted output
    integer
        an integer valued between 0 to 25 that represents the prediction
    """

    data = np.asarray(image, dtype="int32")
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def crop_image(image, start_x, start_y, width, height):
    """
    Crops an image
    Parameters
    ----------
    image: image
        the image that has to be cropped
    start_x: integer
        x co-ordinate of the starting point for cropping
    start_y: integer
        y co-ordinate of the starting point for cropping
    width: integer
        expected width of the cropped image
    height: integer
        expected height of the cropped image
    
    Returns
    -------
    """

    return image[start_y : start_y + height, start_x : start_x + width]




@api_view(['POST'])
def signToText(request):


    validated = False
    #img = _grab_image(stream=request.FILES['image'])
    image = request.data['image']
    format, imgstr = image.split(';base64,') 
    ext = format.split('/')[-1]  

    data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
    print(data)
    img = _grab_image(stream=data)

    # image_frame = cv2.imread('E:/arivu sign language/one.png',0)
    print(img)

    hand_image = crop_image(img, 20, 150, 300, 300)
    image_grayscale = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15, 15), 0)
    hand_image = cv2.resize(
        image_grayscale_blurred, (28, 28), interpolation=cv2.INTER_AREA
    )

    hand_image = np.resize(hand_image, (28, 28, 1))
    hand_image = np.expand_dims(hand_image, axis=0)

    pred_probab, pred_class = predict(model, hand_image)
    # print("prob " + pred_probab)
    print("class " + str(pred_class))
    # cv2.rectangle(image_frame, (20, 150), (320, 450), (255, 255, 00), 2)

    pred = ""

    if pred_probab >= 0.560:
        pred = prediction_to_char(pred_class)
        print(pred)
    else:
        pred="  Please try after few minutes "


    cv2.destroyAllWindows()
    return Response({"value": pred})


def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image
