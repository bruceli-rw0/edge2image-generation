"""
The following are adapted from the blog "Holistically-Nested Edge Detection with 
OpenCV and Deep Learning" by Adrian Rosebrock

Link: https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
"""

import cv2

class CropLayer:
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0
    
    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        
        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        
        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]
    
    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]

class HEDModel:

    def __init__(self):
        cv2.dnn_registerLayer("Crop", CropLayer)
        self.net = cv2.dnn.readNetFromCaffe(
            prototxt='image_processing/deploy.prototxt',
            caffeModel='image_processing/hed_pretrained_bsds.caffemodel'
        )

    def compute_hed(self, image):
        H, W, _ = image.shape

        # preprocess input
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=1.0, 
            size=(W, H),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, 
            crop=False
        )

        self.net.setInput(blob)
        hed = self.net.forward()
        return hed
