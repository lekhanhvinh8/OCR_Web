#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from itertools import groupby
from imutils.object_detection import non_max_suppression


# In[2]:


class CTCLayer(layers.Layer):

    def __init__(self, name=None):

        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


# In[79]:


class OCRService():
    def __init__(self):
        self.char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.model_detection = self.load_detection_model()
        self.model_recognition = self.load_recognition_model()

    def load_detection_model(self):
        model_detection = cv2.dnn.readNet(r'services/frozen_east_text_detection.pb')
        return model_detection

    def load_recognition_model(self):
        model_recognition = self.create_crnn()

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
        model_recognition.compile(optimizer = optimizer)

        model_recognition.load_weights(r'services/C_LSTM_best_3.hdf5')

        # Get the prediction model by extracting layers till the output layer
        prediction_model_recognition = keras.models.Model(
            model_recognition.input[0], model_recognition.get_layer(name="dense").output #model.input[0] corresponses model.get_layer(name="image").input
        )

        return prediction_model_recognition

    def create_crnn(self):

        # input with shape of height=32 and width=128
        inputs = Input(shape=(32, 128, 1), name="image")

        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        conv_1 = Conv2D(32, (3,3), activation = "selu", padding='same')(inputs)
        pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)

        conv_2 = Conv2D(64, (3,3), activation = "selu", padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)

        conv_3 = Conv2D(128, (3,3), activation = "selu", padding='same')(pool_2)
        conv_4 = Conv2D(128, (3,3), activation = "selu", padding='same')(conv_3)

        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(256, (3,3), activation = "selu", padding='same')(pool_4)

        # Batch normalization layer
        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(256, (3,3), activation = "selu", padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(64, (2,2), activation = "selu")(pool_6)

        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

        # bidirectional LSTM layers with units=128
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True))(blstm_1)

        softmax_output = Dense(len(self.char_list) + 1, activation = 'softmax', name="dense")(blstm_2)

        output = CTCLayer(name="ctc_loss")(labels, softmax_output) #y_true = labels, y_pred = softmax_output

        #model to be used at training time
        model = Model(inputs=[inputs, labels], outputs=output)

        return model

    def ctc_decoder(self, predictions):
        '''
        input: given batch of predictions from text rec model
        output: return lists of raw extracted text

        '''
        text_list = []

        pred_indcies = np.argmax(predictions, axis=2)

        for i in range(pred_indcies.shape[0]):
            ans = ""

            ## merge repeats
            merged_list = [k for k,_ in groupby(pred_indcies[i])]

            ## remove blanks
            for p in merged_list:
                if p != len(self.char_list): # len(char_list) = 62, which is a number that be pass as a padding of labels
                    ans += self.char_list[int(p)]

            text_list.append(ans)

        return text_list

    def preprocessing_before_detect(self, img):
        # use multiple of 32 to set the new img shape
        height, width, _ = img.shape

        new_height = (height//32)*32
        new_width = (width//32)*32

        # get the ratio change in width and height
        h_ratio = height/new_height
        w_ratio = width/new_width

        blob = cv2.dnn.blobFromImage(img, 1, (new_width, new_height),(123.68, 116.78, 103.94), True, False)

        return blob, h_ratio, w_ratio

    def crop_images(self, image, fin_boxes):
        crop_imgs = []

        for boundingbox in fin_boxes:

            x1 = boundingbox[0]
            y1 = boundingbox[1]
            x2 = boundingbox[2]
            y2 = boundingbox[3]

            crop_img = image[y1:y2, x1:x2]

            crop_imgs.append(crop_img)

        return crop_imgs

    def preprocess_before_recognit(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 32))
        image = image / 255
        image = np.expand_dims(image, axis=0)
        image = image.reshape([32, 128, 1])
        image = image[np.newaxis]

        return image

    def recognit(self, img):
        preds = self.model_recognition.predict(img)
        pred_texts = self.ctc_decoder(preds)

        return pred_texts

    def detect(self, img):
        blob, h_ratio, w_ratio = self.preprocessing_before_detect(img)

        # ## Pass the image to network and extract score and geometry map
        self.model_detection.setInput(blob)

        self.model_detection.getUnconnectedOutLayersNames()

        (geometry, scores) = self.model_detection.forward(self.model_detection.getUnconnectedOutLayersNames())

        # ## Post-Processing
        rectangles = []
        confidence_score = []
        for i in range(geometry.shape[2]):
            for j in range(0, geometry.shape[3]):

                if scores[0][0][i][j] < 0.1:
                    continue

                bottom_x = int(j*4 + geometry[0][1][i][j])
                bottom_y = int(i*4 + geometry[0][2][i][j])


                top_x = int(j*4 - geometry[0][3][i][j])
                top_y = int(i*4 - geometry[0][0][i][j])

                rectangles.append((top_x, top_y, bottom_x, bottom_y))
                confidence_score.append(float(scores[0][0][i][j]))

        # use Non-max suppression to get the required rectangles
        fin_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.5)

        fin_boxes = [[int(x1*w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)] for x1, y1, x2, y2 in fin_boxes]

        return fin_boxes

    def recognit_many(self, image, fin_boxes):
        final_texts = []

        image_copy = image.copy()

        for crop_img in self.crop_images(image_copy, fin_boxes):
            crop_img_pre = self.preprocess_before_recognit(crop_img)

            result = self.recognit(crop_img_pre)

            final_texts.append(result)

        return final_texts

    def restructure(self, img, fin_boxes, final_texts):

        if(len(fin_boxes) == 0 or len(fin_boxes) != len(final_texts)):
            return []

        height, width, _ = img.shape

        min_width = fin_boxes[0][2] - fin_boxes[0][0]
        min_height = fin_boxes[0][3] - fin_boxes[0][1]

        for box in fin_boxes:
            if box[2] - box[0] < min_width:
                min_width = box[2] - box[0]
            if box[3] - box[1] < min_height:
                min_height = box[3] - box[1]

        new_width = int(width / min_width) + 1
        new_height = int(height / min_height) + 1
        restructure_box = [[' ' for x in range(new_width)] for y in range(new_height)]

        for i in range(0, len(fin_boxes)):
            x = int(fin_boxes[i][0] / min_width)
            y = int(fin_boxes[i][1] / min_height)
            text = final_texts[i][0]

            restructure_box[y][x] = text

        #merge k neighboring lines
        i = 0
        max_neighbors = 1

        while i < new_height - max_neighbors:
            #print(i, "----------------------------")
            #print('.'.join(restructure_box[i]))
            if set(''.join(restructure_box[i])) == {' '}:
                i += 1
                continue

            can_merge = True
            for j in range(new_width):
                count_not_space = 0
                for k in range(0, max_neighbors + 1):
                     if restructure_box[i+k][j] != ' ':
                        count_not_space += 1
                if count_not_space > 1:
                    can_merge = False
                    break

            if can_merge:
                for j in range(new_width):
                    if restructure_box[i][j] == ' ':
                        for k in range(1, max_neighbors + 1):
                            if restructure_box[i+k][j] != ' ':
                                #print(k, restructure_box[i+k][j])
                                restructure_box[i][j] = restructure_box[i+k][j]
                                break


                for k in range(1, max_neighbors + 1):
                    #print(i + 1, ".".join(restructure_box[i + 1]))
                    del restructure_box[i + 1]

                new_height -= max_neighbors


            i += 1

        return restructure_box
