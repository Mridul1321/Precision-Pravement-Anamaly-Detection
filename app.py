import random
# import serial
import time
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image

my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO(r"./output_model/best.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480


def detect(frame):
    # ret, frame = cap.read()
    # if frame is read correctly ret is True

    

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.55, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            frame,
            class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )
            

    
    #s.write(v.encode())
    #print(s.readline().decode('ascii'))
    # Display the resulting frame
    
    #s.write(v.encode())
    #print(s.readline().decode('ascii'))
    # Display the resulting frame
    
    # Terminate run when "Q" pressed
    st.image(frame, caption="Image with Predicted Boxes", use_column_width=True)
# When everything done, release the capture
def main():
    st.title("Rode defect detection")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read uploaded image as PIL image
        
        image = Image.open(uploaded_image)
    
    # Step 3: Save the Image
        image.save("saved_image.jpg")  
        cap = cv2.imread(r"saved_image.jpg")
        detect(cap)
if __name__ == '__main__':
    main()