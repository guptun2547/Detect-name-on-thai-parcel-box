import tkinter  as  tk
from PIL import ImageTk, Image
import cv2
import numpy as np
import os
from tesserocr import PyTessBaseAPI, PSM, OEM
import imutils
import re
import pandas as pd
from thefuzz import fuzz, process
from tkinter import filedialog



# Create an instance of tkinter window
win = tk.Tk()
win.title = "DetectParcelBox"
win.geometry("400x300")


def open_file():
    
    file_path = tk.filedialog.askopenfilename()
    
    file_name = os.path.basename(file_path)
    string_var.set(file_name)

    image = Image.open(file_path)
    image.thumbnail((200,200))

    image = ImageTk.PhotoImage(image)
    
    label_image.configure(image=image)
    label_image.image = image
    
    ocr(file_path)

def rotate_img():
    with PyTessBaseAPI(lang="osd", psm=PSM.OSD_ONLY, oem=OEM.TESSERACT_LSTM_COMBINED) as api:
        api.SetImageFile('out.jpg')
        os = api.DetectOrientationScript()
        # print(type(os))
        if isinstance(os, dict):
            img = cv2.imread('out.jpg')
            rotated_img = imutils.rotate_bound(img, 360 - os['orient_deg'])
            cv2.imwrite('out.jpg', rotated_img)

def getText():
    with PyTessBaseAPI(lang="tha") as api:
        api.SetVariable('preserve_interword_spaces', '1')

        api.SetImageFile("out.jpg")
        return str(api.GetUTF8Text())


def ocr(file_path):
    img = cv2.imread(file_path)
    # scale_percent = 0.8
    # width = int(img.shape[1] * scale_percent)
    # height = int(img.shape[0] * scale_percent)
    # img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x,y,w,h = cv2.boundingRect(contours[0])
    out = gray_img[y:y+h, x:x+w]
    eroded_img = cv2.erode(out, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)
    cv2.imwrite('out.jpg', dilated_img)

    # NLP
    rotate_img()
    text_encoded = getText()
    chk = False
    confidence = 0

    for t in text_encoded.split('\n'):
        if chk: break
        t = t.replace(" ", "")
        for r in receiver_list:
            if chk: break
            if r in t:
                char_to_remove = re.findall(pattern, t)
                list_with_char_removed = [char for char in t if not char in char_to_remove]
                result_string = ''.join(list_with_char_removed)
                r_matching = re.search(r, result_string)

                p = result_string[r_matching.end():]
                best_matching = process.extract(p, name, limit=1, scorer=fuzz.partial_ratio)
                confidence = best_matching[0][1]

                chk=True
                # print(p)
                # print(best_matching) 
                log.write("[Result] {}\n".format(result_string))
    log.write("[Text Encoded] {}".format(text_encoded))

    # print(chk)
    if confidence < 70:
        cv2.imwrite('out.jpg', out)
        rotate_img()
        text_encoded = getText()
        chk = False

        for t in text_encoded.split('\n'):
            if chk: break
            t = t.replace(" ", "")
            for r in receiver_list:
                if chk: break
                if r in t:
                    char_to_remove = re.findall(pattern, t)
                    list_with_char_removed = [char for char in t if not char in char_to_remove]
                    result_string = ''.join(list_with_char_removed)
                    r_matching = re.search(r, result_string)

                    p = result_string[r_matching.end():]
                    best_matching = process.extract(p, name, limit=1, scorer=fuzz.partial_ratio)
                    # print(p)
                    # print(best_matching)
                    log.write("[Result] {}\n".format(result_string))
                    chk=True
        log.write("[Text Encoded] {}".format(text_encoded))
        
        if not chk:
            # print("None")
            log.write("[Result] None \n")
            result.write("None\n")

    # Check confidence of matching
    confidence = best_matching[0][1]
    if confidence >= 70:
        result.write("{}\n".format(best_matching[0][0]))
        name_result = best_matching[0][0]
    else :
        result.write("None\n")
        name_result = "None"
    
    string_var2.set(name_result)
    print(name_result)



df = pd.read_csv('./StudentList.csv') # path of data base for fuzzy algorithm
name = df.iloc[:,4].to_list()
log = open("log.txt", "w")
result = open("result.txt", "w")
receiver_list = ["ผู้รับ", "ผู่รับ", "ผุ้รับ", "ผ้รับ", "ผู้รัน", "ฝับรับ"]
pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")
kernel = np.ones((5,5), np.uint8)

label_button = tk.Button(text="อัพโหลด",command=open_file).pack()

# variables
string_var = tk.StringVar()
string_var2 = tk.StringVar()

# Label image name
label_name = tk.Label(win, textvariable=string_var).pack()

# Label image
label_image = tk.Label(win)
label_image.pack()

# Load an image file using the Image module from Pillow
image = Image.open("parcel.jpg")
image.thumbnail((200,200))

# Convert the image to a PhotoImage object
photo = ImageTk.PhotoImage(image)

# Set the image of the label to the PhotoImage object
label_image.config(image=photo)



# Label result
label_result = tk.Label(win, textvariable=string_var2,font=("Courier",24)).pack()



win.mainloop()
