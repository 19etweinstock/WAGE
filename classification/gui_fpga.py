import tkinter as tk
import numpy as np
import cv2
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import ImageTk,Image,ImageDraw
from urllib.request import urlopen
import time
from network import runNetwork
import serial

ser = serial.Serial('/COM3', 115200)

def event_function(event):
    
    x=event.x
    y=event.y
    
    x1=x-20
    y1=y-20
    x2=x+20
    y2=y+20

    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')

def save():
    
    img_array=np.array(img)
    img_array=cv2.resize(img_array,(256,256)) 
    cv2.imwrite(str(time.time())+'image.jpg',img_array)
    messagebox.showinfo("Save", "Image saved successfully !")


def clear():
    
    global img,img_draw
    canvas.delete('all')
    img=Image.new('RGB',(500,500),(0,0,0))
    img_draw=ImageDraw.Draw(img)    
    entry.delete(0 , tk.END)
    label_status.config(text='Draw a Digit')


def upload():

    global img , dispimg
    path=fd.askopenfilename(filetypes=[("Image File",'.jpg')])
    if path is not None and path != '':
        img = Image.open(path)
        dispimg = ImageTk.PhotoImage(img.resize((400,400)))
        canvas.create_image(250,250,image= dispimg)
        predict()


def predict():

    global img , dispimg
    if(len(str(entry.get())) > 5 ):
        img = Image.open(urlopen(str(entry.get())))
        dispimg = ImageTk.PhotoImage(img.resize((400,400)))
        canvas.create_image(250,250,image= dispimg)


    img_array=np.array(img)
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array=cv2.resize(img_array,(28,28))
        
    img_array=img_array/256.0

    # set up python serial port for talking to fpga
    value = 0
    index = 2 ** 7
    count = 0
    for j in range (27, -1, -1):
        for k in range(27, -1, -1):
            if (img_array[j][k] > 1/4):
                value = value + index
            index = index // 2
            count = count + 1
            if (count == 8):
                s = bin(value).replace("0b",'')
                if (len(s) != 8):
                    diff = 8 - len(s)
                    s = "0" * diff + s
                # print(s, end='')
                uart_tx = bytes([value])
                ser.write(uart_tx)
                value = 0
                index = 2 ** 7
                count = 0

    result = runNetwork(img_array)
    print([result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9]])
    max_hw = int.from_bytes(ser.read(1),'little')
    if (max_hw == 255):
        max_hw = int.from_bytes(ser.read(1),'little')
    nine = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    eight = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    seven = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    six = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    five = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    four = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    three = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    two = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    one = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    zero = int.from_bytes(ser.read(1), 'little', signed=True) / 16
    print([zero, one, two, three, four, five, six, seven, eight, nine])
    label=np.argmax(result)
    max = np.max(result)
    if (result[label] >= 1 and np.sum(result <=0) == 9):
        label = f'strong {label}'
    elif (np.sum(result<=0) == 9):
        label = f'weak {label}'
    elif (np.sum(result == 0.) == 10):
        label = 'no prediction'
    elif (np.sum(result <=0) != 9 and np.sum(result == max) == 1):
        label = f'confused {label}'
    elif (np.sum(result == max) != 1):
        label = []
        for i in range(0,10):
            if(result[i] == max):
                label.append(i)
    label_status.config(text='SW : '+str(label) + ' HW: ' + str(max_hw))
    
    
win=tk.Tk()
win.title("Quantized MNIST Digit Prediction by Ethan Weinstock")
win.resizable(height=False,width=False)


label_status=tk.Label(win,text='Draw a Digit',bg='pink',font='Helvetica 24 bold')
label_status.grid(row=0,column=0,columnspan=4 )

canvas=tk.Canvas(win,width=500,height=500,bg='white')
canvas.grid(row=1,column=0,columnspan=4)

button_save=tk.Button(win,text='SAVE',bg='green',fg='white',font='Helvetica 20 bold',command=save)
button_save.grid(row=2,column=0)

button_predict=tk.Button(win,text='PREDICT',bg='blue',fg='white',font='Helvetica 20 bold',command = predict)
button_predict.grid(row=2,column=1)

button_clear=tk.Button(win,text='CLEAR',bg='cyan',fg='white',font='Helvetica 20 bold',command=clear)
button_clear.grid(row=2,column=2)

button_exit=tk.Button(win,text='UPLOAD',bg='red',fg='white',font='Helvetica 20 bold',command=upload)
button_exit.grid(row=2,column=3)

dummy = tk.Label(win, font = 'Helvetica 15 bold' , text=" ")
dummy.grid(row = 3 , column = 0)

label_link = tk.Label(win, font = 'Helvetica 13 bold' , text="Image link :")
label_link.grid(row = 4 , column = 0)


entry = tk.Entry(win , width = 60)
entry.grid(row=4, column = 1, columnspan=3)

dummy = tk.Label(win, font = 'Helvetica 15 bold' , text=" ")
dummy.grid(row = 5 , column = 0)

canvas.bind('<B1-Motion>',event_function)
img=Image.new('RGB',(500,500),(0,0,0))
img_draw=ImageDraw.Draw(img)

win.mainloop()