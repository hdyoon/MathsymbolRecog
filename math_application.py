# -*- coding: utf-8 -*-
#"""
#    <프로그램 개요>
#    GUI Application 클래스
#    Instance 생성시 CNN Model Instance 전달 받음.,
#    50x50 Image 입력 -> 배열 변환 -> 모델 적용(predict함수)
#    
#    <Methods>
#    __init__ : 초기화 시 CNN 모델 적용
#    arange() : 화면 구성
#    binding_widget : 화면 입력 위젯과 이벤트 연결
#    take_snapshot() : Image Array를 모델에 Query
#    clear() : Canvas, Image 모두 초기화
#    b1up(), b1down(), motion() : Canvas, Image 입력
#    destructor() : 화면 root 삭제(마지막 호)
#"""
from PIL import Image, ImageDraw
import tkinter as tk
import numpy
import math_mnist

class Application:
    def __init__(self, model):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        
        self.b1 = "up"
        self.xold = None
        self.yold = None
        
        self.current_image = None  # current image from the camera
        
        self.arrange()
        self.model = model
        
    def arrange(self):
        
        self.root = tk.Tk()  # initialize root window
        self.root.title("Handwritten Math Symbol Recognition")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        self.panel = tk.Canvas(self.root, width=50, height=50, bd=0, bg="black")
        self.panel.pack()
        
        self.result_label = tk.Label(self.root, text=" ")
        self.result_label.pack()
        
        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self.memory_image = Image.new("L", (50, 50), color='white')
        self.draw = ImageDraw.Draw(self.memory_image)
        
        # create a button, that when pressed, will take the current frame and save it to file
        self.btn_check = tk.Button(self.root, text="Snapshot!", command=self.take_snapshot)
        self.btn_check.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.btn_clear = tk.Button(self.root, text="Clear", command=self.clear)
        self.btn_clear.pack(fill="both", expand=True, padx=10, pady=10)

        self.binding_widget()
    
    def take_snapshot(self):
        x_data = numpy.array(self.memory_image).reshape(1,2500)
        hot_x_data = numpy.argmax(self.model.predict(x_data))
        
        df_labels = math_mnist.read_categories()
        predict_symbol = math_mnist.get_category_char(hot_x_data, df_labels, one_hot=False)
        
        self.result_label['text'] = predict_symbol
        print(self.model.predict(x_data))
        print("Prediction Symbol: ", predict_symbol)
        
    def clear(self):
        self.panel.delete("all")
        self.memory_image = None
        self.draw = None
        self.memory_image = Image.new("L", (50, 50), color='white')
        self.draw = ImageDraw.Draw(self.memory_image)
        self.result_label.config(text=" ")
        
    def b1down(self, event):
        self.b1 = "down"           # you only want to draw when the button is down
    
    def b1up(self, event):
        self.b1 = "up"
        self.xold = None           # reset the line when you let go of the button
        self.yold = None
    
    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line((self.xold,self.yold),(event.x,event.y),smooth=False,fill="white")
                # do the PIL image/draw (in memory) drawings
                self.draw.line((self.xold,self.yold,event.x,event.y),fill=0)
            self.xold = event.x
            self.yold = event.y
    
    def binding_widget(self):
        self.panel.bind("<Motion>", self.motion)
        self.panel.bind("<ButtonPress-1>",self.b1down)
        self.panel.bind("<ButtonRelease-1>", self.b1up)
        
    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
