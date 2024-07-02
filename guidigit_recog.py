from keras.models import load_model
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np

model = load_model('mnist.h5')

def predict_digit(img):

	#resize image to 28x28 pixels
	img = img.resize((28,28))

    #convert rgb to grayscale
	img = img.convert('L')
	img = ImageOps.invert(img)
	img = np.array(img)

	#reshaping for model normalization
	img = img.reshape(1,28,28,1)
	img - img.astype('float32')
	img = img/255.0

	#predicting the class
	res = model.predict([img])[0]
	return np.argmax(res), max(res)

class App(tk.Tk):

	def __init__(self):
		super().__init__()
		self.x = self.y = 0
		
		#Configure window
		self.title("Regonize Digits(0-9)")
		self.geometry("500x400")
		self.configure(bg='lightblue')

        	# Creating elements
		self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross", bd=5, relief="sunken")
		self.label = tk.Label(self, text="draw a digit", font=("Helvetica", 48), bg='lightblue')
		self.classify_btn = tk.Button(self, text = "Recognize", command = self.classify_handwriting) 
		self.button_clear = tk.Button(self, text = "clear", command = self.clear_all, bg='red', fg='white', font=("Helvetica", 14))
        
		# Grid structure
		self.canvas.grid(row=0, column=0, padx=20, pady=20, columnspan=2)
		self.label.grid(row=1, column=0, pady=10, padx=20)
		self.classify_btn.grid(row=2, column=1, pady=10, padx=20)
		self.button_clear.grid(row=2, column=0, pady=10, padx=20) 
        	
		self.canvas.bind("<B1-Motion>", self.draw_lines)

	def clear_all(self):
		self.canvas.delete("all")
		self.label.configure(text="Draw a digit")

	def classify_handwriting(self):
		x = self.winfo_rootx() + self.canvas.winfo_x()
		y = self.winfo_rooty() + self.canvas.winfo_y()
		x1 = x + self.canvas.winfo_width()
		y1 = y + self.canvas.winfo_height()
		im = ImageGrab.grab().crop((x, y, x1, y1))
		digit, acc = predict_digit(im)
		self.label.configure(text=f'{digit}, {int(acc * 100)}%')

	def draw_lines(self, event):
		self.x = event.x
		self.y = event.y
		r = 8
		self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
		self.canvas.create_line(self.x, self.y, event.x, event.y, width=20, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)

app = App()
tk.mainloop()
