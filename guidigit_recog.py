from keras.models import load_model
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np

model = load_model('final_model.h5')

def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert rgb to grayscale
    img = img.convert('L')
    
    # Invert the image (MNIST digits are black on white background)
    img = ImageOps.invert(img)
    
    # Convert image to a numpy array
    img = np.array(img)
    
    # Reshape and normalize the image
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255.0
    
    # Predict the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.x = self.y = 0
        
        # Configure window
        self.title("Recognize Digits (0-9)")
        self.geometry("400x400")
        self.configure(bg='white')

        # Creating elements
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white", cursor="cross", bd=5, relief="sunken")
        self.label = tk.Label(self, text="Draw a digit", font=("Courier", 24), bg='lightblue')
        self.classify_btn = tk.Button(self, text="Recognize", font=("Courier", 14), command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all, bg='white', fg='black', font=("Courier", 14))
        
        # Grid structure
        self.canvas.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        self.label.grid(row=1, column=0, pady=10, padx=10, columnspan=2)
        self.classify_btn.grid(row=2, column=1, pady=10, padx=10)
        self.button_clear.grid(row=2, column=0, pady=10, padx=10)
        
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
        r = 10  # Adjust the thickness of the strokes
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

app = App()
tk.mainloop()
