import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

IMG_HEIGHT = 128
IMG_WIDTH = 128

img_path = ""

class_indices = {'A': 0, 'B': 1, 'AB': 2, 'O': 3}

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read the image. Please ensure the file is a valid image.")
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def is_fingerprint(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    edges = cv2.Canny(img, threshold1=100, threshold2=200)

    edge_pixels = np.sum(edges == 255)
    total_pixels = edges.size
    edge_percentage = (edge_pixels / total_pixels) * 100

    if edge_percentage > 20:
        return True
    return False

def predict_blood_group():
    if img_path == "":
        result_label.config(text="Please upload a fingerprint image first.", fg="red")
        return
    
    if not is_fingerprint(img_path):
        result_label.config(text="Uploaded image is not a valid fingerprint.", fg="red")
        return

    predict_button.config(state="disabled")
    
    loading_label.config(text="Prediction in Progress...")
    loading_label.grid(row=4, column=0, columnspan=3, pady=10)

    try:
        model = tf.keras.models.load_model("fingerprint_blood_group_model.h5")
        
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)

        predicted_class = list(class_indices.keys())[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        result_label.config(text=f"Predicted Blood Group: {predicted_class} ({confidence:.2f}% confidence)", fg="#4CAF50")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="red")
    finally:
        loading_label.grid_forget()
        predict_button.config(state="normal")

def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")

root = tk.Tk()
root.title("Fingerprint Blood Group Prediction")
root.geometry("600x600")
root.configure(bg="#ffffff")

title_label = tk.Label(root, text="Fingerprint Blood Group Prediction", font=("Helvetica Neue", 20, "bold"), bg="#ff4d4d", fg="white", padx=20, pady=10)
title_label.grid(row=0, column=0, columnspan=3, pady=20)

upload_box = tk.Label(root, text="Click to Upload Fingerprint Image", width=30, height=3, relief="solid", bg="#f2f2f2", font=("Arial", 14), fg="#333", bd=0, highlightthickness=2, highlightbackground="#ff4d4d", highlightcolor="#ff4d4d", cursor="hand2")
upload_box.grid(row=1, column=0, columnspan=3, pady=20)
upload_box.bind("<Button-1>", lambda e: upload_image())

image_label = tk.Label(root, bg="#f2f2f2", relief="solid", width=20, height=15)
image_label.grid(row=2, column=0, columnspan=3, pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image, relief="flat", width=20, height=2, bg="#ff4d4d", fg="white", font=("Arial", 12, "bold"), bd=0, highlightthickness=0, cursor="hand2")
upload_button.grid(row=3, column=0, columnspan=3, pady=10)

loading_label = tk.Label(root, text="", font=("Arial", 12), bg="#ffffff")

predict_button = tk.Button(root, text="Predict Blood Group", command=predict_blood_group, relief="flat", width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), bd=0, highlightthickness=0, cursor="hand2")
predict_button.grid(row=5, column=0, columnspan=3, pady=20)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#ffffff", fg="black")
result_label.grid(row=6, column=0, columnspan=3, pady=20)

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)
root.grid_rowconfigure(6, weight=1)

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

root.mainloop()
