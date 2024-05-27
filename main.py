import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

MAX_WIDTH = 1000
MAX_HEIGHT = 800
MAX_WIDTH_IMG = 600
MAX_HEIGHT_IMG = 400

file_path = ""

# Load model
yolov8_model = YOLO('model/yolov8/weights/best.pt') 


def open_file():
    global file_path  # Use global variable for file_path

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        # Clear previous image and prediction
        imgLabel.configure(image=None)
        predictLabel.configure(text="")

        # Display new image
        display_image(file_path)
        display_image_name(file_path)

def display_image(image_path):
    image = Image.open(image_path)
    image = resize_image(image, MAX_WIDTH_IMG, MAX_HEIGHT_IMG)
    photo = ImageTk.PhotoImage(image)

    imgLabel.configure(image=photo)
    imgLabel.image = photo  # Keep a reference to the image object

def display_image_name(image_path):
    image_name = os.path.basename(image_path)
    imgNameLabel.configure(text=f"Image: {image_name}")

def resize_image(image, max_width, max_height):
    width_ratio = max_width / image.width
    height_ratio = max_height / image.height
    min_ratio = min(width_ratio, height_ratio)

    if min_ratio < 1:
        new_width = int(image.width * min_ratio)
        new_height = int(image.height * min_ratio)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def predict():
    global file_path  # Use global variable for file_path

    if not file_path:
        predictLabel.configure(text="No image selected")
        return

    # Predict on image
    results = yolov8_model(file_path)

    # Assuming results[0].names is a dictionary like {0: 'COVID-19', 1: 'Normal', 2: 'Pneumonia'}
    name_class = results[0].names
    probs = results[0].probs.data.tolist()

    # Format the results into a string
    result_text = "Predictions:\n"
    for i, prob in enumerate(probs):
        result_text += f"{name_class[i]}: {prob:.5f}\n"

    # Get the class of the image with the highest probability
    highest_prob = max(probs)
    max_index = probs.index(highest_prob)
    status = name_class[max_index]

    # Display the result on screen
    result_text += f"\nStatus: {status} with probability {highest_prob:.5f}"
    predictLabel.configure(text=result_text)

# Initialize customtkinter
ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue") 

# Create the main window with an initial size
root = ctk.CTk()
root.title("Pneumonia Prediction Application")
root.geometry(f"{MAX_WIDTH}x{MAX_HEIGHT}")

# Set the same background color for root and button frame
root_bg_color = root.cget("bg")

# Create a frame for the buttons with the same background color
button_frame = ctk.CTkFrame(master=root, fg_color=root_bg_color, corner_radius=0)
button_frame.pack(pady=20)

# Create a button to open the file dialog
open_button = ctk.CTkButton(master=button_frame, text="Open Image", command=open_file, font=("JetBrains Mono", 20))
open_button.pack(side="left", pady=10)

# Create a button to predict
predict_button = ctk.CTkButton(master=button_frame, text="Predict", command=predict, font=("JetBrains Mono", 20))
predict_button.pack(side='left', padx=10)

# Create a label widget to display the image
imgLabel = ctk.CTkLabel(master=root, text="")
imgLabel.pack(pady=20)

# Create a label to show image name
imgNameLabel = ctk.CTkLabel(master=root,text="",font=("JetBrains Mono", 18))
imgNameLabel.pack(pady=5)

# Create a label to predict the image
predictLabel = ctk.CTkLabel(master=root, text="", font=("JetBrains Mono", 18))
predictLabel.pack(pady=10)

# Start the main event loop
root.mainloop()
