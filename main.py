import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

MAX_WIDTH = 1000
MAX_HEIGHT = 800
MAX_WIDTH_IMG = 700
MAX_HEIGHT_IMG = 500

file_path = ""

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

def display_image(image_path):
    image = Image.open(image_path)
    image = resize_image(image, MAX_WIDTH_IMG, MAX_HEIGHT_IMG)
    photo = ImageTk.PhotoImage(image)

    imgLabel.configure(image=photo)
    imgLabel.image = photo  # Keep a reference to the image object



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
    ## YOLOV8
    # Load model
    model = YOLO('model/yolov8/weights/best.pt') 

    # Predict on image
    results = model(file_path)

    # Get the class names and probabilities list
    name_class = results[0].names
    probs = results[0].probs.data.tolist()

    # Get the class of the image and the probability
    highest_prob = max(probs)
    max_index = probs.index(highest_prob)
    print(highest_prob)
    print(name_class[max_index])

    # Display the result on screen
    predictLabel.configure(text="Status: " + name_class[max_index] + "\nProbability: " + str(highest_prob))

# Initialize customtkinter
ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue") 

# Create the main window with an initial size
root = ctk.CTk()
root.title("Pneumonia Prediction Application")
root.geometry(f"{MAX_WIDTH}x{MAX_HEIGHT}")

# Create a frame for the buttons
button_frame = ctk.CTkFrame(master=root,bg_color="black")
button_frame.pack(pady=20)

# Create a button to open the file dialog
open_button = ctk.CTkButton(master=button_frame, text="Open Image", command=open_file,font=("JetBrains Mono", 20))
open_button.pack(side="left", pady=20)

# Create a button to predict
predict_button = ctk.CTkButton(master=button_frame, text="Predict", command=predict,font=("JetBrains Mono", 20))
predict_button.pack(side='left', padx=10)

# Create a label widget to display the image
imgLabel = ctk.CTkLabel(master=root, text="")
imgLabel.pack(pady=20)

# Create a label to predict the image
predictLabel = ctk.CTkLabel(master=root, text="",font=("JetBrains Mono", 24))
predictLabel.pack(pady=10)

# Start the main event loop
root.mainloop()
