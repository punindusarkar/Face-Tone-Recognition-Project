import cv2
import numpy as np
import face_recognition
from tkinter import *
from PIL import Image, ImageTk

# Color options dictionary based on skin tone
color_options = {
    'fair': ["Crimson", "Scarlet", "Rose", "Fuchsia", "Lavender", "White", "Ivory"],
    'medium': ["Coral", "Amber", "Peach", "Copper", "Dusty Rose", "Mauve", "Plum"],
    'olive': ["Emerald", "Seafoam", "Turquoise", "Aqua", "Periwinkle", "Slate", "Steel Blue"],
    'dark': ["Burgundy", "Purple", "Charcoal", "Royal Blue", "Cobalt", "Magenta", "Black"],
    'light': ["Goldenrod", "Mustard", "Beige", "Cream", "Buff", "Peach", "Champagne"],
    'very_fair': ["Snow", "Alabaster", "Porcelain", "Ivory", "Milk", "Cotton", "Frost"],
    'tan': ["Copper", "Bronze", "Caramel", "Tawny", "Sienna", "Chestnut", "Honey"],
    'deep_olive': ["Olive Drab", "Moss", "Army Green", "Cucumber", "Lime", "Seaweed", "Shamrock"],
    'light_brown': ["Hazel", "Tan", "Brown Sugar", "Warm Taupe", "Cinnamon", "Chestnut", "Sable"],
    'dark_brown': ["Espresso", "Dark Chocolate", "Mahogany", "Mocha", "Coffee", "Ebony", "Onyx"]
}

# Function to detect faces
def detect_face(frame):
    # Ensure frame is in the correct format
    if len(frame.shape) != 3 or frame.shape[2] != 3:  # Check if frame is not 3-channel (BGR)
        raise ValueError("Frame must be a 3-channel image (BGR/RGB format)")
    
    print(f"Frame type before conversion: {frame.dtype}, shape: {frame.shape}")

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ensure it's now a proper RGB image (8-bit per channel)
    if rgb_frame.dtype != np.uint8:
        raise ValueError(f"Converted frame is not in the correct image type. It must be uint8. Got {rgb_frame.dtype}.")

    print(f"Frame type after conversion: {rgb_frame.dtype}, shape: {rgb_frame.shape}")

    # Detect faces in the RGB frame
    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations

# Function to estimate skin tone based on face region
def estimate_skin_tone(frame, face_location):
    top, right, bottom, left = face_location
    face_region = frame[top:bottom, left:right]
    face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    
    # Calculate average color of the face region
    avg_color = np.mean(face_region, axis=(0, 1))
    r, g, b = avg_color
    
    # Define skin tone based on average color
    if r > 180 and g > 140 and b > 120:
        return 'very_fair'  # Light skin tones
    elif r > 140 and g > 110 and b > 90:
        return 'fair'
    elif r > 120 and g > 100 and b > 80:
        return 'light'
    elif r > 100 and g > 80 and b > 60:
        return 'medium'
    elif r > 80 and g > 70 and b > 50:
        return 'tan'
    elif r > 60 and g > 50 and b > 40:
        return 'dark'
    else:
        return 'dark_brown'

# Function to extract dominant color from outfit
def extract_dominant_color(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = centers[0].astype(int)
    return dominant_color

# Function to update the Tkinter UI
def update_ui(frame, label):
    # Detect face
    try:
        face_locations = detect_face(frame)
    except ValueError as e:
        print(f"Error during face detection: {e}")
        return "Error during face detection"

    # For simplicity, use the first face detected
    if face_locations:
        face_location = face_locations[0]
        # Estimate skin tone
        skin_tone = estimate_skin_tone(frame, face_location)
        
        # Get color options based on skin tone
        colors = color_options.get(skin_tone, ["No suggestion available"])

        # Display the detected skin tone and color suggestions
        color_info = f"Skin Tone: {skin_tone.capitalize()}\nSuggested Colors: {', '.join(colors)}"
        
        # Draw rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Extract outfit color (Optional: can add additional code for the outfit color)
        dominant_color = extract_dominant_color(frame)
        color_info += f"\nOutfit Color: RGB {tuple(int(c) for c in dominant_color)}"

    else:
        color_info = "No face detected."

    # Convert frame to ImageTk format for display
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.config(image=img)
    label.image = img
    
    return color_info

# Function to start camera and display in Tkinter window
def start_camera():
    cap = cv2.VideoCapture(0)
    
    # Create main Tkinter window
    root = Tk()
    root.title("Face Detection and Outfit Color Recommendation")
    
    # Label to display the camera feed
    label = Label(root)
    label.pack()
    
    # Text label to show color information
    color_label = Label(root, text="", font=("Arial", 12))
    color_label.pack()

    # Loop to capture frames from the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update the UI with the current frame and color info
        color_info = update_ui(frame, label)
        color_label.config(text=color_info)
        
        root.update()
    
    cap.release()
    root.mainloop()

if __name__ == "__main__":
    start_camera()
