import cv2
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to ask for student details
def get_student_details():
    student_id = student_id_entry.get()
    student_name = student_name_entry.get()
    return student_id, student_name

# Define directory for storing the images and training data
def create_student_directory(student_id):
    image_dir = f"student_images/{student_id}"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    return image_dir

# Create the 'student_models' directory if it doesn't exist
def create_model_directory():
    if not os.path.exists("student_models"):
        os.makedirs("student_models")

# Initialize camera
cap = cv2.VideoCapture(0)

# Function to capture 100 images from the camera
def capture_images():
    student_id, student_name = get_student_details()
    if not student_id or not student_name:
        messagebox.showwarning("Input Error", "Please enter both student ID and name.")
        return

    image_dir = create_student_directory(student_id)
    messagebox.showinfo("Capture", f"Capturing 100 images for {student_name}...")
    image_count = 0
    while image_count < 100:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Convert the frame to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's face detector to find faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the face image
            face_image = gray[y:y+h, x:x+w]  # Convert to grayscale for LBPH
            image_path = os.path.join(image_dir, f"{student_id}_{image_count}.jpg")
            cv2.imwrite(image_path, face_image)
            image_count += 1
            print(f"Captured image {image_count}/100")
        
        # Display the frame with the rectangle
        cv2.imshow("Capturing Images", frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release camera after capturing
    cap.release()
    cv2.destroyAllWindows()

    messagebox.showinfo("Capture Completed", f"Captured 100 images for {student_name}.")

# Function to train the face recognition model
def train_model():
    student_id, student_name = get_student_details()
    if not student_id or not student_name:
        messagebox.showwarning("Input Error", "Please enter both student ID and name.")
        return

    create_model_directory()  # Make sure the 'student_models' directory exists
    image_dir = f"student_images/{student_id}"
    messagebox.showinfo("Training", f"Training model for {student_name}...")
    
    # Create a list of images and labels
    faces = []
    labels = []
    
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
        
        # Get the label (student_id)
        label = int(student_id)  # In this case, student_id is used as the label
        
        # Append the face image and corresponding label
        faces.append(image)
        labels.append(label)
    
    # Create an LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    
    # Save the trained model
    model_path = f"student_models/{student_id}_model.yml"
    recognizer.save(model_path)
    print(f"Model trained and saved as {model_path}.")
    messagebox.showinfo("Training Completed", f"Model for {student_name} trained successfully.")

# Function to recognize faces and mark attendance
def recognize_face():
    messagebox.showinfo("Recognition", "Starting face recognition for attendance...")
    
    create_model_directory()  # Make sure the 'student_models' directory exists
    
    # Load all the trained models from student_models folder
    trained_models = [f for f in os.listdir("student_models") if f.endswith("_model.yml")]
    
    # Create a face cascade for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)

    recognized_students = set()  # Set to track students whose attendance has already been marked
    all_students = {model_file.split('_')[0] for model_file in trained_models}  # All student IDs from models

    # This dictionary will track if a student has been marked present or absent
    attendance_record = {student_id: "Absent" for student_id in all_students}

    # This dictionary will track the previous attendance status (to detect change)
    previous_attendance_status = {student_id: "Absent" for student_id in all_students}

    # Get the current date for saving the attendance file
    current_date = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"attendance_{current_date}.csv"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Convert the frame to grayscale (for recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Track if any face was recognized
        face_recognized = False

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Get the region of interest (the face)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Try to recognize the face
            recognized = False
            for model_file in trained_models:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(f"student_models/{model_file}")
                
                student_id = model_file.split('_')[0]
                label, confidence = recognizer.predict(roi_gray)
                
                if label == int(student_id) and confidence < 100:
                    # If student is recognized and hasn't been marked yet, mark them Present
                    if attendance_record[student_id] == "Absent" and previous_attendance_status[student_id] == "Absent":
                        student_name = f"Student {student_id}"
                        attendance_record[student_id] = "Present"  # Update from Absent to Present
                        previous_attendance_status[student_id] = "Present"  # Update previous status
                        mark_attendance(student_name, "Present", attendance_file)  # Mark in real-time
                    recognized_students.add(student_id)  # Mark student as recognized
                    recognized = True
                    face_recognized = True
                    break

            if not recognized:
                cv2.putText(frame, "Unknown Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Now mark Absent for all students who were not recognized yet and whose status is still "Absent"
        for student_id in all_students:
            if student_id not in recognized_students and attendance_record[student_id] == "Absent" and previous_attendance_status[student_id] != "Absent":
                student_name = f"Student {student_id}"
                attendance_record[student_id] = "Absent"
                previous_attendance_status[student_id] = "Absent"
                mark_attendance(student_name, "Absent", attendance_file)

        # Reset the recognized students for the next cycle
        recognized_students.clear()

        # Display the frame
        cv2.imshow("Face Recognition", frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to mark attendance in a file
def mark_attendance(student_name, status, file_name):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, "a") as f:
        f.write(f"{time_now},{student_name},{status}\n")
    print(f"Attendance Marked: {student_name} is {status}")

# Create main window
root = tk.Tk()
root.title("Student Attendance System")
root.geometry("500x400")
root.config(bg="#A0D6B4")

# Add labels
label = tk.Label(root, text="Student Attendance System", font=("Arial", 20, "bold"), bg="#A0D6B4")
label.pack(pady=20)

# Add input fields for student ID and name
student_id_label = tk.Label(root, text="Student ID", font=("Arial", 14), bg="#A0D6B4")
student_id_label.pack()
student_id_entry = tk.Entry(root, font=("Arial", 14), width=20)
student_id_entry.pack(pady=5)

student_name_label = tk.Label(root, text="Student Name", font=("Arial", 14), bg="#A0D6B4")
student_name_label.pack()
student_name_entry = tk.Entry(root, font=("Arial", 14), width=20)
student_name_entry.pack(pady=5)

# Add buttons
register_button = tk.Button(root, text="Register Student", font=("Arial", 14), bg="#76C7C0", command=capture_images)
register_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", font=("Arial", 14), bg="#76C7C0", command=train_model)
train_button.pack(pady=10)

recognize_button = tk.Button(root, text="Recognize and Mark Attendance", font=("Arial", 14), bg="#76C7C0", command=recognize_face)
recognize_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
