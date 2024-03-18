import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis with CPU context
face_analysis = FaceAnalysis()

# Load pre-trained models
face_analysis.prepare(ctx_id=-1)

# Define source and destination folders
source_folder = "/content"
destination_folder = "/content"

# Reference image for similarity comparison
reference_image_path = "/content/similar_faces/refx.jpg"
reference_image = cv2.imread(reference_image_path)

# Detect faces in the reference image
reference_faces = face_analysis.get(reference_image)

# Assuming there is only one face in the reference image
reference_face = reference_faces[0]
reference_embedding = reference_face.embedding

# Function to align faces using facial landmarks
def align_face(image, landmarks):
    if landmarks is None:
        # If no landmarks detected, return the original image
        return image
    
    # Define reference landmarks for alignment (e.g., eyes)
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Calculate angle between eyes
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    angle = np.degrees(angle)

    # Rotate image around center of eyes
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return aligned_face

# Function to calculate similarity between faces
def calculate_similarity(reference_embedding, face_embedding):
    # Compute cosine similarity between embeddings
    similarity_score = np.dot(reference_embedding, face_embedding) / (np.linalg.norm(reference_embedding) * np.linalg.norm(face_embedding))
    return similarity_score

# Threshold for similarity comparison
similarity_threshold = 0.35

# Initialize counters for statistics
total_images = 0
similar_images = 0
different_images = 0

# Iterate over images in source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        total_images += 1
        image_path = os.path.join(source_folder, filename)
        image = cv2.imread(image_path)

        # Detect faces in the image
        faces = face_analysis.get(image)

        for face in faces:
            # Align face using facial landmarks
            aligned_face = align_face(image, face.landmarks)

            # Extract face embedding
            face_embedding = face.embedding

            # Compare face with reference face
            similarity_score = calculate_similarity(reference_embedding, face_embedding)

            # Move image to appropriate folder based on similarity score
            if similarity_score >= similarity_threshold:
                destination_path = os.path.join(destination_folder, "similar_faces")
                similar_images += 1
            else:
                destination_path = os.path.join(destination_folder, "different_faces")
                different_images += 1

            # Move the image
            os.makedirs(destination_path, exist_ok=True)
            os.rename(image_path, os.path.join(destination_path, filename))

            # Display live updates
            print("Processed:", filename)

# Display statistics
print(f"Total images processed: {total_images}")
print(f"Number of similar images: {similar_images}")
print(f"Number of different images: {different_images}")
