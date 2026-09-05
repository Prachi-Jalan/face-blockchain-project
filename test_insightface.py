import cv2
from insightface.app import FaceAnalysis

# Initialize the model — downloads model weights on first run
app = FaceAnalysis(name="buffalo_l")  # buffalo_l = a strong general-purpose model pack
app.prepare(ctx_id=0, det_size=(640, 640))

# Load an image with OpenCV (note: BGR format, not RGB)
img = cv2.imread("test_images/test2.jpg")

faces = app.get(img)

print(f"Found {len(faces)} face(s)")
for face in faces:
    print("Bounding box:", face.bbox)
    print("Embedding shape:", face.embedding.shape)
