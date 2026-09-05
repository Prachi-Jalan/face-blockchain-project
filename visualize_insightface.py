import cv2
from insightface.app import FaceAnalysis

image_path = "test_images/test3.jpg"
output_path = "output_images/test3_insightface_detected.jpg"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread(image_path)
faces = app.get(img)

print(f"Found {len(faces)} face(s)")

for face in faces:
    box = face.bbox.astype(int)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

cv2.imwrite(output_path, img)
print(f"Saved visualization to {output_path}")
