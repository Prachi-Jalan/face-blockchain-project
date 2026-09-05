import face_recognition
from PIL import Image, ImageDraw

image_path = "test_images/test4.jpg"
output_path = 'output_images/output4.jpg'

image = face_recognition.load_image_file(image_path)
face_locations = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=2)

print(f"Found {len(face_locations)} face(s)")

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left) in face_locations:
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=4)

pil_image.save(output_path)
print("Saved visualization to {output_path}")
