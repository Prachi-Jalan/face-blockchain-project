import face_recognition

image = face_recognition.load_image_file("test_images/test1.jpg")
face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s)")
print(face_locations)
