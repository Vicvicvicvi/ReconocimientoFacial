import face_recognition

# Cargar las imágenes
imagen_conocida = face_recognition.load_image_file("persona_conocida.jpg")
imagen_desconocida = face_recognition.load_image_file("persona_desconocida.jpg")

# Obtener los embeddings de las imágenes
embedding_conocido = face_recognition.face_encodings(imagen_conocida)[0]
embedding_desconocido = face_recognition.face_encodings(imagen_desconocida)[0]

# Comparar los embeddings para determinar si las imágenes son de la misma persona
resultado = face_recognition.compare_faces([embedding_conocido], embedding_desconocido)

if resultado[0]:
    print("Las imágenes son de la misma persona")
else:
    print("Las imágenes no son de la misma persona")
