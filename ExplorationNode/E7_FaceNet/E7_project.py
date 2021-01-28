
import face_recognition
import os
import matplotlib.pyplot as plt
import numpy as np

def get_gropped_face(image_file):
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return None
    a, b, c, d = face_locations[0]
    cropped_face = image[a:c, d:b, : ]

    return cropped_face

def get_face_embedding(face):
    return face_recognition.face_encodings(face)

def get_face_embedding_dict(dir_path):
    file_list = os.listdir(dir_path)
    embedding_dict = {}

    for file in file_list:
        img_path = os.path.join(dir_path, file)
        face = get_gropped_face(img_path)
        if face is None:
            continue
        embedding = get_face_embedding(face)
        if len(embedding) > 0:  # 얼굴영역 face가 제대로 detect되지 않으면
            embedding_dict[os.path.splitext(file)[0]] = embedding[0]

    return embedding_dict


def get_distance(emb1, name2):
    return np.linalg.norm(emb1 - embedding_dict[name2], ord=2)

def get_nearest_face(fe, top=5):
    sort_key_func = get_sort_key_func(fe)
    sorted_faces = sorted(embedding_dict.items(), key=lambda x:sort_key_func(x[0]))

    for i in range(top+1):
        if sorted_faces[i]:
            print('순위 {} : 이름({}), 거리({})'.format(i, sorted_faces[i][0], sort_key_func(sorted_faces[i][0])))



def get_sort_key_func(name1):
    def get_distance_from_name1(name2):
        return get_distance(name1, name2)
    return get_distance_from_name1


embedding_dict = get_face_embedding_dict(os.getcwd()+'/korean/')

cropped_face = get_gropped_face(os.getcwd()+'/me.jpeg')
embedding = get_face_embedding(cropped_face)
get_nearest_face(embedding)
