import cv2
import dlib
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def save_face_data(encodings, names, cpfs):
    with open('face_data.pkl', 'wb') as f:
        pickle.dump((encodings, names, cpfs), f)

def load_face_data():
    if os.path.exists('face_data.pkl'):
        with open('face_data.pkl', 'rb') as f:
            return pickle.load(f)
    return [], [], []

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

known_face_encodings, known_face_names, known_face_cpfs = load_face_data()

def register_face(name, cpf):
    face_images = []
    for i in range(50): 
        ret, frame = video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                for face in faces:
                    shape = shape_predictor(gray, face)
                    face_encoding = np.array(encoder.compute_face_descriptor(frame, shape))
                    face_images.append(face_encoding)

                for face_encoding in face_images:
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(f"{name}_{cpf}")
                    known_face_cpfs.append(cpf)

                save_face_data(known_face_encodings, known_face_names, known_face_cpfs)
                messagebox.showinfo("Sucesso", f"Rosto de {name} registrado com sucesso!")
                break
            else:
                messagebox.showwarning("Aviso", "Nenhum rosto detectado. Tente novamente.")
        else:
            messagebox.showwarning("Erro", "Erro ao capturar a imagem. Tente novamente.")

def recognize_faces():
    frame_count = 0  
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % 5 == 0:  
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_small_frame)

            face_names = []
            face_cpfs = []  
            for face in faces:
                shape = shape_predictor(rgb_small_frame, face)
                face_encoding = np.array(encoder.compute_face_descriptor(rgb_small_frame, shape))
                matches = [np.linalg.norm(face_encoding - known_face) < 0.6 for known_face in known_face_encodings]   
                name = "Desconhecido"
                cpf = ""

                if True in matches:
                    best_match_index = matches.index(True)
                    name = known_face_names[best_match_index].split('_')[0]
                    cpf = known_face_cpfs[best_match_index]

                face_names.append(name)
                face_cpfs.append(cpf)  

            for (face, name, cpf) in zip(faces, face_names, face_cpfs):
                top, right, bottom, left = (int(face.top() / 0.25), int(face.right() / 0.25), 
                                            int(face.bottom() / 0.25), int(face.left() / 0.25))

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 50), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, f"CPF: {cpf}", (left + 6, bottom - 25), font, 1.0, (255, 255, 255), 1)  

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            panel.imgtk = imgtk
            panel.configure(image=imgtk)

        frame_count += 1
        window.update()

def on_register():
    nome = simpledialog.askstring("Entrada", "Digite o nome da pessoa:")
    cpf = simpledialog.askstring("Entrada", "Digite o CPF:")
    if nome and cpf:
        register_face(nome, cpf)

def on_recognize():
    recognize_faces()

window = tk.Tk()
window.title("Reconhecimento Facial")

panel = tk.Label(window)
panel.pack(pady=10)

frame_register = tk.Frame(window)
frame_register.pack(pady=10)

btn_register = tk.Button(frame_register, text="Registrar Novo Rosto", command=on_register, height=2, width=30)
btn_register.pack(pady=5)

frame_recognize = tk.Frame(window)
frame_recognize.pack(pady=10)

btn_recognize = tk.Button(frame_recognize, text="Reconhecer Rostos", command=on_recognize, height=2, width=30)
btn_recognize.pack(pady=5)

def on_closing():
    video_capture.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing) 

window.mainloop()
