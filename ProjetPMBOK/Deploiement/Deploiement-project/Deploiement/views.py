from django.http import StreamingHttpResponse, HttpResponseServerError, response
from django.views.decorators import gzip
import joblib
from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
import pickle
import os
import cv2
import time
import imutils

import warnings
warnings.filterwarnings('ignore')
from owlready2 import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np

names = []



def face(request):
    global name
    curr_path = os.getcwd()

    # print("Loading face detection model")
    proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
    model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
    face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

    # print("Loading face recognition model")
    recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
    face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

    recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
    le = pickle.loads(open('le.pickle', "rb").read())
    print("Starting test video file")
    vs = cv2.VideoCapture(2)
    time.sleep(1)
    tries = 0
    access = False
    name = ""
    while tries < 5:

        ret, frame = vs.read()
        frame = imutils.resize(frame, width=600)

        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False,
                                           False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence >= 0.7:
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                (fH, fW) = face.shape[:2]

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)

                face_recognizer.setInput(face_blob)
                vec = face_recognizer.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

    if name != "unknown":
        return render(request, "main.html", {'name': name})
    else:
        return redirect("login")


def rempalcement(m):
    m = str(m).replace('PMBOK1.', '').replace('_', ' ')
    return m


def prediction(request):
    if request.method == 'POST':
        query = request.POST['query']
        df_concept = joblib.load('C:/Users/ASUS/Notebooks/Projet 5eme/df_concept')

        onto = get_ontology("C:/Users/ASUS/Notebooks/Projet 5eme/PMBOK1.owl").load()
        list_classe = list(onto.classes())
        newlist = [rempalcement(classe) for classe in list_classe]

        docs = [(classe) for classe in newlist]

        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matris = tfidf_vectorizer.fit_transform(docs)

        query_vector = tfidf_vectorizer.transform([query])
        list_similarity = cosine_similarity(query_vector, tfidf_matris)[0].tolist()

        list_similarity_tri = sorted(list_similarity, reverse=True)

        list_similarity_tri = [i for i in list_similarity_tri if i != 0]

        top = []
        for i, element in enumerate(list_similarity_tri):
            top.append(list_similarity.index(list_similarity_tri[i]))

        top = list(dict.fromkeys(top))

        classe = np.array(newlist)[top]

        classeO = np.array(list_classe)[top]

        sousClasse = []
        for i, element in enumerate(classeO):
            sousClasse.append(list(onto.get_instances_of(element)))

        souslist = [rempalcement(classe) for classe in sousClasse]

        doc2 = [(classe) for classe in souslist]

        tfidf_matris2 = tfidf_vectorizer.fit_transform(doc2)

        query_vector2 = tfidf_vectorizer.transform([query])
        list_similarity2 = cosine_similarity(query_vector2, tfidf_matris2)[0].tolist()

        list_similarity_tri2 = sorted(list_similarity2, reverse=True)
        sous = sousClasse[list_similarity2.index(list_similarity_tri2[0])]

        sousClient = []
        for i in sous:
            i = str(i).replace('_', ' ').replace('PMBOK1.', '')
            sousClient.append(i)

        respTitle = []
        respAnn = []
        for i in sousClient:
            respTitle.append(i)
            respAnn.append(''.join(list(
                df_concept[((df_concept['individuals'] == i) & (df_concept['concept'] == classe[0]))]['annotation'])))
        import pandas as pd
        df = pd.read_excel('C:/Users/ASUS/Notebooks/Projet 5eme//dict.xlsx')

        page = list(df[df['classe'] == classe[0]]['page'])[0]
        pagepr = list(df[df['classe'] == classe[0]]['pageprocess'])[0]
        process = list(df[df['classe'] == classe[0]]['process'])[0]
        if request.is_ajax():
            return JsonResponse({'classe':classe[0],'Titles': respTitle
                                    ,'Ann':respAnn,'page':page,'pagepr':pagepr,'process':process})

    return render(request, "recom.html")




def index(request):
    return render(request, "main.html")
