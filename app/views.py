from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Employee, Detected
from .forms import EmployeeForm
import cv2
import pickle
import face_recognition
import datetime
from cachetools import TTLCache
import pandas as pd
from django.conf import settings
import csv
import gmplot
import numpy as np
import pytesseract
import os
import re
vid = cv2.VideoCapture(0)

cache = TTLCache(maxsize=20, ttl=60)
# with open('innovators.csv', 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["date", "name", "lat", "lon"])

def identify1(frame, name, buf, buf_length, known_conf):

    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)


    if count >= known_conf:
        timestamp = datetime.datetime.now(tz=timezone.utc)
        print(name, timestamp)
        cache[name] = 'detected'
        path = 'detected/{}_{}.jpg'.format(name, timestamp)
        write_path = 'media/' + path
        cv2.imwrite(write_path, frame)
        latt=37.773097
        long=122.471789
        with open('innovators.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp,name,latt,long])
        try:
            emp = Employee.objects.get(name=name)
            emp.detected_set.create(time_stamp=timestamp, photo=path)

        except:
            pass




def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(closest_distances)
    #print(knn_clf.predict_proba(face_encodings))
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



def identify_faces(video_capture):

    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = vid.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="app/facerec/models/trained_model.clf")
            # print(predictions)

        process_this_frame = not process_this_frame

        face_names = []

        for name, (top, right, bottom, left) in predictions:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            identify1(frame, name, buf, buf_length, known_conf)

            face_names.append(name)

        buf[i] = face_names
        i = (i + 1) % buf_length


        # print(buf)


        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    vid.release()
    cv2.destroyAllWindows()


def n_plate1(request):
    return render(request, 'app/vehicle.html')



def n_plate(request):
    str = request.POST['nplate']
    str = str.replace(" ","")
    str = str.split(',')
    # img='app/cc1.jpg'
    # carplate_img = cv2.imread(img)
    # isFile = os.path.isfile(img)
    # print(isFile)
    # cv2.imshow('image', carplate_img)
    # carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
        # Function to enlarge the plt display for user to view more clearly
    # def enlarge_plt_display(image, scale_factor):
    #     width = int(image.shape[1] * scale_factor / 100)
    #     height = int(image.shape[0] * scale_factor / 100)
    #     dim = (width, height)
    #     plt.figure(figsize = dim)
    #     plt.axis('off')
    #     plt.imshow(image)
    #
    # carplate_haar_cascade = cv2.CascadeClassifier('app/haarcascade_russian_plate_number.xml')
    # # carplate_haar_cascade = cv2.CascadeClassifier('./haarcascade_licence_plate_rus_16stages.xml')
    #
    # def carplate_detect(image):
    #     carplate_overlay = image.copy() # Create overlay to display red rectangle of detected car plate
    #     carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1, minNeighbors=5)
    #
    #     for x,y,w,h in carplate_rects:
    #         cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (255,0,0), 5)
    #
    #     return carplate_overlay
    #
    # def carplate_extract(image):
    #
    #     carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    #
    #     for x,y,w,h in carplate_rects:
    #          carplate_img = image[y+15:y+h-10 ,x+15:x+w-20]
    #
    #     return carplate_img
    #
    # # Enlarge image for further image processing later on
    # def enlarge_img(image, scale_percent):
    #     width = int(image.shape[1] * scale_percent / 100)
    #     height = int(image.shape[0] * scale_percent / 100)
    #     dim = (width, height)
    #     resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #     return resized_image



    carplate_haar_cascade = cv2.CascadeClassifier('app/haarcascade_russian_plate_number.xml')
    while True:
        ret, img = vid.read()

        # Display the resulting frame
        # cv2.imshow('frame', frame)
        #  carplate_img = cv2.imread(frame)
        carplate_img = img
        carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
        carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb,scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in carplate_rects:
             carplate_img = carplate_img_rgb[y+15:y+h-10 ,x+15:x+w-20]
        #carplate_extract_img = carplate_extract(carplate_img_rgb)
        scale_percent = 150
        width = int(carplate_img.shape[1] * scale_percent / 100)
        height = int(carplate_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(carplate_img, dim, interpolation = cv2.INTER_AREA)
        #carplate_extract_img = enlarge_img(carplate_extract_img, 150)
        carplate_extract_img_gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)
        cv2.imshow("carplate_img", carplate_img)

        plate = pytesseract.image_to_string(carplate_extract_img_gray_blur,config=f"--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",)
        plate = re.sub('[\W_]+', '', plate)
        if len(plate)>=9:
            print(plate)
            for k in str :
                if k in plate :
                    timestamp = datetime.datetime.now()
                    timestamp = timestamp.strftime("%d-%m-%y_%H:%M:%S")
                    timestamp = timestamp.strip()
                    print(plate, timestamp)
                    cache[plate] = 'v_detected'
                    path = 'v_detected/{}_{}.jpg'.format(plate,timestamp)
                    write_path = 'media/' + path
                    cv2.imwrite(write_path,img)
                    latt=37.773097
                    long=122.471789
                    with open('vehicle.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp,plate,latt,long,path])
                    vid.release()
                    cv2.destroyAllWindows()
                    return render(request,'app/res.html',{'pn':plate,'timestamp':timestamp,'latt':latt,'long':latt})




        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    video_capture.release()
    cv2.destroyAllWindows()


def v_view(request):
    with open('vehicle.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        a= []
        for row in reader:
            dict =[]
            dict.append(row[0])
            dict.append(row[1])
            dict.append(row[2])
            dict.append(row[3])
            dict.append(row[4])
            a.append({'timestamp':row[0],'plate':row[1],'latt':row[2],'long':row[3],'path':row[4]})
            print(dict)

    return render(request,'app/res1.html',{'a':a[1:]})








def map(request):
    strr = request.POST['myinput']
    type = request.POST['type']
    if type == 'person' :
        c = 'innovators.csv'
    else :
        c = 'vehicle.csv'

    with open(c, 'a', newline='') as file:
        writer = csv.writer(file)
    apikey = '' # (your API key here)
    gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13, apikey=apikey)

    b=[]
    with open(c) as csvf:
        csv_reader = csv.reader(csvf, delimiter=',')
        for row in csv_reader:
    	       if(row[1]==strr):
    		             b.append((float(row[2]),float(row[3])))

    print(b)
    print(b[0])
    path = zip(*b)
    var1,var2 = zip(*b)

    gmap.plot(*path, edge_width=2, color='red')
    gmap.scatter(var1,var2, color='#7CFC00	', size=40, marker=False)
    gmap.draw('/home/phoenix/Godseye/app/templates/map.html')
    return render(request,'/home/phoenix/Godseye/app/templates/map.html')



def index(request):
    return render(request, 'app/index.html')

def prev_map(request):
    return render(request, 'app/prev_map.html')

def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))


def add_photos(request):
	emp_list = Employee.objects.all()
	return render(request, 'app/add_photos.html', {'emp_list': emp_list})


def click_photos(request, emp_id):
	cam = cv2.VideoCapture(0)
	emp = get_object_or_404(Employee, id=emp_id)
	click(emp.name, emp.id, cam)
	return HttpResponseRedirect(reverse('add_photos'))


def train_model(request):
	trainer()
	return HttpResponseRedirect(reverse('index'))


def detected(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected.objects.filter(time_stamp__date=date_formatted).order_by('time_stamp').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})


def identify(request):
	video_capture = cv2.VideoCapture(0)
	identify_faces(video_capture)
	return HttpResponseRedirect(reverse('index'))


def add_emp(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        if form.is_valid():
            emp = form.save()
            # post.author = request.user
            # post.published_date = timezone.now()
            # post.save()
            return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})
