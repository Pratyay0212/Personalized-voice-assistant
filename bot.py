import json
import nltk
import numpy
import random
import tensorflow
import tflearn
import pickle
import speech_recognition as sr
from gtts import gTTS
import os
import webbrowser
import playsound
import time
import datetime
import smtplib



from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')
stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


    
def speech_rec():
        r= sr.Recognizer()
        with sr.Microphone() as source:
            audio=0
            r.adjust_for_ambient_noise(source)
            print("please say something")
            audio=r.listen(source)
            time.sleep(3)
            if(audio==0):
                speech_rec()

        inp = r.recognize_google(audio)
        audio=0
        return str(inp)

def chat():
    

    output=gTTS(text='Hello again. How can I help You?', lang='en', slow=False)
    output.save("output.mp3")
    playsound.playsound("output.mp3")
    os.remove("output.mp3")
    while True:

        
        inp=speech_rec()
        

        if 'search' in inp:
            output1=gTTS(text='what do you want to search', lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")
            print("say")
            inp1=speech_rec()
            if inp1=='ABORT':
                return
            webbrowser.open('http://google.com/?#q='+ inp1)
                
        elif 'play music' in inp:
            output1=gTTS(text='what do you want to search', lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")
            inp1=speech_rec()
            if inp1=='ABORT':
                return
            webbrowser.open('https://www.youtube.com/results?search_query='+ inp1)

        elif 'date' in inp:
            x = datetime.datetime.now()
            output1=gTTS(text='Todays date is:'+  str(x.day) + ' '+ str(x.month)+ ' ' + str(x.year)  , lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")


        elif 'location' in inp:
            output1=gTTS(text='what do you want to search', lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")
            inp1=speech_rec()
            if inp1=='ABORT':
                return
            webbrowser.open("http://www.google.com/maps/place/"+inp1)

        elif 'send email' in inp:
            output1=gTTS(text='to whom do you want to send this email', lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")

            inp1=speech_rec()
            
            inp1=inp1.replace(" ", "")
            inp1=inp1.lower()
            if inp=='ABORT':
                return
            output1=gTTS(text='is there any number after email', lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")
            inp2=speech_rec()
            if 'one' in inp2:
                inp2=inp2.replace("one","1")
            if 'two' in inp2:
                inp2=inp2.replace("two","2")
            if 'three' in inp2:
                inp2=inp2.replace("three","3")
            if 'four' in inp2:
                inp2=inp2.replace("four","4")
            if 'five' in inp2:
                inp2=inp2.replace("five","5")
            if 'six' in inp2:
                inp2=inp2.replace("six","6")
            if 'seven' in inp2:
                inp2=inp2.replace("seven","7")
            if 'eight' in inp2:
                inp2=inp2.replace("eight","8")
            if 'nine' in inp2:
                inp2=inp2.replace("nine","9")
            if 'no' in inp2:
                inp2=inp2.replace("no","")
            inp2=inp2.replace(" ", "")
            print(inp1+inp2)
            output1=gTTS(text='what is the message', lang='en', slow=False)
            output1.save("output1.mp3")
            playsound.playsound("output1.mp3")
            os.remove("output1.mp3")
            inp3=speech_rec()

            server= smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login("pratyaymazumdar1006@gmail.com","pratyay@007")
            server.sendmail("pratyaymazumdar1006@gmail.com", inp1+inp2+"@gmail.com",inp3)
            server.quit()

                
        else:
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            language='en'
            if (results[results_index] > 0.7):
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                output1=gTTS(text=random.choice(responses), lang=language, slow=False)
                output1.save("output1.mp3")
                playsound.playsound("output1.mp3")
                os.remove("output1.mp3")

            
while(1):

    chat()