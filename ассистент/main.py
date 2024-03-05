import config
import tts
import datetime
from fuzzywuzzy import fuzz
from num2words import num2words
import sounddevice as sd  
import words
from skills import *
from sklearn.feature_extraction.text import CountVectorizer     #pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import vosk 
import queue
import json
import random
from funcscl import inBase
from funcscl import CreateRequestsTable, CreateIMGTable, Fill_keyword_ids, UpdateLink, lastRow
from words import data_set
import omegaconf
# i=1
# CreateIMGTable()
# CreateRequestsTable()
# Fill_keyword_ids(data_set)
# UpdateLink('поздоровайся', f'D\sourse\img{i}')
# i=1
# for keyw in data_set.keys():
#     UpdateLink(keyw, f'D\sourse\img{i}')
#     i += 1




model = vosk.Model('model_small')
q = queue.Queue()
samplerate = 16000
device = sd.default.device
def q_callback(indata, frames, time, status):
    
    q.put(bytes(indata))

def recognize(data, vectorizer, clf):
    '''
    Анализ распознанной речи
    '''

    #проверяем есть ли имя бота в data, если нет, то return
    trg = words.TRIGGERS.intersection(data.split())
    if not trg:
        return

    #удаляем имя бота из текста
    data.replace(list(trg)[0], '')

    #получаем вектор полученного текста
    #сравниваем с вариантами, получая наиболее подходящий ответ
    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]
    inBase(random.randint(1, 100), answer)
    # lastRow()
    #получение имени функции из ответа из data_set
    func_name = answer.split()[0]
    
    #озвучка ответа из модели data_set
    tts.va_speak(answer)

    #запуск функции из skills
    #exec(func_name + '()')



def main():
    '''
    Обучаем матрицу ИИ
    и постоянно слушаем микрофон
    '''

    #Обучение матрицы на data_set модели
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(words.data_set.keys()))
    
    clf = LogisticRegression()
    clf.fit(vectors, list(words.data_set.values()))

    del words.data_set

    #постоянная прослушка микрофона
    with sd.RawInputStream(samplerate=samplerate, blocksize = 16000, device=device[0], dtype='int16',
                                channels=1, callback=q_callback):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                data = json.loads(rec.Result())['text']
                recognize(data, vectorizer, clf)
                
            else:
                 print(rec.PartialResult())
                #  print(rec.FinalResult())
                #  print(json.loads(rec.Result())['text'])
                 


if __name__ == '__main__':
    main()

