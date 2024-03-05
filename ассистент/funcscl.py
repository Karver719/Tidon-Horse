import sqlite3
import config

conn = sqlite3.connect(config.path)
c = conn.cursor()

def CreateIMGTable():
    # Создаем таблицу картинок
    c.execute('''CREATE TABLE IF NOT EXISTS images (
                    img_id INTEGER PRIMARY KEY,
                    keyWord TEXT,
                    imgLink TEXT
                )''')

def CreateRequestsTable():
    # Создаем таблицу с внешним ключом
    c.execute('''CREATE TABLE IF NOT EXISTS requests
                 (req_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 gifNum INTEGER,
                 keyWord TEXT
                )''')

def Fill_keyword_ids(data_set):
    for keyword in data_set.values():
        # Вставляем ключевое слово в таблицу keyword
        c.execute('INSERT INTO images (keyword) VALUES (?)', (keyword,))
    conn.commit()

def UpdateLink(key, link):
    # Обновляем столбец imgLink в таблице images, если значение keyword_id совпадает с arg1
    c.execute('''UPDATE images
                 SET imgLink = ?
                 WHERE keyword = ?''', (link, key))
    conn.commit()

def inBase(num, keyword):
    c.execute("INSERT INTO requests (gifNum, keyWord) VALUES (?, ?)", (num, keyword,))
    conn.commit()

def lastRow():
    #ID последней добавленной строки
    last_row_id = c.lastrowid
    c.execute("SELECT gifNum, keyWord FROM requests WHERE req_id=?", (last_row_id,))
    # Получаем последнюю добавленную строку
    last_row = c.fetchone()
    print('Последний добавленный запрос:', last_row)

# def getGifnum():
#     # Получаем ID последней строки в таблице
#     c.execute("SELECT MAX(req_id) FROM requests")
#     last_row_id = c.fetchone()[0]
#     # Выполняем запрос SELECT для получения значения столбца "gifNum" из последней строки
#     c.execute("SELECT gifNum FROM requests WHERE req_id = ?", (last_row_id,))
#     gifNum_value = c.fetchone()[0]
#     # Выводим значение столбца "gifNum" в последней строке
#     return int(gifNum_value)

# def getKeyword():
#     # Получаем ID последней строки в таблице
#     c.execute("SELECT MAX(req_id) FROM requests")
#     last_row_id = c.fetchone()[0]
#     # Выполняем запрос SELECT для получения значения столбца "keyWord" из последней строки
#     c.execute("SELECT keyWord FROM requests WHERE req_id = ?", (last_row_id,))
#     gifNum_value = c.fetchone()[0]
#     # Выводим значение столбца "keyWord" в последней строке
#     return gifNum_value

# def getLink():
#     c.execute("SELECT MAX(req_id) FROM requests")
#     last_row_id = c.fetchone()[0]
#     c.execute("SELECT imgLink FROM images, requests "
#               "WHERE requests.keyWord=images.keyWord AND req_id = ?", (last_row_id,))
#     # Извлекаем результат запроса
#     link_value = c.fetchone()
#     if link_value is None:
#         return 'Неизвестно'
#     else:
#         return link_value[0]

# def allData():
#     key = getKeyword()
#     link = getLink()
#     if link == 'Неизвестно':
#         num = 10
#     else:
#         num = getGifnum()
#     return num, key, link
