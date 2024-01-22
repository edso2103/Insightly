import flet as ft
import speech_recognition as sr
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from unidecode import unidecode as unidecode_func
from nltk.corpus import stopwords
from sentiment_analysis_spanish import sentiment_analysis
from wordcloud import WordCloud
import nltk
import os
import onnxruntime as ort
from transformers import pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
import cv2
import numpy as np
import time
from tf2onnx import convert
import tensorflow as tf
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode as unidecode_func
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import shutil
import pygetwindow as gw
import pyautogui
import soundfile as sf

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
stemmer = SnowballStemmer("spanish")
tokenizer = RegexpTokenizer(r'\w+')
detokenizer = TreebankWordDetokenizer()
#############################ANALISIS DE ENTREVISTAS###############################
directorio_actual = os.getcwd()
textos = []
data = {"Numero de Texto": [], "Transcripcion": [], "Transcripcion Procesada": [],
        "Resumen": [], "Palabras": [], "Sentimiento": [], "Categoria": []}

# Inicializar objetos
r = sr.Recognizer()
stop_words = set(stopwords.words("spanish"))
lemmatizer = WordNetLemmatizer()
path=None

def extraer_palabras_clave(texto):
    tokens = word_tokenize(texto, language='spanish')
    tagged_tokens = pos_tag(tokens)
    palabras_clave = [token[0].lower() for token in tagged_tokens
                  if token[1] in ('NN', 'JJ', 'VB') and token[0].lower() not in stop_words
                  and token[1] != 'IN']
    return list(set(palabras_clave))

def procesar_texto(texto):
    tokens = word_tokenize(texto, language='spanish')
    lemmatized_text = " ".join([stemmer.stem(token.lower()) for token in tokens])
    palabras_procesadas = [unidecode_func(word.lower()) for word in tokenizer.tokenize(lemmatized_text)
                           if unidecode_func(word.lower()) not in stop_words]
    return detokenizer.detokenize(palabras_procesadas)

def resumir(texto, numero_oraciones=1):
    parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
    summarizer = LsaSummarizer()
    resumen = summarizer(parser.document, numero_oraciones)
    resumen_texto = " ".join([str(oracion) for oracion in resumen])
    return resumen_texto

classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
def categorizar_sentimiento(texto, classifier):
    results = classifier(texto)
    for result in results:
        star_rating = result['label']
        if star_rating in ['1 star', '2 stars']:
            return 'negativo'
        elif star_rating == '3 stars':
            return 'neutro'
        elif star_rating in ['4 stars', '5 stars']:
            return 'positivo'
        
def generar_nube_palabras(texto):
    # Realizar análisis de partes del discurso
    tokens = nltk.word_tokenize(texto)
    pos_tags = nltk.pos_tag(tokens)

    # Filtrar adjetivos, adverbios, verbos y sustantivos
    selected_tags = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS']
    filtered_words = [word for word, pos in pos_tags if pos in selected_tags]


    # Crear la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

    # Guardar la nube de palabras como imagen
    nombre_imagen = os.path.join(directorio_actual, 'nube_palabras.png')
    wordcloud.to_file(nombre_imagen)

    return nombre_imagen

def generar_grafica_y_guardar(resumen,texto_c,textos_,textos_p):
    
    # Guardar el gráfico en un archivo HTML
    ruta_nube_palabras = generar_nube_palabras(texto_c)
    nombre='analisis_entrevista.html'
    nombre_archivo = os.path.join(directorio_actual, nombre)

    with open(nombre_archivo, 'w', encoding='utf-8') as file:
        file.write(f"<img src='{ruta_nube_palabras}' alt='Nube de Palabras'>")
        file.write("<br><br><h2>Detalles de cada texto:</h2>")
        for i in range(len(data["Numero de Texto"])):
            file.write(f"<p><b>Texto:</b></p>")
            file.write(f"<p><b>Transcripción:</b> {data['Transcripcion'][i]}</p>")
            file.write(f"<p>Sentimiento: {data['Sentimiento'][i]}</p>")
            file.write("<br>")
        file.write(f"<p><b>Resumen:</b> {resumen}</p>")
        file.write(f"<p><b>Texto Completo:</b> {textos_}</p>")
        file.write(f"<p><b>Trancripción Procesada:</b> {textos_p}</p>")

def generar_grafica_y_guardar_(resumen,texto_c):
   
    # Guardar el gráfico en un archivo HTML
    ruta_nube_palabras = generar_nube_palabras(texto_c)
    nombre='analisis_entrevista.html'
    nombre_archivo = os.path.join(directorio_actual, nombre)

    with open(nombre_archivo, 'w', encoding='utf-8') as file:
        file.write(f"<img src='{ruta_nube_palabras}' alt='Nube de Palabras'>")
        file.write("<br><br><h2>Detalles de cada texto:</h2>")
        for i in range(len(data["Numero de Texto"])):
            file.write(f"<p><b>Texto:</b></p>")
            file.write(f"<p><b>Transcripción:</b> {data['Transcripcion'][i]}</p>")
            file.write(f"<p>Sentimiento: {data['Sentimiento'][i]}</p>")
            file.write("<br>")
        file.write(f"<p><b>Resumen:</b> {resumen}</p>")

    
def transcribe_audio(path):
    
    with sr.AudioFile(path) as source:
        audio_text = r.listen(source)
        try:
            texto = r.recognize_google(audio_text, language="es-ES")
            return texto
        except (sr.UnknownValueError, sr.RequestError) as e:
            print(f"Error en la transcripción {path}: {str(e)}")
            return ""
        
def get_large_audio_transcription_on_silence(path):
    sound = AudioSegment.from_file(path,format='mp4')
    chunks = split_on_silence(sound,
                              min_silence_len=2000,
                              silence_thresh=sound.dBFS - 14,
                              keep_silence=2000,
                              )
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        text = transcribe_audio(chunk_filename)
        if text:
            text = f"{text.capitalize()}. "
            textos.append(text)
    return textos

def grabar_vivo():
    
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        texto = r.recognize_google(audio, language="es-ES")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

    return texto

def analizar_vivo(texto):
    
    texto_procesado = procesar_texto(texto)
    if len(texto_procesado.split()) > 3:
        sentimiento = categorizar_sentimiento(texto,classifier)
        resumen = resumir(texto)
        palabras_clave_oracion = extraer_palabras_clave(texto)
        palabras=' '.join(palabras_clave_oracion)
        data["Numero de Texto"].append(1)
        data["Transcripcion"].append(texto)
        data["Transcripcion Procesada"].append(texto_procesado)
        data["Sentimiento"].append(sentimiento)

        generar_grafica_y_guardar_(resumen,palabras)

def archivo(path):
    _, extension = os.path.splitext(path)
    extension = extension.lower()[1:]
    if extension in ['mp4', 'avi', 'mpeg', 'wvm', 'mov', 'flv', 'webm', 'ogv']:
        audio = AudioSegment.from_file(path)
        audio.export('audio_temp.wav', format="wav")
        

    else:
        data, samplerate = sf.read(path)
        sf.write('audio_temp.wav', data, samplerate)
    
    p = os.path.join(os.getcwd(), "audio_temp.wav")
    textos = get_large_audio_transcription_on_silence(p)
    return textos

def analizar_archivo(textos):   

    palabras_clave = []
    procesado=[]
    for i, oracion in enumerate(textos, start=1):
    
        texto_procesado = procesar_texto(oracion)
        procesado.append(texto_procesado)
        if len(texto_procesado.split()) > 3:
            sentimiento = categorizar_sentimiento(oracion,classifier)
            palabras_clave_oracion = extraer_palabras_clave(oracion)
            palabras_clave.extend(palabras_clave_oracion)

            data["Numero de Texto"].append(i)
            data["Transcripcion"].append(oracion)
            data["Transcripcion Procesada"].append(texto_procesado)
            data["Sentimiento"].append(sentimiento)
    
    palabras = ' '.join(palabras_clave_oracion)
    
    
    textos_=(' '.join(textos))
    resumen=resumir(textos_)
    textos_p=(' '.join(procesado))
    generar_grafica_y_guardar(resumen,palabras,textos_,textos_p)


##########################RECONOCIMIENTO DE EXPRESIONES###############################
def reconocimiento():
    face_detection = cv2.CascadeClassifier(directorio_actual + "/paquetes/haar_cascade_face_detection.xml")

    labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
    model_tf = tf.keras.models.load_model(directorio_actual + "/paquetes/network-5Labels.h5")
    onnx_model, _ = convert.from_keras(model_tf)
    onnx_path = directorio_actual + "/paquetes/network-5Labels.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    model = ort.InferenceSession(directorio_actual + "/paquetes/network-5Labels.onnx")


    tiempos_emociones = {emocion: 0 for emocion in labels}

    cv2.namedWindow('Facial Expression', cv2.WINDOW_NORMAL)

    num_iteraciones_sin_rostros = 0
    max_iteraciones_sin_rostros = 10

    while True:
        inicio_frame = time.time()
        screenshot = np.array(pyautogui.screenshot())
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        detected = face_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        if len(detected) > 0:
            num_iteraciones_sin_rostros = 0
        else:
            num_iteraciones_sin_rostros += 1

        for x, y, w, h in detected:
            cv2.rectangle(screenshot, (x, y), (x+w, y+h), (245, 135, 66), 2)
            cv2.rectangle(screenshot, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
            face = gray[y+5:y+h-5, x+20:x+w-20]
            face = cv2.resize(face, (48, 48))
            face = face/255.0
            arr = np.array([face.reshape((48, 48, 1))])

            # Realizar la inferencia utilizando onnxruntime
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            prediction = model.run([output_name], {input_name: arr.astype(np.float32)})[0]

            predictions = np.argmax(prediction)
            state = labels[predictions]
            tiempo_actual = time.time() - inicio_frame
            tiempos_emociones[state] += tiempo_actual

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(screenshot, state, (x+10, y+15), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Facial Expression', screenshot)

        key = cv2.waitKey(1)
        if key == ord('q') or num_iteraciones_sin_rostros >= max_iteraciones_sin_rostros:
            break

    cv2.destroyAllWindows()

    tiempos = [tiempos_emociones[emocion] for emocion in labels]
    return tiempos



####################################INTERFAZ#######################################
def main(page: ft.Page):
    archivo_a_borrar='audio_temp.wav'
    if os.path.exists(archivo_a_borrar):
        os.remove(archivo_a_borrar)
        print(f"El archivo '{archivo_a_borrar}' ha sido eliminado.")
    else:
        print(f"El archivo '{archivo_a_borrar}' no existe.")

    carpeta = 'audio-chunks'
    try:
        shutil.rmtree(carpeta)
        print(f'Carpeta {carpeta} eliminada exitosamente.')
    except Exception as e:
        print(f'Error al eliminar la carpeta {carpeta}: {e}')

    page.window_width = 1600        
    page.window_height = 1000   
    page.title = "Insightly"
    page.bgcolor=ft.colors.WHITE
    
    def grabando(e):
        if e.control.icon=='play_circle_filled':
            
            e.control.icon='pause_circle_filled'
            e.control.text='Grabando'
            texto1.value = 'Grabando...'
            texto1.update()
            page.update()
            text_=grabar_vivo()
            texto1.value = text_
            texto1.update()

    
    def analisis_vivo(e):
        try:
       
            if not texto1.value:
                
                msg="Sin grabación para analizar "
                dlg=ft.AlertDialog(title=ft.Text('Aviso'),content=ft.Text(msg))
                page.dialog=dlg
                dlg.open=True
                page.update()
            else:
                
                analizar_vivo(texto1.value)
                
                msg="Análisis realizado y guardado en "+directorio_actual
                dlg=ft.AlertDialog(title=ft.Text('Aviso'),content=ft.Text(msg))
                page.dialog=dlg
                dlg.open=True
                page.update()
        except Exception as e:
            mostrar_alerta(f"Error al realizar análisis: {str(e)}")

    def analisis_texto(e):

        try:
       
            if not texto2.value:
                
                msg="Sin grabación para analizar "
                dlg=ft.AlertDialog(title=ft.Text('Aviso'),content=ft.Text(msg))
                page.dialog=dlg
                dlg.open=True
                page.update()
            else:
                
                analizar_archivo(textoos)
                msg="Análisis realizado y guardado en "+directorio_actual
                dlg=ft.AlertDialog(title=ft.Text('Aviso'),content=ft.Text(msg))
                page.dialog=dlg
                dlg.open=True
                page.update()
        except Exception as e:
            mostrar_alerta(f"Error al realizar análisis: {str(e)}")

            
       

        
    texto1=ft.Text(size=20)
    texto2=ft.Text(size=20)
    
    def pick_files_result(e: ft.FilePickerResultEvent):
        try:
            global textos
            global data
            textos = []
            data = {"Numero de Texto": [], "Transcripcion": [], "Transcripcion Procesada": [],
                "Resumen": [], "Palabras": [], "Sentimiento": [], "Categoria": []}
    
            global path
            global textoos
            
            if e.files:
                path=e.files[0].path
                selected_files.value = ", ".join(map(lambda f: f.name, e.files))
                selected_files.update()
                texto2.value = 'Realizando transcripción. Espere un momento ...'
                texto2.update()
                textoos=archivo(path)
                completo=' '.join(textoos)
                texto2.value = completo
                texto2.update()
            else:
                path=None
                selected_files.value = "Cancelado"
                text_=''
                texto2.value = text_
                texto2.update()
                selected_files.update()
        except Exception as e:
            mostrar_alerta(f"Error al realizar análisis: {str(e)}")

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text(size=20)

    

    def graficar():

        try:
            grafico.content=ft.Column([
                                ft.Container(alignment=ft.alignment.center,width=350,height=100),
                                ft.Row([ft.Text('Espere un momento....\n\nRecuerde abrir iniciar la reunión en zoom \n la cámara se cerrará sino encuentra ningun rostro',size=20),
                                    ],alignment=ft.MainAxisAlignment.CENTER),
                                ft.Row([
                                    ft.ProgressBar(width=400, color="white", bgcolor=ft.colors.AMBER)],alignment=ft.MainAxisAlignment.CENTER)])

            grafico.update()
            labels=['Sorpresa','Neutro','Enojado','Feliz','Triste']
            valores=reconocimiento()

            chart = ft.BarChart(
                bar_groups=[
                    ft.BarChartGroup(
                        x=i,
                        bar_rods=[
                            ft.BarChartRod(
                                from_y=0,
                                to_y=val,
                                width=40,
                                color=ft.colors.AMBER_300,
                                tooltip=label,
                                border_radius=0,
                            ),
                        ],
                    ) for i, (label, val) in enumerate(zip(labels, valores))
                ],
                border=ft.border.all(1, ft.colors.GREY_400),
                left_axis=ft.ChartAxis(
                    labels_size=40, title=ft.Text("Tiempo en segundos"), title_size=40
                ),
                bottom_axis=ft.ChartAxis(
                    labels=[
                        ft.ChartAxisLabel(
                            value=i, label=ft.Container(ft.Text(label))
                        ) for i, label in enumerate(labels)
                    ],
                    labels_size=40,title=ft.Text("")
                ),
                horizontal_grid_lines=ft.ChartGridLines(
                    color=ft.colors.GREY_300, width=1, dash_pattern=[3, 3]
                ),
                tooltip_bgcolor=ft.colors.with_opacity(0.5, ft.colors.GREY_300),
                max_y=max(valores) + 10,
                interactive=True,
                expand=True,
            )
            chart_container = ft.Container(content=chart)
            grafico.content = chart_container
            grafico.bgcolor=ft.colors.GREY_50
            grafico.update()

        except Exception as e:
            mostrar_alerta(f"Error al realizar análisis: {(str(e))}")
        
    grafico= ft.Container(
                    content=ft.Text(''),
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.AMBER_200,
                    width=700,
                    height=400,
                    border_radius=10,padding=10)
    
    def route_change(route):
        t_esfuerzos.value=""
        t_dice.value=""
        t_result.value=""
        t_oye.value=""
        t_ve.value=""
        t_pienso.value=""
        texto1.value=""
        texto2.value=""
        textoos=''
        completo=''
        path=None
        grafico.content=ft.Text('')
        selected_files.value=''
        
        page.views.clear()
        page.views.append(
            ft.View(
                "/",
                [
                    ft.AppBar(
                        leading_width=10,
                        toolbar_height=200,
                        title=ft.Text(
                            "\n\nExplora la colección de herramientas\n      para analizar las entrevistas",
                            size=40,
                            color=ft.colors.BLACK,
                            weight=ft.FontWeight.BOLD,
                            italic=True,

                        ),
                        center_title=True,
                        bgcolor=ft.colors.WHITE
                        
                    ),
                    ft.Column([row_with_alignment(ft.MainAxisAlignment.CENTER,items_3(),50),row_with_alignment(ft.MainAxisAlignment.CENTER,items_2(),70),
                               row_with_alignment(ft.MainAxisAlignment.CENTER,items_3(),50),row_with_alignment(ft.MainAxisAlignment.CENTER,items_1(),160)],spacing=0),
                   
                    
                ],bgcolor=ft.colors.WHITE,
                
            )
        )
        if page.route == '/entrevistas_grab':
            page.overlay.extend([pick_files_dialog])
            page.views.append(
                ft.View(
                    "/entrevistas_grab",  
                    [
                        ft.AppBar(
                            toolbar_height=200,
                            title=ft.Text(
                                "Analizador de grabaciones",
                                size=40,
                                color=ft.colors.BLACK,
                                weight=ft.FontWeight.BOLD,
                                italic=True,
                            ),
                            center_title=True,
                            bgcolor=ft.colors.WHITE
                        ),
                        
                        
                        ft.Column([row_with_alignment(ft.MainAxisAlignment.CENTER,items_4(),50),
                                   row_with_alignment(ft.MainAxisAlignment.CENTER,items_5(),200),row_with_alignment(ft.MainAxisAlignment.CENTER,items_6(),200)],spacing=20),
                        
                    ],bgcolor=ft.colors.WHITE,
                )
            )
        elif page.route == '/entrevistas_vivo':
            page.views.append(
                ft.View(
                    "/entrevistas_vivo",  
                    [
                        ft.AppBar(
                            toolbar_height=200,
                            title=ft.Text(
                                "Analizador en vivo",
                                size=40,
                                color=ft.colors.BLACK,
                                weight=ft.FontWeight.BOLD,
                                italic=True,
                            ),
                            center_title=True,
                            bgcolor=ft.colors.WHITE
                        ),
                        
                        
                        ft.Column([row_with_alignment(ft.MainAxisAlignment.CENTER,items_44(),50),
                                   row_with_alignment(ft.MainAxisAlignment.CENTER,items_55(),200),row_with_alignment(ft.MainAxisAlignment.CENTER,items_66(),200)],spacing=20),
                        
                    ],bgcolor=ft.colors.WHITE
                )
            )
        elif page.route == '/expresiones':
            page.views.append(
                ft.View(
                    "/expresiones", 
                    [
                        ft.AppBar(
                            toolbar_height=200,
                            title=ft.Text(
                                "Reconocimiento de expresiones",
                                size=40,
                                color=ft.colors.BLACK,
                                weight=ft.FontWeight.BOLD,
                                italic=True,
                            ),
                            center_title=True,
                            bgcolor=ft.colors.WHITE
                            
                        ),
                        
                        ft.Column([row_with_alignment(ft.MainAxisAlignment.CENTER,items_8(),0),row_with_alignment(ft.MainAxisAlignment.CENTER,items_7(),200)],spacing=30)
                    ],bgcolor=ft.colors.WHITE
                )
            )
        elif page.route == '/notas':
            page.views.append(
                ft.View(
                    "/notas",  
                    [
                        ft.AppBar(
                            toolbar_height=200,
                            title=ft.Text(
                                "Analizador de notas",
                                size=40,
                                color=ft.colors.BLACK,
                                weight=ft.FontWeight.BOLD,
                                italic=True,
                            ),
                            center_title=True,
                            bgcolor=ft.colors.WHITE
                        ),
                        
                        ft.Column([row_with_alignment(ft.MainAxisAlignment.CENTER,items(400,150,ft.colors.AMBER_200),50), 
                                   row_with_alignment(ft.MainAxisAlignment.CENTER,items_0(),0),
                                   row_with_alignment(ft.MainAxisAlignment.CENTER,items_6_6(),0),
                                   row_with_alignment(ft.MainAxisAlignment.CENTER,itemm(),50)
                                 ])
                     
                        
                    ],bgcolor=ft.colors.WHITE,
                )
            )
      

    def view_pop(view):
        page.views.pop()
        top_view=page.views[-1]
        page.go(top_view.route)

    t_esfuerzos=ft.TextField(label='Esfuerzos: ',border_color=ft.colors.TRANSPARENT,max_lines=5,multiline=True)
    t_dice=ft.TextField(label='¿Qué dice y hace?: ',border_color=ft.colors.TRANSPARENT,max_lines=5,multiline=True)
    t_result=ft.TextField(label='Resultados: ',border_color=ft.colors.TRANSPARENT,max_lines=5,multiline=True)
    t_oye=ft.TextField(label='¿Qué oye?: ',border_color=ft.colors.TRANSPARENT,max_lines=5,multiline=True)
    t_ve=ft.TextField(label='¿Qué ve?: ',border_color=ft.colors.TRANSPARENT,max_lines=5,multiline=True)
    t_pienso=ft.TextField(label='¿Qué piensa y siente?: ',border_color=ft.colors.TRANSPARENT,max_lines=5,multiline=True)
    

    def itemm():
        items = []
        items.append(
            ft.Container(
                content=t_esfuerzos,
                alignment=ft.alignment.center,
                width=400,
                height=150,
                bgcolor=ft.colors.AMBER_200,padding=10
            )
        )
        items.append(
            ft.Container(
                content=t_dice,
                alignment=ft.alignment.center,
                width=400,
                height=150,
                bgcolor=ft.colors.AMBER_200,padding=10
            )
        )
        items.append(
            ft.Container(
                content=t_result,
                alignment=ft.alignment.center,
                width=400,
                height=150,
                bgcolor=ft.colors.AMBER_200,padding=10
            )
        )

        return items
        

    def items(w,h,color):

        items = []
        items.append(
            ft.Container(
                content=t_pienso,
                alignment=ft.alignment.center,
                width=w,
                height=h,
                bgcolor=color,padding=10
            )
        )
        return items
    
    def items_0():
        items = []
        items.append(
            ft.Container(
                content=t_oye,
                alignment=ft.alignment.center,
                width=500,
                height=150,
                bgcolor=ft.colors.AMBER_200,padding=10
            )

        )
        items.append(
            ft.Image(
                src=r"\static\empatia.png",
                width=100,
                height=100,
                fit=ft.ImageFit.CONTAIN,
            )
        )

        items.append(
            ft.Container(
                content=t_ve,
                alignment=ft.alignment.center,
                width=500,
                height=150,
                bgcolor=ft.colors.AMBER_200,padding=10
            )

        )
        return items
    
      
    def items_1():
        items = []
        items.append(
            ft.ElevatedButton("Empezar",on_click=lambda _: page.go(f"/expresiones"),bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50),
        )
        items.append(
            ft.ElevatedButton("Empezar",on_click=lambda _: page.go(f"/entrevistas_grab"),bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50)
        )
        items.append(
            ft.ElevatedButton("Empezar",on_click=lambda _: page.go(f"/notas"),bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50)
        )
        items.append(
            ft.ElevatedButton("Empezar",on_click=lambda _: page.go(f"/entrevistas_vivo"),bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50)
        )
        return items
    

    def items_2():
        items = []
        items.append(
            ft.Card(
                content=ft.Container(
                    
                    content=ft.Column(
                        [
                        
                            ft.Container(
                                alignment=ft.alignment.center,
                                width=160,
                                height=160,
                                image_src=r"\static\reconocimiento.png",
                                bgcolor=ft.colors.TRANSPARENT,
                            ),
                    
                            
                            ft.ListTile(
                                title=ft.Text("Reconocimiento de expresiones",weight=ft.FontWeight.BOLD,text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK,),
                                subtitle=ft.Text(
                                    "¿Qué dice el rostro de tus usuarios?",text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK,
                                    
                                ),
                            ),
                            
                           
                        ]
                    ),
                    width=280,
                    height=350,
                    bgcolor=ft.colors.AMBER_300,
                    padding=50,
                    border_radius=20
                ),color=ft.colors.AMBER_300,
            ),
        )
        items.append(
            ft.Card(
                content=ft.Container(
                    
                    content=ft.Column(
                        [
                        
                            ft.Container(
                                alignment=ft.alignment.center,
                                width=160,
                                height=160,
                                image_src=r"\static\grabaciones.png",
                                bgcolor=ft.colors.TRANSPARENT,
                            ),
                    
                            
                            ft.ListTile(
                                title=ft.Text("Análisis de grabaciones",weight=ft.FontWeight.BOLD,text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK),
                                subtitle=ft.Text(
                                    "Sube la grabación de la entrevista y analízala",text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK
                                ),
                            ),
                        ]
                    ),
                    width=280,
                    height=350,
                    bgcolor=ft.colors.AMBER_300,
                    padding=50,
                    border_radius=20
                ),color=ft.colors.AMBER_300,
            )
        )
        items.append(
            ft.Card(
                content=ft.Container(
                    
                    content=ft.Column(
                        [
                        
                            ft.Container(
                                alignment=ft.alignment.center,
                                width=160,
                                height=160,
                                image_src=r"\static\notas.png",
                                bgcolor=ft.colors.TRANSPARENT,
                            ),
                    
                            
                            ft.ListTile(
                                title=ft.Text("Análisis de notas",weight=ft.FontWeight.BOLD,text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK),
                                subtitle=ft.Text(
                                    "Escribe, organiza y analiza tus notas",text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK
                                ),
                            ),
                        ]
                    ),
                    width=280,
                    height=350,
                    bgcolor=ft.colors.AMBER_300,
                    padding=50,
                    border_radius=20
                ),color=ft.colors.AMBER_300,
            )
        )
        items.append(
            ft.Card(
                    content=ft.Container(
                        
                        content=ft.Column(
                            [
                            
                                ft.Container(
                                    alignment=ft.alignment.center,
                                    width=160,
                                    height=160,
                                    image_src=r"\static\voz.png",
                                    bgcolor=ft.colors.TRANSPARENT,
                                ),
                        
                                
                                ft.ListTile(
                                    title=ft.Text("Análisis en vivo",weight=ft.FontWeight.BOLD,text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK),
                                    subtitle=ft.Text(
                                        "Documenta y analiza las sesiones en tiempo real",text_align=ft.TextAlign.CENTER,color=ft.colors.BLACK
                                    ),
                                ),
                            ]
                        ),
                        width=280,
                        height=350,
                        bgcolor=ft.colors.AMBER_300,
                        padding=50,
                        border_radius=20
                    ),color=ft.colors.AMBER_300,
                        
            )
        )
        return items
    def items_3():
        items = []
        items.append(
            ft.Container(
                alignment=ft.alignment.center,
                width=10,
                height=10,
            )
        )
        items.append(
            ft.Container(
                alignment=ft.alignment.center,
                width=50,
                height=50,
            )
        )
        return items
    
    def items_4():
        items = []
        items.append(
            ft.ElevatedButton(
                            "Subir entrevista",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: pick_files_dialog.pick_files(
                                allowed_extensions=['mp4','avi','mpeg','wvm','mov','flv','webm','ogv','mp3','m4a','wav'],
                                allow_multiple=False,
                                
                            ),
                            bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50
                        ),
        )
        items.append(
            selected_files
        )
        return items

    def items_44():
        items = []
        items.append(
            ft.ElevatedButton(
                "Grabar",
                icon=ft.icons.PLAY_CIRCLE_FILLED,
                on_click=grabando,bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50
                )
        )
        return items
    
    def items_5():
        items = []
        items.append(
            ft.Container(
                    content=texto2,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.AMBER_200,
                    width=800,
                    height=400,
                    border_radius=10,padding=40)
        )
        return items
    
    def items_55():
        items = []
        items.append(
            ft.Container(
                    content=texto1,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.AMBER_200,
                    width=800,
                    height=400,
                    border_radius=10,padding=40)
        )
        return items
    
    
    def analizar_texto(e):
        try:
            t_field = [
                {"label": 'Esfuerzos', "field": t_esfuerzos.value},
                {"label": '¿Qué dice y hace?', "field": t_dice.value},
                {"label": 'Resultados', "field": t_result.value},
                {"label": '¿Qué oye?', "field": t_oye.value},
                {"label": '¿Qué ve?', "field": t_ve.value},
                {"label": '¿Qué piensa y siente?', "field": t_pienso.value}
            ]
            with open('analisis_entrevista.html', 'w', encoding='utf-8') as file:
                file.write("<br><br><h2>Detalles de cada texto:</h2>")

                for i, text_data in enumerate(t_field, start=1):
                    label = text_data["label"]
                    field = text_data["field"]

                    texto_procesado = procesar_texto(field)
                    if len(texto_procesado.split()) >= 3:
                        sentimiento = categorizar_sentimiento(field,classifier)
                        palabras_clave_oracion = extraer_palabras_clave(field)

                        file.write(f"<h3>{label}</h3>")
                        file.write(f"<p><b>Transcripción:</b> {field}</p>")
                        file.write(f"<p><b>Sentimiento:</b> {sentimiento}</p>")
                        file.write(f"<p><b>Palabras Clave:</b> {', '.join(palabras_clave_oracion)}</p>")
                        file.write(f"<p><b>Resumen:</b> {resumir(field)}</p>")
                        file.write("<hr>")

            msg="Análisis realizado y guardado en "+directorio_actual
            dlg=ft.AlertDialog(title=ft.Text('Aviso'),content=ft.Text(msg))
            page.dialog=dlg
            dlg.open=True
            page.update()
        except Exception as e:
            mostrar_alerta(f"Error al realizar análisis{str(e)}")
    def items_6():
        items = []
        items.append(
            ft.ElevatedButton("Analizar entrevista",on_click=analisis_texto,bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50)
        )
        return items
    def items_6_6():
        items = []
        items.append(
            ft.ElevatedButton("Analizar entrevista",on_click=analizar_texto,bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50)
        )
        return items
    
    def items_66():
        items = []
        items.append(
            ft.ElevatedButton("Analizar entrevista",on_click=analisis_vivo,bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=200,height=50)
        )
        return items


    def items_7():
        items=[]
        items.append(grafico)
        return items
    
    def items_8():
        items=[]
        items.append(ft.ElevatedButton("Comenzar reconocimiento",on_click=lambda _:graficar(),bgcolor=ft.colors.BLACK,color=ft.colors.WHITE,width=300,height=50))
        return items

    def row_with_alignment(align: ft.MainAxisAlignment,r,space):
        return ft.Row( spacing=space,controls=r, alignment=align)

    def mostrar_alerta(mensaje):
        dlg = ft.AlertDialog(title=ft.Text('Error'), content=ft.Text(mensaje))
        page.dialog = dlg
        dlg.open = True
        page.update()

    page.on_route_change=route_change
    page.on_view_pop=view_pop
    page.go(page.route)  

        

ft.app(target=main,assets_dir="assets")