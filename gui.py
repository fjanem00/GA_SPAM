from tkinter import *
import pickle
from tkinter import messagebox

#Inicialización de las ventanas 
emisor = Tk()
receptor = Tk()

emisor.title("Emisor Window")
receptor.title("Receptor Window")

emisor.geometry('850x600')
emisor.config(bg = "grey")
receptor.geometry('400x400')
receptor.config(bg = "grey")
 
#Creación de la ventana de receptor inicial
global black_list 
black_list = []
global valor_recibidos
valor_recibidos = 0
recibidos = "Recibidos("+str(valor_recibidos)+")"
recibidos_lbl = Label(receptor, text=recibidos, font=("Arial Bold", 50))
recibidos_lbl.place(x=10, y=50)

global valor_spam
valor_spam = 0
spam = "SPAM("+str(valor_spam)+")"
spam_lbl = Label(receptor, text=spam, font=("Arial Bold", 50))
spam_lbl.place(x=10, y=200)

#Creación de la ventana emisor
emisor_lbl = Label(emisor, text="Emisor:    ", bg = "Grey")
emisor_lbl.place(x=10, y=50)
receptor_lbl = Label(emisor, text="Receptor:", bg = "Grey")
receptor_lbl.place(x=10, y=100)
asunto_lbl = Label(emisor, text="Asunto:    ", bg = "Grey")
asunto_lbl.place(x=10, y=150)

txt_emisor = Entry(emisor,width=80,)
txt_emisor.place(x=100, y=50)
txt_receptor = Entry(emisor,width=80)
txt_receptor.place(x=100, y=100)
txt_asunto = Entry(emisor, width=80)
txt_asunto.place(x=100, y=150)

txt = Text(emisor, insertborderwidth = 10, bd = 2, height = 20, width = 116)
txt.place(x=10, y=200)

emisor_list = ""
content = ""
#Cargar el clasificador
def load_classifier(obj_name):
    return pickle.load(open(obj_name, 'rb'))
#Definir funcion cambiar spam
def spamChange():

	global valor_spam
	valor_spam = valor_spam + 1
	spam_change = "SPAM("+str(valor_spam)+")"
	spam_lbl = Label(receptor, text=spam_change, font=("Arial Bold", 50))
	spam_lbl.place(x=10, y=200)

#Definir funcion clicked()
def clicked():

	#Inicialización
	global black_list 
	count = 0
	clf = load_classifier("./tfidf_lr_clf.pkl")
	correo = True
	noBL = True
	emisor_list = txt_emisor.get()
	
	if emisor_list != "":
		count = count + 1
	receptor_list = txt_receptor.get()
	if receptor_list != "":
		count = count + 1
	
	print(receptor_list.find('@') != -1)
	if emisor_list.find('@') != -1 and receptor_list.find('@') != -1: 
			correo = True
	else:
		messagebox.showinfo('!Atención!', 'El emisor y el receptor deben ser correos electrónicos')

	asunto_text = txt_asunto.get()

	if asunto_text != "":
		count = count + 1

	if txt.get("1.0",'end-1c') != "":
		count = count + 1
	print(count)
	if count == 4 and correo:
		#content usa el contenido de texto más el asunto y se sometera a un preprocesado y al modelo entrenado 
		content = txt.get("1.0",'end-1c') + " "+ asunto_text
  
		#Uso del modelo para predecir 
		
		for black in black_list:
			if black == emisor_list:
				noBL = False
				break

		if noBL:
			predictions = clf.predict_proba([content])[0]

			if predictions[1] < predictions[0]:
				global valor_recibidos
				valor_recibidos = valor_recibidos + 1
				recibidos = "Recibidos("+str(valor_recibidos)+")"
				recibidos_lbl = Label(receptor, text=recibidos, font=("Arial Bold", 50))
				recibidos_lbl.place(x=10, y=50)
			else:
				spamChange()
				black_list.append(emisor_list)
			
		else:
	 		spamChange()
	else:
	 	messagebox.showinfo('!Atención!', 'No están todos lo campos rellenos')

sent_btn = Button(emisor, text="Enviar", command=clicked)
sent_btn.place(x=775, y=550)





#creación de los bucles
emisor.mainloop()
#receptor.mainloop()