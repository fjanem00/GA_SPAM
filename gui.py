from tkinter import *

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
#Definir funcion cambiar spam
def spamChange():

	global valor_spam
	valor_spam = valor_spam + 1
	spam_change = "SPAM("+str(valor_spam)+")"
	spam_lbl = Label(receptor, text=spam_change, font=("Arial Bold", 50))
	spam_lbl.place(x=10, y=200)

#Definir funcion clicked()
def clicked():

	global black_list 
	
	noBL = True
	emisor_list = txt_emisor.get()
	r = txt_receptor.get()
	asunto_text = txt_asunto.get()
	content = txt.get("1.0",'end-1c') + " "+ asunto_text

	#content usa el contenido de texto más el asunto y se sometera a un preprocesado y al modelo entrenado 
	
	for black in black_list:
		if black == emisor_list:
			noBL = False
			break

	if noBL:
		if r == "fran":
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

sent_btn = Button(emisor, text="Enviar", command=clicked)
sent_btn.place(x=775, y=550)





#creación de los bucles
emisor.mainloop()
#receptor.mainloop()