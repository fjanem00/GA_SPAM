from tkinter import *

#Inicialización de las ventanas 
emisor = Tk()
#receptor = Tk()
emisor.title("Emisor Window")
#receptor.title("Receptor Window")
emisor.geometry('850x600')
emisor.config(bg = "grey")
#receptor.geometry('400x400')
 
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

texto = ""
#Definir funcion clicked()
def clicked():
 	texto = txt_asunto.get()
 	print(texto)

sent_btn = Button(emisor, text="Enviar", command=clicked)
sent_btn.place(x=775, y=550)





#creación de los bucles
emisor.mainloop()
#receptor.mainloop()