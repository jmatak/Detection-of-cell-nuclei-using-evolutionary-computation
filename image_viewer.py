from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import copy
import joblib
import image_process as ipss

root, panelA, panelB, panelC, individual, indString = None, None, None, None, None, None
currentImage = None


def select_transformation():
    global individual, indString
    path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                      filetypes=(("Transformation files", "*.ind"), ("all files", "*.*")))

    if len(path) > 0:
        individual = joblib.load(path)
        indString.set(str(individual))
        setImage()


def select_image():
    """
    Metoda za dohvaćanje slike s računala i njeno daljnje procesuiranje
    """

    path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                      filetypes=(("Image files", "*.png"), ("all files", "*.*")))

    if len(path) > 0:
        global currentImage
        currentImage = cv2.imread(path)
        setImage()


def setImage():
    global panelA, panelB, panelC, root, individual, currentImage
    if currentImage is None:
        return

    image = currentImage
    processed = cv2.cvtColor(ipss.process_image(ipss.revert_img(copy.copy(image)), individual), cv2.COLOR_GRAY2RGB)
    contours = cv2.cvtColor(ipss.draw_contours(ipss.revert_img(copy.copy(image)), ipss.find_contours(processed)),
                            cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image)
    processed = Image.fromarray(processed)
    contours = Image.fromarray(contours)

    image = ImageTk.PhotoImage(image)
    processed = ImageTk.PhotoImage(processed)
    contours = ImageTk.PhotoImage(contours)

    if panelA is None or panelB is None or panelC is None:
        panelA = Label(root, image=image)
        panelA.image = image
        panelA.pack(side="top", padx=10, pady=10)

        panelB = Label(root, image=processed)
        panelB.image = processed
        panelB.pack(side="right", padx=10, pady=10)

        panelC = Label(root, image=contours)
        panelC.image = contours
        panelC.pack(side="left", padx=10, pady=10)

    else:
        panelA.configure(image=image)
        panelB.configure(image=processed)
        panelC.configure(image=contours)
        panelA.image = image
        panelB.image = processed
        panelC.image = contours


def viewer(ind):
    """
    Metoda koja nakon evolucijske metode iscrtava korisničko sučelje za prepoznavanje jezgri stanica.

    :param ind: Zadana jedinka transformacije
    """
    global root, individual, indString

    root = Tk()
    individual = ind
    indString = StringVar()
    indString.set(str(ind))
    root.title("Detekcija")
    label = Label(root, textvariable=indString)
    label.pack(side="top")

    downPanel = Frame(root)
    btnImage = Button(downPanel, text="Odaberi sliku", command=select_image)
    btnImage.pack(side="left", padx=10, pady=10)

    btnTrans = Button(downPanel, text="Odaberi transformaciju", command=select_transformation)
    btnTrans.pack(side="right", padx=10, pady=10)

    downPanel.pack(side="bottom")
    root.mainloop()
