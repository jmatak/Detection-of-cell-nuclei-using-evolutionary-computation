from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import image_process as ipss

root, panelA, panelB, individual = None, None, None, None


def select_image():
    global panelA, panelB, root, individual

    path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                      filetypes=(("Image files", "*.png"), ("all files", "*.*")))

    if len(path) > 0:
        image = cv2.imread(path)

        processed = cv2.cvtColor(ipss.process_image(image, individual), cv2.COLOR_GRAY2RGB)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        processed = Image.fromarray(processed)

        image = ImageTk.PhotoImage(image)
        processed = ImageTk.PhotoImage(processed)

        if panelA is None or panelB is None:
            panelA = Label(root, image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            panelB = Label(root, image=processed)
            panelB.image = processed
            panelB.pack(side="right", padx=10, pady=10)

        else:
            panelA.configure(image=image)
            panelB.configure(image=processed)
            panelA.image = image
            panelB.image = processed


def viewer(ind):
    global panelA, panelB, root, individual

    root = Tk()
    individual = ind
    root.title("Detekcija")
    label = Label(root, text=str(ind))
    label.pack(side="top")
    btn = Button(root, text="Odaberi sliku", command=select_image)
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    root.mainloop()
