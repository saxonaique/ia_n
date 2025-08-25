import json
from tkinter import filedialog

class TFCCardLoader:
    def load_card(self):
        filepath = filedialog.askopenfilename(filetypes=[("TFC Files", "*.json")])
        if filepath:
            with open(filepath, 'r') as f:
                return json.load(f)
        return None