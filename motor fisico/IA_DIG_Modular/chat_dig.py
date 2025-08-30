import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, Toplevel
import threading
import time

# Importamos el motor principal y, crucialmente, la clase LearningMemory para poder modificarla
from neo import Metamodulo, LearningMemory

# --- MODO DE DEPURACIÓN ---
# Si pones esto en True, la consola imprimirá los vectores y las puntuaciones de similitud.
# Muy útil para ver por qué la IA toma una decisión.
DEBUG_MODE = True

# --- SOLUCIÓN: Modificación de LearningMemory en tiempo de ejecución ---
# "Monkey Patching" para arreglar el bug de comparación sin tocar el archivo original.
def find_similar_pattern_fixed(self, current_vector: list, input_data_text: str):
    """
    Versión corregida de find_similar_pattern.
    Ahora recibe el vector actual directamente y evita la auto-comparación.
    """
    if not self.patterns:
        return None
    
    best_match = None
    best_similarity = 0.0
    
    if DEBUG_MODE: print(f"\n--- Buscando similitud para vector: { [f'{v:.2f}' for v in current_vector[:4]] }... ---")

    for pattern_id, pattern in self.patterns.items():
        # SALVAGUARDA CLAVE: Si el texto de entrada es idéntico a un patrón guardado,
        # lo ignoramos para forzar que encuentre asociaciones, no duplicados exactos.
        if pattern["input_data"] == input_data_text:
            continue

        similarity = self._calculate_similarity(current_vector, pattern["pattern_vector"])
        if DEBUG_MODE: print(f"  Comparando con '{pattern['user_label']}' -> Similitud: {similarity:.3f}")

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = pattern
    
    # Solo devolvemos un resultado si supera el umbral de confianza
    if best_match and best_similarity >= self.similarity_threshold:
        best_match["similarity_score"] = best_similarity
        # Actualizar contador de uso
        self.patterns[best_match["id"]]["usage_count"] += 1
        self.patterns[best_match["id"]]["last_used"] = time.time()
        return best_match
    
    return None

# Reemplazamos el método problemático en la clase LearningMemory con nuestra versión corregida
LearningMemory.find_similar_pattern = find_similar_pattern_fixed


# --- Lógica del Chat (Adaptada del script de línea de comandos) ---
class ChatDIG:
    def __init__(self, simulation_steps=5):
        print("🧠 Inicializando el motor del ChatDIG...")
        self.metamodulo = Metamodulo()
        self.simulation_steps = simulation_steps
        print("✅ Motor DIG listo.")

    def _generate_vector_from_text(self, text: str) -> list:
        """Función aislada para generar un vector a partir de un texto."""
        self.metamodulo.current_cycle = 0
        self.metamodulo.fingerprint_system.reset_session()
        self.metamodulo.receive_input(text, 'text')
        
        summary = None
        for _ in range(self.simulation_steps):
            summary = self.metamodulo.process_step()
        
        # Devolvemos el vector de características de la huella final
        return self.metamodulo.fingerprint_system.learning_memory._extract_pattern_vector(
            self.metamodulo.fingerprint_system.fingerprints
        )

    def train(self, label: str, text: str) -> bool:
        """Entrena a la IA asociando un texto con una etiqueta."""
        # Genera las huellas y el vector para el nuevo concepto
        self._generate_vector_from_text(text) 
        # Aprende usando las huellas que se generaron en el paso anterior
        pattern_id = self.metamodulo.learn_from_session(
            input_data=text, input_type='text', user_label=label
        )
        return bool(pattern_id)

    def get_response(self, user_input: str) -> str:
        # 1. Generar el vector para la entrada actual del usuario de forma limpia
        current_vector = self._generate_vector_from_text(user_input)
        
        # 2. Buscar un patrón similar usando la función corregida
        recognized_pattern = self.metamodulo.fingerprint_system.learning_memory.find_similar_pattern(
            current_vector, user_input
        )
        
        # 3. Decidir si asociar o aprender
        if recognized_pattern:
            label = recognized_pattern.get('user_label', 'un concepto previamente aprendido')
            original_text = recognized_pattern.get('input_data', '')
            similarity = recognized_pattern.get('similarity_score', 0.0)
            
            response = (
                f"Eso me recuerda a una idea que tengo guardada: '{label}' (Similitud: {similarity:.1%}).\n\n"
                f"El texto original de esa idea es: \"{original_text}\""
            )
            return response
        else:
            label = user_input if len(user_input) < 50 else user_input[:47] + "..."
            # Aprende usando las huellas que ya se generaron en _generate_vector_from_text
            self.metamodulo.learn_from_session(user_input, 'text', user_label=label)
            return f"He aprendido y guardado esa idea: \"{label}\".\n\nIntenta escribir algo similar y veré si lo asocio."

# --- Interfaz Gráfica (GUI) ---
# (El código de la GUI no necesita cambios, se pega aquí para que sea un solo archivo)
class ChatDIG_App(tk.Tk):
    def __init__(self, chat_logic: ChatDIG):
        super().__init__()
        
        self.chat_logic = chat_logic
        self.processing_lock = threading.Lock() 

        self.title("ChatDIG - IA Conversacional")
        self.geometry("700x550")
        
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        self._create_menu()
        self._create_widgets()
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        welcome_message = (
            "¡Hola! Soy ChatDIG. Cada frase que escribas la aprenderé como un nuevo concepto. "
            "Intenta escribir algo, y luego una frase similar para ver cómo las asocio."
        )
        self._add_message("DIG", welcome_message, "ai_tag")

    def _create_menu(self):
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Salir", command=self._on_closing)

        actions_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Acciones", menu=actions_menu)
        actions_menu.add_command(label="Entrenar Concepto Específico", command=self._open_training_window)
        actions_menu.add_command(label="Mostrar Estadísticas", command=self._show_stats)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Acerca de", command=self._show_about)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        history_frame = ttk.Frame(main_frame)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.chat_history = scrolledtext.ScrolledText(
            history_frame, state='disabled', wrap=tk.WORD, font=("Helvetica", 11)
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        self.chat_history.tag_configure("user_tag", foreground="#007bff", font=("Helvetica", 11, "bold"))
        self.chat_history.tag_configure("ai_tag", foreground="#28a745", font=("Helvetica", 11, "bold"))
        self.chat_history.tag_configure("text_tag", foreground="black")

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X)

        self.user_input = ttk.Entry(input_frame, font=("Helvetica", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.user_input.bind("<Return>", self._send_message)

        self.send_button = ttk.Button(input_frame, text="Enviar", command=self._send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0))

        self.status_bar = ttk.Label(self, text="Lista.", relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _add_message(self, sender: str, message: str, sender_tag: str):
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, f"{sender}: ", (sender_tag,))
        self.chat_history.insert(tk.END, f"{message}\n\n", ("text_tag",))
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)

    def _send_message(self, event=None):
        user_message = self.user_input.get().strip()
        if not user_message:
            return

        self._add_message("Usuario", user_message, "user_tag")
        self.user_input.delete(0, tk.END)

        self.user_input.config(state='disabled')
        self.send_button.config(state='disabled')
        self.status_bar.config(text="Analizando...")

        threading.Thread(target=self._get_response_worker, args=(user_message,), daemon=True).start()

    def _get_response_worker(self, user_message: str):
        with self.processing_lock:
            ai_response = self.chat_logic.get_response(user_message)
            self.after(0, self._display_ai_response, ai_response)

    def _display_ai_response(self, response: str):
        self._add_message("DIG", response, "ai_tag")
        
        self.user_input.config(state='normal')
        self.send_button.config(state='normal')
        self.status_bar.config(text="Lista.")
        self.user_input.focus_set()

    def _open_training_window(self):
        train_window = Toplevel(self)
        train_window.title("Entrenar Concepto Específico")
        train_window.geometry("450x300")
        train_window.transient(self) 
        train_window.grab_set() 

        main_frame = ttk.Frame(train_window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Etiqueta (Concepto Clave):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        label_entry = ttk.Entry(main_frame, font=("Helvetica", 10))
        label_entry.pack(fill=tk.X, pady=(5, 15))
        label_entry.focus_set()
        
        ttk.Label(main_frame, text="Texto Descriptivo:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        text_entry = scrolledtext.ScrolledText(main_frame, height=5, wrap=tk.WORD, font=("Helvetica", 10))
        text_entry.pack(fill=tk.BOTH, expand=True, pady=(5, 15))

        def on_train():
            label = label_entry.get().strip()
            text = text_entry.get("1.0", tk.END).strip()
            if not label or not text:
                messagebox.showwarning("Datos incompletos", "Debes proporcionar una etiqueta y un texto.", parent=train_window)
                return
            
            success = self.chat_logic.train(label, text)
            if success:
                messagebox.showinfo("Éxito", f"¡Concepto '{label}' aprendido correctamente!", parent=train_window)
                train_window.destroy()
            else:
                messagebox.showerror("Error", "No se pudo aprender el concepto.", parent=train_window)

        train_button = ttk.Button(main_frame, text="Entrenar", command=on_train)
        train_button.pack(pady=(5, 0))

    def _show_stats(self):
        stats = self.chat_logic.metamodulo.get_learning_stats()
        stats_text = (
            f"Total de ideas guardadas: {stats.get('total_patterns', 0)}\n"
            f"Uso de memoria: {stats.get('memory_usage', 'N/A')}\n\n"
            "Ideas más recordadas:\n"
        )
        top_patterns = stats.get('top_patterns', [])
        if not top_patterns:
            stats_text += "  (Aún no hay ideas suficientemente usadas)"
        else:
            for p in top_patterns:
                stats_text += f"  - '{p['label']}' (usado {p['usage_count']} veces)\n"
        
        messagebox.showinfo("Estadísticas de Aprendizaje", stats_text)
    
    def _show_about(self):
        messagebox.showinfo(
            "Acerca de ChatDIG",
            "ChatDIG es una IA conversacional experimental.\n\n"
            "Utiliza el sistema de 'Organismo Informacional' para analizar la 'forma' o 'huella' "
            "de un texto en lugar de su significado semántico, y asocia ideas basadas en la similitud de estas huellas."
        )

    def _on_closing(self):
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
            self.destroy()


if __name__ == "__main__":
    # Para obtener mejores resultados, se recomienda eliminar el archivo
    # 'learning_memory.json' si existe, para empezar con una memoria limpia.
    print("Iniciando la aplicación ChatDIG GUI...")
    chat_logic_instance = ChatDIG()
    app = ChatDIG_App(chat_logic_instance)
    app.mainloop()