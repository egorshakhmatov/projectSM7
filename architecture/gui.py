"""Desktop shell for SM7. Data integration and production training are separate."""
import queue
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import traceback

from run import ROOT, get_status, show_architecture


class Application:
    def __init__(self, window):
        self.window = window
        self.messages = queue.Queue()
        window.title("СМ7 — выбор светофора")
        window.geometry("980x740")
        window.minsize(800, 620)
        window.configure(bg="#f3f6f7")
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6f7")
        style.configure("TLabel", background="#f3f6f7", foreground="#182e3b", font=("Segoe UI", 11))
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"))
        style.configure("TButton", font=("Segoe UI", 11), padding=(14, 12))
        frame = ttk.Frame(window, padding=28)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="СМ7 · Выбор светофора", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="История за час → GRU → направление на следующие 5 минут").pack(anchor="w", pady=(8, 18))
        self.notice = tk.StringVar(value="Среда подготовлена. Можно проверить модель и изучить архитектуру.")
        ttk.Label(frame, textvariable=self.notice, wraplength=860).pack(anchor="w", pady=(0, 16))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.verify_button = ttk.Button(buttons, text="Проверить модель GRU", command=self.start_verification)
        self.verify_button.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=4)
        ttk.Button(buttons, text="Почему выбрана GRU", command=self.explain_model).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=4)
        ttk.Button(buttons, text="Схема архитектуры", command=self.open_architecture).grid(row=0, column=2, sticky="ew", pady=4)
        ttk.Button(buttons, text="Состояние проекта", command=self.show_status).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=4)
        ttk.Button(buttons, text="Обучение · ожидает данные", state="disabled").grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=4)
        ttk.Button(buttons, text="Прогноз · ожидает модель", state="disabled").grid(row=1, column=2, sticky="ew", pady=4)
        for column in range(3):
            buttons.columnconfigure(column, weight=1)
        ttk.Label(frame, text="Рабочая область", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(20, 8))
        self.text = scrolledtext.ScrolledText(frame, wrap="word", font=("Segoe UI", 11), bg="white", fg="#182e3b", relief="flat", padx=18, pady=16)
        self.text.pack(fill="both", expand=True)
        ttk.Label(frame, text="Данные готовит другой участник. Проверка GRU — технический тест, а не прогноз.", wraplength=870).pack(anchor="w", pady=(14, 0))
        self.show_status()
        window.after(100, self.poll_messages)

    def display(self, content):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def explain_model(self):
        self.display("ПОЧЕМУ ВЫБРАНА GRU\n\n"
                     "Нам нужно учитывать, как плотность менялась за последний час. GRU читает наблюдения по порядку "
                     "и обновляет внутреннюю память. Одинаковая последняя плотность может завершать рост или снижение; "
                     "история помогает различать такие случаи.\n\n"
                     "КАК ПРОХОДИТ ОДИН ПРОГНОЗ\n\n"
                     "1. Вход: 12 наблюдений, в каждом три плотности.\n"
                     "2. Значения делятся на 200, чтобы попасть в диапазон от 0 до 1.\n"
                     "3. GRU обрабатывает их по порядку. Её память — 32 внутренних числа.\n"
                     "4. Выходной слой формирует оценки для трёх светофоров.\n"
                     "5. Получаем три вероятности и выбираем наибольшую.\n\n"
                     "ПОЧЕМУ ЭТО РЕШЕНИЕ ПОДХОДИТ ПРОЕКТУ\n\n"
                     "Такая модель прямо задана в ТЗ. Она компактна: 3 651 обучаемый параметр. "
                     "У аналогичной LSTM с тем же размером памяти — 4 835 параметров. "
                     "Видеокарта и облачный сервис не требуются.\n\n"
                     "Точность заранее неизвестна. После получения данных сравним GRU с простым выбором "
                     "минимальной плотности в последней строке. Сейчас доступна техническая проверка модели; "
                     "рабочие прогнозы появятся после обучения.")

    def open_architecture(self):
        try:
            if show_architecture():
                messagebox.showinfo("Схема", f"Откройте файл:\n{ROOT / 'docs/overview.html'}")
        except OSError as error:
            messagebox.showerror("Не удалось открыть схему", str(error))

    def show_status(self):
        state = get_status()
        self.display("ГОТОВО НА ЭТОМ ПК\n\n"
                     "• Окно программы с кнопками.\n• Среда Python и CPU-версия PyTorch.\n"
                     "• GRU по ТЗ: один слой, 32 скрытых признака.\n"
                     "• Проверка вычислений, шага обучения и загрузки весов.\n\n"
                     "СЛЕДУЮЩИЙ ЭТАП\n\n"
                     "Подключить данные коллеги, обучение на полном наборе и сохранённую модель. "
                     "После этого станут доступны обучение и рабочий прогноз.\n\n"
                     f"Python: {state['python_version']}\n"
                     "Вход: 12 наблюдений × 3 светофора.\n"
                     "Выход будущего прогноза: номер светофора и три вероятности.\n\n"
                     "Нажмите «Проверить модель GRU», чтобы проверить установленную среду и модель без дорожных данных.")

    def start_verification(self):
        self.verify_button.state(["disabled"])
        self.notice.set("Проверяю среду и GRU. Окно остаётся доступным…")
        self.display("Проверка библиотек и вычислений на CPU…")
        def worker():
            try:
                from environment import check_environment
                environment = check_environment()
                if not environment["ready"]:
                    raise RuntimeError("\n".join(environment["errors"]))
                from model import verify_model
                result = verify_model()
                versions = "\n".join(f"{key}: {value}" for key, value in environment["versions"].items())
                self.messages.put((True, "ПРОВЕРКА ПРОЙДЕНА\n\n"
                    f"Обучаемых параметров: {result['parameters']}\n"
                    "Вычисления: CPU\nФорма выхода для двух примеров: 2 × 3\n"
                    "Один шаг обучения: выполнен\nСуммы вероятностей: корректны\n"
                    "Сохранение и повторная загрузка весов: результат совпал\n\n"
                    f"УСТАНОВЛЕННЫЕ КОМПОНЕНТЫ\n{versions}\n\n"
                    "Это техническая проверка на небольших фиксированных тензорах. "
                    "Качество выбора светофора ещё не измерялось; обученные веса проекта не создавались."))
            except Exception as error:
                (ROOT / ".runtime").mkdir(exist_ok=True)
                (ROOT / ".runtime/app-error.log").write_text(traceback.format_exc(), encoding="utf-8")
                self.messages.put((False, f"Проверка не завершена: {error}\n\nПовторно запустите START.cmd для проверки среды."))
        threading.Thread(target=worker, daemon=True).start()

    def poll_messages(self):
        try:
            while True:
                success, content = self.messages.get_nowait()
                self.display(content)
                self.notice.set("Модель и среда проверены." if success else "Проверка обнаружила проблему.")
                self.verify_button.state(["!disabled"])
        except queue.Empty:
            pass
        self.window.after(100, self.poll_messages)


def main():
    window = tk.Tk()
    Application(window)
    window.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        log = runtime / "app-error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        messagebox.showerror("СМ7", f"Не удалось открыть программу. Подробности: {log}")
        raise SystemExit(1)
