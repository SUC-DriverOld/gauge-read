import json
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EvalAnnotationApp:
    def __init__(self, root, image_dir=None):
        self.root = root
        self.root.title("Label Tool")
        self.root.geometry("900x600")

        self.image_dir = Path(image_dir).resolve()
        self.image_files = self._load_image_files()
        self.current_index = None
        self.current_photo = None

        self.start_var = tk.StringVar(value="0")
        self.end_var = tk.StringVar(value="")
        self.value_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="准备就绪")
        self.input_widgets = []
        self.entry_validate_command = None

        self._build_ui()
        self._bind_shortcuts()

        if not self.image_files:
            self.status_var.set("未在指定目录中找到图片")
            messagebox.showwarning("提示", f"未在目录中找到图片: {self.image_dir}")
            return

        self._refresh_file_list()
        self._load_image_by_index(0, save_current=False)
        if self.input_widgets:
            self.input_widgets[0].focus_set()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.entry_validate_command = (self.root.register(self._validate_numeric_input), "%P")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(self.root, padding=8)
        left_frame.grid(row=0, column=0, sticky="ns")
        left_frame.rowconfigure(1, weight=1)

        ttk.Label(left_frame, text="标注列表").grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.file_listbox = tk.Listbox(left_frame, width=38, exportselection=False)
        self.file_listbox.grid(row=1, column=0, sticky="ns")
        self.file_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        list_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.file_listbox.yview)
        list_scrollbar.grid(row=1, column=1, sticky="ns")
        self.file_listbox.configure(yscrollcommand=list_scrollbar.set)

        right_frame = ttk.Frame(self.root, padding=8)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(right_frame)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top_bar.columnconfigure(1, weight=1)

        self.index_label = ttk.Label(top_bar, text="0 / 0")
        self.index_label.grid(row=0, column=0, sticky="w")

        self.filename_label = ttk.Label(top_bar, text="", anchor="center")
        self.filename_label.grid(row=0, column=1, sticky="ew")

        image_frame = ttk.Frame(right_frame)
        image_frame.grid(row=1, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        form_frame = ttk.Frame(right_frame)
        form_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        form_frame.columnconfigure(0, weight=1)

        form_content = ttk.Frame(form_frame)
        form_content.grid(row=0, column=0)

        ttk.Label(form_content, text="起始读数").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.start_entry = ttk.Entry(
            form_content, textvariable=self.start_var, width=16, validate="key", validatecommand=self.entry_validate_command
        )
        self.start_entry.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ttk.Label(form_content, text="终止读数").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.end_entry = ttk.Entry(
            form_content, textvariable=self.end_var, width=16, validate="key", validatecommand=self.entry_validate_command
        )
        self.end_entry.grid(row=0, column=3, sticky="w", padx=(0, 20))

        ttk.Label(form_content, text="当前读数").grid(row=0, column=4, sticky="w", padx=(0, 8))
        self.value_entry = ttk.Entry(
            form_content, textvariable=self.value_var, width=16, validate="key", validatecommand=self.entry_validate_command
        )
        self.value_entry.grid(row=0, column=5, sticky="w")
        self.input_widgets = [self.start_entry, self.end_entry, self.value_entry]

        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        button_frame.columnconfigure(0, weight=1)

        button_content = ttk.Frame(button_frame)
        button_content.grid(row=0, column=0)

        self.prev_button = ttk.Button(button_content, text="上一张", command=self._prev_image)
        self.prev_button.grid(row=0, column=0, padx=(0, 8))

        self.next_button = ttk.Button(button_content, text="下一张", command=self._next_image)
        self.next_button.grid(row=0, column=1, padx=(0, 8))

        self.save_button = ttk.Button(button_content, text="保存", command=self._save_current_annotation)
        self.save_button.grid(row=0, column=2, padx=(0, 8))

        self.jump_annotated_button = ttk.Button(button_content, text="跳到下一未标注", command=self._jump_to_next_unlabeled)
        self.jump_annotated_button.grid(row=0, column=3)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 4))
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _bind_shortcuts(self):
        for widget in self.input_widgets:
            widget.bind("<Tab>", self._focus_next_input)
        self.root.bind_all("<KeyPress-u>", self._on_jump_shortcut)
        self.root.bind_all("<KeyPress-U>", self._on_jump_shortcut)

    def _focus_next_input(self, event):
        if not self.input_widgets:
            return "break"
        current_widget = event.widget
        try:
            idx = self.input_widgets.index(current_widget)
        except ValueError:
            idx = -1
        next_widget = self.input_widgets[(idx + 1) % len(self.input_widgets)]
        next_widget.focus_set()
        next_widget.select_range(0, tk.END)
        return "break"

    def _on_jump_shortcut(self, _event):
        self._jump_to_next_unlabeled()
        return "break"

    def _validate_numeric_input(self, proposed_value):
        allowed_chars = set("-.0123456789")
        return all(char in allowed_chars for char in proposed_value)

    def _load_image_files(self):
        if not self.image_dir.exists():
            return []
        return sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()])

    def _annotation_path(self, image_path):
        return image_path.with_suffix(".json")

    def _load_annotation_data(self, image_path):
        annotation_path = self._annotation_path(image_path)
        if not annotation_path.exists():
            return None
        try:
            with annotation_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _has_valid_annotation(self, data):
        if not isinstance(data, dict):
            return False
        for field_name in ("start", "end", "value"):
            value = data.get(field_name)
            if value is None:
                return False
            try:
                float(value)
            except (TypeError, ValueError):
                return False
        return True

    def _is_annotated(self, image_path):
        data = self._load_annotation_data(image_path)
        return self._has_valid_annotation(data)

    def _refresh_file_list(self):
        current_selection = self.current_index
        self.file_listbox.delete(0, tk.END)
        for idx, image_path in enumerate(self.image_files):
            self.file_listbox.insert(tk.END, image_path.name)
            if self._is_annotated(image_path):
                self.file_listbox.itemconfig(idx, foreground="green")
            else:
                self.file_listbox.itemconfig(idx, foreground="black")

        if current_selection is not None and 0 <= current_selection < len(self.image_files):
            self.file_listbox.selection_set(current_selection)
            self.file_listbox.see(current_selection)

    def _parse_number(self, text, field_name, allow_empty):
        text = text.strip()
        if text == "":
            if allow_empty:
                return None
            raise ValueError(f"{field_name}不能为空")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name}必须是数字") from exc

    def _save_current_annotation(self):
        if self.current_index is None:
            return True

        image_path = self.image_files[self.current_index]
        try:
            start = self._parse_number(self.start_var.get(), "起始读数", allow_empty=False)
            end = self._parse_number(self.end_var.get(), "终止读数", allow_empty=True)
            value = self._parse_number(self.value_var.get(), "当前读数", allow_empty=True)
        except ValueError as exc:
            messagebox.showerror("输入错误", str(exc))
            return False

        data = {"filename": image_path.name, "start": start, "end": end, "value": value}

        annotation_path = self._annotation_path(image_path)
        with annotation_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.status_var.set(f"已保存: {annotation_path.name}")
        self._refresh_file_list()
        return True

    def _load_annotation(self, image_path):
        data = self._load_annotation_data(image_path)
        if data is None:
            self.start_var.set("0")
            self.end_var.set("")
            self.value_var.set("")
            return

        self.start_var.set("" if data.get("start") is None else str(data.get("start")))
        self.end_var.set("" if data.get("end") is None else str(data.get("end")))
        self.value_var.set("" if data.get("value") is None else str(data.get("value")))

    def _display_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        max_width = 980
        max_height = 680
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo)

    def _load_image_by_index(self, index, save_current=True):
        if not self.image_files:
            return
        if index < 0 or index >= len(self.image_files):
            return

        if save_current and not self._save_current_annotation():
            return

        self.current_index = index
        image_path = self.image_files[index]
        self._display_image(image_path)
        self._load_annotation(image_path)

        self.index_label.configure(text=f"{index + 1} / {len(self.image_files)}")
        self.filename_label.configure(text=image_path.name)
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.see(index)
        self.status_var.set(f"当前图片: {image_path.name}")

    def _prev_image(self):
        if self.current_index is None:
            return
        self._load_image_by_index(max(0, self.current_index - 1))

    def _next_image(self):
        if self.current_index is None:
            return
        self._load_image_by_index(min(len(self.image_files) - 1, self.current_index + 1))

    def _jump_to_next_unlabeled(self):
        if not self.image_files:
            return
        start_index = 0 if self.current_index is None else self.current_index + 1
        for idx in range(start_index, len(self.image_files)):
            if not self._is_annotated(self.image_files[idx]):
                self._load_image_by_index(idx)
                return
        messagebox.showinfo("提示", "后面没有未标注图片了")

    def _on_list_select(self, _event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index == self.current_index:
            return
        self._load_image_by_index(index)

    def _on_close(self):
        if self._save_current_annotation():
            self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Lable Tool")
    parser.add_argument("-d", "--directory", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    EvalAnnotationApp(root, image_dir=args.directory)
    root.mainloop()


if __name__ == "__main__":
    main()
