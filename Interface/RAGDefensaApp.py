import asyncio

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QTabWidget, QLineEdit, QDateEdit, QMessageBox)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import yaml
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

with open(base_dir / "config_data.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)


class RAGDefensaApp(QWidget):
    def __init__(self,response_without_dates,response_with_dates,crawl_and_store,migrate_cassandra_to_chroma,test_evaluation):
        super().__init__()
        self.response_without_dates = response_without_dates
        self.response_with_dates = response_with_dates
        self.crawl_and_store = crawl_and_store
        self.migrate_cassandra_to_chroma = migrate_cassandra_to_chroma
        self.test_evaluation = test_evaluation
        self.setWindowTitle("RAG de Defensa")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()

        # Cabecera
        header_layout = QHBoxLayout()
        self.logo_label = QLabel()
        pixmap = QPixmap("./Images/LogoUJA.jpg").scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio)
        self.logo_label.setPixmap(pixmap)
        self.title_label = QLabel("RAG de Defensa")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        header_layout.addWidget(self.logo_label)
        header_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #006D38; padding: 10px;")
        main_layout.addWidget(header_widget)

        # Pestañas
        self.tabs = QTabWidget()
        self.create_query_tab()
        self.create_query_with_date_tab()
        self.create_advanced_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def create_query_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.query_entry = QLineEdit()
        self.query_entry.setPlaceholderText("Introduce tu consulta")
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        execute_button = QPushButton("Ejecutar Consulta")
        execute_button.clicked.connect(self.execute_query)

        layout.addWidget(self.query_entry)
        layout.addWidget(execute_button)
        layout.addWidget(self.response_text)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Consulta sin Fecha")

    def create_query_with_date_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.query_entry_date = QLineEdit()
        self.query_entry_date.setPlaceholderText("Introduce tu consulta")
        self.date_entry = QDateEdit()
        self.date_entry.setCalendarPopup(True)
        self.date_entry2 = QDateEdit()
        self.date_entry2.setCalendarPopup(True)
        self.response_text_date = QTextEdit()
        self.response_text_date.setReadOnly(True)
        execute_button = QPushButton("Ejecutar Consulta")
        execute_button.clicked.connect(self.execute_query_with_date)

        layout.addWidget(self.query_entry_date)
        layout.addWidget(QLabel("Fecha inicial:"))
        layout.addWidget(self.date_entry)
        layout.addWidget(QLabel("Fecha final:"))
        layout.addWidget(self.date_entry2)
        layout.addWidget(execute_button)
        layout.addWidget(self.response_text_date)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Consulta con Fecha")

    def create_advanced_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        scraping_button = QPushButton("Ejecutar Scraping")
        scraping_button.clicked.connect(self.run_scraping)
        migrate_button = QPushButton("Migrar Datos a ChromaDB")
        migrate_button.clicked.connect(self.migrate_data)
        evaluate_button = QPushButton("Evaluar Base de Datos")
        evaluate_button.clicked.connect(self.evaluate_data)

        layout.addWidget(scraping_button)
        layout.addWidget(migrate_button)
        layout.addWidget(evaluate_button)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Opciones Avanzadas")

    def execute_query(self):
        question = self.query_entry.text()
        if not question:
            self.response_text.setText("Por favor, introduce una consulta.")
            return
        self.response_text.setText("Ejecutando consulta...")
        answer = self.response_without_dates(question)
        self.response_text.setText(answer.get("content"))

    def execute_query_with_date(self):
        question = self.query_entry_date.text()
        date = self.date_entry.date().toString("dd/MM/yyyy")
        date2 = self.date_entry2.date().toString("dd/MM/yyyy")
        if not question or not date or not date2:
            self.response_text_date.setText("Por favor, completa la consulta y selecciona las fechas.")
            return
        self.response_text_date.setText(f"Ejecutando consulta entre {date} y {date2}...")
        answer = self.response_with_dates(question, date, date2)
        self.response_text_date.setText(answer.get("content"))

    def run_scraping(self):
        print("Ejecutando scraping...")
        self.crawl_and_store(data.get("news_url"))
        self.show_message("Scraping", "Scraping terminado")

    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)  # Botón "OK"
        msg.exec()

    def migrate_data(self):
        print("Migrando datos a ChromaDB...")
        self.migrate_cassandra_to_chroma()
        self.show_message("Migración", "Migración de datos completada.")

    def evaluate_data(self):
        print("Evaluando base de datos...")
        asyncio.run(self.test_evaluation())
        self.show_message("Evaluación", "Evaluación completada.")
