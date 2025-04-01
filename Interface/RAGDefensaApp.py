import asyncio

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QTabWidget, QLineEdit, QDateEdit, QMessageBox,
                             QTableWidget, QTableWidgetItem, QScrollArea)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QDate
import yaml
from pathlib import Path

from PyQt6.QtWidgets import QDialog


class AdvancedOptionsWindow(QDialog):
    def __init__(self, parent=None, run_scraping=None, migrate_data=None, evaluate_data=None, show_context=None,
                 show_context_date=None):
        super().__init__(parent)
        self.setWindowTitle("Opciones Avanzadas")
        self.setGeometry(200, 200, 400, 250)

        layout = QVBoxLayout()

        crawl_button = QPushButton("Ejecutar el Crawler")
        if run_scraping:
            crawl_button.clicked.connect(run_scraping)

        migrate_button = QPushButton("Migrar Datos a ChromaDB")
        if migrate_data:
            migrate_button.clicked.connect(migrate_data)

        evaluate_button = QPushButton("Evaluar Base de Datos")
        if evaluate_data:
            evaluate_button.clicked.connect(evaluate_data)

        show_context_button = QPushButton("Mostrar Contexto Usado (Sin Fecha)")
        if show_context:
            show_context_button.clicked.connect(show_context)

        show_context_button_date = QPushButton("Mostrar Contexto Usado (Con Fecha)")
        if show_context_date:
            show_context_button_date.clicked.connect(show_context_date)

        layout.addWidget(crawl_button)
        layout.addWidget(migrate_button)
        layout.addWidget(evaluate_button)
        layout.addWidget(show_context_button)
        layout.addWidget(show_context_button_date)

        self.setLayout(layout)


base_dir = Path(__file__).resolve().parent.parent

with open(base_dir / "config_data.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)
    urls = data["news_url"].split(", ")


class RAGDefensaApp(QWidget):
    def __init__(self, response_without_dates, response_with_dates, crawl_and_store, migrate_cassandra_to_chroma,
                 test_evaluation, get_articles_from_db):
        super().__init__()
        self.last_context = []
        self.last_context_date = []
        self.advanced_window = None
        self.response_without_dates = response_without_dates
        self.response_with_dates = response_with_dates
        self.crawl_and_store = crawl_and_store
        self.migrate_cassandra_to_chroma = migrate_cassandra_to_chroma
        self.test_evaluation = test_evaluation
        self.get_articles_from_db = get_articles_from_db
        self.current_page = 0
        self.max_pages = 0
        self.articles_per_page = 10

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
        self.menu_label = QLabel()
        pixmap = QPixmap("./Images/base-de-datos-2.png").scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio)
        self.menu_label.setPixmap(pixmap)
        self.menu_label.setToolTip("Opciones avanzadas")
        self.menu_label.mousePressEvent = self.open_advanced_options

        header_layout.addWidget(self.logo_label)
        header_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.menu_label, alignment=Qt.AlignmentFlag.AlignCenter)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #006D38; padding: 10px;")
        main_layout.addWidget(header_widget)

        # Pestañas
        self.tabs = QTabWidget()
        self.create_query_tab()
        self.create_query_with_date_tab()
        self.create_documents_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def create_query_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        query_layout = QHBoxLayout()
        self.query_entry = QLineEdit()
        self.query_entry.setPlaceholderText("Introduce tu consulta")
        self.query_entry.returnPressed.connect(self.execute_query)

        execute_button = QPushButton("Ejecutar Consulta")
        execute_button.clicked.connect(self.execute_query)
        query_layout.addWidget(self.query_entry)
        query_layout.addWidget(execute_button)

        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)

        layout.addWidget(self.response_text)
        layout.addLayout(query_layout)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Consulta sin Fecha")

    def create_query_with_date_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        query_date_layout = QHBoxLayout()
        self.query_entry_date = QLineEdit()
        self.query_entry_date.setPlaceholderText("Introduce tu consulta")
        self.query_entry_date.returnPressed.connect(self.execute_query_with_date)

        execute_button = QPushButton("Ejecutar Consulta")
        execute_button.clicked.connect(self.execute_query_with_date)
        query_date_layout.addWidget(self.query_entry_date)
        query_date_layout.addWidget(execute_button)

        date_layout = QHBoxLayout()
        self.date_entry = QDateEdit()
        self.date_entry.setCalendarPopup(True)
        self.date_entry.setFixedWidth(120)
        self.date_entry.setDate(QDate.currentDate())
        self.date_entry2 = QDateEdit()
        self.date_entry2.setCalendarPopup(True)
        self.date_entry2.setFixedWidth(120)
        self.date_entry2.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("Fecha inicial:"))
        date_layout.addWidget(self.date_entry)
        date_layout.addWidget(QLabel("Fecha final:"))
        date_layout.addWidget(self.date_entry2)

        self.response_text_date = QTextEdit()
        self.response_text_date.setReadOnly(True)

        layout.addWidget(self.response_text_date)
        layout.addLayout(date_layout)
        layout.addLayout(query_date_layout)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Consulta con Fecha")

    def create_documents_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.articles_table = QTableWidget()
        self.articles_table.setColumnCount(1)
        self.articles_table.setHorizontalHeaderLabels(["Titular"])
        self.articles_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.articles_table)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Anterior")
        self.prev_button.clicked.connect(self.previous_page)
        self.next_button = QPushButton("Siguiente")
        self.next_button.clicked.connect(self.next_page)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Ver Documentos")
        self.load_articles()

    def open_advanced_options(self, event):
        self.advanced_window = AdvancedOptionsWindow(
            self,
            run_scraping=self.run_scraping,
            migrate_data=self.migrate_data,
            evaluate_data=self.evaluate_data,
            show_context=self.show_context,
            show_context_date=self.show_context_date
        )
        self.advanced_window.exec()

    def open_advanced_options(self, event):
        self.advanced_window = AdvancedOptionsWindow(
            self,
            run_scraping=self.run_scraping,
            migrate_data=self.migrate_data,
            evaluate_data=self.evaluate_data,
            show_context=self.show_context,
            show_context_date=self.show_context_date
        )
        self.advanced_window.exec()

    def show_context(self):
        if not self.last_context:
            self.show_message("Contexto", "No hay contexto disponible. Ejecuta una consulta primero.")
            return
        self.show_context_window("Contexto Usado (Sin Fecha)", "\n\n---\n\n".join(self.last_context))

    def show_context_date(self):
        if not self.last_context_date:
            self.show_message("Contexto", "No hay contexto disponible. Ejecuta una consulta con fecha primero.")
            return
        self.show_context_window("Contexto Usado (Con Fecha)", "\n\n---\n\n".join(self.last_context_date))

    def show_context_window(self, title, context_text):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(150, 150, 600, 400)

        layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        text_container = QWidget()
        text_layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(context_text)
        text_layout.addWidget(text_edit)

        text_container.setLayout(text_layout)
        scroll_area.setWidget(text_container)

        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(dialog.accept)

        layout.addWidget(scroll_area)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec()

    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def load_articles(self):
        articles = self.get_articles_from_db(data.get("table_name"))

        self.max_pages = (len(articles) / 10) - 1
        start = self.current_page * self.articles_per_page
        end = start + self.articles_per_page
        self.articles_table.setRowCount(0)

        for article in articles[start:end]:
            row_position = self.articles_table.rowCount()
            self.articles_table.insertRow(row_position)
            self.articles_table.setItem(row_position, 0, QTableWidgetItem(article["titular"]))

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.load_articles()

    def next_page(self):
        if self.current_page < self.max_pages:
            self.current_page += 1
            self.load_articles()

    def execute_query(self):
        question = self.query_entry.text()
        self.response_text.setText("Ejecutando la consulta...")
        QApplication.processEvents()
        if not question:
            self.response_text.setText("Por favor, introduce una consulta.")
            return
        answer, context = self.response_without_dates(question)
        self.last_context = context
        self.response_text.setText(answer.get("content"))

    def execute_query_with_date(self):
        question = self.query_entry_date.text()
        date = self.date_entry.date().toString("dd/MM/yyyy")
        date2 = self.date_entry2.date().toString("dd/MM/yyyy")
        self.response_text_date.setText(f"Ejecutando la consulta entre {date} and {date2}...")
        QApplication.processEvents()
        if not question or not date or not date2:
            self.response_text_date.setText("Por favor, completa la consulta y selecciona las fechas.")
            return
        answer, context = self.response_with_dates(question, date, date2)
        self.last_context_date = context
        self.response_text_date.setText(answer.get("content"))

    def run_scraping(self):
        print("Ejecutando scraping...")
        self.crawl_and_store(urls)
        self.show_message("Scraping", "Scraping terminado")

    def migrate_data(self):
        print("Migrando datos a ChromaDB...")
        self.migrate_cassandra_to_chroma()
        self.show_message("Migración", "Migración de datos completada.")

    def evaluate_data(self):
        print("Evaluando base de datos...")
        asyncio.run(self.test_evaluation())
        self.show_message("Evaluación", "Evaluación completada.")
