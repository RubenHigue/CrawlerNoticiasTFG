from abc import ABC, abstractmethod
from bs4 import BeautifulSoup


class InterfazEstrategia(ABC):
    @abstractmethod
    def extract_articles(self, soup: BeautifulSoup):
        """Extrae los enlaces de los artículos de la página principal."""
        pass

    @abstractmethod
    def extract_details(self, article_soup: BeautifulSoup):
        """Extrae los detalles del artículo."""
        pass
