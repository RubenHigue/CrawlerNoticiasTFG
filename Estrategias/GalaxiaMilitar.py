from .InterfazEstrategia import InterfazEstrategia
from bs4 import BeautifulSoup
import csv


class GalaxiaMilitar(InterfazEstrategia):

    def extract_articles(self, soup: BeautifulSoup):
        articles = soup.find_all("article", id=True)
        links = []
        for article in articles:
            links.append(article.find("a", title=True, class_=False, href=lambda href: href and href.startswith("https://galaxiamilitar.es/")))
        return links

    def extract_details(self, article_soup: BeautifulSoup):
        fuente = "Galaxia Militar"

        fecha_tag = article_soup.find("time", class_="entry-date published")
        fecha = fecha_tag.get_text(strip=True) if fecha_tag else "Fecha no encontrada"
        hora = fecha_tag["datetime"].split("T")[1].split("+")[0] if fecha_tag and "datetime" in fecha_tag.attrs else ""

        autor_tag = article_soup.find("span", class_="author vcard")
        autor_nombre = autor_tag.get_text(strip=True) if autor_tag else "Redacción"
        autor_url = autor_tag.find("a")["href"] if autor_tag and autor_tag.find("a") else "URL no encontrada"

        titular_tag = article_soup.find("h1", class_="entry-title")
        titular = titular_tag.get_text(strip=True) if titular_tag else None

        contenido_tag = article_soup.find("div", class_="entry-content")
        noticia = " ".join([p.get_text(strip=True) for p in
                            contenido_tag.find_all("p")]) if contenido_tag else "Contenido no encontrado"

        return {
            "fuente": fuente,
            "fecha": fecha,
            "hora": hora,
            "titular": titular,
            "autor": autor_nombre,
            "autor_url": autor_url,
            "noticia": noticia,
            "articulo_original": article_soup.prettify(),
        }
