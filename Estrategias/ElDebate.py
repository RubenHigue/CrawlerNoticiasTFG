from .InterfazEstrategia import InterfazEstrategia
from bs4 import BeautifulSoup


class ElDebate(InterfazEstrategia):
    def extract_articles(self, soup: BeautifulSoup):
        base_url = "https://www.eldebate.com"
        articles = soup.find_all("a", id=True, href=lambda href: href and href.startswith("/espana/defensa/"))

        for article in articles:
            article["href"] = base_url + article["href"]

        return articles

    def extract_details(self, article_soup: BeautifulSoup):
        fuente = "ElDebate.com"
        fecha_hora_tag = article_soup.find("time")
        fecha_hora = fecha_hora_tag.get_text(strip=True) if fecha_hora_tag else "Fecha no encontrada"

        if " " in fecha_hora:
            fecha, hora = fecha_hora.split(" ", 1)
        else:
            fecha, hora = fecha_hora, " "

        autor_tag = article_soup.find("a", class_="c-detail__author__name")
        autor_nombre = autor_tag.get_text(strip=True) if (autor_tag and autor_tag.get_text()) else ("Nombre no "
                                                                                                    "encontrado")
        autor_url = autor_tag["href"] if autor_tag and "href" in autor_tag.attrs else "URL no encontrada"

        titular_tag = article_soup.find("div", class_="default-title")
        titular = titular_tag.find("h1").get_text(strip=True) if titular_tag and titular_tag.find("h1") else "Titular no encontrado"

        parrafos = article_soup.find_all("p")
        if parrafos and "Buscar en" in parrafos[0].get_text(strip=True):
            parrafos = parrafos[1:]
        noticia = " ".join([p.get_text(strip=True) for p in parrafos])

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
