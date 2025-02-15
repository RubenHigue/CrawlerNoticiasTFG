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
        fecha, hora = fecha_hora.split(" ") if " " in fecha_hora else (fecha_hora, "")

        autor_tag = article_soup.find("span", class_="author")
        autor_nombre = autor_tag.get_text(strip=True) if autor_tag else "Nombre no encontrado"
        autor_url = "URL no encontrada"

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
