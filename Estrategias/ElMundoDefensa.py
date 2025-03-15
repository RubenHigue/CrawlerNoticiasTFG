from datetime import datetime

from .InterfazEstrategia import InterfazEstrategia


class ElMundoDefensa(InterfazEstrategia):
    def extract_articles(self, soup):
        articles = soup.find_all("a", class_="ue-c-cover-content__link",
                                 href=lambda href: href and href.startswith("https://www.elmundo.es/")
                                                   and "https://www.elmundo.es/autor" not in href)
        return articles

    def extract_details(self, article_soup):
        fuente = "El Mundo"

        # Extraer fecha
        time_tag = article_soup.find("time")

        fecha, hora = "", ""

        if time_tag and "datetime" in time_tag.attrs:
            fecha_iso = time_tag["datetime"]

            fecha_obj = datetime.strptime(fecha_iso, "%Y-%m-%dT%H:%M:%SZ")

            fecha = fecha_obj.strftime("%d/%m/%Y")
            hora = fecha_obj.strftime("%H:%M:%S")

        # Extraer autor
        autor_tag = article_soup.find("div", class_="ue-c-article__author-name-item")
        link_autor = autor_tag.find("a")
        autor = link_autor.get_text(strip=True) if autor_tag and autor_tag.find("a") else "Redacción"
        autor_url = link_autor["href"] if link_autor and "href" in link_autor.attrs else "URL no encontrada"

        # Extraer titular
        titular_tag = article_soup.find("h1")
        titular = titular_tag.get_text(strip=True) if titular_tag else "Titular no encontrado"

        # Extraer contenido del artículo
        noticia = " ".join([p.get_text(strip=True) for p in
                            article_soup.find_all("p", class_="ue-c-article__paragraph")])

        return {
            "fuente": fuente,
            "fecha": fecha,
            "hora": hora,
            "titular": titular,
            "autor": autor,
            "autor_url": autor_url,
            "noticia": noticia,
            "articulo_original": article_soup.prettify(),
        }
