from .InterfazEstrategia import InterfazEstrategia


class LibertadDigital(InterfazEstrategia):
    def extract_articles(self, soup):
        return soup.find_all("a",
                             href=lambda href: href and href.startswith("https://www.libertaddigital.com/defensa/"))

    def extract_details(self, article_soup):
        fuente = "Libertad Digital"
        fecha_hora = article_soup.find("time").get_text(strip=True) if article_soup.find(
            "time") else "Fecha no encontrada"
        fecha, hora = fecha_hora.split(" - ") if " - " in fecha_hora else (fecha_hora, "")

        autor_tag = article_soup.find("a", rel="author")
        autor_nombre = autor_tag.find("span").get_text(strip=True) if autor_tag and autor_tag.find(
            "span") else "Nombre no encontrado"
        autor_url = autor_tag["href"] if autor_tag and "href" in autor_tag.attrs else "URL no encontrada"

        titular_tag = article_soup.find("div", class_="heading")
        titular = titular_tag.find("h1").get_text(strip=True) if titular_tag and titular_tag.find(
            "h1") else "Titular no encontrado"

        noticia = " ".join([p.get_text(strip=True) for p in article_soup.find_all("p")])

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
