import requests
from bs4 import BeautifulSoup
from Factoria import Factoria
import csv
import time


def crawl_website(url: str, output_file: str):
    domain = url.split("//")[-1].split("/")[0]
    scraper = Factoria.get_scraper(domain)

    with open(output_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["fuente", "fecha", "hora", "titular", "autor", "autor_url", "noticia", "articulo_original", "url"])

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        articles = scraper.extract_articles(soup)
        for article in articles:
            article_url = article["href"]
            article_response = requests.get(article_url)
            article_soup = BeautifulSoup(article_response.content, "html.parser")

            details = scraper.extract_details(article_soup)
            details["url"] = article_url
            writer.writerow(details.values())

            time.sleep(1)
