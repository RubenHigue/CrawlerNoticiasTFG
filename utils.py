def extract_domain(url: str) -> str:
    """Extrae el dominio de una URL."""
    return url.split("//")[-1].split("/")[0]
