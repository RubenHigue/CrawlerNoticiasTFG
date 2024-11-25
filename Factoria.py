from Estrategias.LibertadDigital import LibertadDigital


class Factoria:
    @staticmethod
    def get_scraper(domain: str):
        if "libertaddigital.com" in domain:
            return LibertadDigital()
        # Añadir más dominios aquí
        raise ValueError(f"No hay scraper definido para el dominio: {domain}")
