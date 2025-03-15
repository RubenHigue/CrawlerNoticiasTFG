from Estrategias.ElDebate import ElDebate
from Estrategias.ElMundoDefensa import ElMundoDefensa
from Estrategias.GalaxiaMilitar import GalaxiaMilitar
from Estrategias.LibertadDigital import LibertadDigital


class Factoria:
    @staticmethod
    def get_scraper(domain: str):
        if "libertaddigital.com" in domain:
            return LibertadDigital()
        if "eldebate.com" in domain:
            return ElDebate()
        if "galaxiamilitar.es" in domain:
            return GalaxiaMilitar()
        if "elmundo.es" in domain:
            return ElMundoDefensa()
        # Añadir más dominios aquí
        raise ValueError(f"No hay scraper definido para el dominio: {domain}")
