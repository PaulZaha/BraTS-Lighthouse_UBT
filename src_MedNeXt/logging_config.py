import logging

# Logger konfigurieren
logging.basicConfig(
    filename="training.log",
    encoding="utf-8",
    filemode="a",
    format="{name} - {asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)

# Den Logger ohne Verwendung des 'name' Parameter holen
logger = logging.getLogger(__name__)
