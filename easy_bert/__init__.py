import logging
import sys

logger = logging.getLogger('easy-bert')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)
