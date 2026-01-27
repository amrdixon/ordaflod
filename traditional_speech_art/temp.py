import certifi
import ssl
import urllib.request

ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())