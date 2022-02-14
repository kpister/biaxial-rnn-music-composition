from bs4 import BeautifulSoup
import requests
import os
from pathlib import Path

seen = set()


def download_midi_recursive(website: str, page: str, folder: Path):
    if page in seen:
        return

    seen.add(page)
    print("Downloading page " + page)

    html_page = requests.get(f"{website}/{page}")
    soup = BeautifulSoup(html_page.content)

    for link in soup.findAll("a"):
        url = link.attrs.get("href")

        if url and url.endswith(".mid"):
            filename = os.path.basename(url)
            midiurl = requests.get(f"{website}/{url}")
            fullpath = folder / filename

            if os.path.exists(fullpath):
                print("Skipping " + filename)
            else:
                print("Downloading " + filename)
                with open(fullpath, "wb") as local_file:
                    local_file.write(midiurl.content)

        if url and url.endswith(".htm") or url.endswith(".html"):
            try:
                relative_url = os.path.basename(url)
                download_midi_recursive(website, relative_url, folder)
            except Exception as e:
                print(e)


# website = "http://www.midiworld.com"
# page = "classic.htm"

website = "http://www.piano-midi.de"
page = "midi_files.htm"

folder = Path("./music")
download_midi_recursive(website, page, folder)
