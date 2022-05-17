from urllib.request import urlopen

# https://nlp.stanford.edu/data/glove.6B.zip
url = "https://nlp.stanford.edu/data/glove.6B.zip"
save_as = "glove.6B.zip"

import time
def get_time():
    return time.perf_counter()

def time_since(t):
    ts = get_time()-t
    print(f"{ts:.2f} seconds")

t = get_time()
# Download from URL
with urlopen(url) as file:
    # Read & Decode: The .read() first downloads the data in a binary format
    # then the .decode() converts it to a string (if it's not binary)
    content = file.read() #.decode()
time_since(t)

# Save to file
with open(save_as, 'wb') as download:
    download.write(content)