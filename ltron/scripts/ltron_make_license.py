import ltron.license as license

def main():
    with open('./LICENSE', 'w') as f:
        f.write(license.generate_license())
