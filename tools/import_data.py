import s3fs
import os
import zipfile
import json

# Configuration

MON_BUCKET = "omarboumhaousse" 
DOSSIER_CIBLE = "diffusion"

# Créer un dossier 'pdfs' et 'images' pour y mettre les données
PDFS_LOCAL_DIR = "pdfs"
IMAGES_LOCAL_DIR = "images"
os.makedirs(PDFS_LOCAL_DIR, exist_ok=True)
os.makedirs(IMAGES_LOCAL_DIR, exist_ok=True)

# CONNEXION AU CLOUD

print("Connexion à MinIO...")
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

# TÉLÉCHARGEMENT DU JSON

print("\n Téléchargement de OmniDocBench.json...")
chemin_json_s3 = f"{MON_BUCKET}/{DOSSIER_CIBLE}/OmniDocBench.json"
chemin_json_local = os.path.join("./", "OmniDocBench.json")

if fs.exists(chemin_json_s3):
    fs.get(chemin_json_s3, chemin_json_local)
    
else:
    print("JSON introuvable sur le cloud.")

# TÉLÉCHARGEMENT ET EXTRACTION DES PDFs

print("\n Téléchargement de pdfs.zip...")
chemin_zip_s3 = f"{MON_BUCKET}/{DOSSIER_CIBLE}/pdfs.zip"
chemin_zip_local = os.path.join(PDFS_LOCAL_DIR, "pdfs.zip")
# dossier_pdf_local = "./"

if fs.exists(chemin_zip_s3):
    # Téléchargement
    fs.get(chemin_zip_s3, chemin_zip_local)
    
    # Unzip
    os.system(f'unzip -o "{chemin_zip_local}" -d "./"')
    
    # Suppression du zip
    os.remove(chemin_zip_local)
    
    
else:
    print("Le fichier pdfs.zip est introuvable sur le cloud.")

# TÉLÉCHARGEMENT ET EXTRACTION DES IMAGES

print("\n Téléchargement de images.zip...")
chemin_zip_s3 = f"{MON_BUCKET}/{DOSSIER_CIBLE}/images.zip"
chemin_zip_local = os.path.join(IMAGES_LOCAL_DIR, "images.zip")
# dossier_images_local = "./"

if fs.exists(chemin_zip_s3):
    # Téléchargement
    fs.get(chemin_zip_s3, chemin_zip_local)

    # Unzip  
    os.system(f'unzip -o "{chemin_zip_local}" -d "./"')
    
    # Suppression du zip
    os.remove(chemin_zip_local)

else:
    print("Le fichier images.zip est introuvable sur le cloud.")
