import os
import requests
import tarfile


def load_img_data():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"

    data_path = '../data/'
    tar_path = os.path.join(data_path, 'images_dataSAT.tar')

    # Img checking.
    base_dir = os.path.join(data_path, 'images_dataSAT/')
    dir_non_agri = os.path.join(base_dir, 'class_0_non_agri/')
    dir_agri = os.path.join(base_dir, 'class_1_agri/')

    os.makedirs(dir_non_agri, exist_ok=True)
    os.makedirs(dir_agri, exist_ok=True)

    amount_of_class_0_img = len(os.listdir(dir_non_agri))
    amount_of_class_1_img = len(os.listdir(dir_agri))

    downloaded_data = False
    if amount_of_class_0_img > 0 and amount_of_class_1_img > 0:
        downloaded_data = True

    if not downloaded_data:
        print('Data not found, downloading...')
        r = requests.get(url, stream=True)

        with open(tar_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        with tarfile.open(tar_path) as tar:
            tar.extractall(path=data_path)
    else:
        print('Data already downloaded')

    amount_of_class_0_img = len(os.listdir(dir_non_agri))
    amount_of_class_1_img = len(os.listdir(dir_agri))
    print(f'Class 0 images = {amount_of_class_0_img}')
    print(f'Class 1 images = {amount_of_class_1_img}')
