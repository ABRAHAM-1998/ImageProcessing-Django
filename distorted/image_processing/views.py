import cv2
import numpy as np
import pickle
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
import os


from PIL import Image
from scipy.ndimage import map_coordinates
import imageio
from tqdm import tqdm
from django.http import JsonResponse

current_dir = os.path.dirname(os.path.abspath(__file__))

pickle_file_path = os.path.join(current_dir, 'static', 'wide_dist_pickle.p')

dist_pickle = pickle.load(open(pickle_file_path, "rb"))

objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]


# code SevenZeroNine


def cal_undistort(img, objpoints, imgpoints):

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def upload_and_process_image(request):

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            file_url = fs.url(filename)

            img_path = settings.MEDIA_ROOT / filename
            img = cv2.imread(str(img_path))
            undistorted_img = cal_undistort(img, objpoints, imgpoints)

            undistorted_path = settings.MEDIA_ROOT / 'undistorted_image.png'
            cv2.imwrite(str(undistorted_path), undistorted_img)

            return render(request, 'result.html', {
                'original_image': file_url,
                'undistorted_image': settings.MEDIA_URL + 'undistorted_image.png',
            })
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})

# code SevenZeroNine

def map_to_sphere(x, y, z, yaw_radian, pitch_radian):
    theta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
    phi = np.arctan2(y, x)

    theta_prime = np.arccos(
        np.sin(theta) * np.sin(phi) * np.sin(pitch_radian)
        + np.cos(theta) * np.cos(pitch_radian)
    )
    phi_prime = np.arctan2(
        np.sin(theta) * np.sin(phi) * np.cos(pitch_radian)
        - np.cos(theta) * np.sin(pitch_radian),
        np.sin(theta) * np.cos(phi),
    )
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()

def interpolate_color(coords, img, method="bilinear"):

    order = {"nearest": 0, "bilinear": 1, "bicubic": 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode="reflect")
    green = map_coordinates(img[:, :, 1], coords, order=order, mode="reflect")
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode="reflect")
    return np.stack((red, green, blue), axis=-1)

def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    panorama = Image.open(panorama_path).convert("RGB")
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)

    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)

    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)

    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    x = u - W / 2
    y = H / 2 - v
    z = f

    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)
    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi

    coords = np.vstack((V.flatten(), U.flatten()))
    colors = interpolate_color(coords, pano_array)
    return Image.fromarray(colors.reshape((H, W, 3)).astype("uint8"), "RGB")


result =0
def generate_panorama_video(request):
    global result

    if request.method == "POST" and request.FILES.get("panorama_image"):
        result =0   

        panorama_image = request.FILES["panorama_image"]
        panorama_path = os.path.join(settings.MEDIA_ROOT, "panorama_input.jpg")

        with open(panorama_path, "wb") as f:
            for chunk in panorama_image.chunks():
                f.write(chunk)


        output_video_path = os.path.join(settings.MEDIA_ROOT, "output_panorama.mp4")
        writer = imageio.get_writer(output_video_path, fps=30, codec="libx264")

        total_frames = len(np.arange(0, 360, 0.25))
        i =0

        for deg in tqdm(np.arange(0, 360, 0.25)):

            output_image = panorama_to_plane(panorama_path, 90, (600, 600), deg, 90)
            writer.append_data(np.array(output_image))

            i+=1
            progress = int((i + 1) / total_frames * 100)
            request.session['progress'] = progress 
            result = progress
            if progress == 100:
                result=0


        writer.close()

        return render(request, 'result.html', {

            'original_image': settings.MEDIA_URL + "panorama_input.jpg",
            'undistorted_image': None,  # Optional
            'panorama_video_url': settings.MEDIA_URL + "output_panorama.mp4",

        })
    else:
        return render(request, 'upload.html', {'form': ImageUploadForm()})


def get_progress(request):

    progress = request.session.get('progress', 0)  

    return JsonResponse({'progress': result})