import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
from models import *
import time

def load_model(var1, var2):

    def decorator(func):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load('model1.pth', map_location='cpu')
        CP = circleParametrizer(spatial=True, device=device)
        CP.load_state_dict(state['model'])
        CP = CP.to(device)
        CP.eval()
        setattr(func, var1, CP)
        setattr(func, var2, device)
        return func

    return decorator


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


@load_model("CP", "device")
def find_circle(img):
    # Fill in this function
    img = torch.FloatTensor(img).view(1,1,img.shape[0],-1).to(find_circle.device)
    with torch.no_grad():
        PP = find_circle.CP(img)
    return int(PP[0][0].item()), int(PP[0][1].item()), int(PP[0][2].item())


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())


if __name__ == "__main__":
    main()
