import yaml
from cameras import *
from color_segmentation import ColorSegmentation

with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
cam = ImageWrapperCamera("../resources/images/shoes_near.jpeg", 0.2)
hue = np.array(config["color segmentation"]["hue"])
sat = np.array(config["color segmentation"]["sat"])
val = np.array(config["color segmentation"]["val"])
color_segmentation = ColorSegmentation(hue, sat, val)

cv2.namedWindow("video")

if cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) != 1.0:
    cv2.namedWindow("video")

while True:
    img = cam.take_picture()
    cv2.imshow("video", img)
    k = cv2.waitKey(1) & 0xFF
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
cv2.destroyAllWindows()