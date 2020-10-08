import yaml
from utils.cameras import *
with open('config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
cam = ImageWrapperCamera("../resources/images/shoes_near.jpeg", 0.2)

cv2.namedWindow("Feed")

while True:

    img = cam.take_picture()
    cv2.imshow("Feed", img)

    k = cv2.waitKey(1) & 0xFF
    if k % 256 == 27:
        break

cv2.destroyAllWindows()