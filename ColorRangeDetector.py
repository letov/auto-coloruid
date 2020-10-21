import cv2
import numpy as numpy

image_path = "tests_data/SELECT_LEVEL.png"
hsv_max_upper = 0, 0, 0
hsv_min_lower = 255, 255, 255


def bite_range(value):
    value = 255 if value > 255 else value
    return 0 if value < 0 else value


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global hsv_max_upper
        global hsv_min_lower
        global image_hsv
        hsv_pixel = image_hsv[y, x]
        hsv_max_upper = bite_range(max(hsv_max_upper[0], hsv_pixel[0]) + 1), \
                        bite_range(max(hsv_max_upper[1], hsv_pixel[1]) + 1), \
                        bite_range(max(hsv_max_upper[2], hsv_pixel[2]) + 1)
        hsv_min_lower = bite_range(min(hsv_min_lower[0], hsv_pixel[0]) - 1), \
                        bite_range(min(hsv_min_lower[1], hsv_pixel[1]) - 1), \
                        bite_range(min(hsv_min_lower[2], hsv_pixel[2]) - 1)
        print('HSV range: ', (hsv_min_lower, hsv_max_upper))
        hsv_mask = cv2.inRange(image_hsv, numpy.array(hsv_min_lower), numpy.array(hsv_max_upper))
        cv2.imshow("HSV Mask", hsv_mask)


image = cv2.imread(image_path)
cv2.namedWindow('Original')
cv2.setMouseCallback('Original', pick_color)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
