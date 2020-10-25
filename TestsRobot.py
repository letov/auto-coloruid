import unittest
from Robot import Robot
import os
import cv2
import numpy

TESTS_DATA_PATH = "tests_data/"
IMAGE_DATA_PATH = "data/"
DEMO_DELAY = 1

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.robot = Robot()
        self.tests_data = {}
        self.images_data = {}
        for image_file in os.listdir(TESTS_DATA_PATH):
            self.tests_data[os.path.splitext(image_file)[0]] = cv2.imread(TESTS_DATA_PATH + image_file)
        for image_file in os.listdir(IMAGE_DATA_PATH):
            self.images_data[os.path.splitext(image_file)[0]] = cv2.imread(IMAGE_DATA_PATH + image_file)

    def test_get_dilate_contours_by_square_inx(self):
        for image_inx, image in self.images_data.items():
            color_inx = self.robot.COLOR_RED if image_inx == 'button_win' else self.robot.COLOR_ALL
            dilate_contours = self.robot.get_dilate_contours_by_square_inx(image, color_inx,
                                                                           self.robot.SQUARE_BIG_SYMBOL)
            self.assertEqual(len(dilate_contours), 1)
        image = self.tests_data['FAILED']
        image = self.robot.crop_image_by_points(image, (161, 285), (430, 330))
        dilate_contours = self.robot.get_dilate_contours_by_square_inx(image, self.robot.COLOR_RED,
                                                                       self.robot.SQUARE_BIG_SYMBOL)
        self.assertEqual(len(dilate_contours), 6)
        image = self.tests_data['STATE_GAME_SELECT_AREA']
        image = self.robot.crop_image_by_points(image, (292, 19), (357, 52))
        dilate_contours = self.robot.get_dilate_contours_by_square_inx(image, self.robot.COLOR_RED,
                                                                       self.robot.SQUARE_SMALL_SYMBOL)
        self.assertEqual(len(dilate_contours), 2)
        image = self.tests_data['TRAINING_SELECT_AREA']
        image = self.robot.crop_image_by_points(image, (292, 19), (357, 52))
        dilate_contours = self.robot.get_dilate_contours_by_square_inx(image, self.robot.COLOR_RED,
                                                                       self.robot.SQUARE_SMALL_SYMBOL)
        self.assertEqual(len(dilate_contours), 2)

    def test_get_button_of_squares_click(self):
        image = self.tests_data['START']
        button_play = self.robot.get_button_of_squares_click(image, 'button_play', self.robot.COLOR_BLUE)
        self.assertIsNot(button_play, False)
        button_failed = self.robot.get_button_of_squares_click(image, 'button_failed', self.robot.COLOR_RED)
        self.assertIs(button_failed, False)
        image = self.tests_data['FAILED']
        button_play = self.robot.get_button_of_squares_click(image, 'button_play', self.robot.COLOR_BLUE)
        self.assertIs(button_play, False)
        button_failed = self.robot.get_button_of_squares_click(image, 'button_failed', self.robot.COLOR_RED)
        self.assertIsNot(button_failed, False)
        image = self.tests_data['WIN']
        button_play = self.robot.get_button_of_squares_click(image, 'button_play', self.robot.COLOR_BLUE)
        self.assertIs(button_play, False)
        button_failed = self.robot.get_button_of_squares_click(image, 'button_failed', self.robot.COLOR_RED)
        self.assertIs(button_failed, False)

    def test_get_win_button_click(self):
        image = self.tests_data['WIN']
        button_win = self.robot.get_win_button_click(image)
        self.assertIsNot(button_win, False)

    def test_scan_digits(self):
        image = self.tests_data['SELECT_LEVEL']
        image = self.robot.crop_image_by_rectangle(image, numpy.array(self.robot.SELECT_LEVEL_AREA))
        red_digits = self.robot.scan_digits(image, self.robot.COLOR_RED, self.robot.SQUARE_BIG_SYMBOL)
        self.assertEqual(len(red_digits), self.robot.SELECT_LEVEL_COUNT - 1)
        for digit in range(0, len(red_digits)):
            self.assertEqual(digit + 2, red_digits[digit][0])
        image = self.tests_data['STATE_GAME_SELECT_AREA']
        image = self.robot.crop_image_by_points(image, (292, 19), (357, 52))
        red_digits = self.robot.scan_digits(image, self.robot.COLOR_RED, self.robot.SQUARE_SMALL_SYMBOL)
        self.assertEqual(red_digits[0][0], 93)
        image = self.tests_data['TRAINING_SELECT_AREA']
        red_digits = self.robot.scan_digits(image, self.robot.COLOR_RED, self.robot.SQUARE_SMALL_SYMBOL)
        self.assertEqual(red_digits[0][0], 0)
        self.assertEqual(red_digits[1][0], 1)
        self.assertEqual(red_digits[2][0], 79)

    def test_scan_game_statistic(self):
        image = self.tests_data['TRAINING_SELECT_COLOR']
        self.robot.screenshot = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self.robot.scan_game_statistic()
        self.assertEqual(self.robot.stat_step_current, 0)
        self.assertEqual(self.robot.stat_step_count, 1)
        self.assertEqual(self.robot.stat_score_current, 99)

    def test_get_training_cursor(self):
        image = self.tests_data['TRAINING_SELECT_COLOR']
        training_cursor = self.robot.get_training_cursor(image)
        self.assertIsNot(training_cursor, False)

    def test_state_select_color_condition(self):
        image = self.tests_data['TRAINING_SELECT_AREA']
        self.robot.screenshot = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self.robot.set_select_color_next(self.robot.COLOR_BLUE)
        self.assertIs(self.robot.state_select_area_condition(), True)

    def test_analysis_color_areas(self):
        image = self.tests_data['LEVEL_4_1']
        self.robot.screenshot = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self.robot.scan_color_areas()
        self.robot.analysis_color_areas()
        self.assertEqual(self.robot.select_color_next, self.robot.COLOR_GREEN)
        self.assertEqual(self.robot.color_area_inx_next, 0)

if __name__ == '__main__':
    unittest.main()
