import os
import time
import cv2
import numpy
import math
from PIL import Image
from io import BytesIO

class Robot:
    STATE_START = 0x01, 'START'  # cостояния игрового процесса
    STATE_SELECT_LEVEL = 0x02, 'SELECT_LEVEL'
    STATE_TRAINING_SELECT_COLOR = 0x03, 'TRAINING_SELECT_COLOR'
    STATE_TRAINING_SELECT_AREA = 0x04, 'TRAINING_SELECT_AREA'
    STATE_GAME_SELECT_COLOR = 0x05, 'GAME_SELECT_COLOR'
    STATE_GAME_SELECT_AREA = 0x06, 'GAME_SELECT_AREA'
    STATE_GAME_RESULT = 0x07, 'GAME_RESULT'

    SQUARE_BIG_SYMBOL = 0x01
    SQUARE_SMALL_SYMBOL = 0x02
    SQUARE_BIG_ELEMENT = 0x03

    SQUARE_SIZES = {
        SQUARE_BIG_SYMBOL: 9,  # большие символы, выбор уровня
        SQUARE_SMALL_SYMBOL: 7,  # маленькие символы, очки игры
        SQUARE_BIG_ELEMENT: 12,  # квадраты выбора цвета
    }

    MAX_MATCH_SHAPES_BUTTON = 0.1
    MAX_MATCH_SHAPES_DIGITS = 0.15

    MAX_STATE_TIMEOUT = 30  # максимальный таймаут текущего состояния
    DEFAULT_STATE_TIMEOUT = 10  # дефолтный таймаут ожидания успешной смены состояния

    IMAGE_DATA_PATH = "data/"  # файлы кнопок и цифр в игре

    SELECT_LEVEL_AREA = 41, 64, 531, 521  # область выбора уровней
    SELECT_LEVEL_COUNT = 20  # количество уровней
    GAME_STEP_AREA = 352, 2, 148, 63  # область с состоянием шагов
    GAME_SCORE_AREA = 284, 2, 71, 63  # область с очками
    SELECT_COLOR_AREA = 35, 530, 527, 68  # область выбора цветов
    SELECT_COLOR_COUNT = 5  # количество цветов для выбора
    GAME_MAIN_AREA = 27, 73, 544, 452  # главное игровое поле

    CLICK_AREA = 20

    COLOR_BLUE = 0x01  # индексы цветов
    COLOR_ORANGE = 0x02
    COLOR_RED = 0x03
    COLOR_GREEN = 0x04
    COLOR_YELLOW = 0x05
    COLOR_WHITE = 0x06
    COLOR_ALL = 0x07

    COLOR_HSV_RANGE = {  # диапазоны цветов HSV для фильтров [min,max]
        COLOR_BLUE: ((112, 151, 216), (128, 167, 255)),
        COLOR_ORANGE: ((8, 251, 93), (14, 255, 255)),
        COLOR_RED: ((167, 252, 223), (171, 255, 255)),
        COLOR_GREEN: ((71, 251, 98), (77, 255, 211)),
        COLOR_YELLOW: ((27, 252, 51), (33, 255, 211)),
        COLOR_WHITE: ((0, 0, 159), (7, 7, 255)),
    }

    class ColorArea:  # область цвета на игровом поле
        def __init__(self, color_inx, click_point, contour, select_color_weights):
            self.color_inx = color_inx  # индекс цвета
            self.click_point = click_point  # клик поинт области
            self.contour = contour  # контур области
            self.neighbors = []  # индексы соседей
            self.select_color_weights = select_color_weights  # [COLOR_BLUE...COLOR_YELLOW] веса цветов

        def set_neighbors(self, neighbors):  # установить соседей
            self.neighbors = neighbors

        def set_select_color_weights(self, select_color_weights):  # установить веса цветов
            self.select_color_weights = select_color_weights

    def __init__(self):
        self.screenshot = 0  # скриншот
        self.states = {
            self.STATE_START[0]: self.state_start,
            self.STATE_SELECT_LEVEL[0]: self.state_select_level,
            self.STATE_TRAINING_SELECT_COLOR[0]: self.state_training_select_color,
            self.STATE_TRAINING_SELECT_AREA[0]: self.state_training_select_area,
            self.STATE_GAME_RESULT[0]: self.state_game_result,
            self.STATE_GAME_SELECT_COLOR[0]: self.state_game_select_color,
            self.STATE_GAME_SELECT_AREA[0]: self.state_game_select_area,
        }
        self.state_next_success_condition = None  # условие успешного изменения состояния
        self.state_next_success_condition_check_count = 0  # количество попыток state_next_success_condition
        self.state_start_time = 0  # время установки state_next
        self.state_timeout = 0  # таймаут state_next
        self.state_current = 0  # текущее состояние игры
        self.state_next = 0  # выбранное состояние игры
        self.color_areas = []  # список ColorArea текушего шага игры
        self.color_area_inx_next = 0  # индекс color_areas для следующего хода
        self.level_current = 0  # уровень
        self.stat_step_count = 0  # доступно ходов на уровне
        self.stat_step_current = 0  # текущий ход на уровне
        self.stat_step_last = 0  # последний ход на уровне, для ожидания анимации перекраски
        self.stat_score_current = 0  # очки
        self.select_color_current = 0  # текущий цвет
        self.select_color_next = 0  # выбранный цвет
        self.dilate_contours_bi_data = {}  # контуры всех изображений из IMAGE_DATA_PATH
        for image_file in os.listdir(self.IMAGE_DATA_PATH):
            image = cv2.imread(self.IMAGE_DATA_PATH + image_file)
            contour_inx = os.path.splitext(image_file)[0]
            color_inx = self.COLOR_RED if contour_inx == 'button_win' else self.COLOR_ALL
            dilate_contours = self.get_dilate_contours_by_square_inx(image, color_inx, self.SQUARE_BIG_SYMBOL)
            self.dilate_contours_bi_data[contour_inx] = dilate_contours[0]
        self.set_state_next(self.STATE_START, self.state_start_condition, 500)

    """
    Внешние методы
    """

    def run(self, screenshot):
        self.set_screenshot(screenshot)
        if self.state_current != self.state_next:
            if self.state_next_success_condition_check_count == 0:
                print("check state condition: " + str(self.state_next[1]))
            self.state_next_success_condition_check_count += 1
            if self.state_next_success_condition():
                self.set_state_current()
            elif time.time() - self.state_start_time >= self.state_timeout:
                print("state failed: " + str(self.state_next[1]))
                if self.state_next == self.STATE_GAME_RESULT:
                    self.state_next = self.STATE_GAME_SELECT_COLOR
                    self.set_state_current()
                else:
                    self.state_next = self.state_current
            return False
        else:
            try:
                click_point = self.states[self.state_current[0]]()
                if click_point is not False:
                    print("make click (" + str(click_point[0]) + "," + str(click_point[1]) + ")")
                return click_point
            except KeyError:
                self.__del__()

    def emergency_timeout(self):
        """
        Проверяет не ушло ли текущее состояние в максимальный таймаут
        :return: true/false
        """
        return time.time() - self.state_start_time >= self.MAX_STATE_TIMEOUT and self.state_next != self.STATE_START

    """
    Сеттеры
    """

    def set_screenshot(self, screenshot):
        self.screenshot = cv2.cvtColor(numpy.array(Image.open(BytesIO(screenshot))), cv2.COLOR_BGR2RGB)

    def set_state_current(self):
        """
        Подтвердить выбранное состояние игрового процесса
        """
        if self.state_current == self.state_next:
            return
        print("current state " + str(self.state_next[1]))
        self.state_current = self.state_next

    def set_state_next(self, state_next, state_next_success_condition, state_timeout):
        """
        Установить выбранное состояние игрового процесса
        :param state_next: следующее состояние игрового процесса
        :param state_next_success_condition: условие принятия состояния
        :param state_timeout: таймаут принятия состояния
        """
        if self.state_current == state_next:
            return
        self.state_next_success_condition = state_next_success_condition
        self.state_start_time = time.time()
        self.state_timeout = state_timeout
        print("next state " + str(state_next[1]))
        self.state_next = state_next
        self.state_next_success_condition_check_count = 0

    def set_new_level(self, new_level):
        print("new level: " + str(new_level))
        self.level_current = new_level
        self.stat_step_count = 0
        self.stat_step_current = 0
        self.stat_score_current = 0
        self.select_color_current = 0
        self.select_color_next = 0

    def set_select_color_next(self, color_next):
        """
        Новый выбранный цвет
        """
        self.select_color_next = color_next

    def set_select_color_current(self):
        """
        Подтвердить выбранный цвет
        """
        print("selected color: " + str(self.select_color_next))
        self.select_color_current = self.select_color_next
        self.select_color_next = 0

    """
    Сканеры
    """

    def scan_game_statistic(self):
        """
        Сканируем статистику игры
        :return:
        """
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array((self.GAME_STEP_AREA)))
        game_step = self.scan_digits(image, self.COLOR_RED, self.SQUARE_BIG_SYMBOL)
        if game_step is False or len(game_step) != 2:
            return False
        self.stat_step_current = game_step[0][0]
        self.stat_step_count = game_step[1][0]
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array((self.GAME_SCORE_AREA)))
        game_score = self.scan_digits(image, self.COLOR_RED, self.SQUARE_SMALL_SYMBOL)
        if game_score is False or len(game_score) != 1:
            return False
        self.stat_score_current = game_score[0][0]
        return True

    def scan_color_areas(self):
        """
        Сканируем игровую область и собираем self.color_areas
        """
        self.color_areas = []
        self.color_areas_color_count = [0] * self.SELECT_COLOR_COUNT
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array(self.GAME_MAIN_AREA))
        for color_inx in range(1, self.SELECT_COLOR_COUNT + 1):
            dilate_contours = self.get_dilate_contours(image, color_inx, 10)
            for dilate_contour in dilate_contours:
                click_point = tuple(
                    dilate_contour[dilate_contour[:, :, 1].argmin()].flatten() + [0, int(self.CLICK_AREA)])
                self.color_areas_color_count[color_inx - 1] += 1
                color_area = self.ColorArea(color_inx, click_point, dilate_contour, [0] * self.SELECT_COLOR_COUNT)
                self.color_areas.append(color_area)
        blank_image = numpy.zeros_like(image)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        for color_area_inx_1 in range(0, len(self.color_areas)):
            for color_area_inx_2 in range(color_area_inx_1 + 1, len(self.color_areas)):
                color_area_1 = self.color_areas[color_area_inx_1]
                color_area_2 = self.color_areas[color_area_inx_2]
                if color_area_1.color_inx == color_area_2.color_inx:
                    continue
                common_image = cv2.drawContours(blank_image.copy(), [color_area_1.contour, color_area_2.contour],
                                                -1, (255, 255, 255), cv2.FILLED)
                kernel = numpy.ones((15, 15), numpy.uint8)
                common_image = cv2.dilate(common_image, kernel, iterations=1)
                common_contour, _ = cv2.findContours(common_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(common_contour) == 1:
                    self.color_areas[color_area_inx_1].neighbors.append(color_area_inx_2)
                    self.color_areas[color_area_inx_2].neighbors.append(color_area_inx_1)

    def analysis_color_areas(self):
        """
        Устанавливаем веса цветов
        select_color_weights: строка - индекс области, столбец - индекс цвета,
            ячейка - вес цвета
        Распределение весов:
        соседи области одинаковых цветов, +1 к ячейке данного цвета
        соседи области одинаковых цветов и это все цвета на уровне +10 к ячейке данного цвета
        область с уникальным цветом на карте, все ячейки строки +10
        """
        select_color_weights = []
        for color_area_inx in range(0, len(self.color_areas)):
            color_area = self.color_areas[color_area_inx]
            select_color_weight = numpy.array([0] * self.SELECT_COLOR_COUNT)
            for select_color_weight_inx in color_area.neighbors:
                neighbor_color_area = self.color_areas[select_color_weight_inx]
                select_color_weight[neighbor_color_area.color_inx - 1] += 1
            for color_inx in range(0, len(select_color_weight)):
                color_count = select_color_weight[color_inx]
                if color_count != 0 and self.color_areas_color_count[color_inx] == color_count:
                    select_color_weight[color_inx] += 10
            if self.color_areas_color_count[color_area.color_inx - 1] == 1:
                select_color_weight = [x + 10 for x in select_color_weight]
            color_area.set_select_color_weights(select_color_weight)
            select_color_weights.append(select_color_weight)
        select_color_weights = numpy.array(select_color_weights)
        # ищем максимальный вес и определяем по какой обласит каким цветом будет клик
        max_index = select_color_weights.argmax()
        self.color_area_inx_next = max_index // self.SELECT_COLOR_COUNT
        select_color_next = (max_index % self.SELECT_COLOR_COUNT) + 1
        self.set_select_color_next(select_color_next)

    """ 
    Методы состояния игрового процесса и проверки корректности состояний
    """

    def state_start_condition(self):
        """
        Готовность стартового экрана
        """
        return self.get_button_of_squares_click(self.screenshot, 'button_play', self.COLOR_BLUE) is not False

    def state_start(self):
        """
        Экран приветствия, поиск и клик кнопки button_play
        """
        button_play = self.get_button_of_squares_click(self.screenshot, 'button_play', self.COLOR_BLUE)
        if button_play is False:
            return False
        self.set_state_next(self.STATE_SELECT_LEVEL, self.state_select_level_condition, self.DEFAULT_STATE_TIMEOUT)
        return button_play

    def state_select_level_condition(self):
        """
        Готовность выбора уровней
        """
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array(self.SELECT_LEVEL_AREA))
        green_digits = self.scan_digits(image, self.COLOR_GREEN, self.SQUARE_BIG_SYMBOL)
        red_digits = self.scan_digits(image, self.COLOR_RED, self.SQUARE_BIG_SYMBOL)
        return green_digits is not False and red_digits is not False and \
               len(green_digits + red_digits) == self.SELECT_LEVEL_COUNT

    def state_select_level(self):
        """
        Выбираем уровень
        """
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array(self.SELECT_LEVEL_AREA))
        green_digits = self.scan_digits(image, self.COLOR_GREEN, self.SQUARE_BIG_SYMBOL)
        if green_digits is False:
            print("No levels")
            return False
        click_point = self.add_rectangle_to_click(self.SELECT_LEVEL_AREA, green_digits[0][1])
        level = green_digits[0][0]
        if level == 1:
            self.set_state_next(self.STATE_TRAINING_SELECT_COLOR, self.state_new_level_condition,
                                self.DEFAULT_STATE_TIMEOUT)
        else:
            self.set_state_next(self.STATE_GAME_SELECT_COLOR, self.state_new_level_condition,
                                self.DEFAULT_STATE_TIMEOUT)
        return click_point

    def state_new_level_condition(self):
        """
        Проверить начали ли новый уровень
        Неправильноый счетчик уровней, если был Failed, все равно прибавляем 1
        """
        game_statistic = self.scan_game_statistic()
        if game_statistic is False:
            return False
        if self.stat_step_current == 0:
            self.set_new_level(self.level_current + 1)
            return True

    def state_training_select_color(self):
        """
        Выбираем цвет на учебном уровне
        """
        game_statistic = self.scan_game_statistic()
        if game_statistic is False:
            return False
        training_cursor = self.get_training_cursor(self.screenshot)
        if training_cursor is False:
            return False
        self.set_select_color_next(self.COLOR_BLUE)
        self.set_state_next(self.STATE_TRAINING_SELECT_AREA, self.state_select_area_condition,
                            self.DEFAULT_STATE_TIMEOUT)
        return training_cursor

    def state_training_select_area(self):
        """
        Кликаем по учебному курсору
        """
        game_statistic = self.scan_game_statistic()
        if game_statistic is False:
            return False
        if self.select_color_current != self.COLOR_BLUE:
            return False
        training_cursor = self.get_training_cursor(self.screenshot)
        if training_cursor is False:
            return False
        self.stat_step_last = self.stat_step_current
        self.set_state_next(self.STATE_GAME_RESULT, self.state_game_result_condition, self.DEFAULT_STATE_TIMEOUT)
        return training_cursor

    def state_game_result_condition(self):
        """
        Проверяем завершен ли уровень
        """
        self.scan_game_statistic()
        if self.stat_step_current == self.stat_step_count:
            button_win = self.get_win_button_click(self.screenshot)
            button_failed = self.get_button_of_squares_click(self.screenshot, 'button_failed', self.COLOR_RED)
            return button_win is not False or button_failed is not False
        elif self.stat_step_last != self.stat_step_current:
            # если сменился игровой шаг, выбираем новый цвет
            self.set_state_next(self.STATE_GAME_SELECT_COLOR, lambda: True, self.DEFAULT_STATE_TIMEOUT)
        else:
            return False

    def state_game_result(self):
        """
        Переходим на следующий уровень или повторно проходим текущий
        """
        button_win = self.get_win_button_click(self.screenshot)
        if button_win is not False:
            self.set_state_next(self.STATE_GAME_SELECT_COLOR, self.state_new_level_condition,
                                self.DEFAULT_STATE_TIMEOUT)
            return button_win
        button_failed = self.get_button_of_squares_click(self.screenshot, 'button_failed', self.COLOR_RED)
        if button_failed is not False:
            self.set_state_next(self.STATE_GAME_SELECT_COLOR, self.state_new_level_condition,
                                self.DEFAULT_STATE_TIMEOUT)
            return button_failed
        self.set_state_next(self.STATE_GAME_SELECT_COLOR, lambda: True,
                            self.DEFAULT_STATE_TIMEOUT)
        return False

    def state_game_select_color(self):
        """
        Анализируем игровую область и выбираем цвет/область для клика
        """
        game_statistic = self.scan_game_statistic()
        if game_statistic is False:
            return False
        self.color_area_inx_next = 0
        self.scan_color_areas()
        self.analysis_color_areas()
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array(self.SELECT_COLOR_AREA))
        color_rectangles = self.get_dilate_contours_by_square_inx(image, self.select_color_next,
                                                                  self.SQUARE_BIG_ELEMENT)
        color_rectangles = [cnt for cnt in color_rectangles if cv2.contourArea(cnt) > 2000]
        if len(color_rectangles) != 1:
            return False
        click_point = self.add_rectangle_to_click(self.SELECT_COLOR_AREA, self.get_contour_centroid(color_rectangles[0]))
        self.set_state_next(self.STATE_GAME_SELECT_AREA, self.state_select_area_condition, self.DEFAULT_STATE_TIMEOUT)
        return click_point

    def state_select_area_condition(self):
        """
        Проверяем выбран ли квадрат цвета self.select_color_next и устанавливаем текущий цвет
        """
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array(self.SELECT_COLOR_AREA))
        color_rectangles = self.get_dilate_contours_by_square_inx(image, self.select_color_next,
                                                                  self.SQUARE_BIG_ELEMENT)
        color_rectangles = [cnt for cnt in color_rectangles if cv2.contourArea(cnt) > 2000]
        if len(color_rectangles) != 1:
            return False
        color_rectangle = color_rectangles[0]
        extreme_top_point = color_rectangle[color_rectangle[:, :, 1].argmin()].flatten()
        centroid = self.get_contour_centroid(color_rectangle)
        if abs(extreme_top_point[0] - centroid[0]) < 4:
            self.set_select_color_current()
            return True
        else:
            return False

    def state_game_select_area(self):
        # кликаем по области и смотрим резульат
        game_statistic = self.scan_game_statistic()
        if game_statistic is False:
            return False
        self.stat_step_last = self.stat_step_current
        click_point = self.add_rectangle_to_click(self.GAME_MAIN_AREA,
                                                  self.color_areas[self.color_area_inx_next].click_point)
        self.set_state_next(self.STATE_GAME_RESULT, self.state_game_result_condition, self.DEFAULT_STATE_TIMEOUT)
        return click_point

    """ 
    Зрение
    """

    def get_dilate_contours(self, image, color_inx, distance):
        """
        Объединяет объекты изображения image в общие контуры
        :param image: входное изображение
        :param color_inx: индекс COLOR_HSV_RANGE для цветофильтра, 0 - бимодальный
        :param distance: расстояние между объектами для слияния
        :return: список контуров / False
        """
        thresh = self.get_color_thresh(image, color_inx)
        if thresh is False:
            return []
        kernel = numpy.ones((distance, distance), numpy.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_dilate_contours_by_square_inx(self, image, color_inx, square_inx):
        """
        Оболочка для get_dilate_contours, distance по square_inx
        :param image: входное изображение
        :param color_inx: индекс COLOR_HSV_RANGE для цветофильтра, 0 - бимодальный
        :param square_inx: индекс размера квадратов SQUARE_SIZES
        :return: список контуров
        """
        distance = math.ceil(self.SQUARE_SIZES[square_inx] / 2)
        return self.get_dilate_contours(image, color_inx, distance)

    def get_color_thresh(self, image, color_inx):
        """
        Возвращает пороговое изображение по color_inx
        :param image: входное изображение
        :param color_inx: индекс COLOR_HSV_RANGE
        :return: выходное изображение / False
        """
        try:
            if color_inx == self.COLOR_ALL:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(image, self.COLOR_HSV_RANGE[color_inx][0], self.COLOR_HSV_RANGE[color_inx][1])
            return thresh
        except cv2.error as e:
            return False

    def get_match_shapes(self, dilate_contour, contour_inx):
        """
        Оболочка для matchShapes
        :param dilate_contour: входной контур
        :param contour_inx: индекс self.dilate_contours_bi_data
        :return: matchShapes result
        """
        return cv2.matchShapes(dilate_contour, self.dilate_contours_bi_data[contour_inx], cv2.CONTOURS_MATCH_I1, 0)

    def filter_contours_of_rectangles(self, contours):
        """
        Фильтруем контуры, оставляя прямоугольники с максимальным соотношением сторон 1/3
        :param contours: входные контуры
        :return: выходные контуры
        """
        squares = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            square = cv2.boxPoints(rect)
            square = numpy.int0(square)
            (_, _, w, h) = cv2.boundingRect(square)
            a = max(w, h)
            b = min(w, h)
            if numpy.unique(square).shape[0] <= 4 and a <= b * 3:
                squares.append(numpy.array([[square[0]], [square[1]], [square[2]], [square[3]]]))
        return squares

    def get_contours_of_squares(self, image, color_inx, square_inx):
        """
        Группирует объекты из квадратов, параллельные осям в единый контур
        :param image: входное изображение
        :param color_inx: индекс COLOR_HSV_RANGE для цветофильтра квадратов
        :param square_inx: индекс размера квадратов SQUARE_SIZES
        :return: список контуров или False
        """
        thresh = self.get_color_thresh(image, color_inx)
        if thresh is False:
            return False
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_of_squares = self.filter_contours_of_rectangles(contours)
        if len(contours_of_squares) < 1:
            return False
        image_zero = numpy.zeros_like(image)
        image_zero = cv2.cvtColor(image_zero, cv2.COLOR_BGR2RGB)
        cv2.drawContours(image_zero, contours_of_squares, -1, (255, 255, 255), -1)
        dilate_contours = self.get_dilate_contours_by_square_inx(image_zero, self.COLOR_ALL, square_inx)
        square_area = pow(self.SQUARE_SIZES[square_inx], 2)
        min_contour_area = square_area * 4
        dilate_contours = [cnt for cnt in dilate_contours if cv2.contourArea(cnt) > min_contour_area]
        if len(dilate_contours) < 1:
            return False
        else:
            return dilate_contours

    def get_button_of_squares_click(self, image, contour_inx, color_inx):
        """
        Возвращает координату клика для кнопки contour_inx, которая состоит из квадратов цвета color_inx
        :param image: входное изображение
        :param contour_inx: индекс self.dilate_contours_bi_data
        :param color_inx: индекс COLOR_HSV_RANGE для цветофильтра квадратов
        :return: координата клика (x,y) или False
        """
        square_inx = self.SQUARE_BIG_SYMBOL
        contours_of_squares = self.get_contours_of_squares(image, color_inx, square_inx)
        if contours_of_squares is False:
            return False
        for contour_of_square in contours_of_squares:
            crop_image = self.crop_image_by_contour(image, contour_of_square)
            dilate_contours = self.get_dilate_contours_by_square_inx(crop_image, self.COLOR_ALL, square_inx)
            if len(dilate_contours) < 1:
                continue
            dilate_contour = dilate_contours[0]
            if self.get_match_shapes(dilate_contour, contour_inx) < self.MAX_MATCH_SHAPES_BUTTON:
                return self.get_contour_centroid(contour_of_square)
        return False

    def get_win_button_click(self, image):
        """
        Возвращает координату клика для кнопки button_win
        :param image: входное изображение
        :return: координата клика (x,y) или False
        """
        square_inx = self.SQUARE_BIG_ELEMENT
        distance = math.ceil(self.SQUARE_SIZES[square_inx] / 2)
        dilate_contours = self.get_dilate_contours(image, self.COLOR_RED, distance)
        if dilate_contours is False:
            return False
        for dilate_contour in dilate_contours:
            if cv2.contourArea(dilate_contour) < 2000:
                continue
            if self.get_match_shapes(dilate_contour, 'button_win') < 0.2:
                return self.get_contour_centroid(dilate_contour)
        return False

    def get_training_cursor(self, image):
        """
        Возвращает координату клика для учебного курсора
        :param image: входное изображение
        :return: координата клика (x,y) или False
        """
        thresh = self.get_color_thresh(image, self.COLOR_WHITE)
        if thresh is False:
            return False
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        if len(contours) != 1:
            return False
        return self.get_contour_centroid(contours[0])

    def scan_digits(self, image, color_inx, square_inx):
        """
        Распознает цифры на изображении
        :param image: входное изображение
        :param color_inx: цвет цифр
        :param square_inx: индекс размера квадратов SQUARE_SIZES
        :return: [[digit, centroid]]
        """
        result = []
        contours_of_squares = self.get_contours_of_squares(image, color_inx, square_inx)
        before_digit_x, before_digit_y = (-100, -100)
        if contours_of_squares is False:
            return result
        for contour_of_square in reversed(contours_of_squares):
            crop_image = self.crop_image_by_contour(image, contour_of_square)
            dilate_contours = self.get_dilate_contours_by_square_inx(crop_image, self.COLOR_ALL, square_inx)
            if len(dilate_contours) < 1:
                continue
            dilate_contour = dilate_contours[0]
            match_shapes = {}
            for digit in range(0, 10):
                match_shapes[digit] = self.get_match_shapes(dilate_contour, 'digit_' + str(digit))
            min_match_shape = min(match_shapes.items(), key=lambda x: x[1])
            if len(min_match_shape) > 0 and (min_match_shape[1] < self.MAX_MATCH_SHAPES_DIGITS):
                digit = min_match_shape[0]
                if digit == 6 or digit == 9:
                    extreme_bottom_point = dilate_contour[dilate_contour[:, :, 1].argmax()].flatten()
                    x_points = dilate_contour[:, :, 0].flatten()
                    extreme_right_points_args = numpy.argwhere(x_points == numpy.amax(x_points))
                    extreme_right_points = dilate_contour[extreme_right_points_args]
                    extreme_top_right_point = extreme_right_points[extreme_right_points[:, :, :, 1].argmin()].flatten()
                    if extreme_top_right_point[1] > round(extreme_bottom_point[1] / 2):
                        digit = 6
                    else:
                        digit = 9
                if digit == 2 or digit == 5:
                    extreme_right_point = dilate_contour[dilate_contour[:, :, 0].argmax()].flatten()
                    y_points = dilate_contour[:, :, 1].flatten()
                    extreme_top_points_args = numpy.argwhere(y_points == numpy.amin(y_points))
                    extreme_top_points = dilate_contour[extreme_top_points_args]
                    extreme_top_right_point = extreme_top_points[extreme_top_points[:, :, :, 0].argmax()].flatten()
                    if abs(extreme_right_point[0] - extreme_top_right_point[0]) > 0.05 * extreme_right_point[0]:
                        digit = 2
                    else:
                        digit = 5
                rect = cv2.minAreaRect(contour_of_square)
                box = cv2.boxPoints(rect)
                box = numpy.int0(box)
                (digit_x, digit_y, digit_w, digit_h) = cv2.boundingRect(box)
                if abs(digit_y - before_digit_y) < digit_y * 0.3 and abs(
                        digit_x - before_digit_x) < digit_w + digit_w * 0.5:
                    result[len(result) - 1][0] = int(str(result[len(result) - 1][0]) + str(digit))
                else:
                    result.append([digit, self.get_contour_centroid(contour_of_square)])
                before_digit_x, before_digit_y = digit_x + (digit_w / 2), digit_y
        return result

    def crop_image_by_contour(self, image, contour):
        """
        Вырезать изображение по контуру, контур преобазуем в прямоугольник
        :param image: входное изображение
        :param contour: прямоугольный конутр
        :return: выходное изображение
        """
        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        box = numpy.int0(box)
        rect = cv2.boundingRect(box)
        return self.crop_image_by_rectangle(image, numpy.array(rect))

    def crop_image_by_rectangle(self, image, rect):
        """
        Вырезать изображение по прямоугольнику
        :param image: входное изображение
        :param rect: прямоугольник numpy.array[x y w h]
        :return: выходное изображение
        """
        pt1, pt2 = tuple(rect[:2]), tuple(rect[:2] + rect[2:])
        return self.crop_image_by_points(image, pt1, pt2)

    def add_rectangle_to_click(self, rect, click):
        return (click[0] + rect[0], click[1] + rect[1])

    def crop_image_by_points(self, image, pt1, pt2):
        """
        Вырезать изображение по точкам
        :param image: входное изображение
        :param pt1: (x,y)
        :param pt2: (x,y)
        :return: выходное изображение
        """
        return image[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    def get_contour_centroid(self, contour):
        """
        Центроид контура для клика
        :param contour: входной контуры
        :return:
        """
        moments = cv2.moments(contour)
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

    """
    Отладка
    """

    def demo_image(self, image, delay=500, text=''):
        if text:
            cv2.putText(image, text, (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        cv2.imshow("Robot Demo", image)
        cv2.waitKey(delay)

    def demo_contours(self, image, contours, delay=500, text='', color=(255, 255, 255)):
        cv2.drawContours(image, contours, -1, color, 1)
        self.demo_image(image, delay, text)

    def demo_rectangles(self, image, rectangles, delay=500, text='', color=(255, 0, 0)):
        for rect in rectangles:
            pt1, pt2 = tuple(rect[:2]), tuple(rect[:2] + rect[2:])
            cv2.rectangle(image, pt1, pt2, color=color, thickness=1)
        self.demo_image(image, delay, text)

    def demo_points(self, image, pts, delay=500, text='', color=(255, 0, 255)):
        for pt in pts:
            cv2.circle(image, pt, 3, color, -1)
        self.demo_image(image, delay, text)

    def demo_color_areas(self, delay=500):
        image = self.crop_image_by_rectangle(self.screenshot, numpy.array((self.GAME_MAIN_AREA)))
        print('BLUE ORANGE RED GREEN YELLOW')
        print('color_areas_color_count: ' + str(self.color_areas_color_count))
        for color_area_inx in range(0, len(self.color_areas)):
            color_area = self.color_areas[color_area_inx]
            print('color_area_inx: ' + str(color_area_inx))
            print('    color_inx: ' + str(color_area.color_inx))
            print('    neighbors: ' + str(color_area.neighbors))
            print('    select_color_weights: ' + str(color_area.select_color_weights))
            cv2.circle(image, color_area.click_point, 3, (0, 255, 255), -1)
            cv2.putText(image, str(color_area_inx), color_area.centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        3)
        self.demo_image(image, delay)