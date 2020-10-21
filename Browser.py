from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.common.by import By


class Browser:
    def __init__(self, game_url):
        self.__driver = webdriver.Firefox()
        self.__driver.get(game_url)
        wait(self.__driver, 20).until(EC.frame_to_be_available_and_switch_to_it((By.ID, "iframe-game")))
        self.__canvas = wait(self.__driver, 20).until(EC.visibility_of_element_located((By.XPATH, "/html/body/canvas")))

    def screenshot(self):
        return self.__canvas.screenshot_as_png

    def quit(self):
        self.__driver.quit()

    def click(self, click_point):
        action = webdriver.common.action_chains.ActionChains(self.__driver)
        action.move_to_element_with_offset(self.__canvas, click_point[0], click_point[1]).click().perform()
