from Robot import Robot
from Browser import Browser
if __name__ == '__main__':
    robot = Robot()
    browser = Browser("https://www.miniclip.com/games/coloruid-2/en/#")
    while not robot.emergency_timeout():
        robot.run(browser)
    browser.quit()