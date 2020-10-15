from Robot import Robot
from Browser import Browser
if __name__ == '__main__':
    robot = Robot()
    browser = Browser("https://www.miniclip.com/games/coloruid-2/en/#")
    while not robot.emergency_timeout():
        click_point = robot.run(browser.screenshot())
        if click_point is not False:
            browser.click(click_point)
    browser.quit()