from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# set up the Chrome driver
driver = webdriver.Chrome()

# navigate to the YouTube video page
driver.get("https://www.youtube.com/watch?v=zz70o2Ia4X0")

# wait for the page to load
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.XPATH, "//h2[text()='Comments']")))

# click on the "Comments" heading to expand the comments section
driver.find_element(By.XPATH, "//h2[text()='Comments']").click()

# get all the comment elements
comments = driver.find_elements(By.CSS_SELECTOR, "#content-text")

# print out the text of each comment
for comment in comments:
    print(comment.text)

# close the driver
driver.quit()
