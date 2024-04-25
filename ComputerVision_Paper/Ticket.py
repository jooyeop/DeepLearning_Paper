import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from bs4 import BeautifulSoup


webdriver = selenium.webdriver
chrome_options = Options()
# chrome_options.add_argument("--headless")  # 이 줄을 주석 처리합니다.

url = 'https://www.yes24.com/Templates/FTLogin.aspx?ReturnURL=http://ticket.yes24.com/Special/49438'
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)

time.sleep(1)

account = 'joo0831'
password = 'joo574301!'

id_selector = '//*[@id="SMemberID"]'
driver.find_element(By.XPATH, id_selector).click()
driver.find_element(By.XPATH, id_selector).send_keys(account)

time.sleep(1)

password_selector = '//*[@id="SMemberPassword"]'
driver.find_element(By.XPATH, password_selector).click()
driver.find_element(By.XPATH, password_selector).send_keys(password)

time.sleep(1)

login_selector = '//*[@id="btnLogin"]/span/em'
driver.find_element(By.XPATH, login_selector).click()

time.sleep(1)

driver.get('http://ticket.yes24.com/Special/49438')

url = 'https://time.navyism.com/?host=ticket.yes24.com'

# 서버로부터 응답을 받아오기
response = requests.get(url)

# 웹 페이지 접속
chrome_options2 = Options()
chrome_options2.add_argument("--headless")
driver2 = webdriver.Chrome(options=chrome_options2)
driver2.get('https://time.navyism.com/?host=ticket.yes24.com')
Time = False

chech = '//*[@id="nowarn_check"]'
driver2.find_element(By.XPATH, chech).click()

while True:
    if Time == True:
        break
    time_area = driver2.find_element(By.ID, "time_area")
    if time_area:
        print(time_area.get_attribute('innerHTML'))
        if time_area.get_attribute('innerHTML') == '2024년 04월 25일 20시 00분 00초':
            Time = True
            driver.execute_script("jsf_pdi_GoPerfSale();")
            time.sleep(800000000)
    else:
        print("Element with id 'time_area' not found.")