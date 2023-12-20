from selenium import webdriver 
from selenium.webdriver.common.by import By
import time
import csv

driver = webdriver.Chrome()
url = "https://talk.washingtonpost.com/embed/stream?storyURL=https%3A%2F%2Fwww.washingtonpost.com%2Fopinions%2F2023%2F01%2F04%2Fheat-pumps-climate-carbon-emission-revolution%2F&v=6.16.2&ts=1699910100000&initialWidth=640&childId=comments&parentTitle=Opinion%20%7C%20Heat%20pumps%20could%20be%20a%20climate%20policy%20revolution%20—%20if%20we%20let%20them%20-%20The%20Washington%20Post&parentUrl=https%3A%2F%2Fwww.washingtonpost.com%2Fopinions%2F2023%2F01%2F04%2Fheat-pumps-climate-carbon-emission-revolution%2F"
driver.get(url)
time.sleep(5)
while True:
    try:
        # Identify and click the "Load More" button
        load_more_button = driver.find_element(By.ID, 'comments-loadMore')
        time.sleep(2)
        load_more_button.click()
        time.sleep(5)
    
    except Exception as e:
        # If the "Load More" button is not found, break out of the loop
        break

elements = driver.find_elements(By.CSS_SELECTOR, '[id^="comment"]')
comments_text = elements[0].text

comments=comments_text.split("share\n")

for i in range(len(comments)-1):
    comments[i] = comments[i].split("ago\n")[1]

comments = [c for c in comments if c!="READ MORE OF THIS CONVERSATION >"]
comments = [c.replace("\n", "") for c in comments]

for i in range(len(comments)-1):
    if comments[i][0:8]=="(Edited)":
        comments[i] = comments[i][8:]

print("Before eliminating short comments, we have a total of:", len(comments), "comments.")
        
comments_long = [c for c in comments if len(c)>=20]

print("After eliminating short comments, we have a total of:", len(comments_long), "comments.")

driver.quit()

with open('comments_long.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Comments'])
    for item in comments_long:
        writer.writerow([item])