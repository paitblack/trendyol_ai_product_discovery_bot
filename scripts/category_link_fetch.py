import os
import json
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

options = Options()
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--log-level=3") 
driver = webdriver.Chrome(options=options)

df = pd.read_csv("")
links = df["link"]
categories = df["category"]

for index, link in enumerate(links):
   category = categories[index]
   print(f"Processing: {category} ({link})")

   folder_path = os.path.join(category)
   os.makedirs(folder_path, exist_ok=True)

   driver.get(link)
   time.sleep(3)

   scroll_pause = 1
   scroll_amount = 3432
   max_scrolls = 250
   page_counter = 1

   for i in range(max_scrolls):
       print(f"Scroll #{i+1}")
       last_height = driver.execute_script("return document.body.scrollHeight")
       driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
       time.sleep(scroll_pause)
       new_height = driver.execute_script("return document.body.scrollHeight")

       if new_height == last_height:
           break
    
       scroll_amount = new_height-last_height
       current_url = driver.current_url

       if current_url.strip() == link.strip():
            break

       product_cards = driver.find_elements(By.CSS_SELECTOR, "div.prdct-cntnr-wrppr div.p-card-wrppr")
       new_products = product_cards[-24:] 

       products = []
       for card in new_products:
           try:
               title = card.get_attribute("title") or ""
               title = title.strip()
               
               link_element = card.find_element(By.CSS_SELECTOR, "a.p-card-chldrn-cntnr")
               href = link_element.get_attribute("href")
               product_link = href if href else ""
               
               products.append({"title": title, "link": product_link})

           except NoSuchElementException:
               print("no prod., skip")
               continue
           except Exception as e:
               print(f"err: {e}")
               continue

       if not products:
           print("no new prod. stop.")
           break

       filename = f"products_{page_counter}.json"
       filepath = os.path.join(folder_path, filename)
       with open(filepath, "w", encoding="utf-8") as f:
           json.dump(products, f, ensure_ascii=False, indent=2)
       print(f"saved: {filename} ({len(products)} pr.)")

       page_counter += 1

       if i % 50 == 0 and i > 0:
           driver.refresh()
           time.sleep(3)

   print(f"{category} done. {page_counter-1} pages")

driver.quit()