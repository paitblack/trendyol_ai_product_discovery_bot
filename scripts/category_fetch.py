import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup

df = pd.read_csv(".csv")  

results = {
    "Ana Kategori": [],
    "Alt 1": [],
    "Alt 2": [],
    "Link": []
}

options = webdriver.ChromeOptions()
options.add_argument("--headless") 
driver = webdriver.Chrome(options=options)

for _, row in df.iterrows():
    ana_kategori = row["Ana kategori"]  
    alt1 = row["Alt 1."]               
    kategori_link = row["links"]         
    
    print(f"{ana_kategori} > {alt1} processing")

    driver.get(kategori_link)
    time.sleep(10) 

    source = driver.page_source
    soup = BeautifulSoup(source, 'html.parser')

    fdiv = soup.find("div", class_="fltrs-wrppr hide-fltrs ctgry")
    if fdiv:
        sdiv = fdiv.find("div", class_="fltrs ctgry")
        if sdiv:
            tdiv = sdiv.find("div", class_="ReactVirtualized__Grid__innerScrollContainer")
            if tdiv:
                a_tags = tdiv.find_all("a", class_="fltr-item-wrppr")
                for a in a_tags:
                    href = a.get("href")
                    text_div = a.find("div", class_="fltr-item-text ctgry")
                    alt2 = text_div.get_text(strip=True) if text_div else "N/A"
                    
                    results["Ana Kategori"].append(ana_kategori)
                    results["Alt 1"].append(alt1)
                    results["Alt 2"].append(alt2)
                    results["Link"].append("https://www.trendyol.com" + href)

driver.quit()

df_alt2 = pd.DataFrame(results)
df_alt2.to_csv("", index=False)
print("done")
