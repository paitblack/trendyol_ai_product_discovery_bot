import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

df = pd.read_csv("trendyol_tüm_alt_kategori_linkleri.csv")

results = {
    "Ana Kategori": [],
    "Alt 1": [],
    "Alt 2": [],
    "Marka": [],
    "Marka Link": []
}

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

for _, row in df.iterrows():
    ana_kategori = row["Ana Kategori"]
    alt1 = row["Alt 1"]
    alt2 = row["Alt 2"]
    kategori_link = row["Link"]

    print(f"{ana_kategori} > {alt1} > {alt2} processing...")

    driver.get(kategori_link)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "fltrs-wrppr"))
        )
    except:
        print(f"{kategori_link} skipping...")
        continue

    source = driver.page_source
    soup = BeautifulSoup(source, 'html.parser')

    filter_sections = soup.find_all("div", class_="fltrs-wrppr hide-fltrs")

    marka_block = None
    for section in filter_sections:
        if section.find("div", attrs={"data-title": "Marka"}):
            marka_block = section
            break

    if not marka_block:
        print(f"{kategori_link}  brand filter not found.")
        continue

    marka_grid = marka_block.find("div", class_="ReactVirtualized__Grid__innerScrollContainer")
    if not marka_grid:
        print(f"{kategori_link} brand grind not found")
        continue

    a_tags = marka_grid.find_all("a", class_="fltr-item-wrppr")
    for a in a_tags:
        href = a.get("href")
        marka_div = a.find("div", class_="fltr-item-text")
        marka_adi = marka_div.get_text(strip=True) if marka_div else "N/A"
        marka_link = "https://www.trendyol.com" + href if href else "N/A"

        results["Ana Kategori"].append(ana_kategori)
        results["Alt 1"].append(alt1)
        results["Alt 2"].append(alt2)
        results["Marka"].append(marka_adi)
        results["Marka Link"].append(marka_link)

    print(f"{len(a_tags)} branf found")

driver.quit()

df_results = pd.DataFrame(results)
df_results.to_csv("trendyol_markalar.csv", index=False, encoding="utf-8-sig")

print("done. -> trendyol_markalar.csv")
