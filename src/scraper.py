import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from config import CHROMEDRIVER_PATH

def get_exercise_text(driver, url):
    """
    Use Selenium to open the given URL and extract the text content
    inside the div with CSS class 'entry clearfix'.
    Returns the stripped text if successful, or None if an error occurs.
    """
    try: 
        driver.get(url)
        time.sleep(2)  # Wait for the page to load
        content_div = driver.find_element(By.CSS_SELECTOR, 'div.entry.clearfix')
        return content_div.text.strip()
    except Exception:
        return None

def crawl_level(driver, level):
    """
    Crawl JLPT reading exercise pages for a specific JLPT level.
    The function tries multiple URL patterns and pages until
    a maximum number of consecutive failures occurs.
    Returns a list of dicts with 'url', 'text', and 'level'.
    """
    base_url_1 = f"https://japanesetest4you.com/japanese-language-proficiency-test-jlpt-{level}-reading-exercise-{{}}"
    base_url_2 = f"https://japanesetest4you.com/jlpt-{level}-reading-exercise-{{}}"
    max_failures = 4  # Stop crawling after 4 consecutive failures
    failure_count = 0
    results = []
    i = 1

    print(f"\nStart of JLPT crawling JLPT {level.upper()}")

    while failure_count < max_failures:
        # Create URL variants with and without leading zero for pages 1-9
        if i < 10:
            urls_to_test = [
                base_url_1.format(f"0{i}/"), base_url_1.format(f"{i}/"),
                base_url_2.format(f"0{i}/"), base_url_2.format(f"{i}/")
            ]
        else:
            urls_to_test = [
                base_url_1.format(f"{i}/"), base_url_2.format(f"{i}/")
            ]

        for url in urls_to_test:
            print(f"URL Test: {url}")
            text = get_exercise_text(driver, url)
            # If text exists and is reasonably long, consider it valid
            if text and len(text) > 100:
                print("  OK")
                results.append({"url": url, "text": text, "level": level.upper()})
                failure_count = 0  # Reset failure count on success
                break
        else:
            failure_count += 1  # Increment failure count if all URLs fail
            print(f"Failure {failure_count} / {max_failures}")

        i += 1  # Go to next page number

    print(f"End of JLPT crawl {level.upper()}, {len(results)} texts scraped\n")
    return results

def main():
    # Setup Chrome driver service and options
    service = Service(CHROMEDRIVER_PATH)
    options = webdriver.ChromeOptions()

    driver = webdriver.Chrome(service=service, options=options)

    levels = ['n1', 'n2', 'n3', 'n4', 'n5']  # JLPT levels to crawl
    all_results = []

    # Crawl exercises for each JLPT level and collect results
    for level in levels:
        results = crawl_level(driver, level)
        all_results.extend(results)

    driver.quit()  # Close the browser when done

    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv('jlpt_reading_exercises_n1_to_n5.csv', index=False, encoding='utf-8')

    print("Finished, results saved in jlpt_reading_exercises_n1_to_n5.csv")

if __name__ == "__main__":  #
    main()
