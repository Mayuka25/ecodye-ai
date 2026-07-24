from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1400})
    page.goto("http://localhost:8010/")
    # wait for websocket data to populate (a few ticks, ~1.5s each)
    time.sleep(8)
    page.screenshot(path="/home/claude/ecodye_prototype/preview_full.png", full_page=True)

    # click a different factory card to show interactivity
    page.click("#fc-erode")
    time.sleep(2)
    page.screenshot(path="/home/claude/ecodye_prototype/preview_erode.png", full_page=True)

    browser.close()
print("done")
