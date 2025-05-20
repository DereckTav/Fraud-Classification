import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, StaleElementReferenceException
from tqdm import tqdm
import random
import os

# --- CONFIGURATION ---
CHROMEDRIVER_PATH = "C:/Users/derec/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"
BRAVE_PATH = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
INPUT_CSV = "train/for_automation_train.csv"
OUTPUT_CSV = "train/output_train.csv"
TEXT_COLUMN = "review"  # Column with the text to process
RETRY_COUNT = 5  # Increased retry count
WAIT_SECONDS = 5
MAX_WAIT_TIME = 40  # Increased max wait time

# --- Setup Brave Browser ---
options = Options()
options.binary_location = BRAVE_PATH
    # Add an option to disable headless mode for debugging
# options.add_argument("--headless=new")  # Use the new headless mode
    
    # Uncomment this line to see the browser window (for debugging)
    # options.add_argument("--headless=false")
options.add_argument("--ignore-certificate-errors")
options.add_argument("--disable-gpu")
options.add_argument("--disable-extensions")
options.add_argument("--log-level=3")
options.add_argument("--disable-notifications")  # Block notifications
options.add_argument("--disable-popup-blocking")  # Handle popups
options.add_argument("--disable-infobars")  # Disable info bars
options.add_experimental_option("excludeSwitches", ["enable-automation"])  # Make browser less detectable

service = Service(executable_path=CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 20)

# --- Load CSV ---
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"CSV loaded successfully with {len(df)} rows")
    
    # Display CSV structure and first few rows for debugging
    print("\nCSV Structure:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head(2).to_string())
    
    # Verify the TEXT_COLUMN exists
    if TEXT_COLUMN not in df.columns:
        print(f"ERROR: Column '{TEXT_COLUMN}' not found in CSV! Available columns: {df.columns.tolist()}")
        available_text_columns = [col for col in df.columns if df[col].dtype == object]
        if available_text_columns:
            suggested_column = available_text_columns[0]
            print(f"Suggesting to use '{suggested_column}' as TEXT_COLUMN instead.")
            TEXT_COLUMN = suggested_column
        else:
            raise ValueError(f"No suitable text columns found in {INPUT_CSV}")
    
    # Add debugging - print the actual content in the TEXT_COLUMN
    print("\nSample text from TEXT_COLUMN:")
    for i, sample_text in enumerate(df[TEXT_COLUMN].head(3)):
        print(f"Row {i+1}: {sample_text[:100]}{'...' if len(sample_text) > 100 else ''}")
    
    # For testing purposes
    # df = df.head(5)
    
except Exception as e:
    print(f"Error loading CSV: {str(e)}")
    exit(1)

url = "https://toolbaz.com/writer/review-generator"

# Add custom prompt to guide the generation
PROMPT_PREFIX = "Write a concise, single-paragraph product review based on this: "

# --- Helper Functions ---
def handle_popups():
    """Try to close common popup types"""
    try:
        # Look for common popup close buttons (×, Close, etc.)
        close_buttons = driver.find_elements(By.XPATH, 
            "//*[contains(text(), 'Close') or contains(text(), 'close') or contains(text(), '×')]")
        
        # Also look for elements that might be close buttons by their class names
        close_by_class = driver.find_elements(By.CSS_SELECTOR, 
            ".close, .closeButton, .close-button, .popup-close, button[aria-label='Close']")
        
        all_possible_close_buttons = close_buttons + close_by_class
        
        for button in all_possible_close_buttons:
            try:
                if button.is_displayed():
                    print("Found popup, attempting to close...")
                    button.click()
                    time.sleep(1)
                    return True
            except Exception:
                pass
        
        # Try pressing ESC key to close popups
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        time.sleep(0.5)
        
        return False
    except Exception as e:
        print(f"Error while handling popups: {str(e)}")
        return False

def refresh_page():
    """Refresh the page and wait for it to load"""
    print("Refreshing page...")
    driver.refresh()
    time.sleep(random.uniform(2, 4))  # Randomized wait to seem more human-like

def extract_first_paragraph(text):
    """Extract only the first paragraph from a multi-paragraph text."""
    # First try to split by double newlines (typical paragraph separator)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        return paragraphs[0].strip()
    
    # If that doesn't work, try single newlines
    paragraphs = text.split('\n')
    if len(paragraphs) > 1:
        return paragraphs[0].strip()
    
    # If there are no newlines, just return the text
    return text.strip()

def generate_review(text):
    for attempt in range(RETRY_COUNT):
        try:
            # Navigate to the page
            driver.get(url)
            
            # Add some randomized delay to appear more human-like
            time.sleep(random.uniform(2, 4))
            
            # Check and handle any popups first
            handle_popups()
            
            # Wait for input field to be properly loaded
            try:
                textarea = wait.until(EC.presence_of_element_located((By.ID, "input")))
            except TimeoutException:
                print("Timeout waiting for input field. Refreshing...")
                refresh_page()
                textarea = wait.until(EC.presence_of_element_located((By.ID, "input")))
            
            # Clear any existing text
            textarea.clear()
            time.sleep(0.5)  # Give it a moment
            
            # Different input methods to try
            if attempt == 0:
                # Method 1: JavaScript direct value setting
                driver.execute_script("arguments[0].value = arguments[1];", textarea, text)
                driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", textarea)
            elif attempt == 1:
                # Method 2: Send keys with small chunks to avoid issues
                for chunk in [text[i:i+50] for i in range(0, len(text), 50)]:
                    textarea.send_keys(chunk)
                    time.sleep(0.1)
            else:
                # Method 3: Combined approach
                textarea.send_keys(text[:100])  # First part with send_keys
                driver.execute_script(f"arguments[0].value = arguments[1];", 
                                      textarea, text)  # Rest with JS
            
            # Trigger events to ensure text is recognized
            textarea.click()
            textarea.send_keys(" " + Keys.BACKSPACE)
            
            # Verify the text is in the input field
            actual_value = driver.execute_script("return arguments[0].value;", textarea)
            print(f"Input length: {len(text)}, Actual input length: {len(actual_value)}")
            
            if len(actual_value) < len(text) * 0.8:  # Text got truncated
                print(f"WARNING: Text significantly truncated. Will retry with different method.")
                # We'll retry with a different method in the next iteration
                time.sleep(1)
                continue
            
            # Handle any new popups that might appear
            handle_popups()
            
            try:
                # Wait for and click generate button
                generate_button = wait.until(EC.element_to_be_clickable((By.ID, "main_btn")))
                generate_button.click()
            except ElementClickInterceptedException:
                # Something is blocking the button - handle popups and try again
                handle_popups()
                time.sleep(1)
                generate_button = wait.until(EC.element_to_be_clickable((By.ID, "main_btn")))
                generate_button.click()
            
            # Wait for output with progressively more aggressive checks
            start_time = time.time()
            content_found = False
            
            while time.time() - start_time < MAX_WAIT_TIME and not content_found:
                # Handle any popups that might appear during generation
                handle_popups()
                
                # Try to find output paragraphs with improved detection methods
                # Method 1: Look for the specific generated content element
                try:
                    # Wait for the specific element containing the generated review
                    generated_element = wait.until(EC.presence_of_element_located(
                        (By.XPATH, "//div[@id='output']//div[contains(@class, 'text-container')]//div[@adcopy]")
                    ))
                    output_content = generated_element.text.strip()
                    
                    if output_content and len(output_content) > 20 and output_content != text.strip():
                        single_paragraph = extract_first_paragraph(output_content)
                        print(f"SUCCESS (Method 1): Generated review ({len(single_paragraph)} chars)")
                        return single_paragraph
                except Exception as e:
                    print(f"Method 1 failed: {str(e)}")
                    
                    # Method 2: Look for paragraphs within the output div
                    try:
                        output_div = driver.find_element(By.ID, "output")
                        paragraphs = output_div.find_elements(By.CSS_SELECTOR, "p")
                        
                        if paragraphs:
                            for p in paragraphs:
                                content = p.text.strip()
                                if content and content != text.strip() and len(content) > 20:
                                    single_paragraph = extract_first_paragraph(content)
                                    print(f"SUCCESS (Method 2): Generated review ({len(single_paragraph)} chars)")
                                    return single_paragraph
                    except Exception as e:
                        print(f"Method 2 failed: {str(e)}")
                    
                    # Method 3: Look for any content that appeared after generation
                    try:
                        # Look for elements that might contain the generated content
                        possible_output_elements = driver.find_elements(By.CSS_SELECTOR, 
                            "#output, .output, .generated-text, .result, .response, div.text-container")
                        
                        for elem in possible_output_elements:
                            content = elem.text.strip()
                            if content and content != text.strip() and len(content) > 20:
                                single_paragraph = extract_first_paragraph(content)
                                print(f"SUCCESS (Method 3): Generated review ({len(single_paragraph)} chars)")
                                return single_paragraph
                    except Exception as e:
                        print(f"Method 3 failed: {str(e)}")
                    
                    # Method 4: Check for any paragraph that appeared anywhere
                    try:
                        all_paragraphs = driver.find_elements(By.CSS_SELECTOR, "p")
                        for p in all_paragraphs:
                            try:
                                content = p.text.strip()
                                # Skip empty paragraphs and original text
                                if content and content != text.strip() and len(content) > 20:
                                    single_paragraph = extract_first_paragraph(content)
                                    print(f"SUCCESS (Method 4): Generated review ({len(single_paragraph)} chars)")
                                    return single_paragraph
                            except StaleElementReferenceException:
                                # Element became stale, will retry on next iteration
                                pass
                    except Exception as e:
                        print(f"Method 4 failed: {str(e)}")
                    
                    # Method 5: Take a screenshot and log page source for debugging
                    if time.time() - start_time > MAX_WAIT_TIME * 0.75:  # If we're close to timeout
                        try:
                            # driver.save_screenshot(f"train/output_waiting_{attempt}.png")
                            print("Saved screenshot while waiting for output")
                            
                            # Save page source
                            with open(f"page_source_{attempt}.html", "w", encoding="utf-8") as f:
                                f.write(driver.page_source)
                            print("Saved page source for debugging")
                        except Exception as sc_err:
                            print(f"Error saving debug info: {str(sc_err)}")
                            
                except Exception as inner_e:
                    print(f"Error while checking output: {str(inner_e)}")
                except Exception as inner_e:
                    print(f"Error while checking output: {str(inner_e)}")
                
                # Wait before checking again
                time.sleep(1)
            
            print(f"WARNING: Timed out waiting for generated content (attempt {attempt+1})")
            if attempt < RETRY_COUNT - 1:
                print("Trying a refresh before next attempt")
                refresh_page()
            
        except Exception as e:
            print(f"[Retry {attempt + 1}] Error: {str(e)}")
            # Take a screenshot to help debug
            try:
                screenshot_path = f"error_screenshot_{attempt}.png"
                driver.save_screenshot(screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")
            except:
                pass
            
            # Wait and try again
            time.sleep(2)
    
    print("ERROR: All retry attempts failed")
    return "[ERROR: Generation failed after multiple attempts]"

# --- Process with Progress Bar ---
successful_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Reviews"):
    print(f"\n{'='*50}\nProcessing review {idx+1}/{len(df)}")
    
    # Try to get the review text - make sure we handle quotes properly
    # Some CSV parsers might have issues with quoted text containing commas
    try:
        input_text = str(row[TEXT_COLUMN]).strip()
    except Exception as e:
        print(f"Error accessing review text: {str(e)}")
        print(f"Row data: {row}")
        # Try to access it as a raw string
        input_text = str(row).strip()
    
    # Remove extra quotes that might be present from CSV parsing
    input_text = input_text.strip('"\'')
    
    # Debug the input text
    print(f"Raw input text: {input_text[:100]}...")
    
    # Check if this looks like valid input text
    if input_text.startswith("Example:") or len(input_text) < 10:
        print(f"WARNING: Input text appears to be a placeholder or too short: '{input_text}'")
        print("This might indicate a problem with your CSV format or column selection.")
        
        # Let's try to create a prompt from other available information
        # For example, if we have product type, we can use that
        if 'product_type' in row:
            product_type = row['product_type']
            suggested_prompt = f"Write a detailed positive review for a {product_type} product"
            print(f"Attempting to use alternative prompt based on product type: '{suggested_prompt}'")
            input_text = suggested_prompt
    
    print(f"Original text: {input_text[:50]}... ({len(input_text)} chars)")
    
    # Add the prompt prefix to guide the review generation
    input_with_prompt = PROMPT_PREFIX + input_text
    
    # For longer reviews, truncate to reasonable length if needed
    if len(input_with_prompt) > 500:
        print(f"Input text is very long ({len(input_with_prompt)} chars). Truncating...")
        input_with_prompt = input_with_prompt[:500] + "..."
    
    # Try to generate the review
    generated_text = generate_review(input_with_prompt)
    
    if not generated_text.startswith("[ERROR"):
        # Create a copy of the row to avoid SettingWithCopyWarning
        new_row = row.copy()
        # Create a new column for the generated text (keep original intact for reference)
        new_row['generated_review'] = generated_text
        successful_rows.append(new_row)
        
        # Save progress after each successful generation
        # Create temp dataframe with successful rows so far
        temp_df = pd.DataFrame(successful_rows)
        temp_df.to_csv(f"tmp_{os.path.dirname(OUTPUT_CSV)}", index=False)
        print(f"✓ Progress saved: {len(successful_rows)}/{len(df)} reviews in tmp_{OUTPUT_CSV}")
    else:
        print(f"✗ FAILED: Could not generate review for row {idx+1}")
    
    # Random sleep time to reduce detection
    sleep_time = random.uniform(WAIT_SECONDS, WAIT_SECONDS + 3)
    print(f"Waiting {sleep_time:.1f} seconds before next request...")
    time.sleep(sleep_time)

# Create new DataFrame with successful rows only
if successful_rows:
    output_df = pd.DataFrame(successful_rows)
    
    # Save it
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Done! {len(output_df)} successful reviews saved to: {OUTPUT_CSV}")
    
    # Display some sample generated reviews
    if 'generated_review' in output_df.columns:
        print("\nSample generated reviews:")
        for i, review in enumerate(output_df['generated_review'].head(3)):
            print(f"Review {i+1}: {review[:100]}...")
else:
    print("\n⚠️ No successful reviews were generated!")

# Try to clean up browser and display summary
try:
    driver.quit()
    print("Browser closed successfully.")
except:
    print("Note: Browser may still be running. Please close it manually.")

print(f"\nSUMMARY:")
print(f"- Total reviews attempted: {len(df)}")
print(f"- Successfully generated: {len(successful_rows)}")
print(f"- Failed: {len(df) - len(successful_rows)}")
if len(successful_rows) > 0:
    print(f"- Output file: {OUTPUT_CSV}")
    if 'generated_review' in output_df.columns:
        print(f"- First successful review: {output_df['generated_review'].iloc[0][:100]}...")