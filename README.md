# NOAA Space Weather Alerts Parser

This project contains a Python script to scrape alerts, watches, and warnings from the NOAA Space Weather Prediction Center website.

## Description

The `data_get.py` script fetches the latest space weather alerts from the [NOAA SWPC Alerts page](https://www.swpc.noaa.gov/products/alerts-watches-and-warnings). It parses the HTML to extract detailed information about each alert and saves the data into a JSON file named `alerts.json`.

## How It Works

1.  The script sends a request to the main alerts page.
2.  It parses the page to find links to individual alert pages.
3.  For each alert link, it fetches the content of the detailed alert page.
4.  It extracts the "Issue Time", "Message Number", and the full text of the alert.
5.  The collected data is structured and saved as a JSON array in the `alerts.json` file.

## Usage

To run the script and fetch the latest alerts, execute the following command in your terminal:

```bash
python data_get.py
```

## Dependencies

The script requires the following Python libraries:

*   `requests`
*   `beautifulsoup4`

You can install them using pip:

```bash
pip install -r requirements.txt
```

## Output

The script generates a file named `alerts.json` in the same directory. This file contains a JSON array of alert objects. Each object has the following structure:

```json
[
    {
        "issue_time": "2024 Jan 12 0123 UTC",
        "message_number": "999",
        "alert_text": "BEGIN... a lot of text ...END"
    }
]
```
