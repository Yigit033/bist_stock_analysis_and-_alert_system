# BIST Stock Analysis and Alert System

This project aims to assist investors by performing fundamental and technical analysis of stocks traded on the Borsa Istanbul (BIST). It also sends email alerts when user-defined price levels are reached.




## Description

This system is a web application developed using Streamlit. Users can add stocks they want to track, set alert preferences, optimize their portfolios, and view detailed analysis results of the stocks.

## Installation

1.  Clone this repository from GitHub:

    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://www.google.com/search?q=https://github.com/your_username/your_repository_name.git)
    ```

2.  Navigate to the project directory:

    ```bash
    cd your_repository_name
    ```

3.  Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

4.  Create a `.env` file and enter your database credentials:

    ```
    MYSQL_USER=your_username
    MYSQL_PASSWORD=your_password
    MYSQL_DATABASE=your_database_name
    MYSQL_HOST=localhost:3306
    ```

5.  Create the `prices` table in your MySQL database. You can do this by running the `create_table.py` file:

    ```bash
    python create_table.py
    ```

## Usage

1.  To start live price tracking and alert checking, run the following commands in separate terminals:

    ```bash
    python live_price_tracker.py
    python alert_checker.py
    ```

2.  To start the Streamlit application, run the following command:

    ```bash
    streamlit run my_app.py
    ```

3.  The application will automatically open in your web browser.

## Features

-   **Add/Remove Stocks:** You can add or remove stocks you want to track.
-   **Alert Settings:** You can set price levels at which you want to receive email alerts.
-   **Live Price Tracking:** You can view the live prices of the stocks you have selected.
-   **Portfolio Optimization:** You can optimize your portfolio to balance risk and return.
-   **Stock Analysis:** You can view detailed fundamental and technical analysis results for the stocks.

## Requirements

-   Python 3.7 or higher
-   Required libraries (listed in `requirements.txt`)
-   MySQL database

#
